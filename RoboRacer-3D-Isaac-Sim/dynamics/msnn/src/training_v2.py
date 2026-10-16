"""
training_v2.py — Enhanced training with 4 improvements:
  Task 1: Weighted loss  (loss_weights param)
  Task 2: Scheduled Sampling  (scheduled_sampling param)
  Task 3: Noise injection  (noise_scale param)
  Task 4: Multi-step Rollout Loss  (multistep_n + multistep_lambda params)

All new params are optional with safe defaults, so existing run scripts work unchanged.
"""
import torch
import torch.nn.functional as F
import random
import numpy as np
import wandb
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
import copy
import RNN_architectures
from Dataset_Classes import *
from tqdm import tqdm
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Unchanged helper functions (copied verbatim) ────────────────────────────

def evaluate_model(model, dataloader, device, history, output_indices):
    model.eval()
    se_sum = None
    ae_sum = None
    count = 0
    with torch.no_grad():
        for inputs_seq, targets_seq in dataloader:
            inputs_seq  = inputs_seq.to(device)
            targets_seq = targets_seq.to(device)
            if inputs_seq.dim() == 2:
                inputs_seq  = inputs_seq.unsqueeze(0)
                targets_seq = targets_seq.unsqueeze(0)
            B, T, F = inputs_seq.shape
            H = history
            if T < H:
                continue
            Dy = targets_seq.size(-1)
            if se_sum is None:
                se_sum = torch.zeros(Dy, device=device)
                ae_sum = torch.zeros(Dy, device=device)
            inputs = inputs_seq[:, :H, :]
            for t in range(H - 1, T):
                outputs = model(inputs)
                targets = targets_seq[:, t, :]
                diff = outputs - targets
                se_sum += (diff.pow(2)).sum(dim=0)
                ae_sum += diff.abs().sum(dim=0)
                count += B
                if t == T - 1:
                    break
                next_inputs = inputs_seq[:, t + 1, :].clone()
                next_inputs[:, output_indices] = outputs
                inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)
    rmse_per_output = torch.sqrt(se_sum / max(1, count)).detach().cpu().numpy()
    mae_per_output  = (ae_sum / max(1, count)).detach().cpu().numpy()
    return rmse_per_output, mae_per_output


def evaluate_model_denormalized(model, dataloader, device, tgt_mean, tgt_std, history, output_indices):
    model.eval()
    se_sum = None
    ae_sum = None
    count = 0
    tgt_mean_tensor = torch.tensor(tgt_mean, dtype=torch.float32, device=device)
    tgt_std_tensor  = torch.tensor(tgt_std,  dtype=torch.float32, device=device)
    with torch.no_grad():
        for inputs_seq, targets_seq in dataloader:
            inputs_seq  = inputs_seq.to(device)
            targets_seq = targets_seq.to(device)
            if inputs_seq.dim() == 2:
                inputs_seq  = inputs_seq.unsqueeze(0)
                targets_seq = targets_seq.unsqueeze(0)
            B, T, F = inputs_seq.shape
            H = history
            if T < H:
                continue
            Dy = targets_seq.size(-1)
            if se_sum is None:
                se_sum = torch.zeros(Dy, device=device)
                ae_sum = torch.zeros(Dy, device=device)
            inputs = inputs_seq[:, :H, :]
            for t in range(H - 1, T):
                outputs = model(inputs)
                targets = targets_seq[:, t, :]
                outputs_den = outputs * tgt_std_tensor + tgt_mean_tensor
                targets_den = targets * tgt_std_tensor + tgt_mean_tensor
                diff = outputs_den - targets_den
                se_sum += (diff.pow(2)).sum(dim=0)
                ae_sum += diff.abs().sum(dim=0)
                count += B
                if t == T - 1:
                    break
                next_inputs = inputs_seq[:, t + 1, :].clone()
                next_inputs[:, output_indices] = outputs
                inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)
    rmse_per_output = torch.sqrt(se_sum / max(1, count)).detach().cpu().numpy()
    mae_per_output  = (ae_sum / max(1, count)).detach().cpu().numpy()
    return rmse_per_output, mae_per_output


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_normalization_stats(csv_files, input_columns):
    all_data = []
    for path in csv_files:
        df = pd.read_csv(path)
        missing = [c for c in input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        data = df[input_columns].to_numpy(dtype=np.float32)
        if np.isnan(data).any():
            raise ValueError(f"{path} contains NaN values.")
        all_data.append(data)
    combined_data = np.vstack(all_data)
    data_mean = combined_data.mean(axis=0)
    data_std  = combined_data.std(axis=0)
    return data_mean, data_std


# ── Modified train_network ───────────────────────────────────────────────────

def train_network(params):
    """Train a model — v2 with optional weighted loss, scheduled sampling,
    noise injection, and multi-step rollout loss.

    New optional params (all have safe defaults):
      loss_weights (list[float]): per-output loss weights [v_x, v_y, ψ̇, a_x, a_y].
                                   Default [1,1,1,1,1] (uniform).
      scheduled_sampling (bool):  linearly decay teacher-forcing ratio from 1→0.
                                   Default False (always use predictions, same as v1).
      noise_scale (float):        std of Gaussian noise added to training inputs.
                                   Default 0.0 (no noise).
      multistep_n (int):          number of steps for auxiliary rollout loss. 0 = off.
                                   Default 0.
      multistep_lambda (float):   weight of multi-step auxiliary loss. Default 0.1.
    """
    required_keys = [
        "model_name", "random_seed", "learning_rate", "batch_size", "num_epochs",
        "criterion", "optimizer", "gradient_clipping_value", "patience", "history",
        "train_files", "val_files", "test_files", "auto_regression_files",
        "outputs_columns", "inputs_columns", "normed", "weight_decay", "seq_length"
    ]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

    # ── New feature flags ──────────────────────────────────────────────────
    loss_weights       = params.get('loss_weights', None)        # Task 1
    scheduled_sampling = params.get('scheduled_sampling', False)  # Task 2
    noise_scale        = params.get('noise_scale', 0.0)           # Task 3
    multistep_n        = params.get('multistep_n', 0)             # Task 4
    multistep_lambda   = params.get('multistep_lambda', 0.1)      # Task 4

    model_name   = params['model_name']
    random_seed  = params['random_seed']
    if random_seed is not None:
        set_seed(random_seed)

    # ── File lists (unchanged logic) ───────────────────────────────────────
    train_files1 = ['processed_data/stanley_0.4_20ms.csv',
                    'processed_data/manual_mode_counterclockwise_20ms_cut.csv',
                    'processed_data/clockwise_20ms.csv',
                    'processed_data/counterclockwise_20ms.csv']
    train_files2 = train_files1 + ['processed_data/Car1_Circles_20ms.csv',
                                    'processed_data/Car1_Oscillation_20ms.csv']
    train_files3 = train_files2 + ['processed_data/Car1_JustDrivingAround_20ms.csv']
    train_files4 = train_files3 + ['processed_data/Car1_BackwardDriving_20ms.csv']
    pf = params['train_files']
    if   pf == 1: train_files = train_files1
    elif pf == 2: train_files = train_files2
    elif pf == 3: train_files = train_files3
    elif pf == 4: train_files = train_files4
    else:         train_files = pf

    val_files            = params['val_files']
    test_files           = params['test_files']
    auto_regression_files = params['auto_regression_files']

    history         = params['history']
    outputs_columns = params['outputs_columns']
    inputs_columns  = params['inputs_columns']
    output_indices  = [inputs_columns.index(c) for c in outputs_columns]
    normed          = params['normed']
    seq_length      = params['seq_length']

    for col in outputs_columns:
        if col not in inputs_columns:
            raise ValueError(f"Output column '{col}' must be included in input_columns.")

    data_mean, data_std = compute_normalization_stats(train_files, inputs_columns)
    safe_std = np.where(data_std < 1e-6, 1.0, data_std)

    train_stride = params.get('stride', 1)
    if normed:
        train_dataset = RNN_Seq_Dataset_norm(train_files, inputs_columns, outputs_columns, seq_length, data_mean, safe_std, stride=train_stride)
        val_dataset   = RNN_Seq_Dataset_norm(val_files,   inputs_columns, outputs_columns, seq_length, data_mean, safe_std)
        test_dataset  = RNN_Seq_Dataset_norm(test_files,  inputs_columns, outputs_columns, seq_length, data_mean, safe_std)
    else:
        train_dataset = RNN_Seq_Dataset(train_files, inputs_columns, outputs_columns, seq_length, stride=train_stride)
        val_dataset   = RNN_Seq_Dataset(val_files,   inputs_columns, outputs_columns, seq_length)
        test_dataset  = RNN_Seq_Dataset(test_files,  inputs_columns, outputs_columns, seq_length)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pin_memory = (device.type == "cuda")

    input_size  = len(inputs_columns)
    output_size = len(outputs_columns)

    # ── Task 1: build weight tensor ────────────────────────────────────────
    if loss_weights is not None:
        assert len(loss_weights) == output_size, \
            f"loss_weights length {len(loss_weights)} != output_size {output_size}"
        w_tensor = torch.tensor(loss_weights, dtype=torch.float32, device=device)
        # normalise so average weight = 1 (keeps loss scale comparable)
        w_tensor = w_tensor / w_tensor.mean()
        print(f"[Task1] Weighted loss enabled: {loss_weights} → normalised {w_tensor.tolist()}")
    else:
        w_tensor = None

    if scheduled_sampling:
        print(f"[Task2] Scheduled sampling enabled (tf_ratio 1.0 → 0.0 over {params['num_epochs']} epochs)")
    if noise_scale > 0:
        print(f"[Task3] Noise injection enabled: scale={noise_scale}")
    if multistep_n > 0:
        print(f"[Task4] Multi-step rollout loss enabled: N={multistep_n}, λ={multistep_lambda}")

    if params.get('pretrained_model') is not None:
        print(f"Using pretrained model: {params['pretrained_model']}")
        pretrained_model_data = torch.load(params['pretrained_model'], weights_only=False)
        pretrained_model_cfg  = pretrained_model_data['config']
        assert input_size   == pretrained_model_cfg['input_size']
        assert output_size  == pretrained_model_cfg['output_size']
        assert normed       == pretrained_model_cfg['normed']
        assert history      == pretrained_model_cfg['history']
        assert model_name   == pretrained_model_cfg['model_class']
        assert inputs_columns  == pretrained_model_cfg['input_columns']
        assert outputs_columns == pretrained_model_cfg['output_columns']
        model = getattr(RNN_architectures, pretrained_model_cfg['model_class'])(
            input_size=input_size, output_size=output_size).to(device)
        model.load_state_dict(pretrained_model_data['model_state_dict'])
    else:
        model = getattr(RNN_architectures, params['model_name'])(
            input_size=input_size, output_size=output_size).to(device)

    batch_size    = params['batch_size']
    num_epochs    = params['num_epochs']
    learning_rate = params['learning_rate']
    weight_decay  = params['weight_decay']

    if params['criterion'] == 'MSE':
        base_criterion = torch.nn.MSELoss(reduction='none')
    elif params['criterion'] == 'Huber':
        base_criterion = torch.nn.HuberLoss(reduction='none')
    else:
        raise ValueError(f"Unknown criterion: {params['criterion']}")

    if params['optimizer'] == 'NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {params['optimizer']}")

    gradient_clipping_value = params['gradient_clipping_value']
    patience        = params['patience']
    best_val_loss   = float('inf')
    patience_counter = 0
    best_model_state = None

    n_workers  = params.get('num_workers', 4)
    pf_factor  = params.get('prefetch_factor', 2) if n_workers > 0 else None
    persist    = (n_workers > 0) and pin_memory

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=n_workers, pin_memory=pin_memory,
                                  persistent_workers=persist, prefetch_factor=pf_factor)
    val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=n_workers, pin_memory=pin_memory,
                                  persistent_workers=persist, prefetch_factor=pf_factor)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=n_workers, pin_memory=pin_memory,
                                  persistent_workers=persist, prefetch_factor=pf_factor)

    for inputs, targets in train_dataloader:
        print("Inputs:", inputs.shape)
        print("Targets:", targets.shape)
        break

    os.makedirs('trained_models', exist_ok=True)
    path_best_model = f"{wandb.run.name}___{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pth"
    wandb.config.update({
        "architecture": model.__class__.__name__,
        "dataset": train_dataset.__class__.__name__,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": optimizer.__class__.__name__,
        "criterion": params['criterion'],
        "inputs": inputs_columns,
        "outputs": outputs_columns,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "patience": patience,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "auto_reg_files": auto_regression_files,
        "gradient_clipping_value": gradient_clipping_value,
        "num_previous_states": history,
        "data_mean": data_mean.tolist(),
        "data_std": safe_std.tolist(),
        "seq_length": seq_length,
        "saved_model": path_best_model,
        "random_seed": random_seed,
        # v2 extras
        "loss_weights": loss_weights,
        "scheduled_sampling": scheduled_sampling,
        "noise_scale": noise_scale,
        "multistep_n": multistep_n,
        "multistep_lambda": multistep_lambda,
    }, allow_val_change=True)
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("test_loss", summary="min")

    H          = history
    accum_every = 100
    pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in pbar:
        # ── Task 2: compute teacher-forcing ratio for this epoch ───────────
        if scheduled_sampling:
            tf_ratio = max(0.0, 1.0 - epoch / max(1, num_epochs - 1))
        else:
            tf_ratio = 0.0   # always use predictions (original behaviour)

        model.train()
        train_loss_sum = 0.0
        train_batches  = 0

        for inputs_seq, targets_seq in tqdm(train_dataloader, leave=False, desc="Processed batches"):
            inputs_seq  = inputs_seq.to(device)    # [B, T, F]
            targets_seq = targets_seq.to(device)   # [B, T, Dy]
            B, T, F     = inputs_seq.shape

            # ── Task 3: add Gaussian noise to training inputs ─────────────
            if noise_scale > 0.0:
                inputs_seq = inputs_seq + torch.randn_like(inputs_seq) * noise_scale

            optimizer.zero_grad()
            loss_accum_val = 0.0
            steps_in_seq   = T - (H - 1)
            accum          = 0

            inputs = inputs_seq[:, :H, :]   # [B, H, F]

            for t in range(H - 1, T):
                outputs = model(inputs)                   # [B, Dy]
                targets = targets_seq[:, t, :]            # [B, Dy]

                # ── Task 1: weighted loss ──────────────────────────────────
                raw_loss = base_criterion(outputs, targets)   # [B, Dy]
                if w_tensor is not None:
                    loss = (raw_loss * w_tensor).mean()
                else:
                    loss = raw_loss.mean()

                # ── Task 4: multi-step rollout auxiliary loss ──────────────
                # Triggered once per sequence at a random start inside the window
                ms_loss_val = 0.0
                if multistep_n > 0 and t == H - 1 and T - H >= multistep_n:
                    t_ms = random.randint(H - 1, T - multistep_n - 1)
                    ms_win = inputs_seq[:, t_ms - H + 1 : t_ms + 1, :].clone()  # [B, H, F]
                    ms_loss = torch.tensor(0.0, device=device)
                    for n in range(multistep_n):
                        ms_pred = model(ms_win)                         # [B, Dy], grad enabled
                        ms_tgt  = targets_seq[:, t_ms + n, :]          # [B, Dy]
                        ms_raw  = base_criterion(ms_pred, ms_tgt)       # [B, Dy]
                        if w_tensor is not None:
                            ms_loss = ms_loss + (ms_raw * w_tensor).mean()
                        else:
                            ms_loss = ms_loss + ms_raw.mean()
                        # slide window WITHOUT detach (allow gradient flow)
                        if n < multistep_n - 1:
                            next_ms = inputs_seq[:, t_ms + n + 1, :].clone()
                            next_ms[:, output_indices] = ms_pred         # no detach
                            ms_win = torch.cat([ms_win[:, 1:, :], next_ms.unsqueeze(1)], dim=1)
                    ms_loss = ms_loss / multistep_n
                    loss = loss + multistep_lambda * ms_loss
                    ms_loss_val = ms_loss.item()

                (loss / accum_every).backward()
                loss_accum_val += loss.item() - multistep_lambda * ms_loss_val
                accum += 1

                if t < T - 1:
                    next_inputs = inputs_seq[:, t + 1, :].clone()    # [B, F]

                    # ── Task 2: scheduled sampling ────────────────────────
                    if tf_ratio > 0 and random.random() < tf_ratio:
                        # teacher forcing: keep ground-truth values (next_inputs already has GT)
                        pass
                    else:
                        # use model predictions (original closed-loop behaviour)
                        next_inputs[:, output_indices] = outputs.detach()

                    inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)

                if (accum % accum_every == 0) or (t == T - 1):
                    if gradient_clipping_value and gradient_clipping_value > 0.0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss_sum += (loss_accum_val / steps_in_seq)
            train_batches  += 1

        train_loss = train_loss_sum / train_batches

        # ── Validation loop (unchanged: always auto-regressive) ────────────
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for inputs_seq, targets_seq in val_dataloader:
                inputs_seq  = inputs_seq.to(device)
                targets_seq = targets_seq.to(device)
                B, T, F     = inputs_seq.shape

                inputs       = inputs_seq[:, :H, :]
                loss_total   = 0.0
                steps_in_seq = T - (H - 1)

                for t in range(H - 1, T):
                    outputs = model(inputs)
                    targets = targets_seq[:, t, :]

                    # val loss uses the SAME weighted criterion so it's comparable
                    raw_loss = base_criterion(outputs, targets)
                    if w_tensor is not None:
                        step_loss = (raw_loss * w_tensor).mean()
                    else:
                        step_loss = raw_loss.mean()
                    loss_total += step_loss.item()

                    if t < T - 1:
                        next_inputs = inputs_seq[:, t + 1, :].clone()
                        next_inputs[:, output_indices] = outputs
                        inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)

                val_loss += (loss_total / steps_in_seq)
                num_val_batches += 1

        val_loss /= num_val_batches

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            wandb.run.summary["best_model_epoch"] = epoch
        else:
            patience_counter += 1

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        pbar.set_postfix({
            'Train':   f'{train_loss:.6f}',
            'Val':     f'{val_loss:.6f}',
            'Best':    f'{best_val_loss:.6f}',
            'Patience': f'{patience_counter}/{patience}',
            'TF':      f'{tf_ratio:.2f}',
        })

        if patience_counter >= patience:
            wandb.config.update({"epochs": epoch + 1}, allow_val_change=True)
            pbar.write("Early stopping triggered.")
            break

    pbar.close()

    print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")
    best_model_data = {
        'model_state_dict': best_model_state,
        'config': {
            'output_columns':  outputs_columns,
            'input_columns':   inputs_columns,
            'history':         history,
            'model_class':     model.__class__.__name__,
            'input_size':      input_size,
            'output_size':     output_size,
            'learning_rate':   learning_rate,
            'weight_decay':    weight_decay,
            'batch_size':      batch_size,
            'best_val_loss':   best_val_loss,
            'dataset_class':   train_dataset.__class__.__name__,
            'normed':          normed,
            'data_mean':       data_mean,
            'data_std':        safe_std,
        }
    }
    torch.save(best_model_data, 'trained_models/' + path_best_model)
    print(f"Best model with validation loss {best_val_loss:.4f} saved to trained_models/{path_best_model}")

    # ── Evaluation (unchanged) ─────────────────────────────────────────────
    model.load_state_dict(best_model_data['model_state_dict'])
    model.eval()
    print("Evaluating model on all datasets...")
    if normed:
        train_rmse, train_mae = evaluate_model_denormalized(model, train_dataloader, device, data_mean[output_indices], safe_std[output_indices], history, output_indices)
        val_rmse,   val_mae   = evaluate_model_denormalized(model, val_dataloader,   device, data_mean[output_indices], safe_std[output_indices], history, output_indices)
        test_rmse,  test_mae  = evaluate_model_denormalized(model, test_dataloader,  device, data_mean[output_indices], safe_std[output_indices], history, output_indices)
    else:
        train_rmse, train_mae = evaluate_model(model, train_dataloader, device, history, output_indices)
        val_rmse,   val_mae   = evaluate_model(model, val_dataloader,   device, history, output_indices)
        test_rmse,  test_mae  = evaluate_model(model, test_dataloader,  device, history, output_indices)

    wandb.run.summary.update({
        "RMSE_train_loss": float(np.mean(train_rmse)),
        "RMSE_val_loss":   float(np.mean(val_rmse)),
        "RMSE_test_loss":  float(np.mean(test_rmse)),
    })

    rows = []
    for col, tr, va, te, tr_mae, va_mae, te_mae in zip(
        outputs_columns, train_rmse, val_rmse, test_rmse, train_mae, val_mae, test_mae
    ):
        rows += [
            [wandb.run.name, "Train_RMSE", col, float(tr)],
            [wandb.run.name, "Val_RMSE",   col, float(va)],
            [wandb.run.name, "Test_RMSE",  col, float(te)],
            [wandb.run.name, "Train_MAE",  col, float(tr_mae)],
            [wandb.run.name, "Val_MAE",    col, float(va_mae)],
            [wandb.run.name, "Test_MAE",   col, float(te_mae)],
        ]
    tbl = wandb.Table(data=rows, columns=["Run", "Metric", "Output", "Loss"])
    wandb.log({"per_output_losses": tbl})

    print("Performing auto-regressive inference...")
    total_inf_rmse = []
    for auto_regression_file in auto_regression_files:
        print(f"Processing auto-regression for: {auto_regression_file}")
        auto_regression_df = pd.read_csv(auto_regression_file)
        inputs_np = auto_regression_df[inputs_columns].to_numpy(dtype=np.float32)
        inputs_t  = torch.from_numpy(inputs_np).to(device)
        if normed:
            mean_t = torch.tensor(data_mean, dtype=torch.float32, device=device)
            std_t  = torch.tensor(safe_std,  dtype=torch.float32, device=device)
            inputs_t = (inputs_t - mean_t) / std_t
        T2 = inputs_t.shape[0]
        if T2 < H:
            print(f"Skipping {auto_regression_file}: T({T2}) < history({H})")
            continue
        current_sequence = inputs_t[:H, :].clone()
        preds = []
        out_idx = list(output_indices)
        with torch.no_grad():
            for step in range(T2 - H):
                current_input = current_sequence.unsqueeze(0)
                pred = model(current_input)
                if pred.dim() == 3:
                    pred = pred[:, -1, :]
                pred = pred.squeeze(0)
                preds.append(pred)
                next_row = inputs_t[H + step, :].clone()
                next_row[out_idx] = pred
                current_sequence = torch.cat([current_sequence[1:, :], next_row.unsqueeze(0)], dim=0)
        preds_t = torch.stack(preds, dim=0)
        if normed:
            out_mean = torch.tensor(data_mean[out_idx], dtype=torch.float32, device=device)
            out_std  = torch.tensor(safe_std[out_idx],  dtype=torch.float32, device=device)
            preds_t  = preds_t * out_std + out_mean
        preds_np = preds_t.cpu().numpy()
        gt_np    = auto_regression_df[outputs_columns].to_numpy(dtype=np.float32)[H:]
        inf_rmse = np.sqrt(np.mean((preds_np - gt_np) ** 2, axis=0))
        total_inf_rmse.append(inf_rmse)
        print(f"  Auto-regressive RMSE: {inf_rmse}")
    if total_inf_rmse:
        mean_inf_rmse = np.mean(np.array(total_inf_rmse), axis=0)
        print(f"  Mean auto-regressive RMSE: {mean_inf_rmse}")

    wandb.finish()
    print("Training completed and results logged to wandb!")
