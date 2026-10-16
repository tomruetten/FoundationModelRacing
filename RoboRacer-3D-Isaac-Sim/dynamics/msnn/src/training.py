import torch
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

# Functions to calculate RMSE and MAE per output
def evaluate_model(model, dataloader, device, history, output_indices):
    """Calculate RMSE and MAE for the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader providing the data.
        device (torch.device): The device to run the model on.
        history (int): The number of past timesteps to consider.
        output_indices (list): The indices of the outputs to evaluate.

    Returns:
        tuple: RMSE and MAE per output.
    """
    
    model.eval()
    se_sum = None  # squared error sum per output [Dy]
    ae_sum = None  # absolute error sum per output [Dy]
    count = 0      # number of samples accumulated (B per timestep)

    with torch.no_grad():
        for inputs_seq, targets_seq in dataloader:
            inputs_seq  = inputs_seq.to(device)  # [B, T, F] or [T, F]
            targets_seq = targets_seq.to(device) # [B, T, Dy] or [T, Dy]

            # ensure batch dimension
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

            # initial window
            inputs = inputs_seq[:, :H, :]  # [B, H, F]

            # closed-loop rollout: predict t = H-1 .. T-1
            for t in range(H - 1, T):
                outputs = model(inputs)           # [B, Dy]
                targets = targets_seq[:, t, :]    # [B, Dy]

                diff = outputs - targets
                se_sum += (diff.pow(2)).sum(dim=0)   # sum over batch
                ae_sum += diff.abs().sum(dim=0)
                count += B

                if t == T - 1:
                    break

                # slide window and inject predictions into output slots
                next_inputs = inputs_seq[:, t + 1, :].clone()     # [B, F]
                next_inputs[:, output_indices] = outputs
                inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)

    rmse_per_output = torch.sqrt(se_sum / max(1, count)).detach().cpu().numpy()
    mae_per_output  = (ae_sum / max(1, count)).detach().cpu().numpy()
    return rmse_per_output, mae_per_output


def evaluate_model_denormalized(model, dataloader, device, tgt_mean, tgt_std, history, output_indices):
    """Denormalize and calculate RMSE and MAE for the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader providing the data.
        device (torch.device): The device to run the model on.
        tgt_mean (array-like): Mean values for each output column for denormalization.
        tgt_std (array-like): Standard deviation values for each output column for denormalization.
        history (int): The number of past timesteps to consider.
        output_indices (list): The indices of the outputs to evaluate.

    Returns:
        tuple: RMSE and MAE per output.
    """    
    model.eval()
    se_sum = None  # squared error sum per output [Dy]
    ae_sum = None  # absolute error sum per output [Dy]
    count = 0

    tgt_mean_tensor = torch.tensor(tgt_mean, dtype=torch.float32, device=device)  # [Dy]
    tgt_std_tensor  = torch.tensor(tgt_std,  dtype=torch.float32, device=device)  # [Dy]

    with torch.no_grad():
        for inputs_seq, targets_seq in dataloader:
            inputs_seq  = inputs_seq.to(device)  # [B, T, F] or [T, F]
            targets_seq = targets_seq.to(device) # [B, T, Dy] or [T, Dy]

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

            inputs = inputs_seq[:, :H, :]  # [B, H, F]

            for t in range(H - 1, T):
                outputs = model(inputs)           # [B, Dy]
                targets = targets_seq[:, t, :]    # [B, Dy]

                # denormalize
                outputs_den = outputs * tgt_std_tensor + tgt_mean_tensor
                targets_den = targets * tgt_std_tensor + tgt_mean_tensor

                diff = outputs_den - targets_den
                se_sum += (diff.pow(2)).sum(dim=0)
                ae_sum += diff.abs().sum(dim=0)
                count += B

                if t == T - 1:
                    break

                next_inputs = inputs_seq[:, t + 1, :].clone()
                # feed the normalized predictions back into the next window (because inputs are normalized)
                next_inputs[:, output_indices] = outputs
                inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)

    rmse_per_output = torch.sqrt(se_sum / max(1, count)).detach().cpu().numpy()
    mae_per_output  = (ae_sum / max(1, count)).detach().cpu().numpy()
    return rmse_per_output, mae_per_output

 
def set_seed(seed):
    """Set fixed random seeds for reproducibility
    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_normalization_stats(csv_files, input_columns):
    """
    Compute per feature mean/std for data from provided CSV files.
    Args:
        csv_files (list): List of paths to CSV files containing the dataset.
        input_columns (list): List of input column names (has to include the output columns).
    Returns:
        tuple: (data_mean, data_std) each of shape [F] where F is number of input features.
    """
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

    # Concatenate all data
    combined_data = np.vstack(all_data)  # [total_timesteps, F]
    
    # Compute column-wise statistics
    data_mean = combined_data.mean(axis=0)  # [F]
    data_std = combined_data.std(axis=0)    # [F]

    return data_mean, data_std

def train_network(params):
    """
    Train a model.
    Args:
        params: dictionary with parameters including keys:
            - model_name (str): Neural network model architecture classname in RNN_architectures
            - random_seed (int): Used random seed, int or None
            - learning_rate (float)
            - batch_size (int)
            - num_epochs (int): Number of training epochs
            - criterion: Loss function 'MSE' or 'Huber'
            - optimizer: 'Adam','NAdam' or 'AdamW'
            - weight_decay (float)
            - gradient_clipping_value (float): Gradient clipping value, 0.0 for none
            - patience (int): Patience for EarlyStopping
            - history (int): Number of previous timesteps
            - seq_length (int): Desired sequence length for training
            - train_files (list): List of paths to training CSV files or int of preset
            - val_files (list): List of paths to validation CSV files
            - test_files (list): List of paths to testing CSV files
            - auto_regression_files (list): List of paths to auto-regression CSV files
            - outputs_columns (list): List of output column names
            - inputs_columns (list): List of input column names
            - normed (bool): whether to normalize inputs and outputs
            - pretrained_model (str): Path to a pretrained model (default: None)
            - num_workers (int): number of data loading workers (default: 4)
            - prefetch_factor (int): Number of samples fetched in advance by each worker (default: 2)
            
    """
    # Check if params has all necessary keys
    required_keys = [
        "model_name", "random_seed", "learning_rate", "batch_size", "num_epochs",
        "criterion", "optimizer", "gradient_clipping_value", "patience", "history",
        "train_files", "val_files", "test_files", "auto_regression_files",
        "outputs_columns", "inputs_columns", "normed", "weight_decay", "seq_length"
    ]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    

    # Infer model kind from name
    model_name = params['model_name']

    random_seed = params['random_seed']
    if random_seed is not None:
        set_seed(random_seed)

    # Files for training, validation, testing and auto-regression
    train_files1 = ['processed_data/stanley_0.4_20ms.csv','processed_data/manual_mode_counterclockwise_20ms_cut.csv', \
                'processed_data/clockwise_20ms.csv', 'processed_data/counterclockwise_20ms.csv']
    train_files2 = train_files1 + ['processed_data/Car1_Circles_20ms.csv', 'processed_data/Car1_Oscillation_20ms.csv']
    train_files3 = train_files2 + ['processed_data/Car1_JustDrivingAround_20ms.csv']
    train_files4 = train_files3 + ['processed_data/Car1_BackwardDriving_20ms.csv']

    if params['train_files'] == 1:
        train_files = train_files1
    elif params['train_files'] == 2:
        train_files = train_files2
    elif params['train_files'] == 3:
        train_files = train_files3
    elif params['train_files'] == 4:
        train_files = train_files4
    else:
        train_files = params['train_files']
    val_files = params['val_files']
    test_files = params['test_files']
    auto_regression_files = params['auto_regression_files']

    # Define the dataset
    history = params['history']
    outputs_columns = params['outputs_columns']
    inputs_columns = params['inputs_columns']
    # list is safer for torch tensor indexing; also works for NumPy slicing
    output_indices = [inputs_columns.index(c) for c in outputs_columns]
    normed = params['normed']
    seq_length = params['seq_length']

    for col in outputs_columns:
        if col not in inputs_columns:
            raise ValueError(f"Output column '{col}' must be included in input_columns.")

    # Creating Datasets
    data_mean, data_std = compute_normalization_stats(train_files, inputs_columns)
    safe_std = np.where(data_std < 1e-6, 1.0, data_std)  # Prevent division by zero

    train_stride = params.get('stride', 1)
    if normed:
        train_dataset = RNN_Seq_Dataset_norm(train_files, inputs_columns, outputs_columns, seq_length, data_mean, safe_std, stride=train_stride)
        val_dataset   = RNN_Seq_Dataset_norm(val_files,   inputs_columns, outputs_columns, seq_length, data_mean, safe_std)
        test_dataset  = RNN_Seq_Dataset_norm(test_files,  inputs_columns, outputs_columns, seq_length, data_mean, safe_std)
    else:
        train_dataset = RNN_Seq_Dataset(train_files, inputs_columns, outputs_columns, seq_length, stride=train_stride)
        val_dataset = RNN_Seq_Dataset(val_files, inputs_columns, outputs_columns, seq_length)
        test_dataset = RNN_Seq_Dataset(test_files, inputs_columns, outputs_columns, seq_length)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pin_memory = (device.type == "cuda")

    # Calculate Dimensions
    input_size = len(inputs_columns) # Number of input features
    output_size = len(outputs_columns)  # Number of output features

    # Define  model and hyperparameters
    if params.get('pretrained_model') is not None:
        print(f"Using pretrained model: {params['pretrained_model']}")
        pretrained_model_data = torch.load(params['pretrained_model'], weights_only=False)
        pretrained_model_cfg = pretrained_model_data['config']
        # Check for fit of pretrained model
        assert input_size == pretrained_model_cfg['input_size']
        assert output_size == pretrained_model_cfg['output_size']
        assert normed == pretrained_model_cfg['normed']
        assert history == pretrained_model_cfg['history']
        assert model_name == pretrained_model_cfg['model_class']
        assert inputs_columns == pretrained_model_cfg['input_columns']
        assert outputs_columns == pretrained_model_cfg['output_columns']
        
        model = getattr(RNN_architectures, pretrained_model_cfg['model_class'])(input_size=input_size, output_size=output_size).to(device)
        model.load_state_dict(pretrained_model_data['model_state_dict'])
    else:
        model = getattr(RNN_architectures, params['model_name'])(input_size=input_size, output_size=output_size).to(device)
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    if params['criterion'] == 'MSE':
        criterion = torch.nn.MSELoss()
    elif params['criterion'] == 'Huber':
        criterion = torch.nn.HuberLoss()
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

    # Early stopping parameters
    patience = params['patience']
    best_val_loss = float('inf')  # Initialize the best validation loss
    patience_counter = 0  # Counter for early stopping

    # Create DataLoaders for train and validation sets
    n_workers = params.get('num_workers', 4)
    pf_factor = params.get('prefetch_factor', 2) if n_workers > 0 else None
    persist = (n_workers > 0) and pin_memory  # persistent_workers only helps on GPU
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=pin_memory, persistent_workers=persist, prefetch_factor=pf_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, persistent_workers=persist, prefetch_factor=pf_factor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, persistent_workers=persist, prefetch_factor=pf_factor)

    # Example: Iterate through the DataLoader
    for inputs, targets in train_dataloader:
        print("Inputs:", inputs.shape)
        print("Targets:", targets.shape)  # Should be [batch_size, len(output_columns)]
        break

    # Initialize wandb and set path to save the best model
    os.makedirs('trained_models', exist_ok=True)
    path_best_model = f"{wandb.run.name}___{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pth"  # filename 
    wandb.config.update({
            "architecture": model.__class__.__name__,
            "dataset": train_dataset.__class__.__name__,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
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
            "num_previous_states" : history,
            "data_mean": data_mean.tolist(),
            "data_std": safe_std.tolist(),  # log the std actually used
            "seq_length": seq_length,
            "saved_model": path_best_model,
            "random_seed": random_seed,
        },
        allow_val_change=True,
    )
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("test_loss", summary="min")

    # Training loop with early stopping
    H = history
    accum_every = 100
    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        model.train()  # Set the model to training mode
        train_loss_sum = 0.0
        train_batches = 0
        for inputs_seq, targets_seq in tqdm(train_dataloader, leave=False, desc="Processed batches"):
            # batch on GPU
            inputs_seq  = inputs_seq.to(device)    # [B, T, F]
            targets_seq = targets_seq.to(device)   # [B, T, Dy]
            B, T, F = inputs_seq.shape

            optimizer.zero_grad()
            loss_accum_val = 0.0
            steps_in_seq   = T - (H - 1)   
            accum = 0
            
            # initial H-window per batch
            inputs = inputs_seq[:, :H, :]          # [B, H, F]

            # we predict targets at indices t = H-1 .. T-1
            for t in range(H - 1, T):
                outputs = model(inputs)     
                targets = targets_seq[:, t, :]     # [B, Dy]

                # accumulate loss; backprop in micro-steps
                loss = criterion(outputs, targets)
                (loss / accum_every).backward()
                loss_accum_val += loss.item()
                accum += 1

                # build next window unless it's the last step
                if t < T - 1:
                    # build next window: slide + append next row with predictions in output slots
                    next_inputs = inputs_seq[:, t + 1, :].clone()           # [B, F]
                    next_inputs[:, output_indices] = outputs.detach()       # overwrite GT with pred
                    inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)  # [B, H, F]

                # micro-update
                if (accum % accum_every == 0) or (t == T - 1):
                    if gradient_clipping_value and gradient_clipping_value > 0.0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)
                    optimizer.step()
                    optimizer.zero_grad()

            # average per-step loss for this sequence/batch
            train_loss_sum += (loss_accum_val / steps_in_seq)
            train_batches  += 1

        train_loss = train_loss_sum / train_batches

        # Validation loop (auto-regressive, same as training)
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for inputs_seq, targets_seq in val_dataloader:
                inputs_seq  = inputs_seq.to(device)   # [B, T, F]
                targets_seq = targets_seq.to(device)  # [B, T, Dy]
                B, T, F = inputs_seq.shape

                inputs = inputs_seq[:, :H, :]         # [B, H, F]
                loss_total = 0.0
                steps_in_seq = T - (H - 1)

                for t in range(H - 1, T):
                    outputs = model(inputs)                  # [B, Dy]
                    targets = targets_seq[:, t, :]           # [B, Dy]
                    loss_total = loss_total + criterion(outputs, targets)

                    if t < T - 1:
                        # build next window: slide + append next row with predictions in output slots
                        next_inputs = inputs_seq[:, t + 1, :].clone()
                        next_inputs[:, output_indices] = outputs  # always feed predictions in val
                        inputs = torch.cat([inputs[:, 1:, :], next_inputs.unsqueeze(1)], dim=1)

                val_loss += (loss_total.item() / steps_in_seq)
                num_val_batches += 1

        val_loss /= num_val_batches

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
            # Store the best model weights
            best_model_state = copy.deepcopy(model.state_dict())
            wandb.run.summary["best_model_epoch"] = epoch
        else:
            patience_counter += 1
        # Log to wandb
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        # Update progress bar with current metrics (AFTER the improvement check)
        pbar.set_postfix({
            'Train': f'{train_loss:.6f}',
            'Val': f'{val_loss:.6f}', 
            'Best': f'{best_val_loss:.6f}',
            'Patience': f'{patience_counter}/{patience}'
        })
        
        # Early stopping condition
        if patience_counter >= patience:
            wandb.config.update({"epochs": epoch + 1}, allow_val_change=True)
            pbar.write("Early stopping triggered.")
            break
    pbar.close()

    print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")
    # Save best model and config parameters
    best_model_data = {
        'model_state_dict': best_model_state,
        'config': {
            'output_columns': outputs_columns,
            'input_columns': inputs_columns,
            'history': history,
            'model_class': model.__class__.__name__,
            'input_size': input_size,
            'output_size': output_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'best_val_loss': best_val_loss,
            'dataset_class': train_dataset.__class__.__name__,
            'normed': normed,
            'data_mean': data_mean,
            'data_std': safe_std,  # persist the std actually used
        }
    }
    torch.save(best_model_data, 'trained_models/' + path_best_model)
    print(f"Best model with validation loss {best_val_loss:.4f} saved to trained_models/{path_best_model}")

    ###############################################################
    # Evaluation
    ###############################################################

    # Load the best model for testing
    model.load_state_dict(best_model_data['model_state_dict'])
    model.eval()
    # Evaluate on all datasets
    print("Evaluating model on all datasets...")
    if normed:
        train_rmse, train_mae = evaluate_model_denormalized(model, train_dataloader, device, data_mean[output_indices], safe_std[output_indices], history, output_indices)
        val_rmse, val_mae = evaluate_model_denormalized(model, val_dataloader, device, data_mean[output_indices], safe_std[output_indices], history, output_indices)
        test_rmse, test_mae = evaluate_model_denormalized(model, test_dataloader, device, data_mean[output_indices], safe_std[output_indices], history, output_indices)
    else:
        train_rmse, train_mae = evaluate_model(model, train_dataloader, device, history, output_indices)
        val_rmse, val_mae = evaluate_model(model, val_dataloader, device, history, output_indices)
        test_rmse, test_mae = evaluate_model(model, test_dataloader, device, history, output_indices)

    wandb.run.summary.update({
        "RMSE_train_loss": float(np.mean(train_rmse)),
        "RMSE_val_loss": float(np.mean(val_rmse)), 
        "RMSE_test_loss": float(np.mean(test_rmse))
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

    tbl = wandb.Table(data=rows, columns=["Run","Metric","Output","Loss"])
    wandb.log({"per_output_losses": tbl})

    # Auto-Regressive Inference
    print("Performing auto-regressive inference...")
    total_inf_rmse = []
    for auto_regression_file in auto_regression_files:
        print(f"Processing auto-regression for: {auto_regression_file}")
        auto_regression_df = pd.read_csv(auto_regression_file)

        # Entire input sequence to device once
        inputs_np = auto_regression_df[inputs_columns].to_numpy(dtype=np.float32)  # [T, F]
        inputs_t  = torch.from_numpy(inputs_np).to(device)                          # [T, F]

        # Optional normalization (must match how the model was trained)
        if normed:
            mean_t = torch.tensor(data_mean, dtype=torch.float32, device=device)     # [F]
            std_t  = torch.tensor(safe_std, dtype=torch.float32, device=device)      # [F]
            inputs_t = (inputs_t - mean_t) / std_t

        T = inputs_t.shape[0]
        H = history
        if T < H:
            print(f"Skipping {auto_regression_file}: T({T}) < history({H})")
            continue

        # Init rolling window
        current_sequence = inputs_t[:H, :].clone()  # [H, F]
        preds = []  # accumulate on device
        out_idx = list(output_indices)

        with torch.no_grad():
            for step in range(T - H):
                current_input = current_sequence.unsqueeze(0)  # [1, H, F]
                pred = model(current_input)                    # expect [1, Dy] or [1, *, Dy]
                if pred.dim() == 3:
                    pred = pred[:, -1, :]                      # [1, Dy] – take last timestep
                # else assume [1, Dy]
                pred = pred.squeeze(0)                         # [Dy]
                preds.append(pred)

                # Prepare next window: GT next row with predicted outputs injected
                next_row = inputs_t[H + step, :].clone()       # [F]
                next_row[out_idx] = pred                       # feed prediction
                current_sequence = torch.cat([current_sequence[1:], next_row.unsqueeze(0)], dim=0)

        # Stack predictions and (optionally) denormalize for reporting
        preds_t = torch.stack(preds, dim=0)  # [T-H, Dy]
        if normed:
            # denormalize only target dims
            out_mean_t = torch.tensor(data_mean[out_idx], dtype=torch.float32, device=device)
            out_std_t  = torch.tensor(safe_std[out_idx],  dtype=torch.float32, device=device)
            preds_np = (preds_t * out_std_t + out_mean_t).detach().cpu().numpy()
        else:
            preds_np = preds_t.detach().cpu().numpy()

        ground_truth_outputs = auto_regression_df[outputs_columns].iloc[H:].to_numpy(dtype=np.float32)  # [T-H, Dy]

        # Plotting and logging per output
        auto_reg_filename = auto_regression_file.split("/")[-1].replace('.csv', '')
        time_steps = list(range(len(ground_truth_outputs)))

        if np.isinf(preds_np).any():
            print("Warning: Infs found in predictions!")
        if np.isnan(preds_np).any():
            print("Warning: NaNs found in predictions!")

        for j, col_name in enumerate(outputs_columns):
            plot = wandb.plot.line_series(
                xs=time_steps[:5000],
                ys=[ground_truth_outputs[:5000, j].tolist(), preds_np[:5000, j].tolist()],
                keys=['Ground Truth', 'Auto-Regressive Prediction'],
                title=f'{col_name} - Inference {auto_reg_filename}',
                xname='Time Steps'
            )
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(ground_truth_outputs[:, j], 'r-', label='Ground Truth', linewidth=2)
            ax.plot(preds_np[:, j], 'b--', label='Auto-Regressive Prediction', linewidth=2)
            ax.set_title(f'{col_name} - Auto-Regressive Inference')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            wandb.log({f"{col_name}_{auto_reg_filename}": wandb.Image(fig)})
            plt.close(fig)

            wandb.log({f"{col_name}_auto_regression_{auto_reg_filename}": plot})

        # RMSE per output and total
        mse_per_output = np.mean((ground_truth_outputs - preds_np) ** 2, axis=0)
        rmse_per_output = np.sqrt(mse_per_output)
        total_rmse_file = float(np.mean(rmse_per_output))

        for j, col_name in enumerate(outputs_columns):
            wandb.run.summary[f"inf/rmse/{auto_reg_filename}/{col_name}"] = float(rmse_per_output[j])
        wandb.run.summary[f"inf/rmse/{auto_reg_filename}/total"] = total_rmse_file
        total_inf_rmse.append(total_rmse_file)

    wandb.run.summary["inf/rmse/overall"] = float(np.mean(total_inf_rmse))
    print("Training completed and results logged to wandb!")
    return np.mean(total_inf_rmse)

