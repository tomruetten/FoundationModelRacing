import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class RNN_Seq_Dataset:
    """
    Dataset with sequences of lengths seq_length. Returns (inputs, targets) where targets are the next time step of the output columns.
    """
    def __init__(self, csv_files, input_columns, output_columns, seq_length, stride=1):
        """
        Dataset with sequences of lengths seq_length. Returns (inputs, targets) where targets are the next time step of the output columns.
        
        args
        - csv_files (list): List of paths to CSV files containing the dataset.
        - output_columns (list): List of column names to predict (outputs); determines the order of the output variables.
        - input_columns (list): List of input column names (has to include the output columns).
        - seq_length (int): Desired sequence length.
        - stride (int): Stride length for sampling (default=1).
        """
        
        self.csv_files = csv_files
        self.seq_length = int(seq_length)
        self.output_columns = output_columns
        self.input_columns = input_columns
        self.stride = int(stride)
        self.file_data = []          
        self.sample_indices = []  
        self.output_indexes = np.array([self.input_columns.index(c) for c in self.output_columns], dtype=np.int32)

        for path in csv_files:
            df = pd.read_csv(path)

            missing = [c for c in self.input_columns if c not in df.columns]
            if missing:
                raise ValueError(f"{path} missing columns: {missing}")

            data = df[self.input_columns].to_numpy(dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError(f"{path} contains NaN or non finite values.")

            N = len(data)
            if self.seq_length > N:
                print(f"WARNING! {path} shorter than  seq_length={self.seq_length}. File skipped.")
                continue
            
            num_samples = N - self.seq_length
            self.file_data.append(data)
            fid = len(self.file_data) - 1
            for s in range(0, num_samples, self.stride):
                self.sample_indices.append((fid, s))

        if not self.sample_indices:
            raise ValueError("No valid samples found in provided CSV files.")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        fid, s = self.sample_indices[idx]
        data = self.file_data[fid]
        inputs = data[s : self.seq_length + s, :]  # [seq_length, F]
        targets = data[s + 1 : self.seq_length + s + 1, self.output_indexes]
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class RNN_Seq_Dataset_norm(RNN_Seq_Dataset):
    """
    Same as RNN_Seq_Dataset but applies normalization using mean/std.
    Fit mean/std externally on train dataset only, then pass here.
    """
    def __init__(self, csv_files, input_columns, output_columns, seq_length, data_mean, data_std, stride=1):
        """
        Dataset with sequences of lengths seq_length. Returns (inputs, targets) where targets are the next time step of the output columns.
        
        args
        - csv_files (list): List of paths to CSV files containing the dataset.
        - output_columns (list): List of column names to predict (outputs); determines the order of the output variables.
        - input_columns (list): List of input column names (has to include the output columns).
        - seq_length (int): Desired sequence length.
        - data_mean (array-like): Mean values for each input column for normalization.
        - data_std (array-like): Standard deviation values for each input column for normalization.
        - stride (int): Stride length for sampling (default=1).
        """
        super().__init__(csv_files, input_columns, output_columns, seq_length, stride)
        # ensure torch tensors (float32)
        self.data_mean = torch.as_tensor(data_mean, dtype=torch.float32)
        self.data_std = torch.as_tensor(data_std, dtype=torch.float32)
        self.out_idx = torch.as_tensor(self.output_indexes, dtype=torch.long)

        # guard against zero std
        self.in_std_safe = torch.where(self.data_std == 0, torch.ones_like(self.data_std), self.data_std)
        self.tgt_mean = self.data_mean[self.out_idx]
        self.tgt_std_safe = torch.where(
            self.data_std[self.out_idx] == 0,
            torch.ones_like(self.data_std[self.out_idx]),
            self.data_std[self.out_idx],
        )

    def __getitem__(self, idx):
        inputs, targets = super().__getitem__(idx)  # inputs: [T,F], targets: [T,Dy]
        inputs = (inputs - self.data_mean) / self.in_std_safe
        targets = (targets - self.tgt_mean) / self.tgt_std_safe
        return inputs, targets

