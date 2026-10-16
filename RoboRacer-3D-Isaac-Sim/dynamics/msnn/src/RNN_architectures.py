import torch
import torch.nn as nn

class RNN_Hermansdorfer(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super(RNN_Hermansdorfer, self).__init__()
        self.hidden_size = 150

        # GRU layer
        self.gru1 = nn.GRU(input_size, self.hidden_size, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # GRU layer
        gru_out, _ = self.gru1(x)  # gru_out has shape (batch_size, sequence_length, hidden_size)

        # Pass last timestep's hidden state through the fully connected layer
        output = self.fc(gru_out[:, -1, :])  # output has shape (batch_size, output_size)

        return output

    
  
class RNN_jomoh_1(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super(RNN_jomoh_1, self).__init__()

        # GRU layer
        self.gru1 = nn.GRU(input_size, 100 , batch_first=True)
        self.fc1 = nn.Linear(100 , 100)
        self.fc3 = nn.Linear(100 , 100)
        self.fc5 = nn.Linear(100 , output_size)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # GRU layers
        gru_out, _ = self.gru1(x)  # gru_out has shape (batch_size, sequence_length, hidden_size)
        x = gru_out[:, -1, :]  # Take the last hidden state
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.LeakyReLU(x)
        output = self.fc5(x)

        return output

class RNN_jomoh_2(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super(RNN_jomoh_2, self).__init__()
        self.hidden_size = 100
        self.hidden_layer_blocks = 3
        # GRU layer
        self.gru_input = nn.GRU(input_size, self.hidden_size , batch_first=True)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size , batch_first=True)
        self.fc = nn.Linear(self.hidden_size , self.hidden_size)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)
        self.fc_output = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # GRU layer
        x, _ = self.gru_input(x)
        x = self.fc(x)
        x = self.LeakyReLU(x)
        for _ in range(self.hidden_layer_blocks):
            x, _ = self.gru(x)
            x = self.fc(x)
            x = self.LeakyReLU(x)
            x = self.fc(x)
            x = self.LeakyReLU(x)
        
        output = self.fc_output(x)[:, -1, :]  # Take only the last  state
        
        return output

import torch
import torch.nn as nn

class RNN_jomoh_3(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """
        Args:
            input_size: Number of input features per time step.
            output_size: Number of output features.
        """
        super().__init__()
        self.hidden_size = 150
        self.hidden_layer_blocks = 3
        self.negative_slope = 0.1

        # First GRU that lifts input_size -> hidden_size (shared across time only)
        self.gru_input = nn.GRU(input_size, self.hidden_size, batch_first=True)

        # Build blocks: each block has its OWN GRU and OWN FC layers
        self.blocks = nn.ModuleList()
        for _ in range(self.hidden_layer_blocks):
            block = nn.ModuleDict({
                "gru": nn.GRU(self.hidden_size, self.hidden_size, batch_first=True),
                "fc1": nn.Linear(self.hidden_size, self.hidden_size),
                "fc2": nn.Linear(self.hidden_size, self.hidden_size),
            })
            self.blocks.append(block)

        self.act = nn.LeakyReLU(negative_slope=self.negative_slope)
        self.fc_output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, output_size)  # last time step
        """
        # Initial GRU
        x, _ = self.gru_input(x)                 # (B, T, H)
        # Per-block processing (each with its own parameters)
        for block in self.blocks:
            x, _ = block["gru"](x)               # (B, T, H)
            x = self.act(block["fc1"](x))        # (B, T, H)
            x = self.act(block["fc2"](x))        # (B, T, H)

        # Take last time step and map to outputs
        out = self.fc_output(x[:, -1, :])        # (B, Dy)
        return out

class DNN_jomoh_1(nn.Module):
    def __init__(self, input_size, output_size, history=10):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super(DNN_jomoh_1, self).__init__()
        self.hidden_size = 100
        self.hidden_layer_blocks = 5
        self.input_size = input_size * history
        # Fully connected layers
        self.fc_input = nn.Linear(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)
        self.fc_output = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Fully connected layers
        x = torch.flatten(x, start_dim=1)  # Flatten the input tensor
        x = self.fc_input(x)
        x = self.LeakyReLU(x)
        
        for _ in range(self.hidden_layer_blocks):
            x = self.fc(x)
            x = self.LeakyReLU(x)
        
        output = self.fc_output(x)


class RNN_h200(nn.Module):
    """RNN_jomoh_1 with hidden_size=200 (Run23 — more capacity)."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gru1       = nn.GRU(input_size, 200, batch_first=True)
        self.fc1        = nn.Linear(200, 200)
        self.fc3        = nn.Linear(200, 200)
        self.fc5        = nn.Linear(200, output_size)
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        gru_out, _ = self.gru1(x)
        x = gru_out[:, -1, :]
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc3(x))
        return self.fc5(x)


class RNN_2layer(nn.Module):
    """2-layer stacked GRU, hidden_size=100, dropout=0.2 (Run24 — more depth)."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gru1       = nn.GRU(input_size, 100, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1        = nn.Linear(100, 100)
        self.fc3        = nn.Linear(100, 100)
        self.fc5        = nn.Linear(100, output_size)
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        gru_out, _ = self.gru1(x)
        x = gru_out[:, -1, :]
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc3(x))
        return self.fc5(x)

        return output