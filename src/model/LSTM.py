import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self
                 , input_dim
                 , seq_length
                 , device
                 , hidden_dim=16
                 , output_dim=1
                 , num_layers=1
                 , **kwargs
                 ):

        super(Model, self).__init__()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.seq_length, output_dim)

    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device),
                       torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        predictions = self.linear(lstm_out.contiguous().view(batch_size, -1))
        return predictions