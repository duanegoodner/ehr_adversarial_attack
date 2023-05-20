import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from standard_model_trainer import ModuleWithDevice


# use this as component in nn.Sequential of full model
class BidirectionalX19LSTM(ModuleWithDevice):
    def __init__(
        self,
        device: torch.device,
        input_size: int = 19,
        lstm_hidden_size: int = 128,
    ):
        super(BidirectionalX19LSTM, self).__init__(device=device)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.tensor, lengths: torch.tensor) -> torch.tensor:
        # convert x to PackedSequence
        x_packed = pack_padded_sequence(
            input=x, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out_packed, (h_n, c_n) = self.lstm(x_packed)
        unpacked_lstm_out, lstm_out_lengths = pad_packed_sequence(
            sequence=lstm_out_packed, batch_first=True
        )
        # select final hidden state from each sequence
        final_lstm_out = unpacked_lstm_out[
            torch.arange(unpacked_lstm_out.shape[0]), lstm_out_lengths - 1, :
        ].squeeze()
        return final_lstm_out


class LSTMSun2018(ModuleWithDevice):
    def __init__(
        self,
        device: torch.device,
        input_size: int = 19,
        lstm_hidden_size: int = 128,
        fc_hidden_size: int = 32,
    ):
        super(LSTMSun2018, self).__init__(device=device)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = BidirectionalX19LSTM(
            device=device,
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
        )
        # self.act_lstm = nn.Tanh()
        self.act_lstm = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(
            in_features=2 * lstm_hidden_size, out_features=fc_hidden_size
        )
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=fc_hidden_size, out_features=2)
        self.act_2 = nn.Softmax(dim=1)
        self.to(device=device)

    def forward(self, x: torch.tensor, lengths: torch.tensor) -> torch.tensor:
        final_lstm_out = self.lstm(x, lengths)
        fc_1_out = self.fc_1(final_lstm_out)
        fc_1_out = self.act_1(fc_1_out)
        fc_2_out = self.fc_2(fc_1_out)
        out = self.act_2(fc_2_out)
        # out = torch.softmax(fc_2_out, dim=1)
        return out
