import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn


# a = torch.Tensor([[1], [2], [3]])
# b = torch.Tensor([[4], [5]])
# c = torch.Tensor([[6]])
# packed = rnn_utils.pack_sequence([a, b, c])
#
# lstm = nn.LSTM(1, 3)
#
# packed_output, (h, c) = lstm(packed)
#
# y = rnn_utils.pad_packed_sequence(packed_output)


a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
b = torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.float)
c = torch.tensor([[17, 18, 19]], dtype=torch.float)

padded_sequences = rnn_utils.pad_sequence(sequences=[a, b, c])
my_lengths = torch.tensor([item.shape[0] for item in [a, b, c]])


packed_input = rnn_utils.pack_padded_sequence(
    input=padded_sequences, lengths=my_lengths
)

lstm = nn.LSTM(3, 5)
packed_output, (hidden, cell) = lstm(packed_input)
padded_output = rnn_utils.pad_packed_sequence(packed_output)
