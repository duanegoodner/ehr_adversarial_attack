# https://discuss.pytorch.org/t/rnn-output-vs-hidden-state-dont-match-up-my-misunderstanding/43280


import torch
import torch.nn as nn


class RnnClassifier(nn.Module):
    def __init__(self, bidirectional=True):
        super(RnnClassifier, self).__init__()
        self.bidirectional = bidirectional

        self.embed_dim = 5
        self.hidden_dim = 3
        self.num_layers = 4

        self.word_embeddings = nn.Embedding(100, self.embed_dim)
        self.num_directions = 2 if bidirectional == True else 1
        self.rnn = nn.LSTM(
            self.embed_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
        )
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
            ),
            torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
            ),
        )

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        # Push through embedding layer and transpose for RNN layer (batch_first=False)
        X = self.word_embeddings(inputs).transpose(0, 1)
        # Push through RNN layer
        output, self.hidden = self.rnn(X, self.hidden)
        # output.shape = (seq_len, batch_size, num_directions*hidden_dim)
        # self.hidden[0].shape = (num_layers*num_directions, batch_size, hidden_dim)

        # Get h_n^w directly from output of the last time step
        output_last_step = output[
            -1
        ]  # (batch_size, num_directions*hidden_dim)

        # Get h_n^w from hidden state
        hidden = self.hidden[0].view(
            self.num_layers, self.num_directions, batch_size, self.hidden_dim
        )
        hidden_last_layer = hidden[
            -1
        ]  # (num_directions, batch_size, hidden_dim)

        if self.bidirectional:
            direction_1, direction_2 = (
                hidden_last_layer[0],
                hidden_last_layer[1],
            )
            direction_full = torch.cat((direction_1, direction_2), 1)
        else:
            direction_full = hidden_last_layer.squeeze(0)

        print("The following 2 tensors should be equal(?)")
        print(output_last_step[0].data)
        print(direction_full[0].data)

        print(self.hidden[0].data)


if __name__ == "__main__":
    model = RnnClassifier(bidirectional=True)

    inputs = torch.LongTensor([[1, 2, 4, 6, 4, 2, 3]])

    model.hidden = model.init_hidden(inputs.shape[0])

    model(inputs)
