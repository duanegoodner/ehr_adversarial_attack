import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_shape, hidden_dim, targets, learn_rate,
                 multiclass):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.targets = targets
        self.multiclass = multiclass
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=hidden_dim,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        if multiclass:
            self.fc = nn.Linear(hidden_dim * 2, targets)
            self.activation = nn.LogSoftmax(dim=1)
            self.loss_fn = nn.NLLLoss()
        else:
            self.fc = nn.Linear(hidden_dim * 2, targets)
            self.activation = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate,
                                          betas=(0.5, 0.999))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        out = self.activation(out)
        return out

    def train_model(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print('Epoch [%d/%d], Loss: %.4f' % (
            epoch + 1, num_epochs, running_loss / (i + 1)))

    def evaluate_model(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy on test set: %d %%' % (accuracy))
