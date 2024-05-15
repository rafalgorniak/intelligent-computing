from torch import nn


class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomMLP, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x