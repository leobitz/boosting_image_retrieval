import torch
import torch.nn as nn

class LastLayer(nn.Module):

    def __init__(self, n_input, n_output):
        super(LastLayer, self).__init__()
        self.fc1 = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.fc1(x)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, emb_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class SingleNet(nn.Module):

    def __init__(self, input_dim, vec_dim, out_dims):
        super(SingleNet, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, vec_dim)
        self.outs = nn.ModuleList(
            [LastLayer(vec_dim, out_dim) for out_dim in out_dims])

    def forward(self, img, tag, vec):
        x = torch.relu(self.fc1(vec))
        outs = [o(x) for o in self.outs]
        return outs

    def embs(self, x):
        x = torch.relu(self.fc1(x))
        return x

class DoubleNet(nn.Module):

    def __init__(self, input_dim, vec_dim, out_dims):
        super(DoubleNet, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, vec_dim)
        self.outs = nn.ModuleList(
            [LastLayer(vec_dim, out_dim) for out_dim in out_dims])

    def forward(self, img, tag, vec):
        x = torch.relu(self.fc1(vec))
        x = torch.relu(self.fc2(x))
        outs = [o(x) for o in self.outs]
        return outs

    def embs(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
