import torch.nn  as nn
import torch
import torch.nn.functional as F


class VFBaseLine(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), activation=F.relu):
        super(VFBaseLine, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = nn.ModuleList()
        self.activation = activation
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)
        self.output_layer = nn.Linear(in_size, output_size)

    def forward(self, input):
        x = input
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        output = self.output_layer(x)
        return output

    def fit(self, episode, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        v = self.forward(episode.observations).squeeze(1)

        loss = F.mse_loss(episode.rtg, v)
        loss.backward()
        optimizer.step()
