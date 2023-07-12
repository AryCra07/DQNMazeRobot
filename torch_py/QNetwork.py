from abc import ABC

import torch.nn as nn
import torch


class QNetwork(nn.Module, ABC):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_hidden = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.final_fc = nn.Linear(256, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input_hidden(state)
        return self.final_fc(x)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = QNetwork(2, 4, 0).to(device)

    x = torch.tensor([1, 1]).float().unsqueeze(0).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        # Forward pass
        outputs = net(x)
        # Compute loss
        loss = criterion(outputs, target)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))