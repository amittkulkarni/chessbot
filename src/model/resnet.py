import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))))

class ChessResNet(nn.Module):
    """
    ResNet-20 Architecture optimized for Chess.
    Outputs Policy (move probabilities) and Value (W/D/L logits).
    """
    def __init__(self, num_features: int = 18, num_moves: int = 4096,
                 num_res_blocks: int = 20, num_channels: int = 256,
                 num_value_targets: int = 3):
        super().__init__()
        self.conv_input = nn.Conv2d(num_features, num_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual Tower
        self.res_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy Head
        self.p_conv = nn.Conv2d(num_channels, 32, 1)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 8 * 8, num_moves)

        # Value Head
        self.v_conv = nn.Conv2d(num_channels, 32, 1)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.v_fc2 = nn.Linear(128, num_value_targets)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.bn_input(self.conv_input(x)))
        x = self.res_tower(x)

        p = self.p_fc(torch.relu(self.p_bn(self.p_conv(x))).view(x.size(0), -1))
        v = self.v_fc2(torch.relu(self.v_fc1(torch.relu(self.v_bn(self.v_conv(x))).view(x.size(0), -1))))

        return p, v