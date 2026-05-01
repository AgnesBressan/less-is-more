import torch
import pickle
import torch.nn as nn
from pathlib import Path

# Taken from https://github.com/leggedrobotics/elevation_mapping_cupy/blob/main/elevation_mapping_cupy/script/elevation_mapping_cupy/traversability_filter.py


class TraversabilityFilter(nn.Module):
    def __init__(self, w1, w2, w3, w_out, use_bias=False):
        super(TraversabilityFilter, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=3, bias=use_bias)
        self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=3, bias=use_bias)
        self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=3, bias=use_bias)
        self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

        self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
        self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
        self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
        self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

    def __call__(self, elevation):
        with torch.no_grad():
            elevation = elevation.unsqueeze(0)
            out1 = self.conv1(elevation)
            out2 = self.conv2(elevation)
            out3 = self.conv3(elevation)

            out1 = out1[:, :, 2:-2, 2:-2]
            out2 = out2[:, :, 1:-1, 1:-1]
            out = torch.cat((out1, out2, out3), dim=1)
            out = self.conv_out(out.abs())
            out = torch.exp(-out)

        return out.squeeze()


def get_filter_torch(device: str = "cuda") -> TraversabilityFilter:
    current_dir = Path(__file__).parent
    weights_path = current_dir / "weights.dat"

    with open(weights_path, "rb") as file:
        weights = pickle.load(file)
        w1 = weights["conv1.weight"]
        w2 = weights["conv2.weight"]
        w3 = weights["conv3.weight"]
        w_out = weights["conv_final.weight"]

    return TraversabilityFilter(w1, w2, w3, w_out).to(device).eval()
