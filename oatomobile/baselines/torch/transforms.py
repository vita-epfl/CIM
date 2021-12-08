# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformations and preprocessing used across the `PyTorch` models."""

import torch
import torch.nn.functional as F
import numpy as np
from oatomobile.core.typing import ShapeLike
import torchvision.transforms as transforming

def convert_color(x, rgb1, rgb2, device):
    r1, g1, b1 = rgb1  # Original value
    r2, g2, b2 = rgb2  # Value that we want to replace it with

    red, green, blue = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
    mask = (red == r1) & (green == g1) & (blue == b1)
    x[:, 0, :, :][mask] = r2
    x[:, 1, :, :][mask] = g2
    x[:, 2, :, :][mask] = b2

    torch.FloatTensor([r2, g2, b2]).to(device)

    return x

def RGB_to_gray(x, device):
    to_gray = transforming.Compose(
        [transforming.Grayscale(num_output_channels=1), transforming.ToTensor()])
         # transforming.Normalize(mean=[0.5], std=[0.5])])

    img = x.cpu()
    img = [transforming.ToPILImage()(x) for x in img]
    img = [to_gray(x) for x in img]
    return torch.stack(img).to(device)

def downsample_target(
    player_future: torch.Tensor,
    num_timesteps_to_keep: int,
) -> torch.Tensor:
  """Downsamples the target sequence."""
  _, T, _ = player_future.shape
  increments = T // num_timesteps_to_keep
  player_future = player_future[:, 0::increments, :]
  return player_future


def downsample_visual_features(
    visual_features: torch.Tensor,
    output_shape: ShapeLike,
) -> torch.Tensor:
  """Downsamples the visual features."""
  return F.interpolate(
      visual_features,
      size=output_shape,
      mode="bilinear",
      align_corners=True,
  )


def transpose_visual_features(visual_features: torch.Tensor) -> torch.Tensor:
  """Transposes the visual features."""
  return torch.transpose(visual_features, dim0=2, dim1=3)  # pylint: disable=no-member
