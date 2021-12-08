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
"""Trains the deep imitative model on expert demostrations."""

import os
from typing import Mapping

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from absl import app
from absl import flags
from absl import logging
import numpy as np

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="dataset_dir",
    default=None,
    help="The full path to the processed dataset.",
)
flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="The full path to the output directory (for logs, ckpts).",
)
flags.DEFINE_integer(
    name="batch_size",
    default=512,
    help="The batch size used for training the neural network.",
)
flags.DEFINE_integer(
    name="num_epochs",
    default=None,
    help="The number of training epochs for the neural network.",
)
flags.DEFINE_integer(
    name="save_model_frequency",
    default=4,
    help="The number epochs between saves of the model.",
)
flags.DEFINE_float(
    name="learning_rate",
    default=1e-3,
    help="The ADAM learning rate.",
)
flags.DEFINE_integer(
    name="num_timesteps_to_keep",
    default=4,
    help="The numbers of time-steps to keep from the target, with downsampling.",
)
flags.DEFINE_float(
    name="weight_decay",
    default=0.0,
    help="The L2 penalty (regularization) coefficient.",
)
flags.DEFINE_bool(
    name="clip_gradients",
    default=False,
    help="If True it clips the gradients norm to 1.0.",
)
flags.DEFINE_integer(
    name="bootstrap_id",
    default=-1,
    help="Bootstrap id",
)
flags.DEFINE_integer(
    name="pretrain_epoch",
    default=-1,
    help="Pretrain Epoch",
)
flags.DEFINE_integer(
    name="gpu",
    default=0,
    help="GPU id",
)
flags.DEFINE_integer(
    name="num_workers",
    default=0,
    help="num_workers",
)
flags.DEFINE_integer(
    name="val_num_workers",
    default=0,
    help="val_num_workers",
)
flags.DEFINE_integer(
    name="validation_interval",
    default=1,
    help="validation_interval",
)
flags.DEFINE_integer(
    name="in_channels",
    default=2,
    help="# of input channels",
)


def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  # Parses command line arguments.
  bootstrap_id = FLAGS.bootstrap_id
  dataset_dir = FLAGS.dataset_dir
  output_dir = FLAGS.output_dir + '_bootstrap{}'.format(bootstrap_id) if bootstrap_id != -1 else FLAGS.output_dir
  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs
  learning_rate = FLAGS.learning_rate
  save_model_frequency = FLAGS.save_model_frequency
  num_timesteps_to_keep = FLAGS.num_timesteps_to_keep
  weight_decay = FLAGS.weight_decay
  clip_gradients = FLAGS.clip_gradients
  pretrain_epoch = FLAGS.pretrain_epoch
  num_workers = FLAGS.num_workers
  val_num_workers = FLAGS.val_num_workers
  validation_interval = FLAGS.validation_interval
  in_channels = FLAGS.in_channels


  print('GPU: ' + str(FLAGS.gpu))
  os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
  from oatomobile.baselines.torch.dim.model import ImitativeModel
  from oatomobile.baselines.torch.logging import Checkpointer
  from oatomobile.baselines.torch.logging import TensorBoardWriter
  from oatomobile.baselines.torch.typing import ArrayLike
  from oatomobile.datasets.carla import CARLADataset



  noise_level = 1e-2

  # Determines device, accelerator.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member

  # Creates the necessary output directory.
  os.makedirs(output_dir, exist_ok=True)
  log_dir = os.path.join(output_dir, "logs")
  os.makedirs(log_dir, exist_ok=True)
  ckpt_dir = os.path.join(output_dir, "ckpts")
  os.makedirs(ckpt_dir, exist_ok=True)

  # Initializes the model and its optimizer.
  output_shape = [num_timesteps_to_keep, 2]
  model = ImitativeModel(output_shape=output_shape, in_channels=in_channels).to(device)
  optimizer = optim.Adam(
      model.parameters(),
      lr=learning_rate,
      weight_decay=weight_decay,
  )
  writer = TensorBoardWriter(log_dir=log_dir)
  checkpointer = Checkpointer(model=model, ckpt_dir=ckpt_dir)
  if pretrain_epoch != -1:
    print('Loading the parameters of epoch {}.'.format(pretrain_epoch))
    checkpointer.load(pretrain_epoch)

  def transform(batch: Mapping[str, ArrayLike]) -> Mapping[str, torch.Tensor]:
    """Preprocesses a batch for the model.

    Args:
      batch: (keyword arguments) The raw batch variables.

    Returns:
      The processed batch.
    """
    # Sends tensors to `device`.
    batch = {key: tensor.to(device) for (key, tensor) in batch.items()}
    # Preprocesses batch for the model.
    batch = model.transform(batch)
    return batch

  # Setups the dataset and the dataloader.
  modalities = (
      "lidar",
      "is_at_traffic_light",
      "traffic_light_state",
      "player_future",
      "velocity",
  )
  
  bootstrap_ids = None
  if bootstrap_id != -1:
    bootstrap_ids = np.load(os.path.join(dataset_dir, 'bootstrap{}.npy'.format(bootstrap_id)), allow_pickle=True)

  dataset_train = CARLADataset.as_torch(
      dataset_dir=os.path.join(dataset_dir, "train"),
      modalities=modalities,
      bootstrap_ids=bootstrap_ids
  )
  dataloader_train = torch.utils.data.DataLoader(
      dataset_train,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
  )
  
  if validation_interval > 0:
    dataset_val = CARLADataset.as_torch(
        dataset_dir=os.path.join(dataset_dir, "val"),
        modalities=modalities,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size * 5,
        shuffle=True,
        num_workers=val_num_workers,
    )

  # Theoretical limit of NLL.
  nll_limit = -torch.sum(  # pylint: disable=no-member
      D.MultivariateNormal(
          loc=torch.zeros(output_shape[-2] * output_shape[-1]),  # pylint: disable=no-member
          scale_tril=torch.eye(output_shape[-2] * output_shape[-1]) *  # pylint: disable=no-member
          noise_level,  # pylint: disable=no-member
      ).log_prob(torch.zeros(output_shape[-2] * output_shape[-1])))  # pylint: disable=no-member

  def train_step(
      model: ImitativeModel,
      optimizer: optim.Optimizer,
      batch: Mapping[str, torch.Tensor],
      clip: bool = False,
  ) -> torch.Tensor:
    """Performs a single gradient-descent optimisation step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    # Perturb target.
    y = torch.normal(  # pylint: disable=no-member
        mean=batch["player_future"][..., :2],
        std=torch.ones_like(batch["player_future"][..., :2]) * noise_level,  # pylint: disable=no-member
    )

    # Forward pass from the model.
    z = model._params(
        velocity=batch["velocity"],
        visual_features=batch["visual_features"],
        is_at_traffic_light=batch["is_at_traffic_light"],
        traffic_light_state=batch["traffic_light_state"],
    )
    _, log_prob, logabsdet = model._inverse(y=y, z=z)

    # Calculates loss (NLL).
    loss = -torch.mean(log_prob - logabsdet, dim=0)  # pylint: disable=no-member

    # Backward pass.
    loss.backward()

    # Clips gradients norm.
    if clip:
      torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    # Performs a gradient descent step.
    optimizer.step()

    return loss

  def train_epoch(
      model: ImitativeModel,
      optimizer: optim.Optimizer,
      dataloader: torch.utils.data.DataLoader,
  ) -> torch.Tensor:
    """Performs an epoch of gradient descent optimization on `dataloader`."""
    model.train()
    loss = 0.0
    with tqdm.tqdm(dataloader) as pbar:
      for batch in pbar:
        # Prepares the batch.
        batch = transform(batch)
        # Performs a gradien-descent step.
        loss += train_step(model, optimizer, batch, clip=clip_gradients)
    return loss / len(dataloader)

  def evaluate_step(
      model: ImitativeModel,
      batch: Mapping[str, torch.Tensor],
  ) -> torch.Tensor:
    """Evaluates `model` on a `batch`."""
    # Forward pass from the model.
    z = model._params(
        velocity=batch["velocity"],
        visual_features=batch["visual_features"],
        is_at_traffic_light=batch["is_at_traffic_light"],
        traffic_light_state=batch["traffic_light_state"],
    )
    _, log_prob, logabsdet = model._inverse(
        y=batch["player_future"][..., :2],
        z=z,
    )

    # Calculates loss (NLL).
    loss = -torch.mean(log_prob - logabsdet, dim=0)  # pylint: disable=no-member

    return loss

  def evaluate_epoch(
      model: ImitativeModel,
      dataloader: torch.utils.data.DataLoader,
  ) -> torch.Tensor:
    """Performs an evaluation of the `model` on the `dataloader."""
    model.eval()
    loss = 0.0
    with tqdm.tqdm(dataloader) as pbar:
      for batch in pbar:
        # Prepares the batch.
        batch = transform(batch)
        # Accumulates loss in dataset.
        with torch.no_grad():
          loss += evaluate_step(model, batch)
    return loss / len(dataloader)

  def write(
      model: ImitativeModel,
      dataloader: torch.utils.data.DataLoader,
      writer: TensorBoardWriter,
      split: str,
      loss: torch.Tensor,
      epoch: int,
  ) -> None:
    """Visualises model performance on `TensorBoard`."""
    # Gets a sample from the dataset.
    batch = next(iter(dataloader))
    # Prepares the batch.
    batch = transform(batch)
    # Turns off gradients for model parameters.
    for params in model.parameters():
      params.requires_grad = False
    # Generates predictions.
    predictions = model(num_steps=20, **batch)
    # Turns on gradients for model parameters.
    for params in model.parameters():
      params.requires_grad = True
    # Logs on `TensorBoard`.
    writer.log(
        split=split,
        loss=loss.detach().cpu().numpy().item(),
        overhead_features=batch["visual_features"].detach().cpu().numpy()[:8],
        predictions=predictions.detach().cpu().numpy()[:8],
        ground_truth=batch["player_future"].detach().cpu().numpy()[:8],
        global_step=epoch,
    )

  with tqdm.tqdm(range(pretrain_epoch + 1, num_epochs)) as pbar_epoch:
    for epoch in pbar_epoch:
      # Trains model on whole training dataset, and writes on `TensorBoard`.
      loss_train = train_epoch(model, optimizer, dataloader_train)
      write(model, dataloader_train, writer, "train", loss_train, epoch)

      # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
      loss_val = None
      if validation_interval > 0 and epoch % validation_interval == 0:
        loss_val = evaluate_epoch(model, dataloader_val)
        write(model, dataloader_val, writer, "val", loss_val, epoch)

      # Checkpoints model weights.
      if epoch % save_model_frequency == 0:
        checkpointer.save(epoch)

      # Updates progress bar description.
      pbar_epoch.set_description(
          "Epoch {} -> TL: {:.2f} | VL: {:.2f} | THEORYMIN: {:.2f}".format(
              epoch,
              loss_train.detach().cpu().numpy().item(),
              loss_val.detach().cpu().numpy().item() if loss_val is not None else 0,
              nll_limit,
          ))


if __name__ == "__main__":
  flags.mark_flag_as_required("dataset_dir")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("num_epochs")
  app.run(main)
