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

import torch
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
    default=256,
    help="The batch size used for training the neural network.",
)
flags.DEFINE_integer(
    name="num_epochs",
    default=None,
    help="The number of training epochs for the neural network.",
)

flags.DEFINE_float(
    name="learning_rate",
    default=8e-3,
    help="The ADAM learning rate.",
)
flags.DEFINE_integer(
    name="gpu",
    default=0,
    help="GPU id",
)


def main(argv):
    # Debugging purposes.
    logging.debug(argv)
    logging.debug(FLAGS)

    # Parses command line arguments.
    dataset_dir = FLAGS.dataset_dir
    output_dir = FLAGS.output_dir
    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs
    learning_rate = FLAGS.learning_rate

    print('GPU: ' + str(FLAGS.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    from oatomobile.baselines.torch.cim.predictor.model import MLP
    from oatomobile.baselines.torch.logging import Checkpointer
    from oatomobile.baselines.torch.logging import TensorBoardWriter

    # Determines device, accelerator.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member

    # Creates the necessary output directory.
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = MLP().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    writer = TensorBoardWriter(log_dir=log_dir)
    checkpointer = Checkpointer(model=model, ckpt_dir=ckpt_dir)


    def train_step(
            model: MLP,
            optimizer: optim.Optimizer,
            batch_size: int,
            X,
            y
    ) -> torch.Tensor:
        """Performs a single gradient-descent optimisation step."""
        # Resets optimizer's gradients.
        optimizer.zero_grad()
        p = np.random.choice(len(X), batch_size, replace=False)
        loss = model.loss(X[p], y[p])
        loss.backward()
        optimizer.step()

        return loss

    def train_epoch(model, optimizer, X, y, batch_size) -> torch.Tensor:
        """Performs an epoch of gradient descent optimization on `dataloader`."""
        model.train()
        losses = {'loss': 0.0}
        for _ in range(len(X) // batch_size):
            step_losses = train_step(model, optimizer, batch_size, X, y)
            losses = {key: losses.get(key, 0) + step_losses.get(key, 0) for key in set(losses) | set(step_losses)}

        return losses

    def write(
            writer: TensorBoardWriter,
            split: str,
            loss: torch.Tensor,
            epoch: int,
    ) -> None:
        """Visualises model performance on `TensorBoard`."""
        writer.log(
            split=split,
            loss=loss.detach().cpu().numpy().item(),
            global_step=epoch,
            overhead_features=None,
            predictions=None,
            ground_truth=None
        )

    with tqdm.tqdm(range(num_epochs)) as pbar_epoch:
        X = torch.FloatTensor(np.load(dataset_dir + 'X.npy'))
        y = torch.FloatTensor(np.load(dataset_dir + 'y.npy'))

        for epoch in pbar_epoch:
            # Trains model on whole training dataset, and writes on `TensorBoard`.
            losses_train = train_epoch(model, optimizer, X, y, batch_size)
            write(writer, "train", losses_train['loss'], epoch)
            checkpointer.save(epoch)

            # Updates progress bar description.
            pbar_epoch.set_description(
                "Epoch {} -> loss: {:.2f}".format(
                    epoch,
                    losses_train['loss'].detach().cpu().numpy().item(),
                ))


if __name__ == "__main__":
    flags.mark_flag_as_required("dataset_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("num_epochs")
    app.run(main)
