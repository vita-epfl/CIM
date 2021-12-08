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
"""Implements the deep imitative model-based agent."""

from collections import deque
import scipy.interpolate
import oatomobile
from oatomobile.baselines.alpha_base import SetPointAgent
from oatomobile.baselines.torch.cim.model import *
from oatomobile.baselines.torch.cim.predictor.model import *


class CIMAgent(SetPointAgent):
    """The deep imitative model agent."""

    def __init__(self, lags, horizon, environment: oatomobile.envs.CARLAEnv, *,
                 model: ImitativeModel, speedmodule: MLP, alpha = 0.2, gamma = 0, trivial = -1, **kwargs) -> None:
        """Constructs a deep imitation model agent.

        Args:
          environment: The navigation environment to spawn the agent.
          model: The deep imitative model.
        """
        super(CIMAgent, self).__init__(environment=environment, alpha=alpha, gamma=gamma, **kwargs)

        # Determines device, accelerator.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
        self._model = model.to(self._device)
        self._speedmodule = speedmodule.to(self._device)
        self._lags = lags
        self._speeds = [deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon)]
        self._states = [deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon)]
        self._speed = [[], [], [], [], []]
        self._preds = [[], [], [], [], []]
        self._trivial = trivial

    def __call__(self, observation: Mapping[str, np.ndarray],
                 **kwargs) -> np.ndarray:
        """Returns the imitative prior."""

        # Prepares observation for the neural-network.
        observation["overhead_features"] = observation[
            "bird_view_camera_cityscapes"]
        for attr in observation:
            if not isinstance(observation[attr], np.ndarray):
                observation[attr] = np.atleast_1d(observation[attr])
            observation[attr] = observation[attr][None, ...].astype(np.float32)

        # Makes `goal` 2D.
        observation["goal"] = observation["goal"][..., :2]
        # Convert image to CHW.
        observation["bird_view_camera_cityscapes"] = np.transpose(observation["bird_view_camera_cityscapes"], (0, 3, 1, 2))
        observation["lidar"] = np.transpose(observation["lidar"], (0, 3, 1, 2))
        # Processes observations for the `ImitativeModel`.
        observation = {
            key: torch.from_numpy(tensor).to(self._device)  # pylint: disable=no-member
            for (key, tensor) in observation.items()
        }
        observation = self._model.transform(observation)

        # Queries model.
        plan, state = self._model(num_steps=kwargs.get("num_steps", 20),
                           epsilon=kwargs.get("epsilon", 1.0),
                           lr=kwargs.get("lr", 5e-2),
                           **observation)
        plan = plan.detach().cpu().numpy()[0]  # [T, 2]

        # Interpolates plan.
        player_future_length = 40
        increments = player_future_length // plan.shape[0]
        time_index = list(range(0, player_future_length, increments))  # [T]
        plan_interp = scipy.interpolate.interp1d(x=time_index, y=plan, axis=0)
        xy = plan_interp(np.arange(0, time_index[-1]))

        # Appends z dimension.
        z = np.zeros(shape=(xy.shape[0], 1))

        # if observation['is_at_traffic_light'] and observation['traffic_light_state'] == carla.TrafficLightState.Red:
        #    xy = np.zeros_like(xy)

        causal_command = self.get_command(observation, state[0])

        return np.c_[xy, z], causal_command

    def get_command(self, observation, state, parents = [21, 31]):
        if self._trivial != -1:
            return self._trivial
        
        i = self._steps_counter % 5
        current_speed = np.linalg.norm(observation["velocity"].cpu().numpy())
        self._speeds[i].append(current_speed)
        self._states[i].append(state[parents])

        if len(self._speeds[i]) < self._lags:
            return -1
        
        states = torch.stack(list(self._states[i])[-self._lags:])[None, ...]
        next_speed = self._speedmodule(torch.transpose(states, 1, 2)).detach().cpu().numpy()
        
        self._speed[i].append(current_speed)
        self._preds[i].append(next_speed[0,0])

        return next_speed[0,0]

class CIMMLPAgent(SetPointAgent):
    """The deep imitative model agent."""

    def __init__(self, lags, horizon, environment: oatomobile.envs.CARLAEnv, *,
                 model: ImitativeModel, speedmodule: FullMLP, alpha = 0.2, **kwargs) -> None:
        """Constructs a deep imitation model agent.

        Args:
          environment: The navigation environment to spawn the agent.
          model: The deep imitative model.
        """
        super(CIMMLPAgent, self).__init__(environment=environment, alpha = alpha, **kwargs)

        # Determines device, accelerator.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
        self._model = model.to(self._device)
        self._speedmodule = speedmodule.to(self._device)
        self._lags = lags
        self._speeds = [deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon)]
        self._states = [deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon), deque(maxlen=horizon)]
        self._speed = [[], [], [], [], []]
        self._preds = [[], [], [], [], []]
        
    def __call__(self, observation: Mapping[str, np.ndarray],
                 **kwargs) -> np.ndarray:
        """Returns the imitative prior."""

        # Prepares observation for the neural-network.
        observation["overhead_features"] = observation[
            "bird_view_camera_cityscapes"]
        for attr in observation:
            if not isinstance(observation[attr], np.ndarray):
                observation[attr] = np.atleast_1d(observation[attr])
            observation[attr] = observation[attr][None, ...].astype(np.float32)

        # Makes `goal` 2D.
        observation["goal"] = observation["goal"][..., :2]
        # Convert image to CHW.
        observation["bird_view_camera_cityscapes"] = np.transpose(observation["bird_view_camera_cityscapes"], (0, 3, 1, 2))
        observation["lidar"] = np.transpose(observation["lidar"], (0, 3, 1, 2))
        # Processes observations for the `ImitativeModel`.
        observation = {
            key: torch.from_numpy(tensor).to(self._device)  # pylint: disable=no-member
            for (key, tensor) in observation.items()
        }
        observation = self._model.transform(observation)

        # Queries model.
        plan, state = self._model(num_steps=kwargs.get("num_steps", 20),
                           epsilon=kwargs.get("epsilon", 1.0),
                           lr=kwargs.get("lr", 5e-2),
                           **observation)
        plan = plan.detach().cpu().numpy()[0]  # [T, 2]

        # Interpolates plan.
        player_future_length = 40
        increments = player_future_length // plan.shape[0]
        time_index = list(range(0, player_future_length, increments))  # [T]
        plan_interp = scipy.interpolate.interp1d(x=time_index, y=plan, axis=0)
        xy = plan_interp(np.arange(0, time_index[-1]))

        # Appends z dimension.
        z = np.zeros(shape=(xy.shape[0], 1))

        # if observation['is_at_traffic_light'] and observation['traffic_light_state'] == carla.TrafficLightState.Red:
        #    xy = np.zeros_like(xy)

        causal_command = self.get_command(observation, state[0])

        return np.c_[xy, z], causal_command

    def get_command(self, observation, state):
        i = self._steps_counter % 5
        current_speed = np.linalg.norm(observation["velocity"].cpu().numpy())
        self._speeds[i].append(current_speed)
        self._states[i].append(state)

        if len(self._speeds[i]) < self._lags:
            return -1

        states = torch.stack(list(self._states[i])[-self._lags:])[None, ...]
        next_speed = self._speedmodule(torch.transpose(states, 1, 2)).detach().cpu().numpy()

        self._speed[i].append(current_speed)
        self._preds[i].append(next_speed[0,0])

        return next_speed[0,0]