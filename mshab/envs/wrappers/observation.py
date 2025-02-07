from collections import deque
from typing import Dict, List, Optional

import gymnasium as gym

import numpy as np
import torch

import sapien.physx as physx

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.common import flatten_state_dict


class FetchDepthObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, cat_state=True, cat_pixels=False) -> None:
        super().__init__(env)

        self.cat_pixels = cat_pixels
        self.cat_state = cat_state
        self._stack_fn = torch.stack if physx.is_gpu_enabled() else np.stack
        self._cat_fn = torch.cat if physx.is_gpu_enabled() else np.concatenate

        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = self._base_env._init_raw_obs

        if physx.is_gpu_enabled():
            init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)

        self._base_env.update_obs_space(common.to_numpy(self.observation(init_raw_obs)))

    def observation(self, observation):
        agent_obs = observation["agent"]
        extra_obs = observation["extra"]

        fetch_hand_depth = observation["sensor_data"]["fetch_hand"]["depth"].permute(
            0, 3, 1, 2
        )
        fetch_hand_rgb = observation["sensor_data"]["fetch_hand"]["rgb"].permute(
            0, 3, 1, 2
        )
        
        fetch_head_depth = observation["sensor_data"]["fetch_head"]["depth"].permute(
            0, 3, 1, 2
        )
        fetch_head_rgb = observation["sensor_data"]["fetch_head"]["rgb"].permute(
            0, 3, 1, 2
        )


        # import pdb
        # pdb.set_trace()

        ############################# SH
        # fetch_hand_pose = observation['sensor_param']['fetch_hand']['cam2world_gl']
        # fetch_head_pose = observation['sensor_param']['fetch_head']['cam2world_gl']

        fetch_hand_pose = observation['sensor_param']['fetch_hand']['extrinsic_cv']
        fetch_head_pose = observation['sensor_param']['fetch_head']['extrinsic_cv']

        depth_pixels = (
            dict(
                all_depth=self._stack_fn([fetch_hand_depth, fetch_hand_rgb], axis=-3)
            )
            if self.cat_pixels
            else dict(
                fetch_hand_depth=fetch_hand_depth,
                fetch_hand_rgb=fetch_hand_rgb,
                fetch_head_depth=fetch_head_depth,
                fetch_head_rgb=fetch_head_rgb,     
                fetch_hand_pose=fetch_hand_pose,
                fetch_head_pose=fetch_head_pose
            )
        )
        return (
            dict(
                state=self._cat_fn(
                    [
                        flatten_state_dict(agent_obs, use_torch=physx.is_gpu_enabled()),
                        flatten_state_dict(extra_obs, use_torch=physx.is_gpu_enabled()),
                    ],
                    axis=int(physx.is_gpu_enabled()),
                ),
                **depth_pixels,
            )
            if self.cat_state
            else dict(
                agent=agent_obs,
                extra=extra_obs,
                **depth_pixels,
            )
        )


# TODO (arth): deprecate this in favor of StackedDictObservationWrapper + stacking_keys
#   will need to update rl and bc train scripts to matchs new output (i.e. no "pixels" key)
class FrameStack(gym.Wrapper):
    def __init__(
        self,
        env,
        num_stack: int,
        stacking_keys: List[str] = ["fetch_head_rgb"],
    ) -> None:
        super().__init__(env)
        self._base_env = env.unwrapped
        self._num_stack = num_stack
        self._stacking_keys = stacking_keys

        assert all([k in env.observation_space.spaces for k in stacking_keys])

        self._frames: Dict[str, deque] = dict()

        init_raw_obs: dict = self._base_env._init_raw_obs
        pixel_init_raw_obs = dict()
        self._stack_dim = dict()
        for sk in self._stacking_keys:
            obs_space = self.observation_space.spaces[sk]
            if physx.is_gpu_enabled():
                init_raw_obs_sk_replace = init_raw_obs.pop(sk)[:, None, ...]
                stack_dim = -len(obs_space.shape[1:]) - 1
            else:
                init_raw_obs_sk_replace = init_raw_obs.pop(sk)[None, ...]
                stack_dim = -len(obs_space.shape) - 1

            pixel_init_raw_obs[sk] = np.repeat(
                init_raw_obs_sk_replace, num_stack, axis=stack_dim
            )
            self._frames[sk] = deque(maxlen=num_stack)
            self._stack_dim[sk] = stack_dim
        init_raw_obs["pixels"] = pixel_init_raw_obs
        self._base_env.update_obs_space(init_raw_obs)

        self._stack_fn = torch.stack if physx.is_gpu_enabled() else np.stack

    def _get_stacked_frames(self):
        return dict(
            (sk, self._stack_fn(tuple(self._frames[sk]), axis=self._stack_dim[sk]))
            for sk in self._stacking_keys
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            frame = obs.pop(sk)
            for _ in range(self._num_stack):
                self._frames[sk].append(frame)
        obs["pixels"] = self._get_stacked_frames()
        return obs, info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            self._frames[sk].append(obs.pop(sk))
        obs["pixels"] = self._get_stacked_frames()
        return obs, rew, term, trunc, info


class StackedDictObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        num_stack: int,
        stacking_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(env)
        self._base_env: BaseEnv = env.unwrapped
        self._num_stack = num_stack

        if stacking_keys is None:
            assert isinstance(env.single_observation_space, gym.spaces.Dict)
            self._stacking_keys = env.single_observation_space.keys()
        else:
            assert all([k in env.observation_space.spaces for k in stacking_keys])
            self._stacking_keys = stacking_keys

        self._running_stacks: Dict[str, deque] = dict()

        init_raw_obs: dict = self._base_env._init_raw_obs
        stacked_init_raw_obs = dict()
        self._stack_dim = dict()
        for sk in self._stacking_keys:
            obs_space = self.observation_space.spaces[sk]
            if physx.is_gpu_enabled():
                init_raw_obs_sk_replace = init_raw_obs.pop(sk)[:, None, ...]
                stack_dim = -len(obs_space.shape[1:]) - 1
            else:
                init_raw_obs_sk_replace = init_raw_obs.pop(sk)[None, ...]
                stack_dim = -len(obs_space.shape) - 1

            stacked_init_raw_obs[sk] = np.repeat(
                init_raw_obs_sk_replace, num_stack, axis=stack_dim
            )
            self._running_stacks[sk] = deque(maxlen=num_stack)
            self._stack_dim[sk] = stack_dim
        init_raw_obs.update(**stacked_init_raw_obs)
        self._base_env.update_obs_space(init_raw_obs)

        self._stack_fn = torch.stack if physx.is_gpu_enabled() else np.stack

    def _get_stacked_obs(self):
        return dict(
            (
                sk,
                self._stack_fn(
                    tuple(self._running_stacks[sk]), axis=self._stack_dim[sk]
                ),
            )
            for sk in self._stacking_keys
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            frame = obs.pop(sk)
            for _ in range(self._num_stack):
                self._running_stacks[sk].append(frame)
        return self._get_stacked_obs(), info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(*args, **kwargs)
        obs: dict
        for sk in self._stacking_keys:
            self._running_stacks[sk].append(obs.pop(sk))
        return self._get_stacked_obs(), rew, term, trunc, info

