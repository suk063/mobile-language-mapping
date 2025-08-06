import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Union

import gymnasium as gym
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import transforms3d as t3d
import trimesh
from dacite import from_dict
from omegaconf import OmegaConf
from sapien import physx
from tqdm import tqdm

from mani_skill import Actor
from mani_skill.agents.base_agent import Keyframe
from mani_skill.sensors.camera import Camera
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.replicacad.rearrange.scene_builder import (
    ReplicaCADRearrangeSceneBuilder,
)
from mani_skill.utils.structs.pose import Pose
from mshab.envs.make import EnvConfig
from mshab.envs.pick import PickSubtaskTrainEnv
from mshab.envs.place import PlaceSubtaskTrainEnv
from mshab.envs.planner import plan_data_from_file
from mshab.utils.config import parse_cfg

image_height = 224
image_width = 224


def get_tqdm_bar(iterable, desc: str, leave: bool = True):
    return tqdm(iterable, desc=desc, ncols=80, leave=leave)


class FakeLink:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

        self.pose = Pose.create(sapien.Pose())
        self.pose.raw_pose = self.pose.raw_pose.expand((self.num_envs, 7))
        self.pose.raw_pose.to(self.device)
        self.linear_velocity = torch.zeros((self.num_envs, 3), device=self.device)


class FakeRobot:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

        self.qpos = torch.zeros((self.num_envs, 15), device=self.device)
        self.qvel = torch.zeros((self.num_envs, 15), device=self.device)

    def set_pose(self, pose):
        # print(f"Setting robot pose to {pose}")
        pass

    def set_qpos(self, qpos):
        # print(f"Setting robot qpos to {qpos}")
        pass

    def get_net_contact_forces(self, link_names):
        f = torch.zeros((self.num_envs, len(link_names), 3))
        if physx.is_gpu_enabled():
            f = f.cuda()
        return f

    def hide_visual(self):
        pass


class FakeController:
    def reset(self):
        pass


class FakeAgent:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

        self._sensor_configs = []
        self.robot = FakeRobot(self.num_envs, self.device)
        self.keyframes = dict(rest=Keyframe(pose=sapien.Pose(), qpos=np.zeros(15)))
        self.controller = FakeController()
        self.tcp = FakeLink(self.num_envs, self.device)
        self.base_link = FakeLink(self.num_envs, self.device)

        self.single_action_space = gym.spaces.Box(0, 1, dtype=np.float32)
        self.action_space = self.single_action_space
        if self.num_envs > 1:
            self.action_space = gym.vector.utils.batch_space(self.action_space, self.num_envs)

    def reset(self, init_qpos: torch.Tensor = None):
        pass

    def is_grasping(self, objects, min_force=0.5, max_angle=85):
        return torch.tensor([False] * self.num_envs, device=self.device)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        return torch.tensor([False] * self.num_envs, device=self.device)

    @property
    def tcp_pose(self) -> Pose:  # tcp: tool control point
        pose = Pose.create(sapien.Pose())
        pose.raw_pose = pose.raw_pose.expand((self.num_envs, 7))
        pose.raw_pose.to(self.device)
        return pose

    def get_proprioception(self):
        return dict(
            qpos=torch.zeros((self.num_envs, 15), device=self.device),
            qvel=torch.zeros((self.num_envs, 15), device=self.device),
        )

    def before_simulation_step(self):
        pass

    def set_action(self, action):
        pass


def create_env_cls(env_name="StaticPickEnv", parent_env=PickSubtaskTrainEnv):
    @register_env(env_name, max_episode_steps=200)
    class StaticEnv(parent_env):

        def __init__(self, *args, load_agent=False, hide_episode_objects=False, **kwargs):
            self.load_agent = load_agent
            self.hide_episode_objects = hide_episode_objects
            self.camera_mount: Actor = None
            self.camera_mount_offset = 0.01  # make sure the mount is behind the camera
            super().__init__(*args, **kwargs)

            self.set_episode_objects()

        def set_episode_objects(self):
            if self.hide_episode_objects:
                scene_builder: ReplicaCADRearrangeSceneBuilder = self.scene_builder
                for env_ycb_objects in scene_builder.ycb_objs_per_env:
                    for obj_name, objs in env_ycb_objects.items():
                        for obj in objs:
                            scene_builder.hide_actor(obj)

        def step(self, action):
            if self.load_agent:
                super().step(action)
                self.set_episode_objects()
                return
            self.set_episode_objects()

        def _load_agent(self, options: dict):
            if self.load_agent:
                super()._load_agent(options)
            else:
                self.agent = FakeAgent(self.num_envs, self.device)
            self.camera_mount = actors.build_sphere(
                self.scene,
                radius=0.001,
                color=np.array([0.0, 0.0, 0.0, 1.0]),
                name="sphere",
                body_type="kinematic",  # kinematic so it doesn't fall
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            # we should not hide the camera mount, otherwise cameras mounted to it will not be rendered
            # self._hidden_objects.append(self.camera_mount)

        def _after_reconfigure(self, options):
            if self.load_agent:
                super()._after_reconfigure(options)
                return
            self.force_articulation_link_ids = []
            self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
            self.spawn_data = torch.load(self.spawn_data_fp, map_location=self.device)

        @property
        def _default_human_render_camera_configs(self):
            # if self.load_agent:
            #     return super()._default_human_render_camera_configs

            # this camera follows the robot around (though might be in walls if the space is cramped)
            robot_camera_pose = sapien_utils.look_at([self.camera_mount_offset, 0, 0], ([1.0, 0.0, 0.0]))
            robot_camera_config = CameraConfig(
                uid="render_camera",
                pose=robot_camera_pose,
                width=image_width,
                height=image_height,
                fov=1.75,
                near=0.01,
                far=10,
                mount=self.camera_mount,
            )
            # if mount is None, then camera.get_local_pose() == camera.get_global_pose()
            # however, the camera is static and camera.set_local_pose() does not work
            # a workaround is to mount the camera to a kinematic object, set the camera pose to identity
            # and set the kinematic object pose to the camera pose when we want to move the camera
            return robot_camera_config

        @property
        def scene_bounding_box(self):
            vertices = []
            for mesh in self.scene_builder.bg.get_collision_meshes():
                mesh: trimesh.Trimesh
                vertices.append(np.array(mesh.vertices))
            if len(vertices) == 1:
                vertices = vertices[0]
            else:
                vertices = np.concatenate(vertices, axis=0)
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            return bbox_min, bbox_max
            # for name, actor in self.scene.actors.items():  # include all scene objects and bg
            #     actor
            #     print(name)
            # # include all scene objects
            # self.scene_builder.scene_objects
            # # include all articulated objects
            # self.scene_builder.articulations

        def render_at_poses(self, poses: np.ndarray = None) -> dict:
            camera = self.scene.human_render_cameras["render_camera"]
            if poses is not None:  # set the camera pose
                # t0 = np.array([self.camera_mount_offset, 0.0, 0.0], dtype=np.float32)
                # t1 = pose.p
                # pose.p = t1 - t3d.quaternions.quat2mat(pose.q) @ t0
                for i in range(len(poses)):  # (N, 7), [p, q]
                    rot = t3d.quaternions.quat2mat(poses[i, 3:])
                    poses[i, :3] -= self.camera_mount_offset * rot[:, 0]
                self.camera_mount.set_pose(torch.from_numpy(poses).to(self.device))
                # assert np.allclose(t1, camera.camera.get_global_pose().p[0].cpu().numpy())
                # camera.camera.set_local_pose(pose)
                # print(pose.to_transformation_matrix())
                # print(camera.camera.get_extrinsic_matrix())  # world to cam, OpenCV convention
                # print(camera.camera.get_model_matrix())  # cam to world, OpenGL convention
                # print(camera.camera.get_local_pose())  # same as get_global_pose() if mount is None
                # print(camera.camera.get_global_pose())
            # https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html
            # self.scene.step()  # env.step() is called so that this line is not needed
            self.scene.update_render()  # update scene: object states, camera, etc.
            camera.camera.take_picture()  # start the rendering process
            # position can be obtained from the camera's intrinsic and depth
            obs = {k: v for k, v in camera.get_obs(position=False).items()}
            # https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html#rgb-depth-segmentation
            # rgb: uint8
            # depth: int16 (in mm)
            # position: int16 (in mm, in optical frame, OpenGL convention)
            # segmentation: int16 (0 for the distant background)
            # OpenGL convention: x to right, y up, z back
            # OpenCV convention: x to right, y down, z forward
            if "position" in obs:
                # OpenGL to OpenCV convention (y-axis and z-axis flipped)
                obs["position"][..., 1] *= -1
                obs["position"][..., 2] *= -1
            obs["cam_pose"] = poses  # np.concatenate([pose.p, pose.q])
            obs["extrinsic_cv"] = camera.camera.get_extrinsic_matrix()
            obs["intrinsic_cv"] = camera.camera.get_intrinsic_matrix()
            return obs

        def render_sensor(self, sensor_name: str) -> dict:
            camera: Camera = self.scene.sensors[sensor_name]
            camera.camera.take_picture()
            obs = {k: v for k, v in camera.get_obs(position=False).items()}
            if "position" in obs:
                # OpenGL to OpenCV convention (y-axis and z-axis flipped)
                obs["position"][..., 1] *= -1
                obs["position"][..., 2] *= -1
            obs["cam_pose"] = camera.camera.global_pose.raw_pose.cpu().numpy()
            obs["extrinsic_cv"] = camera.camera.get_extrinsic_matrix()
            obs["intrinsic_cv"] = camera.camera.get_intrinsic_matrix()
            return obs

    return StaticEnv


StaticPickEnv = create_env_cls("StaticPickEnv", PickSubtaskTrainEnv)
StaticPlaceEnv = create_env_cls("StaticPlaceEnv", PlaceSubtaskTrainEnv)


@dataclass
class Config:
    seed: int
    eval_env: EnvConfig
    episode_fp: str
    load_agent: bool
    hide_episode_objects: bool
    truncate_trajectory_at_success: bool
    fixed_task_plan_idx: int
    fixed_init_config_idx: int
    fixed_spawn_selection_idx: int
    visualize: bool
    visualize_rgb_gt: bool
    traj_h5: str
    output_file: str


def make_config() -> Config:
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    cfg = from_dict(data_class=Config, data=OmegaConf.to_container(cfg))
    cfg.eval_env.env_kwargs["load_agent"] = cfg.load_agent
    cfg.eval_env.env_kwargs["hide_episode_objects"] = cfg.hide_episode_objects
    return cfg


def visualize_pose(env: Union[StaticPickEnv, StaticPlaceEnv], pose: sapien.Pose):
    obs = env.render_at_poses(pose)

    plt.figure(figsize=(10, 10))
    n_rows = 1
    if "position" in obs:
        n_rows = 2

    plt.subplot(n_rows, 3, 1)
    depth = obs["depth"][:, :, 0].cpu().numpy()
    plt.imshow(depth, cmap="jet")
    plt.title("Depth")
    plt.colorbar(orientation="horizontal")

    plt.subplot(n_rows, 3, 2)
    rgb = obs["rgb"].cpu().numpy()
    plt.imshow(rgb)
    plt.title("RGB")

    plt.subplot(n_rows, 3, 3)
    segmentation = obs["segmentation"].cpu().numpy()
    plt.imshow(segmentation)
    plt.title("Segmentation")
    plt.colorbar(orientation="horizontal")

    if "position" in obs:
        positions = obs["position"].to(torch.float32)

        plt.subplot(n_rows, 3, 4)
        position_x = positions[:, :, 0].cpu().numpy()
        plt.imshow(position_x, cmap="jet")
        plt.colorbar(orientation="horizontal")
        plt.title("Position.x")

        plt.subplot(n_rows, 3, 5)
        position_y = positions[:, :, 1].cpu().numpy()
        plt.imshow(position_y, cmap="jet")
        plt.colorbar(orientation="horizontal")
        plt.title("Position.y")

        plt.subplot(n_rows, 3, 6)
        position_z = positions[:, :, 2].cpu().numpy()
        plt.imshow(position_z, cmap="jet")
        plt.colorbar(orientation="horizontal")
        plt.title("Position.z")

    plt.tight_layout()
    plt.show()


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        if self.cfg.load_agent:
            msg = "Loading agent in the environment. The agent actions will affect the scene."
            bar = f"{'WARNING':=^80}"
            print(bar)
            print(f"{msg:^80}")
            print(bar)

        output_dir = os.path.dirname(self.cfg.output_file)
        os.makedirs(output_dir, exist_ok=True)

        if self.cfg.visualize and self.cfg.visualize_rgb_gt:
            # use cache to speed up loading
            cache_file = os.path.join(output_dir, "cache.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    self.sensor_poses, self.actions, self.rgb_gt = pickle.load(f)
            else:
                self.sensor_poses, self.actions, self.rgb_gt = self.load_info_from_h5()
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump((self.sensor_poses, self.actions, self.rgb_gt), f)
        else:
            self.sensor_poses, self.actions, self.rgb_gt = self.load_info_from_h5()

        actions_0 = next(iter(self.actions.values()))
        self.actions_shape = actions_0.shape[1:]
        self.actions_dtype = actions_0.dtype
        self.seed, self.episode_configs = self.read_episode_fp()
        self.env = self.make_env()
        self.unwrapped: Union[StaticPickEnv, StaticPlaceEnv] = self.env.unwrapped

        # batch processing
        self.rgb_data: dict = None
        self.depth_data: dict = None
        self.seg_data: dict = None
        self.cam_pose_data: dict = None
        self.extrinsic_data: dict = None
        self.traj_lens: list = None
        self.traj_actions: Union[np.ndarray, torch.Tensor] = None
        self.traj_sensor_poses: dict = None
        self.max_traj_len = 0
        self.intrinsic: Union[np.ndarray, torch.Tensor] = None

        # visualization
        self.fig = None
        self.axes = None
        self.img_data = None
        if self.cfg.visualize:
            self.fig, self.axes = plt.subplots(1, 3 if self.rgb_gt is None else 4, figsize=(12, 4))
            self.axes[0].set_title("Depth")
            self.axes[1].set_title("Segmentation")
            self.axes[2].set_title("RGB")
            if self.rgb_gt is not None:
                self.axes[3].set_title("RGB GT")

    def load_info_from_h5(self):
        poses = dict()
        actions = dict()
        rgb = dict() if self.cfg.visualize_rgb_gt else None

        oRc_mat = np.array(  # rotation matrix from camera to optical frame
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )[np.newaxis]
        with h5py.File(self.cfg.traj_h5, "r") as f:
            # plt.imshow(f["traj_0"]["obs"]["sensor_data"]["fetch_head"]["rgb"][0])
            # plt.title("RGB Image from fetch_head")
            # plt.show()

            for traj_name in f.keys():
                traj = f[traj_name]["obs"]["sensor_param"]
                traj_poses = dict()
                success_cutoff = None
                if self.cfg.truncate_trajectory_at_success:
                    success = f[traj_name]["success"][:].tolist()
                    success_cutoff = min(success.index(True) + 1, len(success))
                    actions[traj_name] = f[traj_name]["actions"][:success_cutoff]
                else:
                    actions[traj_name] = f[traj_name]["actions"][:]
                traj_rgb = dict()
                for sensor_name in traj.keys():
                    extrinsic_matrices = traj[sensor_name]["extrinsic_cv"][...]
                    if success_cutoff is not None:
                        extrinsic_matrices = extrinsic_matrices[:success_cutoff]
                    cam_poses = np.zeros_like(extrinsic_matrices)
                    cam_poses[:, :3, :3] = extrinsic_matrices[:, :3, :3].transpose(0, 2, 1) @ oRc_mat
                    cam_poses[:, :3, [3]] = (
                        -extrinsic_matrices[:, :3, :3].transpose(0, 2, 1) @ extrinsic_matrices[:, :3, [3]]
                    )
                    if cam_poses.shape[1] == 4:
                        cam_poses[:, 3, 3] = 1.0  # homogeneous coordinate
                    traj_poses[sensor_name] = cam_poses
                    if self.cfg.visualize_rgb_gt:
                        sensor_data = f[traj_name]["obs"]["sensor_data"][sensor_name]
                        traj_rgb[sensor_name] = sensor_data["rgb"][:]
                poses[traj_name] = traj_poses
                if self.cfg.visualize_rgb_gt:
                    rgb[traj_name] = traj_rgb
        return poses, actions, rgb

    def read_episode_fp(self):
        with open(self.cfg.episode_fp, "r") as f:
            episode_data = json.load(f)
        seed = episode_data["episodes"][0]["episode_seed"][0]
        episode_configs = [
            {k: episode[k] for k in ["task_plan_idx", "init_config_idx", "spawn_selection_idx", "subtask_uid"]}
            for episode in episode_data["episodes"]
        ]
        with open(self.cfg.eval_env.task_plan_fp, "r") as f:
            plan_data = json.load(f)["plans"]
        for episode_config in episode_configs:
            tp = plan_data[episode_config["task_plan_idx"]]
            episode_config["build_config_name"] = tp["build_config_name"]
            episode_config["init_config_name"] = tp["init_config_name"]
        return seed, episode_configs

    def make_env(self):
        env_cfg = self.cfg.eval_env
        if env_cfg.task_plan_fp is not None:
            plan_data = plan_data_from_file(env_cfg.task_plan_fp)
            env_cfg.env_kwargs["task_plans"] = env_cfg.env_kwargs.pop("task_plans", plan_data.plans)
            env_cfg.env_kwargs["scene_builder_cls"] = env_cfg.env_kwargs.pop("scene_builder_cls", plan_data.dataset)
        if env_cfg.spawn_data_fp is not None:
            env_cfg.env_kwargs["spawn_data_fp"] = env_cfg.spawn_data_fp
        env = gym.make(
            env_cfg.env_id,
            max_episode_steps=env_cfg.max_episode_steps,
            obs_mode=env_cfg.obs_mode,
            reward_mode="normalized_dense",
            control_mode="pd_joint_delta_pos",
            render_mode=env_cfg.render_mode,
            shader_dir=env_cfg.shader_dir,
            robot_uids="fetch",
            num_envs=env_cfg.num_envs,
            sim_backend="gpu",
            **env_cfg.env_kwargs,
        )
        return env

    def collect_traj_actions(self, traj_start_idx: int, traj_end_idx: int):
        self.traj_lens = [len(self.actions[f"traj_{i}"]) for i in range(traj_start_idx, traj_end_idx)]
        num_envs = self.cfg.eval_env.num_envs
        self.max_traj_len = max(self.traj_lens)
        traj_actions_shape = (num_envs, self.max_traj_len, *self.actions_shape)
        traj_actions_np = np.zeros(traj_actions_shape, dtype=self.actions_dtype)
        for step, traj_len in enumerate(self.traj_lens):
            key = f"traj_{step + traj_start_idx}"
            traj_actions_np[step, :traj_len] = self.actions[key]
        self.traj_actions = torch.from_numpy(traj_actions_np).to(self.unwrapped.device)

    def collect_traj_sensor_poses(self, traj_start_idx: int):
        sensor_poses = {
            k: np.eye(4, dtype=np.float32)[:3][np.newaxis]
            .repeat(self.max_traj_len, axis=0)[np.newaxis]
            .repeat(self.cfg.eval_env.num_envs, axis=0)  # (N, T, 3, 4)
            for k in self.sensor_poses["traj_0"].keys()
        }
        for step, traj_len in enumerate(self.traj_lens):
            poses = self.sensor_poses[f"traj_{step + traj_start_idx}"]
            for sensor_name in sensor_poses.keys():
                sensor_poses[sensor_name][step, :traj_len] = poses[sensor_name]
        self.traj_sensor_poses = sensor_poses

    @staticmethod
    def get_batch_poses(poses: np.ndarray):
        p = np.stack([pose[:3, 3] for pose in poses], axis=0)  # (N, 3)
        q = np.stack([t3d.quaternions.mat2quat(pose[:3, :3]) for pose in poses], axis=0)
        batch_poses = np.concatenate([p, q], axis=-1).astype(np.float32)
        return batch_poses

    def collect_sensor_obs(self, step, sensor_name: str, poses: np.ndarray):
        if self.cfg.load_agent:
            obs = self.unwrapped.render_sensor(sensor_name)
        else:
            obs = self.unwrapped.render_at_poses(self.get_batch_poses(poses[:, step]))

        rgb = obs["rgb"].cpu().numpy()  # (N, H, W, 3)
        depth = obs["depth"].squeeze(-1).cpu().numpy()  # (N, H, W)
        seg = obs["segmentation"].squeeze(-1).cpu().numpy()  # (N, H, W)
        cam_pose = obs["cam_pose"]  # (N, 7)
        extrinsic = obs["extrinsic_cv"].cpu().numpy()  # (N, 3, 4)
        self.intrinsic = obs["intrinsic_cv"][0].cpu()  # (3, 3)

        for env_idx, traj_len in enumerate(self.traj_lens):
            if step >= traj_len:  # this trajectory has ended
                continue
            self.rgb_data[sensor_name][env_idx].append(rgb[env_idx])
            self.depth_data[sensor_name][env_idx].append(depth[env_idx])
            self.seg_data[sensor_name][env_idx].append(seg[env_idx])
            self.cam_pose_data[sensor_name][env_idx].append(cam_pose[env_idx])
            self.extrinsic_data[sensor_name][env_idx].append(extrinsic[env_idx])

    def step_and_render(self):
        num_envs = self.cfg.eval_env.num_envs
        sensor_names = list(self.traj_sensor_poses.keys())

        self.rgb_data = {k: [[] for _ in range(num_envs)] for k in sensor_names}
        self.depth_data = {k: [[] for _ in range(num_envs)] for k in sensor_names}
        self.seg_data = {k: [[] for _ in range(num_envs)] for k in sensor_names}
        self.cam_pose_data = {k: [[] for _ in range(num_envs)] for k in sensor_names}
        self.extrinsic_data = {k: [[] for _ in range(num_envs)] for k in sensor_names}

        for step in get_tqdm_bar(range(self.max_traj_len), "Steps", False):
            if self.cfg.load_agent:
                self.unwrapped.step(self.traj_actions[:, step])  # step the env
            for sensor_name, poses in self.traj_sensor_poses.items():
                self.collect_sensor_obs(step, sensor_name, poses)

    def process_batch(self, traj_start_idx: int, traj_end_idx: int):
        if self.cfg.fixed_task_plan_idx >= 0:
            task_plan_idxs = torch.tensor([self.cfg.fixed_task_plan_idx] * (traj_end_idx - traj_start_idx)).int()
        else:
            task_plan_idxs = torch.tensor(
                [self.episode_configs[i]["task_plan_idx"] for i in range(traj_start_idx, traj_end_idx)]
            ).int()
        if self.cfg.fixed_init_config_idx >= 0:
            init_config_idxs = [self.cfg.fixed_init_config_idx] * (traj_end_idx - traj_start_idx)
        else:
            init_config_idxs = [self.episode_configs[i]["init_config_idx"] for i in range(traj_start_idx, traj_end_idx)]
        if self.cfg.fixed_spawn_selection_idx >= 0:
            spawn_selection_idxs = [self.cfg.fixed_spawn_selection_idx] * (traj_end_idx - traj_start_idx)
        else:
            spawn_selection_idxs = [
                self.episode_configs[i]["spawn_selection_idx"] for i in range(traj_start_idx, traj_end_idx)
            ]
        # reset
        self.unwrapped.reset(
            self.seed,
            dict(
                reconfigure=True,
                task_plan_idxs=task_plan_idxs,
                init_config_idxs=init_config_idxs,
                spawn_selection_idxs=spawn_selection_idxs,
            ),
        )
        # collect actions for the trajectories
        self.collect_traj_actions(traj_start_idx, traj_end_idx)
        # collect poses of the cameras
        self.collect_traj_sensor_poses(traj_start_idx)
        # step the env and render the cameras
        self.step_and_render()

    def visualize(self, traj_idx: int, sensor_name: str, sensor_data: dict):
        plt.suptitle(f"Traj {traj_idx} - {sensor_name}")
        n = len(sensor_data["rgb"])
        rgb_gt = None if self.rgb_gt is None else self.rgb_gt[f"traj_{traj_idx}"]
        for i in range(n):
            if self.img_data is None:
                self.img_data = [
                    self.axes[0].imshow(sensor_data["depth"][i], cmap="jet"),
                    self.axes[1].imshow(sensor_data["segmentation"][i]),
                    self.axes[2].imshow(sensor_data["rgb"][i]),
                ]
                if rgb_gt is not None:
                    self.img_data.append(self.axes[3].imshow(rgb_gt[sensor_name][i]))
            else:
                self.img_data[0].set_data(sensor_data["depth"][i])
                self.img_data[1].set_data(sensor_data["segmentation"][i])
                self.img_data[2].set_data(sensor_data["rgb"][i])
                if rgb_gt is not None:
                    self.img_data[3].set_data(rgb_gt[sensor_name][i])
            plt.pause(0.01)

    def save_to_dict(self, output: dict, traj_start_idx: int, traj_end_idx: int):
        for traj_idx in range(traj_start_idx, traj_end_idx):
            env_idx = traj_idx - traj_start_idx
            traj_sensor_data = dict()
            for sensor_name in self.traj_sensor_poses.keys():
                traj_sensor_data[sensor_name] = dict(
                    rgb=torch.from_numpy(np.stack(self.rgb_data[sensor_name][env_idx], axis=0)),
                    depth=torch.from_numpy(np.stack(self.depth_data[sensor_name][env_idx], axis=0)),
                    segmentation=torch.from_numpy(np.stack(self.seg_data[sensor_name][env_idx], axis=0)),
                    camera_pose=torch.from_numpy(np.stack(self.cam_pose_data[sensor_name][env_idx], axis=0)),
                    extrinsic=torch.from_numpy(np.stack(self.extrinsic_data[sensor_name][env_idx], axis=0)),
                )

                if self.cfg.visualize:
                    self.visualize(traj_idx, sensor_name, traj_sensor_data[sensor_name])
            traj_sensor_data.update(self.episode_configs[traj_idx])
            output[f"traj_{traj_idx}"] = traj_sensor_data

    def save_as_pt(self):
        output = dict()
        num_envs = self.cfg.eval_env.num_envs
        num_traj = len(self.episode_configs)

        for traj_start_idx in get_tqdm_bar(range(0, num_traj, num_envs), "Traj"):
            traj_end_idx = min(traj_start_idx + num_envs, num_traj)
            self.process_batch(traj_start_idx, traj_end_idx)
            self.save_to_dict(output, traj_start_idx, traj_end_idx)

        output["intrinsic"] = self.intrinsic
        output["segmentation_id_map"] = {k: v.name for k, v in self.unwrapped.segmentation_id_map.items()}
        output["scene_bounding_box"] = dict(
            bbox_min=torch.from_numpy(self.unwrapped.scene_bounding_box[0]).float(),
            bbox_max=torch.from_numpy(self.unwrapped.scene_bounding_box[1]).float(),
        )
        output["seed"] = self.seed
        output["episode_configs"] = self.episode_configs
        torch.save(output, self.cfg.output_file)

    def save_to_h5(self, f: h5py.File, traj_start_idx: int, traj_end_idx: int):
        compression = "lzf"
        for traj_idx in range(traj_start_idx, traj_end_idx):
            env_idx = traj_idx - traj_start_idx
            traj_group = f.create_group(f"traj_{traj_idx}")
            for sensor_name in self.traj_sensor_poses.keys():
                sensor_group = traj_group.create_group(sensor_name)
                sensor_group.create_dataset(
                    "rgb",
                    data=np.stack(self.rgb_data[sensor_name][env_idx], axis=0),
                    compression=compression,
                )
                sensor_group.create_dataset(
                    "depth",
                    data=np.stack(self.depth_data[sensor_name][env_idx], axis=0),
                    compression=compression,
                )
                sensor_group.create_dataset(
                    "segmentation",
                    data=np.stack(self.seg_data[sensor_name][env_idx], axis=0),
                    compression=compression,
                )
                sensor_group.create_dataset(
                    "camera_pose",
                    data=np.stack(self.cam_pose_data[sensor_name][env_idx], axis=0),
                    compression=compression,
                )
                sensor_group.create_dataset(
                    "extrinsic",
                    data=np.stack(self.extrinsic_data[sensor_name][env_idx], axis=0),
                    compression=compression,
                )
                if self.cfg.visualize:
                    self.visualize(traj_idx, sensor_name, sensor_group)
        f.flush()

    def save_as_h5(self):
        msg = "Deprecated as other tools prefer PyTorch format. Use save_as_pt() instead."
        bar = f"{'WARNING':=^80}"
        print(bar)
        print(f"{msg:^80}")
        print(bar)

        f = h5py.File(self.cfg.output_file, "w")
        num_envs = self.cfg.eval_env.num_envs
        num_traj = len(self.episode_configs)
        print(f"Processing {num_traj} trajectories in batches of {num_envs} environments")

        for traj_start_idx in get_tqdm_bar(range(0, num_traj, num_envs), "TrajBatch"):
            traj_end_idx = min(traj_start_idx + num_envs, num_traj)
            self.process_batch(traj_start_idx, traj_end_idx)
            self.save_to_h5(f, traj_start_idx, traj_end_idx)

        compression = "lzf"
        f.create_dataset("intrinsic", data=self.intrinsic, compression=compression)
        segmentation_id_map = f.create_group("segmentation_id_map")
        for k, v in self.unwrapped.segmentation_id_map.items():
            segmentation_id_map.attrs[str(k)] = v.name
        bbox_min, bbox_max = self.unwrapped.scene_bounding_box
        bbox_group = f.create_group("scene_bounding_box")
        bbox_group.attrs["bbox_min"] = bbox_min
        bbox_group.attrs["bbox_max"] = bbox_max

        f.attrs["seed"] = self.seed
        f.attrs["episode_configs"] = json.dumps(self.episode_configs)

        f.flush()
        f.close()


def main():
    torch.set_grad_enabled(False)

    cfg = make_config()
    app = App(cfg)
    if cfg.output_file.endswith(".h5"):
        print(f"Saving RGB-D data to {cfg.output_file} as HDF5 format")
        app.save_as_h5()
    elif cfg.output_file.endswith(".pt"):
        print(f"Saving RGB-D data to {cfg.output_file} as PyTorch format")
        app.save_as_pt()
    else:
        raise ValueError(f"Unsupported output file format: {cfg.output_file}")


if __name__ == "__main__":
    main()
