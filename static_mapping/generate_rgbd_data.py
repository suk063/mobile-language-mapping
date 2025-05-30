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
oRc = torch.tensor(  # rotation matrix from camera to optical frame
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def extrinsic_to_cam_pose(extrinsic: torch.Tensor) -> torch.Tensor:
    global oRc
    oRc = oRc.to(extrinsic.device)
    cam_pose = torch.zeros_like(extrinsic)
    cam_pose[:3, :3] = extrinsic[:3, :3].T @ oRc
    cam_pose[:3, 3] = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    if cam_pose.shape[0] == 4:
        cam_pose[3, 3] = 1.0  # homogeneous coordinate
    return cam_pose


def cam_pose_to_extrinsic(cam_pose: torch.Tensor) -> torch.Tensor:
    global oRc
    oRc = oRc.to(cam_pose.device)
    extrinsic = torch.zeros_like(cam_pose)
    extrinsic[:3, :3] = oRc @ cam_pose[:3, :3].T
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ cam_pose[:3, 3]
    if extrinsic.shape[0] == 4:
        extrinsic[3, 3] = 1.0
    return extrinsic


def transform_matrix_to_sapien_pose(matrix: torch.Tensor) -> sapien.Pose:
    matrix = matrix.detach().cpu().numpy()
    q = t3d.quaternions.mat2quat(matrix[:3, :3])
    p = matrix[:3, 3]
    return sapien.Pose(p=p, q=q)


def depth_to_positions(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor = None,
    depth_scale: float = 1.0,
) -> torch.Tensor:
    h, w = depth.shape[0], depth.shape[1]
    device = depth.device
    dtype = intrinsic.dtype
    depth = depth.to(dtype)
    if depth_scale != 1.0:
        depth = depth * depth_scale
    # Create a grid of pixel coordinates
    pix = torch.stack(
        [
            *torch.meshgrid(  # +0.5 to get pixel centers instead of corners
                torch.arange(h, dtype=dtype) + 0.5,
                torch.arange(w, dtype=dtype) + 0.5,
                indexing="xy",
            ),
            torch.ones(h, w, dtype=dtype),  # homogeneous coordinate
        ],
        dim=-1,  # shape (H, W, 3)
    ).to(device)
    # pixel coordinates to normalized device coordinates
    # Note: This assumes the camera's principal point is at the center of the image
    pix = pix @ torch.linalg.inv(intrinsic).T  # shape (H, W, 3)
    # Scale by depth to get 3D points in camera coordinates
    pix = pix * depth[..., torch.newaxis]  # shape (H, W, 3)
    if extrinsic is not None:
        # Now we have 3D points in camera coordinates, we need to transform them to world coordinates
        rot = extrinsic[:3, :3]
        t = -rot.T @ extrinsic[:3, 3]
        pix = pix @ rot + t
    return pix  # shape (H, W, 3) now in world coordinates


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
            self.action_space = gym.vector.utils.batch_space(
                self.action_space, self.num_envs
            )

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


def create_env_cls(env_name="StaticPickEnv", parent_env=PickSubtaskTrainEnv):
    @register_env(env_name, max_episode_steps=2)
    class StaticEnv(parent_env):

        def __init__(
            self, *args, load_agent=False, hide_episode_objects=False, **kwargs
        ):
            self.load_agent = load_agent
            self.hide_episode_objects = hide_episode_objects
            self.camera_mount: Actor = None
            self.camera_mount_offset = 0.01  # make sure the mount is behind the camera
            super().__init__(*args, **kwargs)

            if self.hide_episode_objects:
                scene_builder: ReplicaCADRearrangeSceneBuilder = self.scene_builder
                for env_ycb_objects in scene_builder.ycb_objs_per_env:
                    for obj_name, objs in env_ycb_objects.items():
                        for obj in objs:
                            scene_builder.hide_actor(obj)

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
            if self.load_agent:
                return super()._default_human_render_camera_configs()

            # this camera follows the robot around (though might be in walls if the space is cramped)
            robot_camera_pose = sapien_utils.look_at(
                [self.camera_mount_offset, 0, 0], ([1.0, 0.0, 0.0])
            )
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
            # self.scene_builder.scene_objects  # include all scene objects
            # self.scene_builder.articulations  # include all articulated objects

        def render_at_pose(self, pose: sapien.Pose = None) -> dict:
            camera = self.scene.human_render_cameras["render_camera"]
            if pose is not None:  # set the camera pose
                t0 = np.array([self.camera_mount_offset, 0.0, 0.0], dtype=np.float32)
                t1 = pose.p
                pose.p = t1 - t3d.quaternions.quat2mat(pose.q) @ t0
                self.camera_mount.set_pose(pose)
                # assert np.allclose(t1, camera.camera.get_global_pose().p[0].cpu().numpy())
                # camera.camera.set_local_pose(pose)
                # print(pose.to_transformation_matrix())
                # print(camera.camera.get_extrinsic_matrix())  # world to cam, OpenCV convention
                # print(camera.camera.get_model_matrix())  # cam to world, OpenGL convention
                # print(camera.camera.get_local_pose())  # same as get_global_pose() if mount is None
                # print(camera.camera.get_global_pose())
            # https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html
            self.scene.step()
            self.scene.update_render()  # update scene: object states, camera, etc.
            camera.camera.take_picture()  # start the rendering process
            # position can be obtained from the camera's intrinsic and depth
            obs = {k: v[0] for k, v in camera.get_obs(position=False).items()}
            # https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html#rgb-depth-segmentation
            # rgb: uint8
            # depth: int16 (in mm)
            # position: int16 (in mm, in optical frame, OpenGL convention)
            # segmentation: int16 (0 for distant background)
            # OpenGL convention: x to right, y up, z back
            # OpenCV convention: x to right, y down, z forward
            if (
                "position" in obs
            ):  # OpenGL to OpenCV convention (y-axis and z-axis flipped)
                obs["position"][..., 1] *= -1
                obs["position"][..., 2] *= -1
            obs["cam_pose"] = np.concatenate([pose.p, pose.q])
            obs["extrinsic_cv"] = camera.camera.get_extrinsic_matrix()[0]
            obs["intrinsic_cv"] = camera.camera.get_intrinsic_matrix()[0]
            return obs

    return StaticEnv


StaticPickEnv = create_env_cls("StaticPickEnv", PickSubtaskTrainEnv)
StaticPlaceEnv = create_env_cls("StaticPlaceEnv", PlaceSubtaskTrainEnv)


@dataclass
class Config:
    seed: int
    eval_env: EnvConfig
    load_agent: bool
    hide_episode_objects: bool
    truncate_trajectory_at_success: bool
    traj_h5: str
    output_h5: str


def make_env(env_cfg: EnvConfig):
    if env_cfg.task_plan_fp is not None:
        plan_data = plan_data_from_file(env_cfg.task_plan_fp)
        env_cfg.env_kwargs["task_plans"] = env_cfg.env_kwargs.pop(
            "task_plans", plan_data.plans
        )
        env_cfg.env_kwargs["scene_builder_cls"] = env_cfg.env_kwargs.pop(
            "scene_builder_cls", plan_data.dataset
        )
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


def make_config() -> Config:
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    cfg = from_dict(data_class=Config, data=OmegaConf.to_container(cfg))
    cfg.eval_env.env_kwargs["load_agent"] = cfg.load_agent
    cfg.eval_env.env_kwargs["hide_episode_objects"] = cfg.hide_episode_objects
    return cfg


def load_poses_from_h5(path: str, truncate_trajectory_at_success: bool = True):
    poses = dict()
    oRc = np.array(  # rotation matrix from camera to optical frame
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )[np.newaxis]
    with h5py.File(path, "r") as f:
        # plt.imshow(f["traj_0"]["obs"]["sensor_data"]["fetch_head"]["rgb"][0])
        # plt.title("RGB Image from fetch_head")
        # plt.show()

        for traj_name in f.keys():
            traj = f[traj_name]["obs"]["sensor_param"]
            traj_poses = dict()
            success_cutoff = None
            if truncate_trajectory_at_success:
                success = f[traj_name]["success"][:].tolist()
                success_cutoff = min(success.index(True) + 1, len(success))
            for sensor_name in traj.keys():
                extrinsic_matrices = traj[sensor_name]["extrinsic_cv"][...]
                if success_cutoff is not None:
                    extrinsic_matrices = extrinsic_matrices[:success_cutoff]
                cam_poses = np.zeros_like(extrinsic_matrices)
                cam_poses[:, :3, :3] = (
                    extrinsic_matrices[:, :3, :3].transpose(0, 2, 1) @ oRc
                )
                cam_poses[:, :3, [3]] = (
                    -extrinsic_matrices[:, :3, :3].transpose(0, 2, 1)
                    @ extrinsic_matrices[:, :3, [3]]
                )
                if cam_poses.shape[1] == 4:
                    cam_poses[:, 3, 3] = 1.0  # homogeneous coordinate
                traj_poses[sensor_name] = cam_poses
            poses[traj_name] = traj_poses
    return poses


def visualize_pose(env: Union[StaticPickEnv, StaticPlaceEnv], pose: sapien.Pose):
    obs = env.render_at_pose(pose)

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


def main():
    torch.set_grad_enabled(False)

    cfg = make_config()
    cfg.eval_env.num_envs = 1  # only need one for generating the data
    poses = load_poses_from_h5(cfg.traj_h5, cfg.truncate_trajectory_at_success)

    env = make_env(cfg.eval_env)
    unwrapped: Union[StaticPickEnv, StaticPlaceEnv] = env.unwrapped
    # compression = "gzip"
    compression = "lzf"

    with h5py.File(cfg.output_h5, "w") as f:
        # store unwrapped.segmentation_id_map
        segmentation_id_map = f.create_group("segmentation_id_map")
        for k, v in unwrapped.segmentation_id_map.items():
            segmentation_id_map.attrs[str(k)] = v.name

        # store scene bounding box
        bbox_min, bbox_max = unwrapped.scene_bounding_box
        bbox_group = f.create_group("scene_bounding_box")
        bbox_group.attrs["bbox_min"] = bbox_min
        bbox_group.attrs["bbox_max"] = bbox_max

        intrinsic = None

        for traj_name, traj_poses in tqdm(
            poses.items(), desc="Trajectories", ncols=80, position=0
        ):
            traj_group = f.create_group(traj_name)
            for sensor_name, sensor_poses in tqdm(
                traj_poses.items(), desc="Sensors", ncols=80, position=1, leave=False
            ):
                sensor_group = traj_group.create_group(sensor_name)
                rgb_data = []
                depth_data = []
                segmentation_data = []
                camera_pose_data = []
                extrinsic_data = []

                for pose in tqdm(
                    sensor_poses, desc=sensor_name, ncols=80, position=2, leave=False
                ):
                    pose = sapien.Pose(
                        p=pose[:3, 3], q=t3d.quaternions.mat2quat(pose[:3, :3])
                    )
                    # visualize_pose(unwrapped, pose)
                    obs = unwrapped.render_at_pose(pose)

                    rgb = obs["rgb"].cpu().numpy()
                    depth = obs["depth"][:, :, 0].cpu().numpy()
                    segmentation = obs["segmentation"].cpu().numpy()
                    camera_pose = obs["cam_pose"]
                    extrinsic = obs["extrinsic_cv"].cpu().numpy()
                    intrinsic = obs["intrinsic_cv"].cpu().numpy()

                    rgb_data.append(rgb)
                    depth_data.append(depth)
                    segmentation_data.append(segmentation)
                    camera_pose_data.append(camera_pose)
                    extrinsic_data.append(extrinsic)

                rgb_data = np.stack(rgb_data, axis=0)
                depth_data = np.stack(depth_data, axis=0)
                segmentation_data = np.stack(segmentation_data, axis=0)
                camera_pose_data = np.stack(camera_pose_data, axis=0)
                extrinsic_data = np.stack(extrinsic_data, axis=0)

                sensor_group.create_dataset(
                    "rgb", data=rgb_data, compression=compression
                )
                sensor_group.create_dataset(
                    "depth", data=depth_data, compression=compression
                )
                sensor_group.create_dataset(
                    "segmentation", data=segmentation_data, compression=compression
                )
                sensor_group.create_dataset(
                    "camera_pose", data=camera_pose_data, compression=compression
                )
                sensor_group.create_dataset(
                    "extrinsic", data=extrinsic_data, compression=compression
                )

            f.flush()
        # intrinsic is the same for all cameras
        f.create_dataset("intrinsic", data=intrinsic, compression=compression)


if __name__ == "__main__":
    main()
