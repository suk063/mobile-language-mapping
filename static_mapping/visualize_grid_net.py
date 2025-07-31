from dataclasses import asdict
import numpy as np
import random
import torch
from tqdm import tqdm
from lang_mapping.grid_net.grid_net import GridNet
from lang_mapping.module.mlp import ImplicitDecoder
from train_static_map_per_episode import Config, StaticMappingDataset
from mshab.utils.config import parse_cfg
from dacite import from_dict
import sys
import os
from omegaconf import OmegaConf
import open3d as o3d
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from utils import depth_to_positions
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


@dataclass
class AppConfig:
    cfg: Config
    vertices_fp: Optional[str]
    min_depth: float
    max_depth: float
    frame_downsample_factor: int
    pcd_downsample_factor: int
    batch_size: int
    init_grid_map_id: int = 0


class App:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg.cfg
        self.app_cfg = cfg
        assert self.cfg.test_model_dir is not None, "Test model directory must be specified in the config."

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = self.cfg.torch_deterministic

        # Load datasets
        self.dataset = StaticMappingDataset(self.cfg.data)
        if len(self.dataset.scene_ids) != self.cfg.grid_cfg.grid.n_scenes:
            raise ValueError(
                f"Number of scenes in dataset ({len(self.dataset.scene_ids)}) "
                f"does not match grid_cfg.grid.n_scenes ({self.cfg.grid_cfg.grid.n_scenes})."
            )
        # self.dataset.scene_ids: episode_id -> scene_id (map_id)
        scene_ids = self.dataset.scene_ids
        episode_configs = self.dataset.episode_configs
        self.data_id_to_map_id = {
            i: scene_ids[episode_configs[fp_idx][traj_name]["init_config_name"]]
            for i, (fp_idx, traj_name, _, _) in enumerate(self.dataset.records)
        }
        self.map_id_to_data_ids = defaultdict(list)
        for data_id, map_id in self.data_id_to_map_id.items():
            self.map_id_to_data_ids[map_id].append(data_id)
        self.intrinsic: torch.Tensor = self.dataset.fps[0]["intrinsic"]

        # Create model
        self.grid_net = GridNet(cfg=asdict(self.cfg.grid_cfg))
        self.implicit_decoder = ImplicitDecoder(
            voxel_feature_dim=self.cfg.grid_cfg.grid.feature_dim * self.cfg.grid_cfg.grid.n_levels,
            hidden_dim=self.cfg.decoder_hidden_dim,
            output_dim=self.cfg.decoder_output_dim,
        ).to(self.cfg.device_decoder)

        # Load model state
        state_fp = os.path.join(self.cfg.test_model_dir, "grid_net.pt")
        print(f"Loading GridNet state from {state_fp}")
        state = torch.load(state_fp, map_location="cpu")
        self.grid_net.to("cpu").load_state_dict(state["model"])
        self.grid_net.distribute_to_devices()
        self.grid_net.eval()

        state_fp = os.path.join(self.cfg.test_model_dir, "implicit_decoder.pt")
        print(f"Loading ImplicitDecoder state from {state_fp}")
        state = torch.load(state_fp, map_location="cpu")
        self.implicit_decoder.load_state_dict(state["model"])
        self.implicit_decoder.to(self.cfg.device_decoder)
        self.implicit_decoder.eval()

        # Open3D visualization setup
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self.vis.create_window(window_name="GridNet Visualization", width=1280, height=720)
        # Initialize point cloud
        self.pcd_color = o3d.geometry.PointCloud()
        self.pcd_map = o3d.geometry.PointCloud()
        self.pcd_decoder = o3d.geometry.PointCloud()
        self.current_grid_map = self.app_cfg.init_grid_map_id
        if self.current_grid_map < 0 or self.current_grid_map >= self.cfg.grid_cfg.grid.n_scenes:
            raise ValueError(
                f"Invalid initial grid map ID: {self.current_grid_map}. "
                f"Must be in range [0, {self.cfg.grid_cfg.grid.n_scenes - 1}]."
            )
        self._update_pcd()
        self.vis.add_geometry(self.pcd_color)
        self.vis.add_geometry(self.pcd_map)
        self.vis.add_geometry(self.pcd_decoder)

        # Add coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(self.coord_frame)

    def _register_key_callbacks(self):
        self.vis.register_key_action_callback(262, self._next_grid_map)  # Right
        self.vis.register_key_action_callback(263, self._prev_grid_map)  # Left

    def _next_grid_map(self, vis, action, mod):
        """Navigate to the next grid map."""
        if action != 0:  # Only handle key press events
            return False
        if self.current_grid_map < self.cfg.grid_cfg.grid.n_scenes - 1:
            self.current_grid_map += 1
            self._update_display()
            print(f"Grid map: {self.current_grid_map + 1}/{self.cfg.grid_cfg.grid.n_scenes}")
        else:
            print("Already at the last grid map")
            return False

    def _prev_grid_map(self, vis, action, mod):
        """Navigate to the previous grid map."""
        if action != 0:  # Only handle key press events
            return False
        if self.current_grid_map > 0:
            self.current_grid_map -= 1
            self._update_display()
            print(f"Grid map: {self.current_grid_map + 1}/{self.cfg.grid_cfg.grid.n_scenes}")
        else:
            print("Already at the first grid map")
            return False

    def _collect_vertices(self):
        """Collect vertices that belong to the current grid map."""
        if self.app_cfg.vertices_fp is not None:
            pcd = o3d.io.read_point_cloud(self.app_cfg.vertices_fp)
            vertices = torch.tensor(np.asarray(pcd.points)).float()
            colors = torch.tensor(np.asarray(pcd.colors)).float()
            return vertices, colors

        vertices = []
        colors = []
        data_ids = self.map_id_to_data_ids[self.current_grid_map]
        print(
            f"{len(data_ids)} frames are available for "
            f"grid map {self.current_grid_map + 1}/{self.cfg.grid_cfg.grid.n_scenes}"
        )
        if self.app_cfg.frame_downsample_factor > 1:
            data_ids = random.sample(data_ids, len(data_ids) // self.app_cfg.frame_downsample_factor)
            print(f"Downsampled to {len(data_ids)} frames")
            data_ids = sorted(data_ids)  # Sort to maintain order
        for data_id in data_ids:
            fp_idx, traj_name, sensor_name, i = self.dataset.records[data_id]
            fp = self.dataset.fps[fp_idx]
            sensor_data = fp[traj_name][sensor_name]
            depth = sensor_data["depth"][i] / 1000.0  # Convert depth from mm to m
            extrinsic = sensor_data["extrinsic"][i]
            world_coords = depth_to_positions(depth, self.intrinsic, extrinsic)
            valid_mask = (depth > self.app_cfg.min_depth) & (depth < self.app_cfg.max_depth)
            vertices.append(world_coords[valid_mask])
            colors.append(sensor_data["rgb"][i][valid_mask] / 255.0)
        vertices = torch.cat(vertices, dim=0)
        colors = torch.cat(colors, dim=0)
        if self.app_cfg.pcd_downsample_factor > 1:
            indices = torch.randperm(len(vertices))[: len(vertices) // self.app_cfg.pcd_downsample_factor]
            vertices = vertices[indices]
            colors = colors[indices]
        return vertices, colors

    def _forward_model(self, vertices: torch.Tensor):

        vertices = vertices.float()

        map_features = []
        decoded_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(vertices), self.app_cfg.batch_size), desc="Forward Model", ncols=80):
                j = min(i + self.app_cfg.batch_size, len(vertices))
                batch_vertices = vertices[i:j].to(self.cfg.device_decoder)
                batch_map_features = self.grid_net.query_feature(
                    batch_vertices,
                    torch.tensor([self.current_grid_map] * len(batch_vertices), device=batch_vertices.device),
                )
                batch_decoded_features = self.implicit_decoder(batch_map_features, batch_vertices)

                map_features.append(batch_map_features.cpu())
                decoded_features.append(batch_decoded_features.cpu())
        map_features = torch.cat(map_features, dim=0)
        decoded_features = torch.cat(decoded_features, dim=0)

        scalar = MinMaxScaler()
        pca = PCA(n_components=3)
        map_pca = pca.fit_transform(map_features.cpu().numpy())
        map_pca_norm = scalar.fit_transform(map_pca)
        map_pca_norm = np.clip(map_pca_norm, 0, 1)

        scalar = MinMaxScaler()
        pca = PCA(n_components=3)
        decoded_pca = pca.fit_transform(decoded_features.cpu().numpy())
        decoded_pca_norm = scalar.fit_transform(decoded_pca)
        decoded_pca_norm = np.clip(decoded_pca_norm, 0, 1)

        return map_pca_norm, decoded_pca_norm

    def _update_pcd(self):
        test_model_dir = self.cfg.test_model_dir
        assert test_model_dir is not None, "Test model directory must be specified in the config."

        vertices, colors = self._collect_vertices()
        if len(vertices) == 0:
            print("No valid vertices found for the current grid map.")
            return

        print(
            f"Collected {len(vertices)} vertices for grid map {self.current_grid_map + 1}/{self.cfg.grid_cfg.grid.n_scenes}"
        )

        # Update point cloud for color visualization
        self.pcd_color.points = o3d.utility.Vector3dVector(vertices.cpu().double().numpy())
        self.pcd_color.colors = o3d.utility.Vector3dVector(colors.cpu().double().numpy())
        pcd_fp = os.path.join(test_model_dir, f"color_{self.current_grid_map}.ply")
        o3d.io.write_point_cloud(pcd_fp, self.pcd_color)
        print(f"Saved color point cloud to {pcd_fp}")

        # Forward model to get features
        map_pca_norm, decoded_pca_norm = self._forward_model(vertices)

        # Update point cloud for map features visualization
        self.pcd_map.points = self.pcd_color.points
        self.pcd_map.colors = o3d.utility.Vector3dVector(map_pca_norm.astype(np.float64))
        pcd_fp = os.path.join(test_model_dir, f"grid_map_{self.current_grid_map}.ply")
        o3d.io.write_point_cloud(pcd_fp, self.pcd_map)
        print(f"Saved grid map point cloud to {pcd_fp}")

        # Update point cloud for decoder features visualization
        self.pcd_decoder.points = self.pcd_color.points
        self.pcd_decoder.colors = o3d.utility.Vector3dVector(decoded_pca_norm.astype(np.float64))
        pcd_fp = os.path.join(test_model_dir, f"decoder_features_{self.current_grid_map}.ply")
        o3d.io.write_point_cloud(pcd_fp, self.pcd_decoder)
        print(f"Saved decoder features point cloud to {pcd_fp}")

        # translate the point clouds so that they don't overlap
        self.pcd_color: o3d.geometry.PointCloud
        bound = self.pcd_color.get_max_bound() - self.pcd_color.get_min_bound()
        x = bound[0] * 1.1

        self.pcd_map.translate((x, 0, 0))
        self.pcd_decoder.translate((x * 2, 0, 0))

    def _update_display(self):
        """Update the Open3D visualizer with the current grid map."""
        self._update_pcd()
        self.vis.update_geometry(self.pcd_color)
        self.vis.update_geometry(self.pcd_map)
        self.vis.update_geometry(self.pcd_decoder)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        """Run the Open3D visualizer."""
        print("\n" + "=" * 60)
        print("Grid Net Visualization Controls:")
        print("  Left/Right Arrow: Navigate between grid maps")
        print("=" * 60)

        self.vis.run()
        self.vis.destroy_window()


def main():
    default_cfg_path = None
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        default_cfg_path = sys.argv[1]
    cfg = from_dict(data_class=AppConfig, data=OmegaConf.to_container(parse_cfg(default_cfg_path=default_cfg_path)))

    # reload the config based on the test model directory
    test_model_dir = cfg.cfg.test_model_dir
    assert test_model_dir is not None, "Test model directory must be specified in the config."
    cfg.cfg = from_dict(
        data_class=Config,
        data=OmegaConf.to_container(parse_cfg(default_cfg_path=os.path.join(test_model_dir, "../config.yaml"))),
    )
    cfg.cfg.test_model_dir = test_model_dir

    app = App(cfg)
    app.run()


if __name__ == "__main__":
    main()
