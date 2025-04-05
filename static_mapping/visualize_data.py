import os

import h5py
import numpy as np
from generate_rgbd_data import depth_to_positions
import torch
from tqdm import tqdm
import open3d as o3d


def main():
    vertices = []
    colors = []
    n_traj = 30
    s1 = 10  # trajectory sampling step
    s2 = 10  # image down sampling
    device = "cuda"
    tasks = ["set_table", "tidy_house", "prepare_groceries"]
    sub_tasks = ["pick", "place"]
    h5_folder = "/home/daizhirui/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange-dataset"
    for task in tqdm(tasks, desc="Task", ncols=80, position=0):
        for sub_task in tqdm(
            sub_tasks, desc="SubTask", ncols=80, position=1, leave=False
        ):
            h5_file = os.path.join(h5_folder, f"{task}/{sub_task}/all_static.h5")
            with h5py.File(
                h5_file,
                "r",
            ) as f:
                intrinsic = torch.tensor(f["intrinsic"][:]).to(device)
                traj_names = list(f.keys())
                np.random.shuffle(traj_names)
                traj_names = traj_names[:n_traj]
                for traj_name in tqdm(
                    traj_names, desc="Traj", ncols=80, position=2, leave=False
                ):
                    if not traj_name.startswith("traj"):
                        continue
                    traj_data = f[traj_name]
                    for sensor_name, sensor_data in traj_data.items():
                        n = sensor_data["depth"].shape[0]
                        for i in range(0, n, s1):
                            depth = torch.tensor(sensor_data["depth"][i]).to(device)
                            extrinsic = torch.tensor(sensor_data["extrinsic"][i])
                            mask = (depth > 10).cpu().numpy()
                            vertices.append(
                                depth_to_positions(
                                    depth=depth,
                                    intrinsic=intrinsic,
                                    extrinsic=extrinsic.to(device),
                                    depth_scale=1 / 1000.0,
                                )
                                .cpu()
                                .numpy()[mask]  # remove invalid depth value
                                .reshape(-1, 3)[::s2]
                            )
                            colors.append(
                                sensor_data["rgb"][i][mask].reshape(-1, 3)[::s2]
                            )
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.concatenate(vertices, axis=0).astype(np.float64)
    )
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.concatenate(colors, axis=0).astype(np.float64) / 255.0
    )
    o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
    o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    main()
