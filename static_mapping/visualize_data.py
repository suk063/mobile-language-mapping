import h5py
import numpy as np
from generate_data import depth_to_positions
import torch
from tqdm import tqdm
import open3d as o3d


def main():
    vertices = []
    colors = []
    n_traj = 100
    s1 = 10  # trajectory sampling step
    s2 = 10  # image down sampling
    device = "cuda"
    with h5py.File(
        "/home/daizhirui/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange-dataset/set_table/pick/all_static.h5",
        "r",
    ) as f:
        intrinsic = torch.tensor(f["intrinsic"][:]).to(device)
        traj_names = list(f.keys())
        np.random.shuffle(traj_names)
        traj_names = traj_names[:n_traj]
        for traj_name in tqdm(traj_names, desc="Traj", ncols=80):
            if not traj_name.startswith("traj"):
                continue
            traj_data = f[traj_name]
            for sensor_name, sensor_data in traj_data.items():
                n = sensor_data["depth"].shape[0]
                for i in range(0, n, s1):
                    depth = torch.tensor(sensor_data["depth"][i]).to(device)
                    extrinsic = torch.tensor(sensor_data["extrinsic"][i]).to(device)
                    mask = (depth > 10).cpu().numpy()  # remove invalid depth value
                    vertices.append(
                        depth_to_positions(
                            depth=depth,
                            intrinsic=intrinsic,
                            extrinsic=extrinsic,
                            depth_scale=1 / 1000.0,
                        )
                        .cpu()
                        .numpy()[mask]
                        .reshape(-1, 3)[::s2]
                    )
                    colors.append(sensor_data["rgb"][i][mask].reshape(-1, 3)[::s2])
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
