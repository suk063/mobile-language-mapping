import os

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from utils import depth_to_positions


def main():
    vertices = []
    colors = []
    n_traj = 30
    s1 = 10  # trajectory sampling step
    s2 = 10  # image down sampling
    device = "cuda"
    tasks = ["set_table", "tidy_house", "prepare_groceries"]
    # tasks = ["set_table"]
    # sub_tasks = ["pick", "place"]
    sub_tasks = ["pick"]
    # scene_ids = [10, 13]
    scene_ids = [10]
    # pt_folder = "/home/daizhirui/Data/mobile_language_mapping"
    pt_folder = "/home/daizhirui/Data/mobile_language_mapping_demo"
    for task in tqdm(tasks, desc="Task", ncols=80, position=0):
        for sub_task in tqdm(sub_tasks, desc="SubTask", ncols=80, position=1, leave=False):
            for scene_id in tqdm(scene_ids, desc="Scene", ncols=80, position=2, leave=False):
                pt_file = os.path.join(pt_folder, f"{task}/{sub_task}/all_{scene_id}_static.pt")
                f = torch.load(pt_file, mmap=True)
                intrinsic = f["intrinsic"].to(device)
                traj_names = list([k for k in f.keys() if k.startswith("traj")])
                np.random.shuffle(traj_names)
                traj_names = traj_names[:n_traj]
                for traj_name in tqdm(traj_names, desc="Traj", ncols=80, position=2, leave=False):
                    traj_data = f[traj_name]
                    for sensor_name in ["fetch_hand", "fetch_head"]:
                        sensor_data = traj_data[sensor_name]
                        n = sensor_data["depth"].shape[0]
                        for i in range(0, n, s1):
                            depth = sensor_data["depth"][i].to(device)
                            extrinsic = sensor_data["extrinsic"][i].to(device)
                            mask = (depth > 10).cpu().numpy()
                            vertices.append(
                                depth_to_positions(
                                    depth=depth,
                                    intrinsic=intrinsic,
                                    extrinsic=extrinsic,
                                    depth_scale=1 / 1000.0,
                                )
                                .cpu()
                                .numpy()[mask]  # remove invalid depth value
                                .reshape(-1, 3)[::s2]
                            )
                            colors.append(sensor_data["rgb"][i][mask].reshape(-1, 3)[::s2])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.concatenate(vertices, axis=0).astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0).astype(np.float64) / 255.0)
    o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
    o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    main()
