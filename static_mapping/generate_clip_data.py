import argparse
import os
from dataclasses import dataclass

import open_clip
import torch

from lang_mapping.utils import get_visual_features
from tqdm import tqdm


@dataclass
class Config:
    input_pt_path: str
    output_pt_path: str
    device: str = "cuda"
    clip_model_name: str = "EVA02-L-14"
    clip_model_pretrained: str = "merged2b_s4b_b131k"


def generate_clip_data(cfg: Config):
    clip_model = open_clip.create_model_and_transforms(
        cfg.clip_model_name,
        cfg.clip_model_pretrained,
    )[0].to(cfg.device)
    clip_model.eval()

    input_pt = torch.load(cfg.input_pt_path, mmap=True)
    bs = 128  # batch size
    output_pt = dict()
    for traj_name, traj_data in tqdm(input_pt.items(), ncols=80):
        if not traj_name.startswith("traj"):
            continue
        traj_output = dict()
        for sensor_name, sensor_data in traj_data.items():
            n = sensor_data["depth"].shape[0]
            clip_features = []
            for i in range(0, n, bs):
                j = min(i + bs, n)
                rgb = sensor_data["rgb"][i:j].permute(0, 3, 1, 2)
                rgb = rgb.float().to(cfg.device)
                with torch.no_grad():
                    features = get_visual_features(clip_model, rgb)
                clip_features.append(features.cpu())
            clip_features = torch.cat(clip_features)
            traj_output[sensor_name] = clip_features
        output_pt[traj_name] = traj_output

    os.makedirs(os.path.dirname(cfg.output_pt_path), exist_ok=True)
    torch.save(output_pt, cfg.output_pt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pt-path", type=str, required=True)
    parser.add_argument("--output-pt-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip-model-name", type=str, default="EVA02-L-14")
    parser.add_argument(
        "--clip-model-pretrained", type=str, default="merged2b_s4b_b131k"
    )
    cfg = Config(**vars(parser.parse_args()))
    print(f"Generate CLIP data from {cfg.input_pt_path} to {cfg.output_pt_path}")
    generate_clip_data(cfg)


if __name__ == "__main__":
    main()
