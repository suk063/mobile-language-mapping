import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm
import os


def dict_to_h5(data: dict, h5_file: h5py.File, compression=None):
    def recursive_write(key, _data, parent=None):
        if isinstance(_data, dict):
            group = parent.create_group(key)
            for key, value in _data.items():
                recursive_write(key, value, group)
        elif isinstance(_data, np.ndarray):
            parent.create_dataset(key, data=_data, compression=compression)
        elif isinstance(_data, torch.Tensor):
            parent.create_dataset(
                key, data=_data.cpu().numpy(), compression=compression
            )
        else:
            parent.attrs[key] = _data

    for k, v in tqdm(data.items(), ncols=80):
        recursive_write(k, v, h5_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pt-path", type=str, required=True)
    parser.add_argument("--output-h5-path", type=str)
    parser.add_argument("--compression", type=str)
    args = parser.parse_args()

    input_pt = torch.load(args.input_pt_path, mmap=True)
    output_h5_path = args.output_h5_path
    if output_h5_path is None:
        output_h5_path = os.path.splitext(args.input_pt_path)[0] + ".h5"
    if os.path.exists(output_h5_path):
        print(f"Output file {output_h5_path} already exists, skipping")
        return
    print(f"Converting {args.input_pt_path} to {output_h5_path}")
    with h5py.File(output_h5_path, "w") as h5_file:
        dict_to_h5(input_pt, h5_file, args.compression)


if __name__ == "__main__":
    main()
