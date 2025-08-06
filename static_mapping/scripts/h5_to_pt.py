import h5py
import torch
import tqdm
import numpy as np
import os
import argparse
import pathlib


def numpy_to_torch(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        return {k: numpy_to_torch(v) for k, v in obj.items()}
    return obj  # return as-is for other types (like strings, int, float)


def h5_to_dict(h5_file):
    out = dict()
    with h5py.File(h5_file, "r") as f:

        def recursive_read(group, _out=None, tbar=None):
            if isinstance(group, h5py.Group):
                if _out is None:
                    _out = dict()
                for key in group.keys():
                    if key in _out:
                        raise ValueError(f"Duplicate key found: {key}")
                    if tbar is not None:
                        tbar.set_description(f"Reading {key}")
                    _out[key] = recursive_read(group[key])
                    if tbar is not None:
                        tbar.update(1)
                if len(group.attrs) > 0:
                    for attr_key, attr_value in group.attrs.items():
                        _out[attr_key] = attr_value
                return _out
            if isinstance(group, h5py.Dataset):
                if len(group.attrs) > 0:
                    assert _out is None, "_out should be None when reading a dataset"
                    _out = dict()
                    for attr_key, attr_value in group.attrs.items():
                        _out[attr_key] = attr_value
                    _out["data"] = group[:]
                    return _out
                return group[:]  # return the numpy array of the dataset
            if isinstance(group, h5py.Reference):
                # Handle references (not common in this context)
                return f"<reference>"
            # Handle other types (like strings)
            if isinstance(group, bytes):
                return group.decode("utf-8")
            raise ValueError(f"Unknown group type: {type(group)}")

        out = recursive_read(f, out, tbar=tqdm.tqdm(f.keys(), ncols=80))
    out = numpy_to_torch(out)  # Convert numpy arrays to torch tensors
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-file", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = pathlib.Path(args.h5_file).with_suffix(".pt")

    if os.path.exists(output):
        print(f"Output file already exists: {output}")
        print("Please specify a different output file or remove the existing one.")
        return

    print(f"Converting {args.h5_file} to {output}...")
    out = h5_to_dict(args.h5_file)
    torch.save(out, output)


if __name__ == "__main__":
    main()
