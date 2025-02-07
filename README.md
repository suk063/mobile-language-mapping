
## Setup and Running

First, make your conda environment:

```bash
conda create -n mshab python=3.9
conda activate mshab
```

Then, proceed with the following steps depending on what you'd like to install.

### Envs


```bash
pip install -e ./ManiSkill
pip install -e .
pip install open_clip_torch omegaconf wandb tensorboard tensorboardX  msgpack torchvision
```

We also host an altered version of the ReplicaCAD dataset necessary for low-level manipulation, which can be downloaded with ManiSkill's download utils. This may take some time:
```bash
python -m mani_skill.utils.download_asset ycb ReplicaCAD ReplicaCADRearrange
```

### Dataset Download

Download the pre-generated data (224×224 resolution for a single scene) from the following URL:

[Google Drive Folder](https://drive.google.com/drive/u/1/folders/1euvIuJBM_MZMEQq1eZFMPk8Kip7q9piV)

After downloading, place the data in the following directory:

```bash
$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange-dataset
```

*Note:* The default location for `MS_ASSET_DIR` is `~/.maniskill/data`.

---

### Training and Evaluation

To begin training, run the following script:

```bash
bash scripts/train_bc_point.sh
```


