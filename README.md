
## Setup and Running
First, make your conda environment:

```bash
conda create -n mshab python=3.9
conda activate mshab
```

## Envs
Proceed with the following steps to install

```bash
git clone https://github.com/haosulab/ManiSkill.git -b mshab_b10 --single-branch
pip install -e ManiSkill
```
In [ManiSkill](https://github.com/haosulab/ManiSkill/blob/mshab/mani_skill/agents/robots/fetch/fetch.py#L70-L92), lines 75-76, 85-86, you need to change 128 to 224.

```bash
pip install -e .
pip install open_clip_torch omegaconf wandb tensorboard tensorboardX  msgpack torchvision
```

The ReplicaCAD dataset necessary for low-level manipulation, which can be downloaded with ManiSkill's download utils. This may take some time:
```bash
python -m mani_skill.utils.download_asset ycb ReplicaCAD ReplicaCADRearrange
```

## Dataset Download

Download the pre-generated data (224×224 resolution for a single scene) from the following URL:

[Google Drive Folder](https://drive.google.com/drive/folders/1euvIuJBM_MZMEQq1eZFMPk8Kip7q9piV?usp=drive_link)

After downloading, place the data in the following directory:

```bash
$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange-dataset
```

*Note:* The default location for `MS_ASSET_DIR` is `~/.maniskill/data`.

## Training and Evaluation

To begin training, run the following script:

```bash
bash scripts/train_bc_point.sh
```


