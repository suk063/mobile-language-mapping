
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
(temporary) In [ManiSkill](https://github.com/haosulab/ManiSkill/blob/mshab/mani_skill/agents/robots/fetch/fetch.py#L70-L92), lines 75-76, 85-86, you need to change 128 to 224.
```bash
pip install -e .
pip install open_clip_torch omegaconf wandb tensorboard tensorboardX  msgpack torchvision
```

The ReplicaCAD dataset necessary for low-level manipulation, which can be downloaded with ManiSkill's download utils. This may take some time:
```bash
python -m mani_skill.utils.download_asset ycb
python -m mani_skill.utils.download_asset ReplicaCAD
python -m mani_skill.utils.download_asset ReplicaCADRearrange
```
The above command may not work due to checksum mismatch. We need to update the checksum at L46 in [ManiSkill/mani_skill/utils/assets/data.py](ManiSkill/mani_skill/utils/assets/data.py):
```bash
wget https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip
sha256sum mani_skill2_ycb.zip  # eb6f30642c90203715c178f67bf2288887ef6e7d05a9f3f1e713efcf7c2a541c
```

## Dataset Download

Download the pre-generated data (224×224 resolution for a single scene) from the following URL:

[Google Drive Folder](https://drive.google.com/drive/u/1/folders/1Fq9UX86H1S-Lsfceeje9SC7Ak3bq45v9)

After downloading:
- Place the folder `rearrange_dataset` into the following directory:
```bash
$MS_ASSET_DIR/scene_datasets/replica_cad_dataset
```
- Place the `task_plans` folder into the following directory:
```bash
$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange
```
*Note:* The default location for `MS_ASSET_DIR` is `~/.maniskill/data`.

## Training and Evaluation

To begin training, run the following script:

```bash
bash scripts/train_bc_point.sh
```
