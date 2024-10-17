#!/bin/bash

# Install PyTorch with CUDA support
pip install torch==2.0.0+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Clone the FORTE repository
git clone https://github.com/charlierabea/FORTE.git
cd FORTE

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init
source ~/.bashrc

# Navigate to the checkpoints directory
cd checkpoints

# Update apt repository and install git-lfs
apt update
apt-get install git-lfs -y
git lfs install

# Clone repositories using git-lfs
# git lfs clone https://huggingface.co/luodian/OTTER-MPT7B-Init.git
git lfs clone https://huggingface.co/Charliebear/BrainGPT

# Navigate back to the main directory
cd ../

# Install gdown for downloading files from Google Drive
pip install gdown

# Download the specific file from Google Drive using gdown
gdown "https://drive.google.com/uc?id=1iDLx7NqvTg8sBTVViQu5wq8OhPSovAo4" -O ./data/CQ500p.json

# Create and activate conda environment
conda env create -f environment.yml
conda activate forte

# Install additional Python requirements
pip install -r requirements.txt
export PYTHONPATH="./evaluation:/usr/bin/python"
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file=./evaluation/pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
./evaluation/pipeline/train/eval.py \
--pretrained_model_name_or_path="./checkpoints/BrainGPT/OTTER_CLIP_BRAINGPT_hf/" \
--mimicit_path="./data/CQ500p_instruction.json" \
--images_path="./data/CQ500p.json" \
--batch_size=1 \
--warmup_steps_ratio=0.01 \
--workers=1
