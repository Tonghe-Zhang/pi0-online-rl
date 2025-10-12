

# Install motion planner and octo environments for synthetic data generation
## Install openvla, maniskill, simplerenv.
```bash
# create conda env: rlvla_env
cd RL4VLA
conda create -n rlvla_env -y python=3.10
conda activate rlvla_env

# install dependencies
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla && pip install -e . && cd ..
  # if you meet errors installing the dlimp package, try manually install it with 
  cd ./openvla
  conda install git
  git clone https://github.com/kvablack/dlimp
  cd dlimp
  pip install -e .
  # then get back to install the openvla package from `cd openvla && pip install -e . && cd ..`
pip install -U tyro
pip install datasets==3.3.2

# special install for flash attention from .whl
# if this is slow to download in your terminal, try download elsewhere and push to your remote server. 
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install ManiSkill3
cd ManiSkill && pip install -e . && cd ..
# configure your maniskill asset directory
mkdir path/to/where/to/save/all/mani_skill_assets  # default is ~/.maniskill/
echo 'export MS_ASSET_DIR=path/to/where/to/save/all/mani_skill_assets'>>~/.bashrc
source ~/.bashrc
conda activate rlvla_env
# download some necessary assets to your `path/to/where/to/save/all/mani_skill_assets`
# this maniskill asset folder should look like
# .maniskill/data
# ├── robots
# │   └── widowx
# └── tasks
#     └── bridge_v2_real2sim_dataset

# install SimplerEnv
cd SimplerEnv && pip install -e . && cd ..

# optional: for ubuntu 2204.
# sudo apt-get install libglvnd-dev  
# if you don't have sudo privilege, try the conda version: 
# conda install -c conda-forge libglvnd-devel
# or the faster, mamba version:
# mamba install -c conda-forge libglvnd-devel
```

## Install octo
```bash
conda create -n octo_env -y python=3.10
conda activate octo_env

git clone https://github.com/octo-models/octo.git

cd ManiSkill && pip install -e . && cd ..

cd octo && pip install -e . && pip install -r requirements.txt && cd ..
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "nvidia-cudnn-cu11>=8.7,<9.0" --index-url https://download.pytorch.org/whl/cu118
pip install -U tyro
pip install scipy==1.12.0

cd SimplerEnv && pip install -e . && cd ..
```


# Data: Single object

* Motion planning

```bash
conda activate rlvla_env
cd ManiSkill
cuda=0
CUDA_VISIBLE_DEVICES=$cuda \
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e "PutOnPlateInScene25Single-v1" \
  --save_video --save_data --num_procs 1 --num_traj 75 --seed=0
# in default, we use arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos control mode, which is actually a delta action (joint) control. 
```

* Octo
```bash
conda activate octo_env
cd SimplerEnv
cuda=7
# for OpenVLA warm-up (extra 5 trajectories for performance evaluation)
CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python simpler_env/eval_ms3_collect.py \
  --env_id "PutCarrotOnPlateInScene-v1"\
  --num-episodes 75 --num-envs 64 --seed 0
# try to increase `num-episodes` if not enough successful trajectories is collected
```

* Merge, split, and normalize
```bash
cd RL4VLA
python ./create_octo_mp_dataset.py
```

# Data: Multiple objects
* Generate 128 k samples in total
```bash
conda activate rlvla_env
cd ManiSkill
cuda=5
CUDA_VISIBLE_DEVICES=$cuda \
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e "PutOnPlateInScene25Main-v3" \
  --save_video --save_data --num_procs 16 --num_traj 12800 --seed=100
```
    If you meet this bug:
    ```bash
    Cannot find valid solution because of an error in motion planning solution: 'PutOnPlateInScene25Single' object is not callable
    ```
    Try get to rlvla_env environment, get into /RL4VLA/ManiSkill directory again and retry. 
    ```bash
    conda activate rlvla_env
    cd RL4VLA/ManiSkill
    cuda=0
    ```
    If it is not working, it is because your maniskill repo is wrong, re-install it. 

* Use 12k in train, 800 in val. 
```bash
python resplit_dataset.py --dataset_path="./datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/pi0_sft.pt" \
--train_episodes=12000 \
--val_episodes=800
```



