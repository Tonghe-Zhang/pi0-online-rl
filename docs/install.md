
## Setup the environment
* Basics
```bash
    # git clone git@github.com:wadeKeith/openpi_zgc.git
    cd openpi
    conda create -n pi_r python=3.10.0
    conda activate pi_r
    pip install torch torchvision torchaudio --no-cache-dir
    pip install -r requirements.txt
    # add debug  (turn them off in production to accelerate programs, turn them on in debugging for verbose logging.)
    echo -e "# verbose debug\nexport HYDRA_FULL_ERROR=1\nexport TORCH_USE_CUDA_DSA=1" >> ~/.bashrc
    echo "Turned on verbose debugging for HYDRA, CUDA, and TORCH_USE_CUDA_DSA"
    echo "export PIR_WANDB_ENTITY=<your_wandb_entity>">>~/.bashrc
    source ~/.bashrc
    conda activate pi_r
```
* ManiSkill


* LIBERO
If you wish to install LIBERO simulation environment and the relevant packages (mujoco_py, robosuite, and robomimic), please refer to [LIBERO_installation](./install_LIBERO.md) for help. 

## Download language_tokenizer weights
pi-zero uses the language tokenize of google' paligemma-3b-pt-224. 

Notice that Google has some restrictions to the use of paligemma. To download their models from HuggingFace, you need to first login to HuggingFace and sign an agreement. 
```python
huggingface-cli login
```

```python
huggingface-cli download --resume-download google/paligemma-3b-pt-224 --local-dir ./google/paligemma-3b-pt-224
```
But then you will meet this error
```python
Access to model google/paligemma-3b-pt-224 is restricted. You must have access to it and be authenticated to access it. Please log in.
```
Then it means you have to login to your HuggingFace account and apply for a warrant of the usage of paligema. 

## Download the pytorch weights of pi0
* Things you need to know in advance
1. The official weights of pi-zero are in JAX, and it is converted to PyTorch by the HhuggingFace LeRobot team. 
2. Note that this JAX->PyTorch conversion hurts performance. 
3. The total model takes up around 13.04 GB disk space, and it is sliced into three files. 

* Download the pi-zero weights
```bash
huggingface-cli download --resume-download yinchenghust/openpi_base --local-dir ./physical-intelligence/pi0_base/
```

* Try to load pi-zero on a cuda device
```bash
# merge the sliced checkpoints of pi-zero as a single file, and then load it to a device with at least 15GB memory.
python scripts/setup/merge_and_load_pi0.py --device=cuda:1 \
--safetensors_files=<A_LIST_THAT_CONTAINS_THE_SLICED_CHECKPOINTS_FROM_HUGGINGFACE> \
--output_dir=<PLACE_YOU_WANT_TO_LOAD_THE_MERGED_CKPT>
```
Example:
```bash
PYTHONPATH=/nvme_data/tonghe/Pi-R python scripts/setup/merge_and_load_pi0.py --device=cuda:1 \
--safetensors_files ./physical-intelligence/pi0_base/model-00001-of-00003.safetensors ./physical-intelligence/pi0_base/model-00002-of-00003.safetensors ./physical-intelligence/pi0_base/model-00003-of-00003.safetensors \
--output_dir physical-intelligence/pi0_base/pretrained_model
```
After loading the checkpoint, a full pi-zero weight will be stored in `--output_dir`, which takes up around 13.04 GB disk space. The default Pi0 configuration file will also be saved in the same directory in .json format. 

For SFT and RLFT training in maniskill, replace the automatically generated `config.json` file with 
```json
{
  "device": "cuda",
  "attention_implementation": "eager",
  "empty_cameras": 0,
  "freeze_vision_encoder": true,
  "input_features": {
    "observation.images.top": {
      "shape": [
        3,
        480,
        640
      ],
      "type": "VISUAL"
    },
    "observation.state": {
      "shape": [
        8
      ],
      "type": "STATE"
    }
  },
  "max_action_dim": 32,
  "max_state_dim": 32,
  "n_action_steps": 50,
  "act_steps": 50,
  "n_obs_steps": 1,
  "num_steps": 5,
  "normalization_mapping": {
    "ACTION": "MEAN_STD",
    "STATE": "MEAN_STD",
    "VISUAL": "IDENTITY"
  },
  "output_features": {
    "action": {
      "shape": [
        7
      ],
      "type": "ACTION"
    }
  },
  "resize_imgs_with_padding": [
    224,
    224
  ],
  "tokenizer_max_length": 512,
  "train_expert_only": false,
  "train_state_proj": true,
  "use_cache": true
}
```

## Install ManiSkill3 and SimplerENV

```bash
echo "export MS_ASSET_DIR=/path/to/your/mani_skill_assets" >>~/.bashrc # 
# e.g.
echo "export MS_ASSET_DIR=/nvme_data/tonghe/Pi-R/ManiSkill/.mani_skill" >>~/.bashrc
echo "export MS_ASSET_DIR=/nvme_data/tonghe/Pi-R/ManiSkill/.mani_skill" >>~/.bashrc
source ~/.bashrc
conda activate pi_r
```

If you meet this, just neglect it. 
```bash
/nvme_data/tonghe/anaconda3/envs/pi_r/lib/python3.10/site-packages/sapien/__init__.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
import pkg_resources
```
Put the .maniskill folder under your `~/.maniskill`

```bash
pip install --upgrade --force-reinstall setuptools wheel
cd ./RL4VLA/ManiSkill && python -m pip install -e .
cd ../SimplerEnv && python -m pip install -e . && cd ..
```


# Install the repo:
```bash
pip install -e .
```