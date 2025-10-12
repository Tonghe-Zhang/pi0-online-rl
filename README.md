




# Fine-tune LeRobot $\pi_0$ in SimplerEnv with Online RL
Author: Tonghe Zhang
Date: July, 2025 to October, 2025

* Note that the commands in this readme are outdated. Visit the .sh instead. 

## TODOs
Places need to be changed in ppo_runner.py
1. self.model.effective_action_dim currently we slice it by treating the first several indices as valid motors. 
but maybe these motors ids are discontinuous, then we should replace :eff_act_dim with eff_act_ids
1. Precision issues: the pytorch weights contain float point 32 and bfloat16 may need to make it consistent before using FSDP/deepspeed. 
2.  amp overflow problem bfloat16 paligemma attention mask large constant (`big_neg`)

## Installation

Please refer to [install.md](./docs/install.md) for details. 

## Data curation
### Motion planning 
```bash
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e "PutOnPlateInScene25Single-v1" \
  --save_video --save_data --num_procs 1 --num_traj 75 --seed=0 
```


## Evaluation
* `Partial reset`:
We choose to wrap the environment with ManiSkill3 vectorized env wrapper, 
and stop partial reset and ignore truncation during evaluation. 
This will make each parallel environment only rollout a single episode (since no reset within each environment), and result in one success or failure. 

* `Memory`:
Currently we load the model on one GPU and setup simulation experiments on another. 
For single task like `PutOnPlateInScene25Single-v1`, 100 parallel environments taks up around 30 GB space on one GPU. 

* `ManiSkill vectorized wrapper`:
We by default set `env.use_maniskill_env_wrapper=True` during environment creation. 

* `env.id`: 
`PutOnPlateInScene25Single-v1`
```bash
export CUDA_VISIBLE_DEVICES='0,1'
ENV_NUMBER=20
python scripts/evaluate/eval_pi0_maniskill.py --config-dir=scripts/evaluate/config --config-name=default \
sim.device=cuda:0 model.device=cuda:1 \
env.id="PutOnPlateInScene25Single-v1" \
env.num_envs=$ENV_NUMBER \
model.path=/mnt/public/zhangtonghe/openpi/results/sft_pi0_maniskill/2025-08-05_15-23-04/best/model \
dataset_stats=/mnt/public/zhangtonghe/openpi/normalization/pi0/warmup/PutOnPlateInScene25Single-v1/normalization.pt \
model.model_overrides.act_steps=5 \
model.model_overrides.n_action_steps=50 \
model.model_overrides.num_steps=10 \
verbose=false \
verbose_each_step=false
```







`PutOnPlateInScene25Main-v3`
with normalization files obtained from 12,800 motion planning episodes. 
```bash
# two gpus
export CUDA_VISIBLE_DEVICES='0,1'
python scripts/evaluate/eval_pi0_maniskill.py --config-dir=scripts/evaluate/config --config-name=default \
sim.device=cuda:0 model.device=cuda:1 \
env.id="PutOnPlateInScene25Main-v3" \
env.num_envs=100 \
model.model_overrides.act_steps=5 \
model.model_overrides.n_action_steps=50 \
model.model_overrides.num_steps=10 \
dataset_stats="/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/12800/pi0_sft_normalization.pt"
```


```bash
# one gpu, 10 output action chunk
# SFT
export CUDA_VISIBLE_DEVICES='1'
python scripts/evaluate/eval_pi0_maniskill.py --config-dir=scripts/evaluate/config --config-name=default \
sim.device=cuda:0 model.device=cuda:0 \
env.id="PutOnPlateInScene25Main-v3" \
env.num_envs=16 \
model.model_overrides.act_steps=5 \
model.model_overrides.n_action_steps=10 \
model.model_overrides.num_steps=10 \
model.path=results/sft_pi0_maniskill/2025-07-08_02-42-20/best/model \
dataset_stats="/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/12800/pi0_sft_normalization.pt"

# pre-trained 
export CUDA_VISIBLE_DEVICES='1'
python scripts/evaluate/eval_pi0_maniskill.py --config-dir=scripts/evaluate/config --config-name=default \
sim.device=cuda:0 model.device=cuda:0 \
env.id="PutOnPlateInScene25Main-v3" \
env.num_envs=16 \
model.model_overrides.act_steps=5 \
model.model_overrides.n_action_steps=10 \
model.model_overrides.num_steps=10 \
model.path=physical-intelligence/pi0_base/pretrained_model \
dataset_stats="/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/12800/pi0_sft_normalization.pt"
```



```bash
export CUDA_VISIBLE_DEVICES='4,5'
python scripts/evaluate/eval_pi0_maniskill.py --config-dir=scripts/evaluate/config --config-name=default \
sim.device=cuda:0 model.device=cuda:1 \
env.id="PutOnPlateInScene25Main-v3" \
env.num_envs=100 \
model.path=results/sft_pi0_maniskill/2025-07-07_13-55-15/last/model \
model.model_overrides.act_steps=5 \
model.model_overrides.n_action_steps=50 \
model.model_overrides.num_steps=10 \
dataset_stats="/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/12800/pi0_sft_normalization.pt"
```



`PutOnPlateInScene25MultiPlate-v`

 


## Supervised Fine-tuning

* `training.batch_size=8`: using 16 causes OOM for 40 GB A100s, using 8 consumes approx. 32GB. 
* `horizon_steps`: 50 steps is too much for motion planning datasets, which usually only uses 10-30 steps. currently we do not support automatic mixed precision yet, as paligemma gives you a big_neg = -2.3819763e38 in the attention mask and it definitely overflows when casting to fp16. 

📁 Dataset: /mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft.pt
📈 Total Episodes: 1000
📋 Total Samples: 19169
📏 Episode Length Range: 13-26 (avg: 19.2)
🎯 Recommended horizon_steps: 13 (for 100% data usage)
Maximum safe horizon_steps: 13
Recommended horizon_steps for 100% data usage: 13
Recommended horizon_steps for 95% data usage: 15
Recommended horizon_steps for 90% data usage: 16

Data retention with different horizon_steps:
  horizon_steps= 1: 1000/1000 episodes (100.0%)
  horizon_steps= 5: 1000/1000 episodes (100.0%)
  horizon_steps=10: 1000/1000 episodes (100.0%)
  horizon_steps=15: 993/1000 episodes (99.3%)
  horizon_steps=20: 406/1000 episodes (40.6%)
  horizon_steps=25:  18/1000 episodes (1.8%)
  horizon_steps=30:   0/1000 episodes (0.0%)

### Single GPU:
```bash
export CUDA_VISIBLE_DEVICES='3,7'
DATASET_EPISODE=75
DATASET_BASEDIR=/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft
TASK_NAME=PutOnPlateInScene25Single-v1
python scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
dataset.path=${DATASET_BASEDIR}/${TASK_NAME}/${DATASET_EPISODE}/pi0_sft.pt \
dataset.normalization_path=${DATASET_BASEDIR}/${TASK_NAME}/${DATASET_EPISODE}/pi0_sft_normalization.pt \
device.model_device=cuda:0     \
training.use_amp=false    \
model.config_overrides.n_action_steps=10     \
model.config_overrides.num_steps=10     \
dataset.horizon_steps=10     \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
eval.eval_steps=25 \
eval.test_in_sim=true
```

* train-val split with sharded dataset. 
```bash
export CUDA_VISIBLE_DEVICES='3,7'
TASK_NAME=PutOnPlateInScene25Main-v3
python scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=cuda:0     \
training.use_amp=false    \
model.config_overrides.n_action_steps=10     \
model.config_overrides.num_steps=10     \
dataset.horizon_steps=10     \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
eval.eval_steps=25 \
eval.test_in_sim=true
```




### Multiple GPUs on the same machine

* Single obj

```bash
export CUDA_VISIBLE_DEVICES='1,5,6,7'
TASK_NAME=PutOnPlateInScene25Single-v1
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=auto     \
training.use_amp=false    \
n_action_steps=10     \
num_steps=4     \
act_steps=5 \
training.warmup_steps=10 \
training.learning_rate=1e-5 \
scheduler.min_lr=1e-5 \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
logging.log_freq=10 \
eval.eval_steps=200 \
eval.test_in_sim=true \
sim_num_envs=16 \
domain_randomization=null
```







```bash
export CUDA_VISIBLE_DEVICES='0,2,4,5,6'
torchrun --standalone --nproc-per-node=5 \
scripts/sft/train_sft.py     \
task_name="PutOnPlateInScene25Main-v3" \
dataset.path=/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft.pt     \
dataset.normalization_path=/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft_normalization.pt     \
device.model_device=auto    \
training.use_amp=false    \
model.config_overrides.n_action_steps=10     \
model.config_overrides.num_steps=10     \
dataset.horizon_steps=10     \
training.batch_size=8     \
training.max_steps=50000     \
training.verbose=false \
save_eval_steps=50
```


```bash
# this one is running on 2 gpus. 
# and in fact, we only need to run for 21_000 steps, or 35 epochs to make the eval loss plateau. however the max_steps maybe 30_000 or something since it has a decay. 
export CUDA_VISIBLE_DEVICES='0,7'
torchrun --standalone --nproc-per-node=2 \
scripts/sft/train_sft.py     \
task_name="PutOnPlateInScene25Main-v3" \
dataset.path=/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft.pt     \
dataset.normalization_path=/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft_normalization.pt     \
device.model_device=auto    \
training.use_amp=false    \
model.config_overrides.n_action_steps=10     \
model.config_overrides.num_steps=10     \
dataset.horizon_steps=10     \
training.batch_size=8     \
training.max_steps=50000     \
training.verbose=false \
save_eval_steps=250
```


```bash
export CUDA_VISIBLE_DEVICES='0,1,7'
torchrun --standalone --nproc-per-node=3 \
scripts/sft/train_sft.py     \
task_name="PutOnPlateInScene25Main-v3" \
dataset.path=/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft.pt     \
dataset.normalization_path=/mnt/public/zhangtongh/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/1000/pi0_sft_normalization.pt     \
device.model_device=auto    \
training.use_amp=false    \
model.config_overrides.n_action_steps=10     \
model.config_overrides.num_steps=10     \
dataset.horizon_steps=10     \
training.batch_size=8     \
training.max_steps=50000     \
training.verbose=false \
test_in_sim=true
```



* train-val split with sharded dataset. With `num_steps=4`. 
* 4 GPUs, 1.5 hours/epoch, 10 batchsize--> 36-39GB VRAM 
* 6788 data in training set
* 1425 data in validation set, 6 mins eval
```bash
export CUDA_VISIBLE_DEVICES='1,5,6,7'
TASK_NAME=PutOnPlateInScene25Main-v3
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=auto     \
training.use_amp=false    \
n_action_steps=10     \
num_steps=4     \
act_steps=5 \
training.warmup_steps=10 \
training.learning_rate=1e-5 \
scheduler.min_lr=1e-5 \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
logging.log_freq=10 \
eval.eval_steps=200 \
eval.test_in_sim=true \
sim_num_envs=16
# nohup \ > ./nohup/pi0_sft_${TASK_NAME}.log 2>&1 &
```




* no domain randomization:
```bash
export CUDA_VISIBLE_DEVICES='1,5,6,7'
TASK_NAME=PutOnPlateInScene25Main-v3
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=auto     \
training.use_amp=false    \
n_action_steps=10     \
num_steps=4     \
act_steps=5 \
training.warmup_steps=10 \
training.learning_rate=1e-5 \
scheduler.min_lr=1e-5 \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
logging.log_freq=10 \
eval.eval_steps=200 \
eval.test_in_sim=true \
sim_num_envs=16 \
domain_randomization=null
```


```bash
export CUDA_VISIBLE_DEVICES='1,5,6,7'
TASK_NAME=PutOnPlateInScene25Main-v3
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=auto     \
training.use_amp=false    \
n_action_steps=6     \
num_steps=4     \
act_steps=2 \
training.warmup_steps=10 \
training.learning_rate=1e-5 \
scheduler.min_lr=1e-5 \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
logging.log_freq=10 \
eval.eval_steps=200 \
eval.test_in_sim=true \
sim_num_envs=16
```







* Smaller chunk size to adapt to MP data. No first success rate drops to 6.25% then 0.00%. 
```bash
export CUDA_VISIBLE_DEVICES='1,5,6,7'
TASK_NAME=PutOnPlateInScene25Main-v3
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=auto     \
training.use_amp=false    \
n_action_steps=3     \
num_steps=4     \
act_steps=2 \
training.warmup_steps=10 \
training.learning_rate=1e-5 \
scheduler.min_lr=1e-5 \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
logging.log_freq=10 \
eval.eval_steps=200 \
eval.test_in_sim=true \
sim_num_envs=16 \
domain_randomization=null
```

* Smaller learning rate: the successs rate at first test goes from 0.00 to 12.5%. but the loss on expert data is getting down slower. Then 6.25%.
```bash
export CUDA_VISIBLE_DEVICES='1,5,6,7'
TASK_NAME=PutOnPlateInScene25Main-v3
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=auto     \
training.use_amp=false    \
n_action_steps=10     \
num_steps=4     \
act_steps=5 \
training.warmup_steps=1 \
training.learning_rate=1e-6 \
scheduler.min_lr=1e-6 \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
logging.log_freq=10 \
eval.eval_steps=200 \
eval.test_in_sim=true \
sim_num_envs=16 \
domain_randomization=null
```









```bash
export CUDA_VISIBLE_DEVICES='0,1,5,6,7'
TASK_NAME=PutOnPlateInScene25Main-v3
torchrun --standalone --nproc-per-node=5 \
scripts/sft/train_sft.py \
task_name=${TASK_NAME} \
device.model_device=auto     \
training.use_amp=false    \
n_action_steps=10     \
num_steps=4     \
act_steps=5 \
training.warmup_steps=10 \
training.learning_rate=1e-5 \
scheduler.min_lr=1e-5 \
training.batch_size=10     \
training.max_steps=50000     \
training.verbose=false \
training.save_steps=50_000 \
logging.log_freq=10 \
eval.eval_steps=200 \
eval.test_in_sim=true \
eval.sim_cfg_overrides.env.num_envs=25
```





## Online Parallel RL in Sim
* `domain_randomization` causes each iter to be slower for 20 seconds. 
* `Partial reset`:
For parallel sim we allow for partial reset and require that the 
number of environments be large enough to offset slight distribution shift before and after resets. 

Each environment will rollout multiple episodes, automatically 
reset when a done flag is detected, and resetting one environment will not 
affect resetting another. 

This also requires wrapping up the gymnasium environment with maniskill vectorized env and specify several arguments...

* `Shape of images`
We modify the pi-zero config file to use [3,480,640] as 
```json
"input_features": {
    "observation.images.top": {
      "shape": [
        3,
        480,
        640
      ],
```
instead of the default [3,256,256] to align with ManiSkill3 SimplerEnv's design. 


```bash
export CUDA_VISIBLE_DEVICES='3,4,5'
python scripts/rlft/train_rl.py wandb=null sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.num_envs=100 act_steps=5 n_action_steps=50 num_steps=4 num_steps_eval=4 env.n_steps_rollout=100 env.max_episode_len=100 \
train.batch_size=64

# minimal testing example to see if optimization is working correctly. 
python scripts/rlft/train_rl.py wandb=null sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
act_steps=2 n_action_steps=5 num_steps=2 num_steps_eval=2 \
env.n_steps_rollout=20 env.max_episode_len=20 \
env.num_envs=10 train.batch_size=16 train.skip_initial_eval=true
```
* batch size =32 causes OOM on 40 GB A100

* the `n_action_steps` should align with how we do it in SFT. 

```bash
export CUDA_VISIBLE_DEVICES='3,5,7'
python scripts/rlft/train_rl.py sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:1 \
env.num_envs=64 \
act_steps=5 n_action_steps=10 \
num_steps=4 num_steps_eval=4 \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=16 \
train.skip_initial_eval=true \
train.use_early_stop=false

# wandb=null 
# domain_randomization=null
```






## Understanding the code

### Key implementations

Training OpenVLA with RL: RL4VLA/SimplerEnv/simpler_env/policies/openvla/openvla_training.py



### Problems:


* There is a package conflict in openVLA and pi-zero
To use paligemma you need transformers>=4.40.1 but openvla needs a lower version. 

openvla 0.0.3 requires tokenizers==0.19.1, but you have tokenizers 0.21.1 which is incompatible.
openvla 0.0.3 requires transformers==4.40.1, but you have transformers 4.52.4 which is incompatible.

* Why is this time reversed? in `sample_actions` in [modeling_pi0.py](lerobot/common/policies/pi0/modeling_pi0.py)
```python
dt = -1.0 / self.config.num_steps
```

* Sounds like we need to download the google/paligemma-3b-pt-224 as the language tokenizer, see lerobot/common/policies/pi0/modeling_pi0.py, `self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")`


* This feature scaling is pretty wierd, not sure why not scaling with d**(-0.5) instead. here, it is d**(+0.5)
img_embs = img_embs * torch.tensor(img_emb_dim**0.5, dtype=dtype, device=device)
in `lerobot/common/policies/pi0/modeling_pi0.py`


## Potential issues
* attention implementation: 
currently in `lerobot/common/policies/pi0/configuration_pi0.py` the attention_implementation: str = "eager"  # or fa2, flex is set to eager by default. maybe should use flex?


## Optimize training speed


### Needs revision:

### Revised, but not tested yet:
* Heavily optimize this `def embed_prefix` --> `embed_prefix_fast` 






