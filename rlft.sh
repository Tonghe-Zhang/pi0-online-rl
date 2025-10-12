

export CUDA_VISIBLE_DEVICES='4,5,6,7'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=5
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Single-v1
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
name=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/nvme_data/tonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
dataset.shard_metadata_path_train=/nvme_data/tonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/nvme_data/tonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/val_dataset_sharded.json \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=1600     \
training.verbose=false \
training.save_freq=50 \
eval.eval_steps=50 \
eval.test_in_sim=true \
sim_num_envs=16 \
logging.log_freq=10




```bash
# 7 output action chunk, 5 exec chunk, 4 denoise steps.
export CUDA_VISIBLE_DEVICES='0,1,2'
export OMP_NUM_THREADS=4
python scripts/rlft/train_rl.py sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.id=PutOnPlateInScene25Single-v1 \
dataset.normalization_path=/nvme_data/tonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
model.path=/nvme_data/tonghe/openpi/results/sft_pi0_maniskill/2025-08-05_15-23-04/best/model \
env.num_envs=64 \
act_steps=5 \
n_action_steps=7 \
num_steps=4 num_steps_eval=4 \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=16 \
train.skip_initial_eval=false \
train.use_early_stop=true
```

```bash
# 8 output action chunk, 5 exec chunk, 4 denoise steps.
export CUDA_VISIBLE_DEVICES='3,4,5'
export OMP_NUM_THREADS=4
python scripts/rlft/train_rl.py sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.id=PutOnPlateInScene25Single-v1 \
dataset.normalization_path=/nvme_data/tonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
model.path=/nvme_data/tonghe/openpi/results/sft_pi0_maniskill/2025-08-06_01-06-57/best/model \
env.num_envs=64 \
act_steps=5 \
n_action_steps=8 \
num_steps=4 num_steps_eval=4 \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=16 \
train.skip_initial_eval=false \
train.use_early_stop=true
```















```bash
# 5 output action chunk, 5 exec chunk, 4 denoise steps.
# [1] 2622132
export CUDA_VISIBLE_DEVICES='0,2,3'
export OMP_NUM_THREADS=4
nohup python scripts/rlft/train_rl.py sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.id=PutOnPlateInScene25Single-v1 \
dataset.normalization_path=/nvme_data/tonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
model.path=/nvme_data/tonghe/openpi/results/sft_pi0_maniskill/2025-08-06_01-08-27/best/model \
env.num_envs=64 \
act_steps=5 \
n_action_steps=5 \
num_steps=4 num_steps_eval=4 \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=16 \
train.skip_initial_eval=false \
train.use_early_stop=true \
wandb.offline_mode=true > ./rlft_n5_a5_d4.log 2>&1 &
```



```bash
# 8 output action chunk, 5 exec chunk, 4 denoise steps. 
# Critic learning rate = 3e-5 ~ 1e-4
# Actor learning rate <= 1e-4
cd /mnt/mnt/public/zhangtonghe/Pi-R/
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2'
export OMP_NUM_THREADS=4
ENV_NAME=PutOnPlateInScene25Single-v1
Na=5
Nc=8
Nd=4
lrA=2e-5
lrC=1e-4
lrCmin=3e-5
Cwarmup=30
SEED=0
NAME=${ENV_NAME}_Na${Na}_Nc${Nc}_Nd${Nd}_lrA${lrA}_lrC${lrC}_lrCmin${lrCmin}_Cwarmup${Cwarmup}_seed${SEED}
nohup python scripts/rlft/train_rl.py \
sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.id=${ENV_NAME} \
name=${NAME} \
seed=${SEED}  \
dataset.normalization_path=/mnt/mnt/public/zhangtonghe/Pi-R/results/sft_pi0_maniskill/pi0-maniskill-singletask-sft/normalization.pt \
model.path=/mnt/mnt/public/zhangtonghe/Pi-R/results/sft_pi0_maniskill/pi0-maniskill-singletask-sft \
model.model_config_overrides.tokenizer_path=/mnt/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
env.num_envs=64 \
n_action_steps=${Nc} \
act_steps=${Na} \
num_steps=${Nd} num_steps_eval=${Nd} \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=16 \
train.skip_initial_eval=false \
train.use_early_stop=true \
train.actor_optimizer.lr=${lrA} \
train.critic_optimizer.lr=${lrC} \
train.critic_lr_scheduler.min_lr=${lrCmin} \
train.n_critic_warmup_itr=${Cwarmup} \
wandb.offline_mode=true > ./${NAME}.log 2>&1 &
```

```bash
# 8 output action chunk, 5 exec chunk, 4 denoise steps. 
# Critic learning rate = 3e-5 ~ 1e-4
# Actor learning rate <= 1e-4
cd /mnt/mnt/public/zhangtonghe/Pi-R/
conda activate pi_r
export CUDA_VISIBLE_DEVICES='1,2,3'
export OMP_NUM_THREADS=4
ENV_NAME=PutOnPlateInScene25Single-v1
Na=5
Nc=8
Nd=4
lrA=2e-5
lrC=1e-4
lrCmin=3e-5
Cwarmup=30
SEED=0
NAME=${ENV_NAME}_Na${Na}_Nc${Nc}_Nd${Nd}_lrA${lrA}_lrC${lrC}_lrCmin${lrCmin}_Cwarmup${Cwarmup}_seed${SEED}
nohup python scripts/rlft/train_rl.py \
sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.id=${ENV_NAME} \
name=${NAME} \
seed=${SEED}  \
dataset.normalization_path=/mnt/public/zhangtonghe/Pi-R/results/sft_pi0_maniskill/pi0-maniskill-singletask-sft/normalization.pt \
model.path=/mnt/public/zhangtonghe/openpi/results/sft_pi0_maniskill/2025-09-04_20-58-23/best/model \
model.model_config_overrides.tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
env.num_envs=64 \
n_action_steps=${Nc} \
act_steps=${Na} \
num_steps=${Nd} num_steps_eval=${Nd} \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=16 \
train.skip_initial_eval=false \
train.use_early_stop=true \
train.actor_optimizer.lr=${lrA} \
train.critic_optimizer.lr=${lrC} \
train.critic_lr_scheduler.min_lr=${lrCmin} \
train.n_critic_warmup_itr=${Cwarmup} \
wandb.offline_mode=true > ./${NAME}.log 2>&1 &
```

