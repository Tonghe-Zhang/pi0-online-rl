# test training result:

export CUDA_VISIBLE_DEVICES='0,1'
ENV_NUMBER=100
python scripts/evaluate/eval_pi0_maniskill.py --config-dir=scripts/evaluate/config --config-name=default \
sim.device=cuda:0 model.device=cuda:1 \
env.id="PutOnPlateInScene25Single-v1" \
env.num_envs=$ENV_NUMBER \
model.model_overrides.act_steps=5 \
model.model_overrides.n_action_steps=7 \
model.model_overrides.num_steps=4 \
dataset_stats=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
model.path=/mnt/public/zhangtonghe/openpi/results/sft_pi0_maniskill/2025-08-05_15-23-04/best/model




# * Single obj: carrot
export CUDA_VISIBLE_DEVICES='3,5,6,7'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=7
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
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/val_dataset_sharded.json \
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

# * Single obj: carrot, this one is working with peak success rate=56%. Reacing the lowest validation loss and best success rate at around 600-1200 steps. 
export CUDA_VISIBLE_DEVICES='3,5,6,7'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=7
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
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/val_dataset_sharded.json \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=50000     \
training.verbose=false \
training.save_freq=50_000 \
logging.log_freq=10 \
eval.eval_steps=50 \
eval.test_in_sim=true \
sim_num_envs=16


## [TESTING]
# * Multiple objs: carrot
export CUDA_VISIBLE_DEVICES='3,5,6,7'
export OMP_NUM_THREADS=8
n_ACTION_STEPS=5
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
name=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Single-v1/octo_mp_140/val_dataset_sharded.json \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=50000     \
training.verbose=false \
training.save_freq=50_000 \
logging.log_freq=10 \
eval.eval_steps=50 \
eval.test_in_sim=true \
sim_num_envs=16



# 4 GPUs
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/val_dataset_sharded.json \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
logging.log_freq=50 \
eval.eval_steps=1000 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &



# 3 GPUs (logfilename mistakenly 3 gpus, it should be 2. same goes for tmux session name and wandb run name.)
cd /mnt/public/zhangtonghe/openpi
conda activate pi_r
export CUDA_VISIBLE_DEVICES='1,2,3'
export OMP_NUM_THREADS=3
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_3gpus_2try
clear
nohup torchrun --standalone --nproc-per-node=3 \
scripts/sft/train_sft.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/val_dataset_sharded.json \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=10     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
logging.log_freq=50 \
eval.eval_steps=1000 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &
# [1] 3178889





# 3 GPUs (logfilename mistakenly 3 gpus, it should be 2. same goes for tmux session name and wandb run name.) with parallel validation. 
cd /mnt/public/zhangtonghe/openpi
conda activate pi_r
export CUDA_VISIBLE_DEVICES='1,2,3'
export OMP_NUM_THREADS=3
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_3gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=3 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/val_dataset_sharded.json \
training.num_workers=8 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=10     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=10 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &



# low CPU workload: 

# 3 GPUs (logfilename mistakenly 3 gpus, it should be 2. same goes for tmux session name and wandb run name.) with parallel validation. 
cd /mnt/public/zhangtonghe/openpi
conda activate pi_r
export CUDA_VISIBLE_DEVICES='1,2,3'
export OMP_NUM_THREADS=1
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_3gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=3 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/val_dataset_sharded.json \
training.num_workers=1 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=10     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=10 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &





# 3 GPUs (logfilename mistakenly 3 gpus, it should be 2. same goes for tmux session name and wandb run name.) with parallel validation. 
cd /mnt/public/zhangtonghe/openpi
conda activate pi_r
export CUDA_VISIBLE_DEVICES='1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_3gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=3 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/val_dataset_sharded.json \
training.num_workers=4 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=10     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=10 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &


# 3 GPUs (logfilename mistakenly 3 gpus, it should be 2. same goes for tmux session name and wandb run name.) with parallel validation. 
cd /mnt/public/zhangtonghe/openpi
conda activate pi_r
export CUDA_VISIBLE_DEVICES='1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_3gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=3 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/val_dataset_sharded.json \
training.num_workers=4 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=10     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=1000 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &



















# 3 GPUs (logfilename mistakenly 3 gpus, it should be 2. same goes for tmux session name and wandb run name.) with parallel validation. 
cd /mnt/public/zhangtonghe/openpi
conda activate pi_r
export CUDA_VISIBLE_DEVICES='4,5,6,7'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_4gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/train_dataset_sharded.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_14848/val_dataset_sharded.json \
training.num_workers=8 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=1000 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &






# 4 GPUs
cd /mnt/public/zhangtonghe/Pi-R
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_4gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
model.path=/mnt/public/zhangtonghe/physical-intelligence \
model.config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/train_dataset_sharded_with_lengths.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/val_dataset_sharded_with_lengths.json \
training.num_workers=4 \
training.num_val_workers=0 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=1000 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &


# this one is working without hangs
cd /mnt/public/zhangtonghe/Pi-R
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_4gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
model.path=/mnt/public/zhangtonghe/physical-intelligence \
model.config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/train_dataset_sharded_with_lengths.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/val_dataset_sharded_with_lengths.json \
training.num_workers=4 \
training.num_val_workers=0 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=5 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &


# test everything in a small time scale, successful. 
cd /mnt/public/zhangtonghe/Pi-R
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_4gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
model.path=/mnt/public/zhangtonghe/physical-intelligence \
model.config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/train_dataset_sharded_with_lengths.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/val_dataset_sharded_with_lengths.json \
training.num_workers=4 \
training.num_val_workers=0 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=5 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=5 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &




# test everything in a small time scale with more train workers 
cd /mnt/public/zhangtonghe/Pi-R
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_4gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
model.path=/mnt/public/zhangtonghe/physical-intelligence \
model.config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/train_dataset_sharded_with_lengths.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/val_dataset_sharded_with_lengths.json \
training.num_workers=8 \
training.num_val_workers=0 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=5 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=5 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &

# formal run
cd /mnt/public/zhangtonghe/Pi-R
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_4gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
model.path=/mnt/public/zhangtonghe/physical-intelligence \
model.config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/train_dataset_sharded_with_lengths.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/val_dataset_sharded_with_lengths.json \
training.num_workers=4 \
training.num_val_workers=0 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=1000 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &





# Formal run with more workers
cd /mnt/public/zhangtonghe/Pi-R
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export OMP_NUM_THREADS=4
n_ACTION_STEPS=8
ACT_STEPS=5
NUM_STEPS=4
TASK_NAME=PutOnPlateInScene25Main-v3
NAME=${TASK_NAME}_outchunk${n_ACTION_STEPS}_exechunk${ACT_STEPS}_denoise${NUM_STEPS}_4gpus_disteval
clear
nohup torchrun --standalone --nproc-per-node=4 \
scripts/sft/train_sft_disteval.py \
name=${NAME} \
task_name=${TASK_NAME} \
n_action_steps=${n_ACTION_STEPS} \
act_steps=${ACT_STEPS} \
num_steps=${NUM_STEPS} \
model.path=/mnt/public/zhangtonghe/physical-intelligence \
model.config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
dataset.shard_metadata_path_train=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/train_dataset_sharded_with_lengths.json \
dataset.shard_metadata_path_val=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/val_dataset_sharded_with_lengths.json \
training.num_workers=4 \
training.num_val_workers=0 \
device.model_device=auto     \
training.use_amp=false    \
training.warmup_steps=0 \
training.learning_rate=1e-4 \
scheduler.min_lr=1e-4 \
training.batch_size=8     \
training.max_steps=80000     \
training.save_freq=1000 \
training.verbose=false \
training.verbose_val=false \
logging.log_freq=50 \
eval.eval_steps=1000 \
eval.test_in_sim=true \
sim_num_envs=16 \
wandb.offline_mode=true \
> ${NAME}.log 2>&1 &
# 4 gpus x 1 workers x 4 threads = 16 threads.. just for dataloading.
# 4 gpus x 2 workers x 4 threads = 32 threads.. just for dataloading.
# 4 gpus x 4 workers x 4 threads = 64 threads.. just for dataloading.
# 4 gpus x 16 workers x 4 threads = 128 threads.. just for dataloading.

