
# 5 output action chunk, 5 exec chunk, 4 denoise steps. 
# Critic learning rate = 3e-5 ~ 1e-4
# Actor learning rate <= 1e-4
cd /mnt/public/zhangtonghe/Pi-R/
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2'
export OMP_NUM_THREADS=4
ENV_NAME=PutOnPlateInScene25Main-v3
Na=5
Nc=5
Nd=4
lrA=2e-5
lrC=2e-4
lrCmin=6e-5
Cwarmup=80
SEED=0
NAME=${ENV_NAME}_Na${Na}_Nc${Nc}_Nd${Nd}_lrA${lrA}_lrC${lrC}_lrCmin${lrCmin}_Cwarmup${Cwarmup}_seed${SEED}
nohup python scripts/rlft/train_rl.py \
sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.id=${ENV_NAME} \
name=${NAME} \
seed=${SEED}  \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
model.path=/mnt/public/zhangtonghe/openpi/results/sft_pi0_maniskill/2025-09-04_20-58-23/best/model \
model.model_config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
env.num_envs=100 \
n_action_steps=${Nc} \
act_steps=${Na} \
num_steps=${Nd} num_steps_eval=${Nd} \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=20 \
train.skip_initial_eval=false \
train.use_early_stop=true \
train.actor_optimizer.lr=${lrA} \
train.critic_optimizer.lr=${lrC} \
train.critic_lr_scheduler.min_lr=${lrCmin} \
train.n_critic_warmup_itr=${Cwarmup} \
rlft_config.critic.hidden_dims=[512,512,256] \
train.ent_coef=0.03 \
wandb.offline_mode=true > ./${NAME}.log 2>&1 &
[2] 196392



# 32 batchsize
cd /mnt/public/zhangtonghe/Pi-R/
conda activate pi_r
export CUDA_VISIBLE_DEVICES='0,1,2'
export OMP_NUM_THREADS=4
ENV_NAME=PutOnPlateInScene25Main-v3
Na=5
Nc=5
Nd=4
lrA=2.8e-5
lrC=1.4e-4
lrCmin=3e-5
Cwarmup=80
SEED=0
NAME=${ENV_NAME}_Na${Na}_Nc${Nc}_Nd${Nd}_lrA${lrA}_lrC${lrC}_lrCmin${lrCmin}_Cwarmup${Cwarmup}_seed${SEED}
nohup python scripts/rlft/train_rl.py \
sim.device=cuda:0 model.device=cuda:1 buffer.device=cuda:2 \
env.id=${ENV_NAME} \
name=${NAME} \
seed=${SEED}  \
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
model.path=/mnt/public/zhangtonghe/openpi/results/sft_pi0_maniskill/2025-09-04_20-58-23/best/model \
model.model_config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
env.num_envs=100 \
n_action_steps=${Nc} \
act_steps=${Na} \
num_steps=${Nd} num_steps_eval=${Nd} \
env.n_steps_rollout=50 \
env.max_episode_len=50 \
train.batch_size=32 \
train.skip_initial_eval=false \
train.use_early_stop=true \
train.actor_optimizer.lr=${lrA} \
train.critic_optimizer.lr=${lrC} \
train.critic_lr_scheduler.min_lr=${lrCmin} \
train.n_critic_warmup_itr=${Cwarmup} \
# train.record_video_condition="periodic" \
# train.video_freq=${train.val_freq} \
rlft_config.critic.hidden_dims=[512,512,256] \
train.ent_coef=0.03 \
wandb.offline_mode=true > ./${NAME}.log 2>&1 &


# 8 output action chunk, 5 exec chunk, 4 denoise steps. 
# Critic learning rate = 3e-5 ~ 1e-4
# Actor learning rate <= 1e-4
cd /mnt/public/zhangtonghe/Pi-R/
conda activate pi_r
export CUDA_VISIBLE_DEVICES='1,2,3'
export OMP_NUM_THREADS=4
ENV_NAME=PutOnPlateInScene25Main-v3
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
dataset.normalization_path=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
model.path=/mnt/public/zhangtonghe/openpi/results/sft_pi0_maniskill/2025-09-04_20-58-23/best/model \
model.model_config_overrides.lang_tokenizer_path=/mnt/public/zhangtonghe/google/paligemma-3b-pt-224 \
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






