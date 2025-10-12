

# evaluate success (Single)
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


# Multitask
```bash
cd /mnt/public/zhangtonghe/Pi-R
conda activate pi_r
export CUDA_VISIBLE_DEVICES='2,3'
ENV_NUMBER=100
python scripts/evaluate/eval_pi0_maniskill.py \
--config-dir=scripts/evaluate/config \
--config-name=default \
sim.device=cuda:0 model.device=cuda:1 \
env.id="PutOnPlateInScene25Main-v3" \
env.num_envs=$ENV_NUMBER \
model.path=/mnt/public/zhangtonghe/openpi/results/sft_pi0_maniskill/2025-09-04_20-58-23/best/model \
dataset_stats=/mnt/public/zhangtonghe/RL4VLA/datasets/warmup/pi0_sft/PutOnPlateInScene25Main-v3/mp_18432/normalization.pt \
model.model_overrides.act_steps=5 \
model.model_overrides.n_action_steps=8 \
model.model_overrides.num_steps=4 \
verbose=false \
verbose_each_step=false
```

