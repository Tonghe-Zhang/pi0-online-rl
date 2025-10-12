# Pi0 ManiSkill3 Evaluation with Hydra

This directory contains the evaluation script for pi0 models on ManiSkill3 environments, now using Hydra for configuration management.

## Features

- **Hydra Configuration Management**: Easy parameter overloading from command line
- **Configuration Logging**: All evaluation configs are saved to `log_dir/cfg.yaml`
- **Model Architecture Logging**: Model structure and parameters are saved to `log_dir/architecture.log`
- **Flexible Environment Support**: Easy switching between different ManiSkill3 tasks
- **Multiple Model Support**: Support for different pi0 model variants

## Usage

### Basic Usage

```bash
# Run with default configuration
python eval_pi0_maniskill.py

# Override specific parameters
python eval_pi0_maniskill.py eval.num_episodes=10 model.device=cpu

# Use a different environment
python eval_pi0_maniskill.py env=put_on_plate

# Use a different model
python eval_pi0_maniskill.py model=pi0_large
```

### Configuration Overrides

You can override any configuration parameter from the command line:

```bash
# Environment settings
python eval_pi0_maniskill.py env.num_envs=5 env.max_episode_len=100

# Model settings
python eval_pi0_maniskill.py model.path=your/custom/model model.device=cuda:1

# Evaluation settings
python eval_pi0_maniskill.py eval.num_episodes=50 eval.seed=123

# Output settings
python eval_pi0_maniskill.py output.dir=my_results output.save_videos=false
```

### Using Different Configurations

The script supports different preset configurations:

```bash
# Use PutOnPlate environment
python eval_pi0_maniskill.py env=put_on_plate

# Use PickAndPlace environment
python eval_pi0_maniskill.py env=pick_and_place

# Use pi0_large model
python eval_pi0_maniskill.py model=pi0_large

# Combine different configs
python eval_pi0_maniskill.py env=put_on_plate model=pi0_large
```

## Configuration Structure

### Default Configuration (`config/default.yaml`)

```yaml
# Environment settings
env:
  id: "PutOnPlateInScene25VisionTexture03-v1"
  num_envs: 10
  max_episode_len: 80
  obs_mode: "rgb+segmentation"
  control_mode: "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
  sim_config:
    sim_freq: 500
    control_freq: 5
  sensor_configs:
    shader_pack: "default"

# Model settings
model:
  path: "physical-intelligence/pi0_base"
  device: "cuda:0"
  deterministic: false

# Simulation settings
sim:
  device: "cuda:0"
  backend: "gpu"

# Evaluation settings
eval:
  num_episodes: 20
  seed: 0

# Output settings
output:
  dir: "results/eval_pi0_maniskill"
  save_videos: true
  save_data: true
  info_on_video: true
```

### Environment Configurations

- `config/env/put_on_plate.yaml`: PutOnPlate task configuration
- `config/env/pick_and_place.yaml`: PickAndPlace task configuration

### Model Configurations

- `config/model/pi0_base.yaml`: pi0_base model configuration
- `config/model/pi0_large.yaml`: pi0_large model configuration

## Output Structure

The evaluation creates the following directory structure:

```
results/eval_pi0_maniskill/
├── videos/                    # Evaluation videos
├── data/                      # Episode data (JSON files)
└── logs/
    ├── cfg.yaml              # Evaluation configuration
    ├── architecture.log      # Model architecture details
    ├── evaluation_results.json  # Summary results
    └── prompts_log.json      # Language instructions log
```

## Logged Information

### Configuration (`cfg.yaml`)
Contains the complete evaluation configuration used for the run, including all overrides.

### Model Architecture (`architecture.log`)
Detailed information about the model structure:
- Model type and device
- Total and trainable parameter counts
- Layer-by-layer architecture breakdown
- Key layer properties (input/output features, hidden sizes, etc.)

### Evaluation Results (`evaluation_results.json`)
Summary statistics including:
- Success rates
- Episode lengths
- Task-specific metrics
- Raw data for further analysis

### Prompts Log (`prompts_log.json`)
All language instructions sent to the model during evaluation.

## Examples

### Quick Test Run
```bash
# Run a quick test with fewer episodes
python eval_pi0_maniskill.py eval.num_episodes=5 env.num_envs=2
```

### CPU-only Evaluation
```bash
# Run on CPU for environments without GPU
python eval_pi0_maniskill.py model.device=cpu sim.device=cpu
```

### Custom Model Path
```bash
# Use a locally saved model
python eval_pi0_maniskill.py model.path=/path/to/your/model
```

### Different Output Directory
```bash
# Save results to a custom location
python eval_pi0_maniskill.py output.dir=my_experiment_results
```

## Troubleshooting

### CUDA Issues
If you encounter CUDA memory issues, try:
```bash
python eval_pi0_maniskill.py model.device=cpu sim.device=cpu
```

### Device Mismatch Issues
The script now supports running the model and simulation on different devices with automatic tensor transfers.

**Multi-device support**: You can run the model on one GPU and simulation on another:
```bash
# Model on cuda:1, simulation on cuda:2
python eval_pi0_maniskill.py model.device=cuda:1 sim.device=cuda:2

# Model on cuda:0, simulation on cuda:1
python eval_pi0_maniskill.py model.device=cuda:0 sim.device=cuda:1

# Model on GPU, simulation on CPU
python eval_pi0_maniskill.py model.device=cuda:0 sim.device=cpu
```

**Note**: The script automatically handles tensor transfers between devices. For best performance, use the same device when possible:
```bash
# Recommended: Same device for both (best performance)
python eval_pi0_maniskill.py model.device=cuda:0 sim.device=cuda:0

# Or use CPU for both
python eval_pi0_maniskill.py model.device=cpu sim.device=cpu
```

### Configuration Issues
To see the resolved configuration:
```bash
python eval_pi0_maniskill.py --cfg job
```

### Hydra Help
For more Hydra options:
```bash
python eval_pi0_maniskill.py --help
``` 