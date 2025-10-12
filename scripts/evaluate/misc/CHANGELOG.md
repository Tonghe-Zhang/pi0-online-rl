# Changelog - Pi0 ManiSkill3 Evaluation Script

## Version 2.0.0 - Hydra Integration

### Major Changes

#### 🔧 Configuration Management
- **Replaced tyro with Hydra**: Migrated from tyro-based argument parsing to Hydra configuration management
- **Hierarchical Configuration**: Implemented structured configuration with separate files for different environments and models
- **Command-line Overrides**: Added support for easy parameter overloading from command line

#### 📁 New Configuration Structure
```
config/
├── default.yaml              # Default configuration
├── quick_test.yaml           # Quick test configuration
├── env/
│   ├── put_on_plate.yaml     # PutOnPlate environment config
│   └── pick_and_place.yaml   # PickAndPlace environment config
└── model/
    ├── pi0_base.yaml         # pi0_base model config
```

#### 📊 Enhanced Logging
- **Configuration Logging**: All evaluation configs are now saved to `log_dir/cfg.yaml`
- **Model Architecture Logging**: Model structure and parameters are saved to `log_dir/architecture.log`
- **Improved Output Structure**: Better organized output directories and files

### New Features

#### 🚀 Easy Configuration Overrides
```bash
# Override specific parameters
python eval_pi0_maniskill.py eval.num_episodes=10 model.device=cpu

# Use preset configurations
python eval_pi0_maniskill.py env=put_on_plate model=pi0_large

# Quick test runs
python eval_pi0_maniskill.py --config-name=quick_test
```

#### 📋 Model Architecture Information
The script now logs detailed model information including:
- Total and trainable parameter counts
- Layer-by-layer architecture breakdown
- Key layer properties (input/output features, hidden sizes, etc.)

#### 🔧 Flexible Environment Support
- Easy switching between different ManiSkill3 tasks
- Configurable simulation parameters
- Support for different observation and control modes

### Technical Improvements

#### 🏗️ Code Structure
- **Modular Functions**: Split functionality into focused functions (`save_config`, `save_model_architecture`)
- **Better Error Handling**: Improved error messages and fallback behavior
- **Type Safety**: Added proper type hints with `DictConfig` and `OmegaConf`

#### 📦 Dependencies
- Added `hydra-core==1.3.2` and `omegaconf==2.3.0` to requirements
- Removed dependency on `tyro` for argument parsing

### Usage Examples

#### Basic Usage
```bash
# Run with default configuration
python eval_pi0_maniskill.py

# Quick test run
python eval_pi0_maniskill.py --config-name=quick_test

# CPU-only evaluation
python eval_pi0_maniskill.py model.device=cpu sim.device=cpu
```

#### Advanced Configuration
```bash
# Multiple parameter overrides
python eval_pi0_maniskill.py eval.num_episodes=50 env.num_envs=5 output.save_videos=false

# Custom model path
python eval_pi0_maniskill.py model.path=/path/to/your/model

# Different environment and model
python eval_pi0_maniskill.py env=pick_and_place model=pi0_large
```

### Output Structure
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

### Migration Guide

#### From Version 1.x
- Replace command-line arguments with Hydra configuration overrides
- Use `cfg.env.id` instead of `args.env_id`
- Use `cfg.model.path` instead of `args.model_path`
- Use `cfg.eval.num_episodes` instead of `args.num_episodes`

#### Example Migration
```bash
# Old way (v1.x)
python eval_pi0_maniskill.py --env_id="PutOnPlateInScene25VisionTexture03-v1" --num_episodes=10

# New way (v2.0)
python eval_pi0_maniskill.py env.id="PutOnPlateInScene25VisionTexture03-v1" eval.num_episodes=10
```

### Breaking Changes
- **Removed tyro dependency**: The script no longer uses tyro for argument parsing
- **Changed configuration structure**: All parameters are now organized in a hierarchical structure
- **Updated output format**: Configuration and architecture information are now saved in separate files

### Files Added
- `config/default.yaml` - Default configuration
- `config/quick_test.yaml` - Quick test configuration
- `config/env/put_on_plate.yaml` - PutOnPlate environment config
- `config/env/pick_and_place.yaml` - PickAndPlace environment config
- `config/model/pi0_base.yaml` - pi0_base model config
- `config/model/pi0_large.yaml` - pi0_large model config
- `README.md` - Comprehensive usage documentation
- `example_usage.py` - Example usage demonstrations
- `test_config.py` - Configuration testing script
- `CHANGELOG.md` - This changelog

### Files Modified
- `eval_pi0_maniskill.py` - Complete refactor to use Hydra
- `requirements.txt` - Added hydra-core and omegaconf dependencies 