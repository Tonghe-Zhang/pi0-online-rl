# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#!/usr/bin/env python3
"""
Test script to verify Hydra configuration loading for pi0 ManiSkill3 evaluation.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="default")
def test_config(cfg: DictConfig):
    """Test the configuration loading"""
    
    print("Testing Hydra Configuration Loading")
    print("=" * 50)
    
    # Print the configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Test specific configuration sections
    print("\nEnvironment Configuration:")
    print(f"  ID: {cfg.env.id}")
    print(f"  Num envs: {cfg.env.num_envs}")
    print(f"  Max episode len: {cfg.env.max_episode_len}")
    
    print("\nModel Configuration:")
    print(f"  Path: {cfg.model.path}")
    print(f"  Device: {cfg.model.device}")
    
    print("\nEvaluation Configuration:")
    print(f"  Num episodes: {cfg.eval.num_episodes}")
    print(f"  Seed: {cfg.eval.seed}")
    
    print("\nOutput Configuration:")
    print(f"  Directory: {cfg.output.dir}")
    print(f"  Save videos: {cfg.output.save_videos}")
    print(f"  Save data: {cfg.output.save_data}")
    
    print("\nConfiguration test completed successfully!")

if __name__ == "__main__":
    test_config() 