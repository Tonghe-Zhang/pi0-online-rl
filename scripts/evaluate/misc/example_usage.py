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
Example usage of the Hydra-based pi0 ManiSkill3 evaluation script.

This script demonstrates various ways to run the evaluation with different configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_evaluation(command):
    """Run the evaluation with the given command"""
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ERROR")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    
    print("=" * 50)
    print()

def main():
    """Demonstrate various evaluation configurations"""
    
    # Get the script directory
    script_dir = Path(__file__).parent
    eval_script = script_dir / "eval_pi0_maniskill.py"
    
    print("Pi0 ManiSkill3 Evaluation Examples")
    print("=" * 50)
    print()
    
    # Example 1: Basic usage with default configuration
    print("Example 1: Basic usage with default configuration")
    run_evaluation(f"python {eval_script}")
    
    # Example 2: Quick test run with fewer episodes
    print("Example 2: Quick test run with fewer episodes")
    run_evaluation(f"python {eval_script} eval.num_episodes=5 env.num_envs=2")
    
    # Example 3: CPU-only evaluation
    print("Example 3: CPU-only evaluation")
    run_evaluation(f"python {eval_script} model.device=cpu sim.device=cpu eval.num_episodes=3")
    
    # Example 4: Different environment
    print("Example 4: Different environment (PickAndPlace)")
    run_evaluation(f"python {eval_script} env=pick_and_place eval.num_episodes=3")
    
    # Example 5: Different model
    print("Example 5: Different model (pi0_large)")
    run_evaluation(f"python {eval_script} model=pi0_large eval.num_episodes=3")
    
    # Example 6: Custom output directory
    print("Example 6: Custom output directory")
    run_evaluation(f"python {eval_script} output.dir=example_results eval.num_episodes=3")
    
    # Example 7: Disable video saving
    print("Example 7: Disable video saving")
    run_evaluation(f"python {eval_script} output.save_videos=false eval.num_episodes=3")
    
    # Example 8: Multiple parameter overrides
    print("Example 8: Multiple parameter overrides")
    run_evaluation(f"python {eval_script} eval.num_episodes=3 env.num_envs=5 env.max_episode_len=50 output.save_videos=false")
    
    print("All examples completed!")
    print("\nNote: Some examples may fail if the required models or environments are not available.")
    print("This is expected behavior for demonstration purposes.")

if __name__ == "__main__":
    main() 