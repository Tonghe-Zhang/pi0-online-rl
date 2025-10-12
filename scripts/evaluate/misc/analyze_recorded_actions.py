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
Helper script to analyze recorded actions from pi0 ManiSkill3 evaluation.

Usage:
    python analyze_recorded_actions.py <path_to_batch_episode_data.json>

Example:
    python analyze_recorded_actions.py results/eval_pi0_maniskill/2024_01_01_12_00_00/data/batch_episode_data.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_episode_data(data_path):
    """Load episode data from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def analyze_actions(data):
    """Analyze recorded actions"""
    print("="*60)
    print("ACTION ANALYSIS REPORT")
    print("="*60)
    
    # Basic info
    print(f"Number of environments: {data['num_envs']}")
    print(f"Success rate: {data['success_rate']:.2%}")
    print(f"Total steps recorded: {data['total_steps']}")
    
    # Success breakdown
    if 'individual_success' in data:
        successful_envs = [i for i, success in enumerate(data['individual_success']) if success]
        print(f"Successful environments: {successful_envs} ({len(successful_envs)}/{data['num_envs']})")
    
    # Action statistics
    if 'action_statistics' in data:
        stats = data['action_statistics']
        if 'error' in stats:
            print(f"Error in action statistics: {stats['error']}")
            return
        
        print(f"\nAction Details:")
        print(f"  Total actions recorded: {stats['total_actions']}")
        print(f"  Action dimension: {stats['action_dim']}")
        print(f"  Action array shape: {stats['shape']}")
        
        print(f"\nOverall Action Statistics:")
        print(f"  Mean: {np.array(stats['mean'])}")
        print(f"  Std:  {np.array(stats['std'])}")
        print(f"  Min:  {np.array(stats['min'])}")
        print(f"  Max:  {np.array(stats['max'])}")
        
        # Per-dimension statistics
        if 'per_dimension' in stats:
            print(f"\nPer-Dimension Statistics:")
            for dim_name, dim_stats in stats['per_dimension'].items():
                print(f"  {dim_name}: mean={dim_stats['mean']:.4f}, std={dim_stats['std']:.4f}, "
                      f"range=[{dim_stats['min']:.4f}, {dim_stats['max']:.4f}]")
    
    # Per-environment action analysis
    all_actions = data['all_actions']
    print(f"\nPer-Environment Action Summary:")
    for env_idx, env_actions in enumerate(all_actions):
        if env_actions:
            env_actions_array = np.array(env_actions)
            success = data['individual_success'][env_idx]
            print(f"  Env {env_idx}: {len(env_actions)} actions, "
                  f"success={success}, "
                  f"action_range=[{env_actions_array.min():.3f}, {env_actions_array.max():.3f}]")
        else:
            print(f"  Env {env_idx}: No actions recorded")


def plot_actions(data, output_dir=None):
    """Create plots for action analysis"""
    try:
        all_actions = data['all_actions']
        
        # Flatten all actions
        flattened_actions = []
        env_labels = []
        step_labels = []
        
        for env_idx, env_actions in enumerate(all_actions):
            for step_idx, action in enumerate(env_actions):
                if action is not None:
                    flattened_actions.append(action)
                    env_labels.append(env_idx)
                    step_labels.append(step_idx)
        
        if not flattened_actions:
            print("No actions to plot")
            return
        
        actions_array = np.array(flattened_actions)
        
        # Plot 1: Action trajectories over time
        plt.figure(figsize=(15, 10))
        
        # Plot actions for each environment separately
        num_envs = data['num_envs']
        colors = plt.cm.tab10(np.linspace(0, 1, num_envs))
        
        for env_idx in range(num_envs):
            env_actions = np.array(all_actions[env_idx]) if all_actions[env_idx] else np.array([])
            if len(env_actions) > 0:
                success = data['individual_success'][env_idx]
                label = f"Env {env_idx} ({'Success' if success else 'Fail'})"
                
                for dim in range(min(actions_array.shape[1], 4)):  # Plot first 4 dimensions
                    plt.subplot(2, 2, dim + 1)
                    plt.plot(env_actions[:, dim], color=colors[env_idx], alpha=0.7, label=label if dim == 0 else "")
                    plt.title(f"Action Dimension {dim}")
                    plt.xlabel("Step")
                    plt.ylabel("Action Value")
                    if dim == 0:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(Path(output_dir) / "action_trajectories.png", dpi=150, bbox_inches='tight')
            print(f"Action trajectories plot saved to {output_dir}/action_trajectories.png")
        else:
            plt.show()
        
        # Plot 2: Action distribution histograms
        plt.figure(figsize=(12, 8))
        
        for dim in range(min(actions_array.shape[1], 6)):  # Plot first 6 dimensions
            plt.subplot(2, 3, dim + 1)
            plt.hist(actions_array[:, dim], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f"Action Dim {dim} Distribution")
            plt.xlabel("Action Value")
            plt.ylabel("Frequency")
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(Path(output_dir) / "action_distributions.png", dpi=150, bbox_inches='tight')
            print(f"Action distributions plot saved to {output_dir}/action_distributions.png")
        else:
            plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze recorded actions from pi0 evaluation")
    parser.add_argument("data_path", help="Path to batch_episode_data.json file")
    parser.add_argument("--plot", action="store_true", help="Create action plots")
    parser.add_argument("--output-dir", help="Directory to save plots (if not specified, plots will be displayed)")
    
    args = parser.parse_args()
    
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found: {args.data_path}")
        return
    
    # Load and analyze data
    data = load_episode_data(args.data_path)
    analyze_actions(data)
    
    # Create plots if requested
    if args.plot:
        plot_actions(data, args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main() 