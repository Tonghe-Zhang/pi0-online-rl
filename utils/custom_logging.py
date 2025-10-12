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


import shutil
def create_bordered_text(text, border_char='#'):
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Split the text into lines
    lines = text.split('\n')

    # Determine left and right columns
    left_lines = []
    right_lines = []
    for line in lines:
        if ':' in line:
            left, right = line.split(':', 1)
            left_lines.append(left.strip())
            right_lines.append(right.strip())
        else:
            left_lines.append(line.strip())
            right_lines.append('')

    # Calculate column widths
    left_width = max(len(line) for line in left_lines) + 2  # +2 for ' :'
    right_width = max(len(line) for line in right_lines) + 2  # +2 for ': '

    # Format the lines
    formatted_lines = [
        f"# {left.ljust(left_width)} {right.rjust(right_width)} #"
        for left, right in zip(left_lines, right_lines)
    ]

    # Create top and bottom borders
    border = border_char * terminal_width

    # Construct the final message
    message = (
        f"\n{border}\n"
        + '\n'.join(formatted_lines)
        + f"\n{border}"
    )

    return message

def print_type_and_shape(var_name, var):
    var_type = type(var)
    try:
        var_shape = var.shape  # This assumes var has a 'shape' attribute
    except AttributeError:
        var_shape = "No shape attribute"

    print(f"{var_name}: Type = {var_type}, Shape = {var_shape}")
    

import os 
import wandb
from omegaconf import OmegaConf
def setup_wandb(cfg, logger):
    # Check if offline mode is enabled (e.g., via config or environment variable)
    offline_mode = cfg.wandb.offline_mode                       # Add this to your config if desired
    if offline_mode:
        wandb_dir = cfg.wandb.get("dir", "./wandb_offline")     # Local directory for offline logs
    else:
        wandb_dir = cfg.wandb.get("dir", "./wandb")
    os.makedirs(wandb_dir, exist_ok=True)  # Ensure the directory exists
    print(f"""Initializing wandb on the main process with:\nentity={cfg.wandb.entity}, project={cfg.wandb.project}, name={cfg.wandb.run}, mode={"offline" if offline_mode else "online"}, dir={wandb_dir}""")
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run ,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        mode="offline" if offline_mode else "online",  # Switch to offline mode
        dir=wandb_dir,  # Specify local directory for offline logs
    )
    # Get the exact subfolder for this run
    run_id = wandb.run.id   # type: ignore # Unique ID for the run
    run_dir = wandb.run.dir  # type: ignore # Full path to the run's directory
    if offline_mode:
        logger.info(f"Wandb running in offline mode. Logs will be saved to {os.path.dirname(run_dir)}")
        logger.info(f"Run ID: {run_id}")
    else:
        logger.info(f"Wandb running online. Run ID: {run_id}")



