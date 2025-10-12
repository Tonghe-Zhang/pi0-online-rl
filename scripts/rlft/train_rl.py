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



import os
# When you use a HuggingFace tokenizer before multiprocessing, you may meet this warning:# `The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...`
# This is because HuggingFace tokenizers use internal parallelism, # and when you run multi-GPU training with torchrun, it forks processes after the tokenizer has already been loaded, which is unncessary and may cause deadlocks. 
# Consequently, we disable this parallelism by setting the environment variable TOKENIZERS_PARALLELISM to false when training on multiple GPUs. see https://github.com/huggingface/transformers/issues/5486 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
# MEMORY OPTIMIZATION: Enable PyTorch memory optimizations. This one significantly saved GPU DRAM. Previously if your model is 13GB then it needs 13GB for each gpu to load it, after this, it only needs ~7 GB. 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging 
logger_custom = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import hydra
from omegaconf import DictConfig
# Import Customized helper files
from utils.custom_memory_manager import cleanup_cuda_memory, signal_handler
from utils.custom_dirs import PI_R_ROOT_DIR
from utils.clear_pycache import clean_pycache
# Register cleanup functions
import signal
import atexit
atexit.register(cleanup_cuda_memory)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
# Add the parent directory to path for imports
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
# Import rl algorithm runner base class
from runner.base_runner import BaseRunner

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    """Main wrapper function"""
    logger_custom.info("Starting RLFT main function")
    logger_custom.info(f"Config keys: {list(cfg.keys())}")
    logger_custom.info(f"cfg._target_ = {cfg.get('_target_', 'NOT FOUND')}")
    # Cleanup remaining garbarge. 
    clean_pycache(PI_R_ROOT_DIR)
    cleanup_cuda_memory()
    logger_custom.info("About to try instantiation")
    try:
        # Instantiate the runner from cfg. 
        logger_custom.info(f"\nInstantiate RLFT runner.")
        logger_custom.info(f"Attempting to get class: {cfg._target_}")
        cls = hydra.utils.get_class(cfg._target_)
        logger_custom.info(f"Got class: {cls}")
        logger_custom.info("About to instantiate runner")
        runner: BaseRunner = cls(cfg)
        logger_custom.info("Runner instantiated successfully. ")
        # Start trainining.
        logger_custom.info(f"\nStart RLFT training.")
        runner.run()
        # Finish training, report output dirs. 
        logger_custom.info(f"\nFinished RLFT training. All training output saved to: {runner.output_dir}\nVideos or images saved to: {runner.render_dir}\nCheckpoints saved to: {runner.checkpoint_dir}\nLogs saved to: {runner.log_dir}\n")
    except KeyboardInterrupt:
        logger_custom.info("\nProcess interrupted by user. Cleaning up memory")
        cleanup_cuda_memory()
        return
    except Exception as e:
        logger_custom.info(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        cleanup_cuda_memory()
        raise
    finally:
        if 'runner' in locals() and hasattr(runner, 'env'):
            runner.env.close() # type: ignore
        cleanup_cuda_memory()

if __name__=="__main__":
    main()