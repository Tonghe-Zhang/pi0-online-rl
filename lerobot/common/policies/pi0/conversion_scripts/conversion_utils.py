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


# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import GemmaConfig, PaliGemmaConfig


def get_paligemma_config(precision: str):
    config = {
        "image_token_index": None,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }

    # image_sizes = {"2b-test": 224, "3b-224px": 224, "3b-448px": 448, "3b-896px": 896}

    image_size = 224  # image_sizes[variant]
    patch_size = 14
    num_image_tokens = (image_size**2) // (patch_size**2)

    config["image_token_index"] = 257152
    text_config = {
        "vocab_size": 257152,
        "num_hidden_layers": 18,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "torch_dtype": precision,
        "hidden_size": 2048,
        "hidden_activation": "gelu_pytorch_tanh",
        "num_attention_heads": 8,
        "intermediate_size": 16384,
        "is_encoder_decoder": False,
    }
    vision_config = {
        "torch_dtype": precision,
        "image_size": image_size,
        "patch_size": patch_size,
        "num_image_tokens": num_image_tokens,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
    }
    final_config = PaliGemmaConfig(text_config=text_config, vision_config=vision_config, **config)
    return final_config


def get_gemma_config(precision: str):
    config = {
        "image_token_index": None,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }

    config["image_token_index"] = 257152
    text_config = {
        "vocab_size": 257152,
        "num_hidden_layers": 18,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "torch_dtype": precision,
        "hidden_size": 1024,
        "hidden_activation": "gelu_pytorch_tanh",
        "num_attention_heads": 8,
        "intermediate_size": 4096,
        "is_encoder_decoder": False,
    }
    final_config = GemmaConfig()
    final_config.update(text_config)
    return final_config
