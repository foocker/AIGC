# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
PEFT utilities: Utilities related to peft library
"""

def set_weights_and_activate_adapters(model, adapter_names, weights):
    pass


def set_adapter_layers(model, enabled=True):
    pass

def delete_adapter_layers(model, adapter_name):
    pass

def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True):
    pass

def recurse_remove_peft_layers(model):
    pass

def get_adapter_name(model):
    pass

def scale_lora_layers(model, weight):
    pass