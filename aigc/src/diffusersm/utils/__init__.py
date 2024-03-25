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


from .outputs import BaseOutput
from .constants import (CONFIG_NAME,)
from .logging import get_logger

from .constants import WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME, SAFETENSORS_FILE_EXTENSION

from .import_utils import (_LazyModule,)

from .state_dict_utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,)

from .peft_utils import (
    set_weights_and_activate_adapters,
    set_adapter_layers,
    delete_adapter_layers)