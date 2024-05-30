# from typing import TYPE_CHECKING

# from .utils import (_LazyModule)

# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusersm` provides the names in the namespace without actually importing anything (and especially none of the backends).

# _import_structure = {
#     "configuration_utils": ["ConfigMixin"],
#     "models": [],
#     "pipelines": [],
#     "schedulers": [],
#     "utils": [
#         "logging",
#     ],
# }

# _import_structure["models"].extend(
#         [
#             "AutoencoderKL",
#             "ConsistencyDecoderVAE",
#             "ControlNetModel",
#             "ModelMixin",
#             "MotionAdapter",
#             "MultiAdapter",
#             "T2IAdapter",
#             "Transformer2DModel",
#             "UNet1DModel",
#             "UNet2DConditionModel",
#             "UNet2DModel",
#             "UNetMotionModel",
#             "UVit2DModel",
#             "VQModel",
#         ]
#     )

# _import_structure["optimization"] = [
#     "get_constant_schedule",
#     "get_constant_schedule_with_warmup",
#     "get_cosine_schedule_with_warmup",
#     "get_cosine_with_hard_restarts_schedule_with_warmup",
#     "get_linear_schedule_with_warmup",
#     "get_polynomial_decay_schedule_with_warmup",
#     "get_scheduler",
# ]
# _import_structure["pipelines"].extend(
#     [
#         "AudioPipelineOutput",
#     ]
# )
# _import_structure["schedulers"].extend(
#     [
#         "AmusedScheduler",
#     ]
# )
# _import_structure["training_utils"] = ["EMAModel"]

# _import_structure["pipelines"].extend(
#         [
#         "AltDiffusionImg2ImgPipeline",
#         ]
#     )


# if TYPE_CHECKING:
#     from .configuration_utils import ConfigMixin
#     from .models import (
#             AsymmetricAutoencoderKL,
#         )
#     from .optimization import (
#         get_constant_schedule,
#         get_constant_schedule_with_warmup,
#         get_cosine_schedule_with_warmup,
#         get_cosine_with_hard_restarts_schedule_with_warmup,
#         get_linear_schedule_with_warmup,
#         get_polynomial_decay_schedule_with_warmup,
#         get_scheduler,
#     )
#     from .pipelines import (
#         AudioPipelineOutput,
#     )
#     from .schedulers import (
#         DDPMScheduler,
#     )
#     from .training_utils import EMAModel

#     from .pipelines import (
#             AltDiffusionImg2ImgPipeline,
#             AltDiffusionPipeline,
#             AmusedImg2ImgPipeline,
#             AudioLDMPipeline,
            
#         )
# else:
#     import sys

#     sys.modules[__name__] = _LazyModule(
#         __name__,
#         globals()["__file__"],
#         _import_structure,
#         module_spec=__spec__,
#         extra_objects={"__version__": __version__},
#     )


from .configuration_utils import ConfigMixin
from .models import (
        AsymmetricAutoencoderKL,
        UNet2DConditionModel
    )
from .optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_scheduler,
)
from .pipelines import (
    AudioPipelineOutput,
    StableDiffusionXLPipeline
)
from .schedulers import (
    DDPMScheduler,
    LCMScheduler,
)
from .training_utils import EMAModel

from .pipelines import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        AmusedImg2ImgPipeline,
        AudioLDMPipeline,
        
    )
from .loaders import (
    StableDiffusionXLLoraLoaderMixin,
)
from .callbacks import PipelineCallback, MultiPipelineCallbacks