from typing import Dict, Any, List

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME


class PipelineCallback(ConfigMixin):
    """
    Base class for all the official callbacks used in a pipeline. This class provides a structure for implementing
    custom callbacks and ensures that all callbacks have a consistent interface.

    Please implement the following:
        `tensor_inputs`: This should return a list of tensor inputs specific to your callback. You will only be able to
        include
            variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
        `callback_fn`: This method defines the core functionality of your callback.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(self, cutoff_step_ratio=1.0, cutoff_step_index=None):
        super().__init__()

        if (cutoff_step_ratio is None and cutoff_step_index is None) or (
            cutoff_step_ratio is not None and cutoff_step_index is not None
        ):
            raise ValueError("Either cutoff_step_ratio or cutoff_step_index should be provided, not both or none.")

        if cutoff_step_ratio is not None and (
            not isinstance(cutoff_step_ratio, float) or not (0.0 <= cutoff_step_ratio <= 1.0)
        ):
            raise ValueError("cutoff_step_ratio must be a float between 0.0 and 1.0.")
        
    @property
    def tensor_inputs(self) -> List[str]:
        raise NotImplementedError(f"You need to set the attribute `tensor_inputs` for {self.__class__}")
    
    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs) ->Dict[str, Any]:
        raise NotImplementedError(f"You need to implement the method `callback_fn` for {self.__class__}")
    
    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        return self.callback_fn(pipeline, step_index, timestep, callback_kwargs)
    
    

class MultiPipelineCallbacks:
    def __init__(self, callbacks: List[PipelineCallback]):
        self.callbacks = callbacks
        
    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        for callback in self.callbacks:
            callback_kwargs = callback(pipeline, step_index, timestep, callback_kwargs)
        
        return callback_kwargs