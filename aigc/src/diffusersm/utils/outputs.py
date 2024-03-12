from collections import OrderedDict

class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    Python dictionary.

    <Tip warning={true}>

    You can't unpack a [`BaseOutput`] directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    first.

    </Tip>
    """
    def __init_subclass__(cls) -> None:
        """ 
        """