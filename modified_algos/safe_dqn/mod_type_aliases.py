"""Common aliases for type hints"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th

class ReplayBufferSamplesCost(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    costs: th.Tensor