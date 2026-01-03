from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


Pos = Tuple[int, int]
Dir = Tuple[int, int]

@dataclass
class Hare:
    pos: Pos
    energy: int

@dataclass
class Wolf:
    pos: Pos
    is_hunting: bool
    direction: Dir

@dataclass
class WorldState:
    hare: Hare
    wolves: List[Wolf]
    carrots: List[tuple[int,int]]
    step_count: int
    rng: np.random.Generator