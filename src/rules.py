from typing import List, Tuple
import numpy as np

def sign(x: int) -> int:
    return (x > 0) - (x < 0)

def hare_move(pos:Tuple[int,int], action:Tuple[int,int], W:int, H:int) -> Tuple[int,int]:
    if (0 <= pos[0] + action[0] < W) and (0 <= pos[1] + action[1] < H): 
        return (pos[0]+action[0], pos[1] + action[1])
    else: return pos

def manhattan(a:Tuple[int, int], b:Tuple[int,int]) -> int:
     return abs(a[0] - b[0]) + abs(a[1] - b[1])

def respawn_carrot(occupied_cells:List[Tuple[int,int]], W:int, H:int, rng) -> Tuple[int,int]:
    """
    Returns a random empty cell not in occupied_cells.
    """
    occupied = set(occupied_cells)

    while True:
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, H))
        if (x, y) not in occupied:
            return (x, y)


def wolf_sees_hare(wolf_pos:Tuple[int,int], wolf_dir:Tuple[int,int], hare_pos:tuple[int,int], is_hunting:bool, vision=4) -> bool:
    if manhattan(wolf_pos, hare_pos) > vision: return False
    dx = hare_pos[0] - wolf_pos[0]
    dy = hare_pos[1] - wolf_pos[1]
    dot = dx*wolf_dir[0] + dy * wolf_dir[1]
    return dot > 0

def calm_wolf_move(pos:Tuple[int,int], direction:Tuple[int, int], W:int, H:int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    x, y = pos
    dx, dy = direction
    nx = x + dx
    ny = y + dy
    # reflect x if needed
    if not (0 <= nx < W):
        dx = -dx
        nx = x + dx
    # reflect y if needed
    if not (0 <= ny < H):
        dy = -dy
        ny = y + dy
    # final safety (degenerate grids)
    if 0 <= nx < W and 0 <= ny < H:
        return (nx, ny), (dx, dy)
    return pos, (dx, dy)

def hunting_wolf_move(pos:Tuple[int,int], hare_pos:Tuple[int,int], W:int, H:int) -> Tuple[int,int]:
    """
    Hunting wolf jumps 2 cells toward the hare.
    Uses sign to move in x and/or y direction.
    Blocks at boundary (stays if out of bounds).
    """
    wx, wy = pos
    hx, hy = hare_pos

    dx = sign(hx - wx)
    dy = sign(hy - wy)

    # If already on hare (should be handled elsewhere), don't move.
    if dx == 0 and dy == 0:
        return pos

    # Jump 2 cells in the chosen direction (can be diagonal)
    nx = wx + 2 * dx
    ny = wy + 2 * dy

    # Block at boundary (same convention as hare_move)
    if 0 <= nx < W and 0 <= ny < H:
        return (nx, ny)
    return pos


def chase_direction(wolf_pos: Tuple[int,int], hare_pos: Tuple[int,int]) -> Tuple[int,int]:
    dx = sign(hare_pos[0] - wolf_pos[0])
    dy = sign(hare_pos[1] - wolf_pos[1])

    # If already on hare, keep current direction (caller can handle)
    return (dx, dy)

def snap_to_diagonal(d: Tuple[int,int], rng) -> Tuple[int,int]:
    dx, dy = d
    if dx == 0: dx = 1 if rng.random() < 0.5 else -1
    if dy == 0: dy = 1 if rng.random() < 0.5 else -1
    return (dx, dy)