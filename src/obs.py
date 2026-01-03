from typing import Tuple, List
import math

DIRS_8 = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

def dir8_from_delta(dx: int, dy: int) -> int:
    if dx == 0 and dy == 0:
        return -1
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    return DIRS_8.index((sx, sy))

def bucket_dist(d: int) ->int:
    return d if d <= 5 else 5

def nearest(pos: Tuple[int,int], targets: List[Tuple[int,int]]):
    # returns (target_pos, dx, dy, manhattan_dist)
    best = None
    bx = by = bd = None
    for t in targets:
        dx = t[0] - pos[0]
        dy = t[1] - pos[1]
        d = abs(dx) + abs(dy)
        if best is None or d < bd:
            best = t
            bx, by, bd = dx, dy, d
    return best, bx, by, bd

def encode_discrete(world_state, W: int, H: int):
    hare = world_state.hare
    hx, hy = hare.pos

    # energy bucket (simple)
    E = hare.energy
    Emax = W * H
    e_bucket = min(9, int(10 * E / max(1, Emax)))  # 0..9

    # nearest carrot
    _, cdx, cdy, cd = nearest((hx, hy), world_state.carrots)
    c_dir = dir8_from_delta(cdx, cdy) if cd is not None else -1
    c_dist = bucket_dist(cd) if cd is not None else 6  # 6 means "none"

    # nearest wolf
    wolf_positions = [w.pos for w in world_state.wolves]
    _, wdx, wdy, wd = nearest((hx, hy), wolf_positions)
    w_dir = dir8_from_delta(wdx, wdy) if wd is not None else -1
    w_dist = bucket_dist(wd) if wd is not None else 6

    # optional: whether ANY wolf is within vision
    vision = 4
    wolf_in_vision = int(wd is not None and wd <= vision)

    return (e_bucket, c_dir, c_dist, w_dir, w_dist, wolf_in_vision)
