from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from .state import Hare, Wolf, WorldState
from .rules import (
    hare_move,
    respawn_carrot,
    wolf_sees_hare,
    calm_wolf_move,
    hunting_wolf_move,
    chase_direction,
    snap_to_diagonal
)

Pos = Tuple[int,int]
Dir = Tuple[int,int]

@dataclass
class EnvConfig:
    W: int = 15
    H: int = 15

    n_carrots: int = 2
    n_wolves: int = 2
    n_hunting_wolves: int = 1

    vision: int = 4

    M: int = 10
    K: int = 20

    max_steps: int = 2000
    start_energy: Optional[int] = 200


class HareEnv:
    ACTIONS: List[Dir] = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    DIAGONALS: List[Dir] = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    def __init__(self, config: EnvConfig, seed: int = 0):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.state: Optional[WorldState] = None

    # ---------- Helpers ----------

    def _random_empty_cell(self, occupied: Set[Pos]) -> Pos:
        while True:
            x = int(self.rng.integers(0, self.cfg.W))
            y = int(self.rng.integers(0, self.cfg.H))
            if (x, y) not in occupied:
                return (x, y)

    def _occupied_cells(self, state: WorldState) -> Set[Pos]:
        occ = {state.hare.pos}
        for w in state.wolves:
            occ.add(w.pos)
        for c in state.carrots:
            occ.add(c)
        return occ

    def _make_obs(self, state: WorldState):
        """
        For now return a simple python structure.
        Later replace with obs.encode_discrete(...) or obs.encode_vector(...).
        """
        return {
            "hare_pos": state.hare.pos,
            "energy": state.hare.energy,
            "wolves": [(w.pos, w.is_hunting, w.direction) for w in state.wolves],
            "carrots": list(state.carrots),
            "step": state.step_count,
        }

    def _teleport_toward_center(self, pos: Pos, steps: int) -> Pos:
        """
        Teleport hare 4-5 steps toward center (as per spec).
        This is placed in env.py because it's 'event logic' not a primitive,
        but you can move it into rules.py if you prefer.
        """
        x, y = pos
        cx, cy = self.cfg.W // 2, self.cfg.H // 2

        for _ in range(steps):
            dx = 0 if x == cx else (1 if cx > x else -1)
            dy = 0 if y == cy else (1 if cy > y else -1)

            # If both differ, randomly choose whether to move diagonally or axis
            if dx != 0 and dy != 0:
                # 50% diagonal, 50% axis (random choice)
                if self.rng.random() < 0.5:
                    # diagonal
                    nx, ny = x + dx, y + dy
                else:
                    if self.rng.random() < 0.5:
                        nx, ny = x + dx, y
                    else:
                        nx, ny = x, y + dy
            else:
                nx, ny = x + dx, y + dy

            if 0 <= nx < self.cfg.W and 0 <= ny < self.cfg.H:
                x, y = nx, ny

        return (x, y)
    
    # ---------- Public API ----------

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> object:
        cfg = self.cfg
        start_energy = cfg.start_energy if cfg.start_energy is not None else cfg.W * cfg.H

        # Place hare
        hare_pos = (int(self.rng.integers(0, cfg.W)), int(self.rng.integers(0, cfg.H)))
        hare = Hare(pos=hare_pos, energy=start_energy)

        occupied = {hare.pos}

        # Place wolves
        wolves: List[Wolf] = []
        n_hunt = min(cfg.n_hunting_wolves, cfg.n_wolves)
        hunting_flags = [True] * n_hunt + [False] * (cfg.n_wolves - n_hunt)
        self.rng.shuffle(hunting_flags)

        for is_hunting in hunting_flags:
            wpos = self._random_empty_cell(occupied)
            occupied.add(wpos)

            # Give every wolf a direction (needed for calm movement + hunting "ahead" vision)
            wdir = self.DIAGONALS[int(self.rng.integers(0, len(self.DIAGONALS)))]
            wolves.append(Wolf(pos=wpos, is_hunting=is_hunting, direction=wdir))

        # Place carrots
        carrots: List[Pos] = []
        for _ in range(cfg.n_carrots):
            cpos = self._random_empty_cell(occupied)
            occupied.add(cpos)
            carrots.append(cpos)

        self.state = WorldState(
            hare=hare,
            wolves=wolves,
            carrots=carrots,
            step_count=0,
            rng=self.rng,
        )

        return self._make_obs(self.state)

    def step(self, action_id: int) -> Tuple[object, float, bool, Dict]:
        """
        action_id: int in [0..7]
        Returns: obs, reward, done, info
        """
        assert self.state is not None, "Call reset() before step()."
        cfg = self.cfg
        s = self.state

        reward = 0.0
        caught = False
        carrots_eaten = 0

        # 1) Hare move + step cost
        action = self.ACTIONS[action_id]
        s.hare.pos = hare_move(s.hare.pos, action, cfg.W, cfg.H)
        s.hare.energy -= 1
        reward -= 1

        # 2) Carrot pickup + respawn
        if s.hare.pos in s.carrots:
            carrots_eaten = 1
            s.hare.energy += cfg.M
            reward += cfg.M

            # remove eaten carrot and respawn
            s.carrots.remove(s.hare.pos)
            occ = self._occupied_cells(s)
            new_c = respawn_carrot(list(occ), cfg.W, cfg.H, s.rng)
            s.carrots.append(new_c)

        # 3) Wolves update
        for w in s.wolves:
            sees = wolf_sees_hare(w.pos, w.direction, s.hare.pos, w.is_hunting, vision=cfg.vision)

            if sees:
                # Hunting jump (does not update direction in this simple version)
                w.pos = hunting_wolf_move(w.pos, s.hare.pos, cfg.W, cfg.H)
                w.direction = chase_direction(w.pos, s.hare.pos)
            else:
                # Calm diagonal motion + reflection updates direction
                if w.direction not in self.DIAGONALS: w.direction = snap_to_diagonal(w.direction, s.rng)
                w.pos, w.direction = calm_wolf_move(w.pos, w.direction, cfg.W, cfg.H)

        # 4) Collision check (caught)
        for w in s.wolves:
            if w.pos == s.hare.pos:
                caught = True
                break

        if caught:
            s.hare.energy -= cfg.K
            reward -= cfg.K

            # teleport 4–5 steps toward center
            steps = 4 if self.rng.random() < 0.5 else 5
            s.hare.pos = self._teleport_toward_center(s.hare.pos, steps)

        # 5) Termination / truncation
        s.step_count += 1
        done = s.hare.energy <= 0
        truncated = s.step_count >= cfg.max_steps
        done = done or truncated

        info = {
            "energy": s.hare.energy,
            "caught": caught,
            "carrots_eaten": carrots_eaten,
            "step": s.step_count,
            "truncated": truncated,
        }

        return self._make_obs(s), reward, done, info