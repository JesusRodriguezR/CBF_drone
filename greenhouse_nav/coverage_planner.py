"""
Topological Graph & Boustrophedon Coverage Planner
===================================================

Represents the greenhouse as a graph:
  - Nodes: lane entry/exit points
  - Edges: lanes (traversable corridors) + headland connections

The planner generates a boustrophedon (ox-plowing) path that visits
every lane exactly once.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from greenhouse_env import GreenhouseEnv


@dataclass
class Waypoint:
    """A waypoint in the coverage plan."""
    x: float
    y: float
    label: str           # e.g. 'L1_top', 'L3_bot', 'headland_turn'
    lane_idx: int = -1   # which lane this belongs to (-1 = headland)
    is_entry: bool = False
    is_exit: bool = False


class CoveragePlanner:
    """
    Boustrophedon coverage planner over the greenhouse lane graph.

    Generates an ordered list of waypoints that covers all lanes
    in a serpentine (back-and-forth) pattern without repeating.
    """

    def __init__(self, env: GreenhouseEnv):
        self.env = env
        self.waypoints: List[Waypoint] = []
        self.visited_lanes: set = set()
        self.current_wp_idx: int = 0

    def plan_coverage(self, start_from: str = 'top') -> List[Waypoint]:
        """
        Generate the full boustrophedon coverage path.

        Parameters
        ----------
        start_from : str
            'top' — start from top headland, first lane goes top→bottom
            'bot' — start from bottom headland, first lane goes bot→top

        Returns
        -------
        waypoints : list of Waypoint
        """
        env = self.env
        c = env.config
        self.waypoints = []
        self.visited_lanes = set()
        self.current_wp_idx = 0

        # Margin from lane walls
        margin_y = 0.3  # how deep into headland to go for turns

        for i in range(c.n_lanes):
            cx = env.lane_centers_x[i]

            # Determine direction: alternate each lane
            if start_from == 'top':
                going_down = (i % 2 == 0)
            else:
                going_down = (i % 2 == 1)

            if going_down:
                entry_y = env.lane_y_max + margin_y  # start in top headland
                exit_y = env.lane_y_min - margin_y   # exit into bottom headland
                entry_label = f'L{i+1}_top'
                exit_label = f'L{i+1}_bot'
            else:
                entry_y = env.lane_y_min - margin_y
                exit_y = env.lane_y_max + margin_y
                entry_label = f'L{i+1}_bot'
                exit_label = f'L{i+1}_top'

            # Entry waypoint
            self.waypoints.append(Waypoint(
                x=cx, y=entry_y, label=entry_label,
                lane_idx=i, is_entry=True
            ))

            # Exit waypoint
            self.waypoints.append(Waypoint(
                x=cx, y=exit_y, label=exit_label,
                lane_idx=i, is_exit=True
            ))

            # Headland transition to next lane (if not last)
            if i < c.n_lanes - 1:
                next_cx = env.lane_centers_x[i + 1]
                self.waypoints.append(Waypoint(
                    x=next_cx, y=exit_y,
                    label=f'headland_L{i+1}_to_L{i+2}',
                    lane_idx=-1
                ))

        return self.waypoints

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Return the current target waypoint."""
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return None

    def advance_waypoint(self) -> Optional[Waypoint]:
        """Mark current waypoint as reached, advance to next."""
        wp = self.get_current_waypoint()
        if wp and wp.is_exit and wp.lane_idx >= 0:
            self.visited_lanes.add(wp.lane_idx)

        self.current_wp_idx += 1
        return self.get_current_waypoint()

    def is_complete(self) -> bool:
        """Check if all waypoints have been visited."""
        return self.current_wp_idx >= len(self.waypoints)

    def get_progress(self) -> Tuple[int, int]:
        """Return (lanes_visited, total_lanes)."""
        return len(self.visited_lanes), self.env.config.n_lanes

    def get_waypoint_positions(self) -> np.ndarray:
        """Return all waypoint positions as (N, 2) array."""
        return np.array([[wp.x, wp.y] for wp in self.waypoints])
