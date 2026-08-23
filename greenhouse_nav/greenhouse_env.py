"""
Greenhouse Environment — 2D Top-Down Model
============================================

Models a closed greenhouse with N parallel crop lanes and two headland
corridors (top and bottom). All units in meters.

Layout (top-down view, +x right, +y up):

    ┌───────────────────────────────────────┐
    │            HEADLAND (top)             │  y = lane_length + headland
    ├────┬────┬────┬────┬────┬────┬────┤
    │ L1 │ L2 │ L3 │ ...│    │    │ LN │
    │    │    │    │    │    │    │    │
    ├────┴────┴────┴────┴────┴────┴────┤
    │            HEADLAND (bot)            │  y = 0
    └───────────────────────────────────────┘
    x=0                                   x = total_width
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class GreenhouseConfig:
    """Configuration for the greenhouse geometry."""
    n_lanes: int = 6                # Number of crop lanes
    lane_width: float = 1.2         # Width of each lane [m]
    lane_length: float = 8.0        # Length of each lane [m]
    headland_depth: float = 1.5     # Depth of headland corridors [m]
    wall_thickness: float = 0.05    # Thickness of crop row walls [m]
    crop_row_width: float = 0.3     # Width of crop rows between lanes [m]


class GreenhouseEnv:
    """2D greenhouse environment with lane geometry and wall segments."""

    def __init__(self, config: GreenhouseConfig = None):
        self.config = config or GreenhouseConfig()
        c = self.config

        # Compute geometry
        # Each lane occupies: crop_row_width + lane_width
        # Plus one extra crop_row on the far side
        self.total_width = (
            c.crop_row_width  # left wall
            + c.n_lanes * c.lane_width
            + (c.n_lanes - 1) * c.crop_row_width  # internal crop rows
            + c.crop_row_width  # right wall
        )
        self.total_height = c.lane_length + 2 * c.headland_depth

        # Lane center x-coordinates
        self.lane_centers_x = []
        for i in range(c.n_lanes):
            x = (c.crop_row_width + c.lane_width / 2
                 + i * (c.lane_width + c.crop_row_width))
            self.lane_centers_x.append(x)

        # Lane y-bounds (inside the lane, not headland)
        self.lane_y_min = c.headland_depth
        self.lane_y_max = c.headland_depth + c.lane_length

        # Headland y-bounds
        self.headland_bot_y = (0, c.headland_depth)
        self.headland_top_y = (c.headland_depth + c.lane_length, self.total_height)

        # Build wall segments as line segments [(x1,y1), (x2,y2)]
        self.wall_segments = self._build_walls()

        # Build crop row rectangles for visualization
        self.crop_rows = self._build_crop_rows()

    def _build_walls(self) -> List[Tuple]:
        """Build wall line segments for collision / distance queries."""
        c = self.config
        walls = []

        # Outer walls
        walls.append(((0, 0), (self.total_width, 0)))                          # bottom
        walls.append(((0, self.total_height), (self.total_width, self.total_height)))  # top
        walls.append(((0, 0), (0, self.total_height)))                          # left
        walls.append(((self.total_width, 0), (self.total_width, self.total_height)))  # right

        # Internal crop row walls (vertical segments spanning the lane region)
        for i in range(c.n_lanes + 1):
            if i == 0:
                x_left = 0
                x_right = c.crop_row_width
            elif i == c.n_lanes:
                x_left = (c.crop_row_width + c.n_lanes * c.lane_width
                          + (c.n_lanes - 1) * c.crop_row_width)
                x_right = self.total_width
            else:
                x_left = (c.crop_row_width + i * c.lane_width
                          + (i - 1) * c.crop_row_width)
                x_right = x_left + c.crop_row_width

            # Left edge of crop row
            walls.append(((x_left, self.lane_y_min), (x_left, self.lane_y_max)))
            # Right edge of crop row
            walls.append(((x_right, self.lane_y_min), (x_right, self.lane_y_max)))

        return walls

    def _build_crop_rows(self) -> List[Tuple]:
        """Build crop row rectangles (x, y, w, h) for visualization."""
        c = self.config
        rows = []

        for i in range(c.n_lanes + 1):
            if i == 0:
                x = 0
            else:
                x = (c.crop_row_width + i * c.lane_width
                     + (i - 1) * c.crop_row_width)

            rows.append((x, self.lane_y_min, c.crop_row_width,
                         c.lane_length))
        return rows

    def get_lane_index(self, x: float, y: float) -> Optional[int]:
        """
        Return the lane index (0-based) if the position is inside a lane.
        Returns None if in headland or outside.
        """
        if y < self.lane_y_min or y > self.lane_y_max:
            return None  # in headland or outside

        for i, cx in enumerate(self.lane_centers_x):
            half_w = self.config.lane_width / 2
            if cx - half_w <= x <= cx + half_w:
                return i
        return None  # inside a crop row or outside

    def get_region(self, x: float, y: float) -> str:
        """Return the region name: 'lane_N', 'headland_top', 'headland_bot', 'crop_row', 'outside'."""
        if x < 0 or x > self.total_width or y < 0 or y > self.total_height:
            return 'outside'

        lane = self.get_lane_index(x, y)
        if lane is not None:
            return f'lane_{lane}'

        if y < self.lane_y_min:
            return 'headland_bot'
        if y > self.lane_y_max:
            return 'headland_top'

        return 'crop_row'

    def distance_to_nearest_wall(self, x: float, y: float) -> float:
        """Compute minimum distance from point to any wall segment."""
        min_dist = float('inf')
        for (x1, y1), (x2, y2) in self.wall_segments:
            d = self._point_to_segment_dist(x, y, x1, y1, x2, y2)
            min_dist = min(min_dist, d)
        return min_dist

    def distances_to_lane_walls(self, x: float, y: float,
                                lane_idx: int) -> Tuple[float, float, float, float]:
        """
        Compute distances to the four walls of a specific lane.

        Returns (d_left, d_right, d_front_top, d_front_bot).
        """
        cx = self.lane_centers_x[lane_idx]
        half_w = self.config.lane_width / 2

        d_left = x - (cx - half_w)
        d_right = (cx + half_w) - x
        d_top = self.lane_y_max - y
        d_bot = y - self.lane_y_min

        return d_left, d_right, d_top, d_bot

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
        """Minimum distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return np.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.hypot(px - proj_x, py - proj_y)

    def render(self, ax: plt.Axes = None, show_labels: bool = True) -> plt.Axes:
        """Draw the greenhouse layout on a matplotlib axes."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Background
        ax.set_facecolor('#f5f0e8')

        # Outer boundary
        outer = patches.Rectangle((0, 0), self.total_width, self.total_height,
                                  linewidth=2, edgecolor='#333', facecolor='none')
        ax.add_patch(outer)

        # Headlands
        for (y_lo, y_hi) in [self.headland_bot_y, self.headland_top_y]:
            h = patches.Rectangle((0, y_lo), self.total_width, y_hi - y_lo,
                                  facecolor='#e8dcc8', edgecolor='none', alpha=0.6)
            ax.add_patch(h)

        # Crop rows (green)
        for (rx, ry, rw, rh) in self.crop_rows:
            row_patch = patches.Rectangle(
                (rx, ry), rw, rh,
                facecolor='#2d6a2d', edgecolor='#1a4a1a',
                linewidth=0.5, alpha=0.85
            )
            ax.add_patch(row_patch)

        # Lane labels
        if show_labels:
            for i, cx in enumerate(self.lane_centers_x):
                cy = self.lane_y_min + self.config.lane_length / 2
                ax.text(cx, cy, f'L{i+1}', ha='center', va='center',
                        fontsize=8, color='#888', alpha=0.5, fontweight='bold')

            ax.text(self.total_width / 2, self.headland_bot_y[1] / 2,
                    'Headland', ha='center', va='center',
                    fontsize=7, color='#999', style='italic')
            mid_top = (self.headland_top_y[0] + self.headland_top_y[1]) / 2
            ax.text(self.total_width / 2, mid_top,
                    'Headland', ha='center', va='center',
                    fontsize=7, color='#999', style='italic')

        ax.set_xlim(-0.3, self.total_width + 0.3)
        ax.set_ylim(-0.3, self.total_height + 0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        return ax
