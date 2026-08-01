"""Deterministic MuJoCo rough-field scene family for Sim2Sim evaluation."""

from __future__ import annotations

import math
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2_SCENE_DIR = REPO_ROOT / "reference_repos" / "unitree_mujoco" / "unitree_robots" / "go2"

VARIANT_SPECS: dict[str, int] = {
    "field_a": 11,
    "field_b": 23,
    "field_c": 37,
    "field_d": 51,
}


def _quat_z(deg: float) -> str:
    rad = math.radians(deg) / 2.0
    return f"{math.cos(rad):.7f} 0 0 {math.sin(rad):.7f}"


def _render_scene(name: str, seed: int) -> str:
    rng = random.Random(seed)
    patch_lines: list[str] = []
    bridge_lines: list[str] = []
    island_lines: list[str] = []

    x_positions = [3.00, 5.80, 8.65]
    y_rows = [-1.00, 0.95]

    for row_idx, y in enumerate(y_rows):
        for col_idx, x in enumerate(x_positions):
            hfield_name = "image_hfield" if (row_idx + col_idx + seed) % 2 == 0 else "perlin_hfield"
            x_jitter = rng.uniform(-0.18, 0.18)
            y_jitter = rng.uniform(-0.14, 0.14)
            yaw_deg = rng.uniform(-10.0, 10.0)
            patch_lines.append(
                f'    <geom type="hfield" hfield="{hfield_name}" '
                f'pos="{x + x_jitter:.2f} {y + y_jitter:.2f} 0.0" quat="{_quat_z(yaw_deg)}" />'
            )

    for x in [4.35, 7.15, 9.95]:
        bridge_height = rng.uniform(0.016, 0.026)
        bridge_y = rng.uniform(-0.16, 0.16)
        bridge_lines.append(
            f'    <geom pos="{x:.2f} {bridge_y:.2f} {bridge_height:.3f}" '
            f'type="box" size="0.26 1.60 {bridge_height:.3f}" />'
        )

    for idx in range(7):
        x = rng.uniform(3.2, 11.8)
        y = rng.uniform(-1.35, 1.35)
        z = rng.uniform(0.028, 0.060)
        sx = rng.uniform(0.14, 0.32)
        sy = rng.uniform(0.16, 0.34)
        yaw_deg = rng.uniform(-16.0, 16.0)
        island_lines.append(
            f'    <geom pos="{x:.2f} {y:.2f} {z:.3f}" type="box" '
            f'size="{sx:.2f} {sy:.2f} {z:.3f}" quat="{_quat_z(yaw_deg)}" />'
        )

    xml = f"""<mujoco model="go2 rough field {name}">
  <include file="go2.xml" />

  <statistic center="5.9 0 0.20" extent="3.8" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-130" elevation="-20" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture
      type="2d"
      name="groundplane"
      builtin="checker"
      mark="edge"
      rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8"
      width="300"
      height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="10 10" reflectance="0.2" />
    <hfield name="perlin_hfield" size="1.42 1.18 0.08 0.2" file="../height_field.png" />
    <hfield name="image_hfield" size="1.42 1.18 0.02 0.1" file="../unitree_hfield.png" />
  </asset>

  <worldbody>
    <light pos="0 0 2.1" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />

    <geom pos="0.95 0.00 0.010" type="box" size="0.70 1.80 0.010" />
    <geom pos="1.90 0.00 0.024" type="box" size="0.38 1.80 0.014" />

{chr(10).join(patch_lines)}
{chr(10).join(bridge_lines)}
{chr(10).join(island_lines)}

    <geom pos="12.20 0.00 0.018" type="box" size="0.62 1.65 0.018" />
    <geom pos="12.95 -0.48 0.036" type="box" size="0.24 0.34 0.036" quat="{_quat_z(6.0)}" />
    <geom pos="13.20 0.62 0.043" type="box" size="0.28 0.30 0.043" quat="{_quat_z(-8.0)}" />
  </worldbody>
</mujoco>
"""
    return xml


def ensure_generated_rough_scene_family() -> dict[str, Path]:
    GO2_SCENE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, seed in VARIANT_SPECS.items():
        path = GO2_SCENE_DIR / f"scene_eval_rough_{name}.xml"
        path.write_text(_render_scene(name, seed))
        outputs[name] = path
    return outputs

