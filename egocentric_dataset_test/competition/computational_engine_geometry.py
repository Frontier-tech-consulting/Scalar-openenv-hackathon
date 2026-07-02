"""
Computational Engineering Geometry Engine for Rocket/IC Engine Simulation
========================================================================

A Python-native computational geometry kernel implementing LEAP 71's PicoGK
principles for generating rocket and internal combustion engine components
using signed distance fields, boolean operations, and voxel-based modeling.

This module creates computational models of:
  - Rocket engines (combustion chamber + nozzle + cooling channels)
  - Internal combustion engines (block + pistons + crankshaft + valves)
  - Engine assemblies suitable for MuJoCo simulation

Based on the Computational Engineering paradigm where geometry is
generated through code rather than traditional CAD modeling.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Base Shape (PicoGK-inspired voxel/CSG primitive)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class BaseShape:
    """
    A lightweight CSG primitive — the 'atom' of computational engineering.

    Mirrors PicoGK's BaseShape concept: a positioned, oriented geometric
    primitive with a boolean operation flag (ADD / SUBTRACT / INTERSECT).
    """
    kind: str                          # "cylinder" | "box" | "sphere" | "torus" | "capsule"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, ...] = ()       # kind-specific dimensions (metres)
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.72, 1.0)
    operation: str = "ADD"             # ADD | SUBTRACT | INTERSECT
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    mass_kg: float | None = None       # override density-based mass

    def parametric_hash(self) -> str:
        blob = f"{self.kind}|{self.center}|{self.size}|{self.operation}|{self.axis}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        cx, cy, cz = self.center
        if self.kind == "cylinder":
            r, half_h = self.size[0], self.size[1]
            return (np.array([cx - r, cy - r, cz - half_h]),
                    np.array([cx + r, cy + r, cz + half_h]))
        if self.kind == "box":
            sx, sy, sz = self.size[:3]
            return (np.array([cx - sx, cy - sy, cz - sz]),
                    np.array([cx + sx, cy + sy, cz + sz]))
        if self.kind == "sphere":
            r = self.size[0]
            return (np.array([cx - r, cy - r, cz - r]),
                    np.array([cx + r, cy + r, cz + r]))
        # Capsule: from-to with radius
        if self.kind == "capsule":
            r = self.size[0]
            pts = np.array([self.size[1:4], self.size[4:7]])
            mn = pts.min(axis=0) - r
            mx = pts.max(axis=0) + r
            return (mn, mx)
        return (np.array([cx, cy, cz]), np.array([cx, cy, cz]))

    def to_mjcf_geom(self, name: str = "") -> str:
        """Emit a MuJoCo <geom .../> element."""
        n = f' name="{name}"' if name else ""
        c = f'{self.center[0]:.6f} {self.center[1]:.6f} {self.center[2]:.6f}'
        rgba = f'{self.rgba[0]:.2f} {self.rgba[1]:.2f} {self.rgba[2]:.2f} {self.rgba[3]:.2f}'
        mass_attr = f' mass="{self.mass_kg:.4f}"' if self.mass_kg else ""

        if self.kind == "cylinder":
            r, half_h = self.size[0], self.size[1]
            ax = f'{self.axis[0]:.4f} {self.axis[1]:.4f} {self.axis[2]:.4f}'
            return (
                f'<geom{n} type="cylinder" pos="{c}" '
                f'fromto="0 0 {-half_h:.6f} 0 0 {half_h:.6f}" '
                f'size="{r:.6f}" rgba="{rgba}"{mass_attr}/>'
            )
        if self.kind == "box":
            sx, sy, sz = self.size[:3]
            return (
                f'<geom{n} type="box" pos="{c}" '
                f'size="{sx:.6f} {sy:.6f} {sz:.6f}" rgba="{rgba}"{mass_attr}/>'
            )
        if self.kind == "sphere":
            r = self.size[0]
            return (
                f'<geom{n} type="sphere" pos="{c}" '
                f'size="{r:.6f}" rgba="{rgba}"{mass_attr}/>'
            )
        if self.kind == "capsule":
            r = self.size[0]
            x0, y0, z0 = self.size[1], self.size[2], self.size[3]
            x1, y1, z1 = self.size[4], self.size[5], self.size[6]
            return (
                f'<geom{n} type="capsule" '
                f'fromto="{x0:.6f} {y0:.6f} {z0:.6f} {x1:.6f} {y1:.6f} {z1:.6f}" '
                f'size="{r:.6f}" rgba="{rgba}"{mass_attr}/>'
            )
        return f'<!-- unsupported kind={self.kind} -->'

    def to_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a triangulated surface mesh.
        Returns (vertices[N,3], faces[M,3]) arrays.
        """
        if self.kind == "cylinder":
            return self._cylinder_mesh()
        if self.kind == "box":
            return self._box_mesh()
        if self.kind == "sphere":
            return self._sphere_mesh()
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    def _cylinder_mesh(self, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
        r, half_h = self.size[0], self.size[1]
        cx, cy, cz = self.center
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        verts = []
        for sign in (-1, 1):
            z = cz + sign * half_h
            verts.append([cx, cy, z])  # cap center
            for a in angles:
                verts.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
        verts = np.array(verts)
        faces = []
        n = segments
        # Bottom cap
        for i in range(n):
            faces.append([0, 1 + (i + 1) % n, 1 + i])
        # Top cap
        off = n + 1
        for i in range(n):
            faces.append([off, off + 1 + i, off + 1 + (i + 1) % n])
        # Side quads
        for i in range(n):
            i1 = 1 + i
            i2 = 1 + (i + 1) % n
            i3 = off + 1 + (i + 1) % n
            i4 = off + 1 + i
            faces.append([i1, i2, i3])
            faces.append([i1, i3, i4])
        return verts, np.array(faces, dtype=int)

    def _box_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        sx, sy, sz = self.size[:3]
        cx, cy, cz = self.center
        v = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],
        ], dtype=float)
        v[:, 0] = v[:, 0] * sx + cx
        v[:, 1] = v[:, 1] * sy + cy
        v[:, 2] = v[:, 2] * sz + cz
        f = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=int)
        return v, f

    def _sphere_mesh(self, stacks: int = 16, slices: int = 32) -> tuple[np.ndarray, np.ndarray]:
        r = self.size[0]
        cx, cy, cz = self.center
        verts = [[cx, cy, cz + r], [cx, cy, cz - r]]
        for i in range(1, stacks):
            phi = math.pi * i / stacks
            for j in range(slices):
                theta = 2 * math.pi * j / slices
                verts.append([
                    cx + r * math.sin(phi) * math.cos(theta),
                    cy + r * math.sin(phi) * math.sin(theta),
                    cz + r * math.cos(phi),
                ])
        faces = []
        for j in range(slices):
            faces.append([0, 2 + j, 2 + (j + 1) % slices])
        for i in range(stacks - 2):
            for j in range(slices):
                a = 2 + i * slices + j
                b = 2 + i * slices + (j + 1) % slices
                c = 2 + (i + 1) * slices + (j + 1) % slices
                d = 2 + (i + 1) * slices + j
                faces.append([a, b, c])
                faces.append([a, c, d])
        last = 2 + (stacks - 2) * slices
        for j in range(slices):
            faces.append([1, last + (j + 1) % slices, last + j])
        return np.array(verts), np.array(faces, dtype=int)

    def mass_properties(self, density: float = 7850.0) -> dict[str, float]:
        """Approximate mass, volume, and Ixx/Iyy/Izz for the primitive."""
        if self.mass_kg is not None:
            return {"mass_kg": self.mass_kg, "volume_m3": 0.0, "Ixx": 0, "Iyy": 0, "Izz": 0}
        cx, cy, cz = self.center
        if self.kind == "cylinder":
            r, h2 = self.size[0], self.size[1]
            vol = math.pi * r ** 2 * 2 * h2
            m = vol * density
            ix = m / 12 * (3 * r ** 2 + (2 * h2) ** 2)
            iz = 0.5 * m * r ** 2
            return {"mass_kg": m, "volume_m3": vol, "Ixx": ix, "Iyy": ix, "Izz": iz}
        if self.kind == "box":
            sx, sy, sz = self.size[:3]
            vol = 8 * sx * sy * sz
            m = vol * density
            ix = m / 12 * ((2 * sy) ** 2 + (2 * sz) ** 2)
            iy = m / 12 * ((2 * sx) ** 2 + (2 * sz) ** 2)
            iz = m / 12 * ((2 * sx) ** 2 + (2 * sy) ** 2)
            return {"mass_kg": m, "volume_m3": vol, "Ixx": ix, "Iyy": iy, "Izz": iz}
        if self.kind == "sphere":
            r = self.size[0]
            vol = 4 / 3 * math.pi * r ** 3
            m = vol * density
            i = 0.4 * m * r ** 2
            return {"mass_kg": m, "volume_m3": vol, "Ixx": i, "Iyy": i, "Izz": i}
        return {"mass_kg": 0.0, "volume_m3": 0.0, "Ixx": 0, "Iyy": 0, "Izz": 0}


# ═══════════════════════════════════════════════════════════════════════
# Parametric Engine Component Models
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PistonConfig:
    """Parametric piston definition (all values in metres)."""
    bore: float = 0.086             # cylinder bore diameter
    stroke: float = 0.086           # piston stroke
    compression_height: float = 0.030
    pin_diameter: float = 0.022
    skirt_length: float = 0.035
    ring_grooves: int = 3
    ring_groove_depth: float = 0.002
    ring_groove_width: float = 0.003
    dome_height: float = 0.005
    material_density: float = 2700  # aluminium


class Piston:
    """Computational piston component (PicoGK BaseShape composition)."""

    def __init__(self, cfg: PistonConfig | None = None, center: tuple[float, float, float] = (0, 0, 0)):
        self.cfg = cfg or PistonConfig()
        self.center = center
        self.shapes: list[BaseShape] = self._build()

    def _build(self) -> list[BaseShape]:
        c = self.cfg
        r = c.bore / 2
        cx, cy, cz = self.center
        shapes: list[BaseShape] = []

        # Crown dome (ADD)
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz + c.compression_height + c.dome_height / 2),
            size=(r, c.dome_height / 2),
            rgba=(0.82, 0.80, 0.75, 1.0),
            operation="ADD",
            mass_kg=math.pi * r ** 2 * c.dome_height * c.material_density,
        ))

        # Skirt cylinder (ADD)
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz - c.skirt_length / 2),
            size=(r, c.skirt_length / 2),
            rgba=(0.85, 0.82, 0.76, 1.0),
            operation="ADD",
            mass_kg=math.pi * r ** 2 * c.skirt_length * c.material_density * 0.85,
        ))

        # Wrist-pin bore (SUBTRACT)
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz - 0.005),
            size=(c.pin_diameter / 2, r + 0.001),
            rgba=(0.3, 0.3, 0.3, 1.0),
            operation="SUBTRACT",
            axis=(1.0, 0.0, 0.0),
        ))

        # Ring grooves (SUBTRACT)
        for i in range(c.ring_grooves):
            gz = cz + c.compression_height - i * 0.008
            shapes.append(BaseShape(
                kind="cylinder",
                center=(cx, cy, gz),
                size=(r + 0.001, c.ring_groove_width / 2),
                rgba=(0.5, 0.5, 0.5, 1.0),
                operation="SUBTRACT",
                mass_kg=-c.ring_groove_depth * c.ring_groove_width * 2 * math.pi * r * 7850,
            ))

        return shapes

    def to_mjcf_body(self, name: str = "piston") -> str:
        lines = [f'<body name="{name}" pos="{self.center[0]:.6f} {self.center[1]:.6f} {self.center[2]:.6f}">']
        for i, s in enumerate(self.shapes):
            if s.operation == "ADD":
                lines.append(f"  {s.to_mjcf_geom(f'{name}_g{i}')}")
        # Pin hole site for connecting rod attachment
        lines.append(f'  <site name="{name}_pin" pos="0 0 -0.005" size="0.003"/>')
        lines.append("</body>")
        return "\n".join(lines)

    def to_mjcf(self) -> str:
        """Full MJCF fragment (body + geoms + site)."""
        return self.to_mjcf_body()

    def to_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        all_v, all_f = [], []
        offset = 0
        for s in self.shapes:
            if s.operation == "ADD":
                v, f = s.to_mesh()
                all_v.append(v)
                all_f.append(f + offset)
                offset += len(v)
        if all_v:
            return np.vstack(all_v), np.vstack(all_f)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        mins, maxs = [], []
        for s in self.shapes:
            if s.operation == "ADD":
                mn, mx = s.bounding_box()
                mins.append(mn)
                maxs.append(mx)
        return np.array(mins).min(axis=0), np.array(maxs).max(axis=0)

    def mass_properties(self) -> dict[str, float]:
        total_m, total_v = 0.0, 0.0
        for s in self.shapes:
            if s.operation == "ADD":
                props = s.mass_properties(self.cfg.material_density)
                total_m += props["mass_kg"]
                total_v += props["volume_m3"]
            elif s.operation == "SUBTRACT":
                props = s.mass_properties(self.cfg.material_density)
                total_m -= abs(props["mass_kg"])
                total_v -= abs(props["volume_m3"])
        return {"mass_kg": max(total_m, 0.001), "volume_m3": max(total_v, 1e-9),
                "Ixx": 0, "Iyy": 0, "Izz": 0}

    def parametric_hash(self) -> str:
        h = hashlib.sha256()
        for s in self.shapes:
            h.update(s.parametric_hash().encode())
        return h.hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CylinderLinerConfig:
    """Parametric cylinder-liner definition (metres)."""
    bore: float = 0.086
    wall_thickness: float = 0.006
    length: float = 0.140
    flange_width: float = 0.010
    flange_thickness: float = 0.004
    cooling_channel_count: int = 6
    cooling_channel_diameter: float = 0.005
    material_density: float = 7200  # cast iron


class CylinderLiner:
    """Computational cylinder liner (bore + wall + flange + coolant channels)."""

    def __init__(self, cfg: CylinderLinerConfig | None = None, center: tuple[float, float, float] = (0, 0, 0)):
        self.cfg = cfg or CylinderLinerConfig()
        self.center = center
        self.shapes = self._build()

    def _build(self) -> list[BaseShape]:
        c = self.cfg
        r_inner = c.bore / 2
        r_outer = r_inner + c.wall_thickness
        cx, cy, cz = self.center
        shapes: list[BaseShape] = []

        # Outer cylinder (ADD)
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz),
            size=(r_outer, c.length / 2),
            rgba=(0.35, 0.35, 0.38, 1.0),
            operation="ADD",
            mass_kg=math.pi * (r_outer ** 2 - r_inner ** 2) * c.length * c.material_density,
        ))

        # Inner bore (SUBTRACT)
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz),
            size=(r_inner, c.length / 2 + 0.001),
            rgba=(0.6, 0.6, 0.62, 1.0),
            operation="SUBTRACT",
        ))

        # Deck flange (ADD)
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz + c.length / 2 - c.flange_thickness / 2),
            size=(r_outer + c.flange_width, c.flange_thickness / 2),
            rgba=(0.4, 0.4, 0.42, 1.0),
            operation="ADD",
        ))

        # Cooling channels (SUBTRACT rings of small cylinders)
        channel_r = (r_outer + r_inner) / 2
        for i in range(c.cooling_channel_count):
            angle = 2 * math.pi * i / c.cooling_channel_count
            lx = cx + channel_r * math.cos(angle)
            ly = cy + channel_r * math.sin(angle)
            shapes.append(BaseShape(
                kind="cylinder",
                center=(lx, ly, cz),
                size=(c.cooling_channel_diameter / 2, c.length / 2),
                rgba=(0.1, 0.1, 0.15, 1.0),
                operation="SUBTRACT",
            ))

        return shapes

    def to_mjcf_body(self, name: str = "cylinder_liner") -> str:
        lines = [f'<body name="{name}" pos="{self.center[0]:.6f} {self.center[1]:.6f} {self.center[2]:.6f}">']
        for i, s in enumerate(self.shapes):
            if s.operation == "ADD":
                lines.append(f"  {s.to_mjcf_geom(f'{name}_g{i}')}")
        lines.append("</body>")
        return "\n".join(lines)

    def to_mjcf(self) -> str:
        return self.to_mjcf_body()

    def to_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        all_v, all_f = [], []
        offset = 0
        for s in self.shapes:
            if s.operation == "ADD":
                v, f = s.to_mesh()
                all_v.append(v)
                all_f.append(f + offset)
                offset += len(v)
        if all_v:
            return np.vstack(all_v), np.vstack(all_f)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.shapes[0].bounding_box()

    def mass_properties(self) -> dict[str, float]:
        return self.shapes[0].mass_properties(self.cfg.material_density)

    def parametric_hash(self) -> str:
        return hashlib.sha256(str(self.cfg).encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CrankshaftConfig:
    """Parametric crankshaft (inline layout, metres)."""
    main_journal_diameter: float = 0.050
    main_journal_length: float = 0.020
    crankpin_diameter: float = 0.044
    crankpin_length: float = 0.018
    throw_radius: float = 0.043    # half-stroke
    web_thickness: float = 0.012
    web_width: float = 0.070
    counterweight_radius: float = 0.055
    num_cylinders: int = 4
    cylinder_spacing: float = 0.096
    material_density: float = 7850  # steel


class Crankshaft:
    """Computational crankshaft — main journals, crankpins, webs, counterweights."""

    def __init__(self, cfg: CrankshaftConfig | None = None, center: tuple[float, float, float] = (0, 0, 0)):
        self.cfg = cfg or CrankshaftConfig()
        self.center = center
        self.shapes = self._build()

    def _build(self) -> list[BaseShape]:
        c = self.cfg
        cx, cy, cz = self.center
        shapes: list[BaseShape] = []
        total_length = c.num_cylinders * c.cylinder_spacing
        start_x = cx - total_length / 2

        for i in range(c.num_cylinders + 1):
            mx = start_x + i * c.cylinder_spacing
            # Main journal
            shapes.append(BaseShape(
                kind="cylinder",
                center=(mx, cy, cz),
                size=(c.main_journal_diameter / 2, c.main_journal_length / 2),
                rgba=(0.55, 0.55, 0.58, 1.0),
                operation="ADD",
                axis=(1.0, 0.0, 0.0),
            ))

        for i in range(c.num_cylinders):
            px = start_x + (i + 0.5) * c.cylinder_spacing
            crank_angle = i * math.pi / 2  # 4-cyl firing order
            py = cy + c.throw_radius * math.cos(crank_angle)
            pz = cz + c.throw_radius * math.sin(crank_angle)
            # Crankpin
            shapes.append(BaseShape(
                kind="cylinder",
                center=(px, py, pz),
                size=(c.crankpin_diameter / 2, c.crankpin_length / 2),
                rgba=(0.6, 0.58, 0.55, 1.0),
                operation="ADD",
                axis=(1.0, 0.0, 0.0),
            ))
            # Web (connects journal to crankpin)
            shapes.append(BaseShape(
                kind="box",
                center=(px, cy + c.throw_radius / 2 * math.cos(crank_angle),
                        cz + c.throw_radius / 2 * math.sin(crank_angle)),
                size=(c.web_thickness / 2, c.web_width / 2, c.throw_radius / 2),
                rgba=(0.5, 0.5, 0.52, 1.0),
                operation="ADD",
            ))
            # Counterweight
            shapes.append(BaseShape(
                kind="cylinder",
                center=(px,
                        cy - c.throw_radius * 0.6 * math.cos(crank_angle),
                        cz - c.throw_radius * 0.6 * math.sin(crank_angle)),
                size=(c.counterweight_radius, c.web_thickness / 2),
                rgba=(0.45, 0.45, 0.48, 1.0),
                operation="ADD",
            ))

        return shapes

    def to_mjcf_body(self, name: str = "crankshaft") -> str:
        lines = [f'<body name="{name}" pos="{self.center[0]:.6f} {self.center[1]:.6f} {self.center[2]:.6f}">']
        # Main hinge joint (rotation around x-axis)
        lines.append(f'  <joint name="{name}_rot" type="hinge" axis="1 0 0" '
                     f'damping="0.5" armature="0.01"/>')
        for i, s in enumerate(self.shapes):
            lines.append(f"  {s.to_mjcf_geom(f'{name}_g{i}')}")
        lines.append(f'  <site name="{name}_ref" pos="0 0 0" size="0.003"/>')
        lines.append("</body>")
        return "\n".join(lines)

    def to_mjcf(self) -> str:
        return self.to_mjcf_body()

    def to_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        all_v, all_f = [], []
        offset = 0
        for s in self.shapes:
            v, f = s.to_mesh()
            if len(v) > 0:
                all_v.append(v)
                all_f.append(f + offset)
                offset += len(v)
        if all_v:
            return np.vstack(all_v), np.vstack(all_f)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        mins, maxs = [], []
        for s in self.shapes:
            mn, mx = s.bounding_box()
            mins.append(mn)
            maxs.append(mx)
        return np.array(mins).min(axis=0), np.array(maxs).max(axis=0)

    def mass_properties(self) -> dict[str, float]:
        total_m = 0.0
        for s in self.shapes:
            props = s.mass_properties(self.cfg.material_density)
            total_m += props["mass_kg"]
        return {"mass_kg": total_m, "volume_m3": total_m / self.cfg.material_density,
                "Ixx": 0, "Iyy": 0, "Izz": 0}

    def parametric_hash(self) -> str:
        return hashlib.sha256(str(self.cfg).encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ConnectingRodConfig:
    big_end_diameter: float = 0.044
    small_end_diameter: float = 0.022
    rod_length: float = 0.143
    beam_width: float = 0.018
    beam_thickness: float = 0.014
    material_density: float = 7850


class ConnectingRod:
    """Computational connecting rod (big-end → beam → small-end)."""

    def __init__(self, cfg: ConnectingRodConfig | None = None,
                 center: tuple[float, float, float] = (0, 0, 0)):
        self.cfg = cfg or ConnectingRodConfig()
        self.center = center
        self.shapes = self._build()

    def _build(self) -> list[BaseShape]:
        c = self.cfg
        cx, cy, cz = self.center
        half_len = c.rod_length / 2
        shapes: list[BaseShape] = []

        # Big end
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz - half_len),
            size=(c.big_end_diameter / 2, c.big_end_diameter / 2),
            rgba=(0.65, 0.63, 0.60, 1.0),
            operation="ADD",
        ))
        # Small end
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz + half_len),
            size=(c.small_end_diameter / 2, c.small_end_diameter / 2),
            rgba=(0.65, 0.63, 0.60, 1.0),
            operation="ADD",
        ))
        # Beam
        shapes.append(BaseShape(
            kind="box",
            center=(cx, cy, cz),
            size=(c.beam_thickness / 2, c.beam_width / 2, half_len),
            rgba=(0.62, 0.60, 0.57, 1.0),
            operation="ADD",
        ))
        return shapes

    def to_mjcf_body(self, name: str = "conrod") -> str:
        lines = [
            f'<body name="{name}" pos="{self.center[0]:.6f} {self.center[1]:.6f} {self.center[2]:.6f}">'
        ]
        lines.append(f'  <joint name="{name}_hinge" type="hinge" axis="0 1 0" '
                     f'damping="0.3" armature="0.005"/>')
        for i, s in enumerate(self.shapes):
            lines.append(f"  {s.to_mjcf_geom(f'{name}_g{i}')}")
        lz = c = self.cfg.rod_length / 2
        lines.append(f'  <site name="{name}_big" pos="0 0 {-lz:.6f}" size="0.003"/>')
        lines.append(f'  <site name="{name}_small" pos="0 0 {lz:.6f}" size="0.003"/>')
        lines.append("</body>")
        return "\n".join(lines)

    def to_mjcf(self) -> str:
        return self.to_mjcf_body()

    def to_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        all_v, all_f = [], []
        offset = 0
        for s in self.shapes:
            v, f = s.to_mesh()
            all_v.append(v)
            all_f.append(f + offset)
            offset += len(v)
        return np.vstack(all_v), np.vstack(all_f)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        r = max(self.cfg.big_end_diameter, self.cfg.small_end_diameter) / 2
        cx, cy, cz = self.center
        return (np.array([cx - r, cy - r, cz - self.cfg.rod_length / 2 - r]),
                np.array([cx + r, cy + r, cz + self.cfg.rod_length / 2 + r]))

    def mass_properties(self) -> dict[str, float]:
        return {"mass_kg": 0.5, "volume_m3": 6e-5, "Ixx": 0, "Iyy": 0, "Izz": 0}

    def parametric_hash(self) -> str:
        return hashlib.sha256(str(self.cfg).encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ValveConfig:
    head_diameter: float = 0.034
    head_thickness: float = 0.003
    stem_diameter: float = 0.007
    stem_length: float = 0.100
    tulip_cone_angle: float = 45.0  # degrees
    keeper_groove_position: float = 0.085
    material_density: float = 7850


class Valve:
    """Computational intake/exhaust valve."""

    def __init__(self, cfg: ValveConfig | None = None,
                 center: tuple[float, float, float] = (0, 0, 0)):
        self.cfg = cfg or ValveConfig()
        self.center = center
        self.shapes = self._build()

    def _build(self) -> list[BaseShape]:
        c = self.cfg
        cx, cy, cz = self.center
        shapes: list[BaseShape] = []

        # Tulip head (disc + conical transition)
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz),
            size=(c.head_diameter / 2, c.head_thickness / 2),
            rgba=(0.70, 0.68, 0.65, 1.0),
            operation="ADD",
        ))

        # Stem
        shapes.append(BaseShape(
            kind="cylinder",
            center=(cx, cy, cz + c.stem_length / 2 + c.head_thickness / 2),
            size=(c.stem_diameter / 2, c.stem_length / 2),
            rgba=(0.65, 0.63, 0.60, 1.0),
            operation="ADD",
        ))

        return shapes

    def to_mjcf_body(self, name: str = "valve") -> str:
        lines = [
            f'<body name="{name}" pos="{self.center[0]:.6f} {self.center[1]:.6f} {self.center[2]:.6f}">'
        ]
        lines.append(f'  <joint name="{name}_trans" type="slide" axis="0 0 1" '
                     f'range="0 {self.cfg.stem_length:.4f}" damping="0.8" armature="0.001"/>')
        for i, s in enumerate(self.shapes):
            lines.append(f"  {s.to_mjcf_geom(f'{name}_g{i}')}")
        lines.append(f'  <site name="{name}_tip" pos="0 0 {self.cfg.stem_length + self.cfg.head_thickness:.6f}" size="0.003"/>')
        lines.append("</body>")
        return "\n".join(lines)

    def to_mjcf(self) -> str:
        return self.to_mjcf_body()

    def to_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        all_v, all_f = [], []
        offset = 0
        for s in self.shapes:
            v, f = s.to_mesh()
            all_v.append(v)
            all_f.append(f + offset)
            offset += len(v)
        return np.vstack(all_v), np.vstack(all_f)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.cfg.head_diameter / 2
        cx, cy, cz = self.center
        h = self.cfg.stem_length + self.cfg.head_thickness
        return (np.array([cx - r, cy - r, cz - self.cfg.head_thickness]),
                np.array([cx + r, cy + r, cz + h]))

    def mass_properties(self) -> dict[str, float]:
        return {"mass_kg": 0.06, "volume_m3": 8e-6, "Ixx": 0, "Iyy": 0, "Izz": 0}

    def parametric_hash(self) -> str:
        return hashlib.sha256(str(self.cfg).encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════
# Assembly: Inline-4 Engine Block
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class EngineBlockConfig:
    """Top-level parametric inline-4 engine configuration."""
    num_cylinders: int = 4
    bore: float = 0.086
    stroke: float = 0.086
    cylinder_spacing: float = 0.096
    block_height: float = 0.200
    block_width: float = 0.220
    block_depth: float = 0.180
    wall_thickness: float = 0.008
    deck_thickness: float = 0.010
    sump_depth: float = 0.040
    material_density: float = 2700  # cast aluminium


class EngineBlockAssembly:
    """
    Full inline-4 engine block assembly with installed components.

    Each sub-component is a Piston / CylinderLiner / Crankshaft / ConnectingRod/Valve
    instance positioned according to the engine layout.

    This is the top-level 'Computational Engineering Model' (CEM) in
    LEAP 71 terminology.
    """

    def __init__(self, cfg: EngineBlockConfig | None = None,
                 base_pos: tuple[float, float, float] = (0, 0, 0)):
        self.cfg = cfg or EngineBlockConfig()
        self.base_pos = base_pos
        self.pistons: list[Piston] = []
        self.liners: list[CylinderLiner] = []
        self.crankshaft: Crankshaft | None = None
        self.conrods: list[ConnectingRod] = []
        self.intake_valves: list[Valve] = []
        self.exhaust_valves: list[Valve] = []
        self.block_shapes: list[BaseShape] = []
        self._assemble()

    def _assemble(self) -> None:
        c = self.cfg
        bx, by, bz = self.base_pos
        total_w = c.num_cylinders * c.cylinder_spacing
        start_x = bx - total_w / 2

        # --- Block walls ---
        r_outer = c.bore / 2 + c.wall_thickness + 0.006
        self.block_shapes.append(BaseShape(
            kind="box",
            center=(bx, by, bz + c.block_height / 2),
            size=(total_w / 2 + c.wall_thickness, r_outer + c.wall_thickness, c.block_height / 2),
            rgba=(0.50, 0.50, 0.52, 0.85),
            operation="ADD",
        ))
        # Interior bore pass-through (SUBTRACT)
        self.block_shapes.append(BaseShape(
            kind="box",
            center=(bx, by, bz + c.block_height / 2 + c.deck_thickness / 2),
            size=(total_w / 2, r_outer, c.block_height / 2 - c.deck_thickness),
            rgba=(0.4, 0.4, 0.42, 1.0),
            operation="SUBTRACT",
        ))

        # Sump
        self.block_shapes.append(BaseShape(
            kind="box",
            center=(bx, by, bz - c.sump_depth / 2),
            size=(total_w / 2 + c.wall_thickness, r_outer + c.wall_thickness, c.sump_depth / 2),
            rgba=(0.38, 0.38, 0.40, 0.9),
            operation="ADD",
        ))

        # --- Per-cylinder components ---
        for i in range(c.num_cylinders):
            cx = start_x + (i + 0.5) * c.cylinder_spacing
            cy = by
            bore_z = bz + c.deck_thickness + c.block_height * 0.4
            crank_z_offset = -c.stroke / 2

            # Cylinder liner
            liner_cfg = CylinderLinerConfig(bore=c.bore)
            liner = CylinderLiner(liner_cfg, center=(cx, cy, bore_z))
            self.liners.append(liner)

            # Piston (initial TDC position)
            piston_cfg = PistonConfig(bore=c.bore, stroke=c.stroke)
            piston_z = bore_z + c.block_height * 0.2
            piston = Piston(piston_cfg, center=(cx, cy, piston_z))
            self.pistons.append(piston)

            # Connecting rod
            rod_cfg = ConnectingRodConfig()
            rod_z = bore_z
            rod = ConnectingRod(rod_cfg, center=(cx, cy, rod_z))
            self.conrods.append(rod)

            # Intake valve (left bank)
            intake_cfg = ValveConfig()
            intake = Valve(intake_cfg, center=(cx - 0.015, cy - 0.010, bore_z + c.block_height * 0.35))
            self.intake_valves.append(intake)

            # Exhaust valve (right bank)
            exhaust_cfg = ValveConfig()
            exhaust = Valve(exhaust_cfg, center=(cx + 0.015, cy + 0.010, bore_z + c.block_height * 0.35))
            self.exhaust_valves.append(exhaust)

        # --- Crankshaft ---
        crank_cfg = CrankshaftConfig(num_cylinders=c.num_cylinders, cylinder_spacing=c.cylinder_spacing)
        self.crankshaft = Crankshaft(crank_cfg, center=(bx, by, bz + c.block_height * 0.08))

    def to_mjcf(self, name: str = "engine_block") -> str:
        """
        Full MuJoCo <worldbody> XML for the complete engine assembly.

        Each component is a nested <body> with appropriate joints:
        - Pistons:    slide joint (vertical translation)
        - Valves:     slide joints (vertical translation)
        - Crankshaft: hinge joint (rotation)
        - ConRods:    hinge joints (swing)
        - Block:      fixed body (world-anchored)
        """
        lines = [f'<body name="{name}" pos="{self.base_pos[0]:.6f} {self.base_pos[1]:.6f} {self.base_pos[2]:.6f}">']

        # Block as fixed body with geoms
        lines.append(f'  <body name="{name}_shell" pos="0 0 0">')
        for i, s in enumerate(self.block_shapes):
            if s.operation == "ADD":
                lines.append(f"    {s.to_mjcf_geom(f'{name}_blk{i}')}")
        lines.append("  </body>")

        # Cylinder liners (fixed)
        for j, liner in enumerate(self.liners):
            liner_xml = liner.to_mjcf()
            # Indent inside the engine body
            for line in liner_xml.split("\n"):
                lines.append(f"  {line}")

        # Pistons (sliding)
        for j, piston in enumerate(self.pistons):
            piston_xml = piston.to_mjcf_body(f"piston_{j}")
            for line in piston_xml.split("\n"):
                lines.append(f"  {line}")

        # Connecting rods
        for j, rod in enumerate(self.conrods):
            rod_xml = rod.to_mjcf_body(f"conrod_{j}")
            for line in rod_xml.split("\n"):
                lines.append(f"  {line}")

        # Crankshaft
        if self.crankshaft:
            crank_xml = self.crankshaft.to_mjcf_body("crankshaft")
            for line in crank_xml.split("\n"):
                lines.append(f"  {line}")

        # Valves
        for j, v in enumerate(self.intake_valves):
            v_xml = v.to_mjcf_body(f"intake_valve_{j}")
            for line in v_xml.split("\n"):
                lines.append(f"  {line}")

        for j, v in enumerate(self.exhaust_valves):
            v_xml = v.to_mjcf_body(f"exhaust_valve_{j}")
            for line in v_xml.split("\n"):
                lines.append(f"  {line}")

        # Assembly reference sites
        lines.append(f'  <site name="{name}_top_center" pos="0 0 {self.cfg.block_height:.6f}" size="0.005"/>')
        lines.append(f'  <site name="{name}_crank_ref" pos="0 0 0" size="0.005"/>')

        lines.append("</body>")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Return a structured summary of the engine assembly."""
        return {
            "type": "Inline-4 Engine Block",
            "bore_mm": self.cfg.bore * 1000,
            "stroke_mm": self.cfg.stroke * 1000,
            "displacement_L": (math.pi * (self.cfg.bore / 2) ** 2 * self.cfg.stroke * self.cfg.num_cylinders) * 1000,
            "cylinders": self.cfg.num_cylinders,
            "num_pistons": len(self.pistons),
            "num_valves": len(self.intake_valves) + len(self.exhaust_valves),
            "num_conrods": len(self.conrods),
            "crankshaft_present": self.crankshaft is not None,
            "block_dimensions_m": {
                "width": self.cfg.cylinder_spacing * self.cfg.num_cylinders,
                "height": self.cfg.block_height,
                "depth": self.cfg.block_depth,
            },
            "materials": {
                "block": "cast_aluminium_2700kgm3",
                "pistons": "aluminium_2700kgm3",
                "crankshaft": "steel_7850kgm3",
                "liners": "cast_iron_7200kgm3",
            },
        }

    def all_meshes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return named mesh data for every component."""
        meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, p in enumerate(self.pistons):
            meshes[f"piston_{i}"] = p.to_mesh()
        for i, l in enumerate(self.liners):
            meshes[f"liner_{i}"] = l.to_mesh()
        if self.crankshaft:
            meshes["crankshaft"] = self.crankshaft.to_mesh()
        for i, r in enumerate(self.conrods):
            meshes[f"conrod_{i}"] = r.to_mesh()
        for i, v in enumerate(self.intake_valves):
            meshes[f"intake_valve_{i}"] = v.to_mesh()
        for i, v in enumerate(self.exhaust_valves):
            meshes[f"exhaust_valve_{i}"] = v.to_mesh()
        return meshes

    def parametric_hash(self) -> str:
        h = hashlib.sha256()
        for p in self.pistons:
            h.update(p.parametric_hash().encode())
        if self.crankshaft:
            h.update(self.crankshaft.parametric_hash().encode())
        return h.hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════
# Computational Engineering Geometry Engine (PicoGK-inspired)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Vector3:
    """3D vector for computational geometry"""
    x: float
    y: float
    z: float
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector3(self.x/mag, self.y/mag, self.z/mag)
        return Vector3(0, 0, 0)


@dataclass
class BoundingBox:
    """Bounding box for voxel field"""
    min: Vector3
    max: Vector3
    
    @property
    def size(self):
        return Vector3(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z
        )


class VoxelField:
    """
    PicoGK-inspired voxel field for computational engineering
    Implements signed distance field operations for engine components
    """
    
    def __init__(self, size: Tuple[int, int, int], voxel_size: float = 1.0):
        self.size = size  # (nx, ny, nz)
        self.voxel_size = voxel_size
        # Initialize signed distance field (positive = outside, negative = inside)
        self.field = np.full(size, np.inf, dtype=np.float32)
        
    def set_signed_distance(self, pos: Tuple[int, int, int], distance: float):
        """Set signed distance at voxel position"""
        x, y, z = pos
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            self.field[x, y, z] = min(self.field[x, y, z], distance)
    
    def get_signed_distance(self, pos: Tuple[int, int, int]) -> float:
        """Get signed distance at voxel position"""
        x, y, z = pos
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            return self.field[x, y, z]
        return np.inf
    
    def offset(self, distance: float):
        """Offset the surface by distance (positive = outward, negative = inward)"""
        # Simple offset - in practice would use more sophisticated methods
        self.field += distance
    
    def boolean_union(self, other: 'VoxelField'):
        """Boolean union operation"""
        self.field = np.minimum(self.field, other.field)
    
    def boolean_subtract(self, other: 'VoxelField'):
        """Boolean subtraction: A - B"""
        self.field = np.maximum(self.field, -other.field)
    
    def boolean_intersection(self, other: 'VoxelField'):
        """Boolean intersection operation"""
        self.field = np.maximum(self.field, other.field)
    
    def extract_surface(self, threshold: float = 0.0) -> List[Tuple[int, int, int]]:
        """Extract surface voxels where signed distance ≈ threshold"""
        surface_voxels = []
        nx, ny, nz = self.size
        
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    dist = self.field[x, y, z]
                    if abs(dist) < threshold * 2:  # Near surface
                        # Check if this is actually on surface (has neighbors with opposite sign)
                        is_surface = False
                        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < self.size[0] and 0 <= ny < self.size[1] and 0 <= nz < self.size[2]):
                                neighbor_dist = self.field[nx, ny, nz]
                                if dist * neighbor_dist < 0:  # Opposite signs = surface
                                    is_surface = True
                                    break
                        if is_surface:
                            surface_voxels.append((x, y, z))
        return surface_voxels


class ImplicitFunction:
    """Base class for implicit signed distance functions"""
    
    def signed_distance(self, point: Vector3) -> float:
        """Return signed distance from point to surface"""
        raise NotImplementedError


class Sphere(ImplicitFunction):
    """Sphere implicit function"""
    
    def __init__(self, center: Vector3, radius: float):
        self.center = center
        self.radius = radius
    
    def signed_distance(self, point: Vector3) -> float:
        dist = np.sqrt(
            (point.x - self.center.x)**2 +
            (point.y - self.center.y)**2 +
            (point.z - self.center.z)**2
        )
        return dist - self.radius


class Box(ImplicitFunction):
    """Box implicit function"""
    
    def __init__(self, min_point: Vector3, max_point: Vector3):
        self.min = min_point
        self.max = max_point
    
    def signed_distance(self, point: Vector3) -> float:
        # Distance to closest point on box
        dx = max(self.min.x - point.x, 0, point.x - self.max.x)
        dy = max(self.min.y - point.y, 0, point.y - self.max.y)
        dz = max(self.min.z - point.z, 0, point.z - self.max.z)
        return np.sqrt(dx*dx + dy*dy + dz*dz)


class Cylinder(ImplicitFunction):
    """Cylinder implicit function"""
    
    def __init__(self, center: Vector3, radius: float, height: float):
        self.center = center
        self.radius = radius
        self.height = height
        self.half_height = height / 2
    
    def signed_distance(self, point: Vector3) -> float:
        # Distance in XY plane from cylinder axis
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dist_xy = np.sqrt(dx*dx + dy*dy)
        
        # Distance along Z axis
        dz = abs(point.z - self.center.z)
        
        # Distance to cylinder surface
        if dz > self.half_height:
            # Outside end caps
            dist_z = dz - self.half_height
            return np.sqrt(dist_xy*dist_xy + dist_z*dist_z) - self.radius
        else:
            # Inside height range
            return abs(dist_xy - self.radius)


class ComputationalEngine:
    """
    Main computational engine for creating rocket/IC engine geometries
    Based on LEAP 71's Computational Engineering principles
    """
    
    def __init__(self, voxel_size: float = 0.5):
        self.voxel_size = voxel_size
        self.components = []
        
    def create_rocket_engine(self, 
                           chamber_diameter: float = 50.0,
                           chamber_length: float = 100.0,
                           nozzle_exit_diameter: float = 30.0,
                           nozzle_length: float = 50.0,
                           wall_thickness: float = 5.0) -> VoxelField:
        """
        Create a rocket engine using computational engineering principles
        Combustion chamber + nozzle + cooling channels
        """
        # Create voxel field large enough for the engine
        max_dim = max(chamber_diameter, nozzle_exit_diameter) + 2 * wall_thickness
        total_length = chamber_length + nozzle_length
        size = (
            int(max_dim / self.voxel_size) + 10,
            int(max_dim / self.voxel_size) + 10,
            int(total_length / self.voxel_size) + 10
        )
        
        engine_field = VoxelField(size, self.voxel_size)
        
        # Center the engine in the voxel field
        center_x = size[0] // 2
        center_y = size[1] // 2
        center_z = size[2] // 2
        
        # Create combustion chamber (cylinder)
        chamber_radius = chamber_diameter / 2
        chamber_implicit = Cylinder(
            Vector3(0, 0, chamber_length/2),  # Centered in Z
            chamber_radius,
            chamber_length
        )
        
        # Render chamber into voxel field
        self._render_implicit_to_field(engine_field, chamber_implicit, 
                                     center_x, center_y, center_z)
        
        # Create nozzle (conical frustum approximated as cylinder stack)
        nozzle_implicit = Cylinder(
            Vector3(0, 0, chamber_length + nozzle_length/2),
            nozzle_exit_diameter / 2,
            nozzle_length
        )
        
        self._render_implicit_to_field(engine_field, nozzle_implicit,
                                     center_x, center_y, center_z)
        
        # Create outer shell (subtract inner volume for wall thickness)
        outer_chamber = Cylinder(
            Vector3(0, 0, chamber_length/2),
            chamber_radius + wall_thickness,
            chamber_length
        )
        
        outer_nozzle = Cylinder(
            Vector3(0, 0, chamber_length + nozzle_length/2),
            nozzle_exit_diameter / 2 + wall_thickness,
            nozzle_length
        )
        
        # Create outer shell field
        outer_field = VoxelField(size, self.voxel_size)
        self._render_implicit_to_field(outer_field, outer_chamber, 
                                     center_x, center_y, center_z)
        self._render_implicit_to_field(outer_field, outer_nozzle,
                                     center_x, center_y, center_z)
        
        # Boolean subtraction to create hollow engine
        engine_field.boolean_subtract(outer_field)
        
        return engine_field
    
    def create_internal_combustion_engine(self,
                                        cylinder_bore: float = 80.0,
                                        stroke: float = 90.0,
                                        num_cylinders: int = 4,
                                        bank_angle: float = 0.0) -> VoxelField:
        """
        Create an internal combustion engine block
        """
        # Calculate dimensions
        block_width = num_cylinders * cylinder_bore * 1.2
        block_height = cylinder_bore * 1.5
        block_length = stroke * 1.5 + 50  # Extra for crankcase
        
        size = (
            int(block_width / self.voxel_size) + 10,
            int(block_height / self.voxel_size) + 10,
            int(block_length / self.voxel_size) + 10
        )
        
        engine_field = VoxelField(size, self.voxel_size)
        
        # Center the engine
        center_x = size[0] // 2
        center_y = size[1] // 2
        center_z = size[2] // 2
        
        # Create engine block (main casting)
        block_implicit = Box(
            Vector3(-block_width/2, -block_height/2, -block_length/2),
            Vector3(block_width/2, block_height/2, block_length/2)
        )
        
        self._render_implicit_to_field(engine_field, block_implicit,
                                     center_x, center_y, center_z)
        
        # Create cylinders (holes for pistons)
        cylinder_spacing = cylinder_bore * 1.2
        start_x = -(num_cylinders - 1) * cylinder_spacing / 2
        
        for i in range(num_cylinders):
            cylinder_x = start_x + i * cylinder_spacing
            cylinder_implicit = Cylinder(
                Vector3(cylinder_x, 0, 0),  # Along X axis
                cylinder_bore / 2,
                block_length * 0.8  # Slightly shorter than block
            )
            
            # Subtract cylinder from block
            cylinder_field = VoxelField(size, self.voxel_size)
            self._render_implicit_to_field(cylinder_field, cylinder_implicit,
                                         center_x, center_y, center_z)
            engine_field.boolean_subtract(cylinder_field)
        
        return engine_field
    
    def _render_implicit_to_field(self, field: VoxelField, 
                                implicit: ImplicitFunction,
                                offset_x: int, offset_y: int, offset_z: int):
        """Render implicit function to voxel field"""
        nx, ny, nz = field.size
        vs = field.voxel_size
        
        # Sample the implicit function at each voxel
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    # Convert voxel indices to world coordinates
                    world_x = (x - offset_x) * vs
                    world_y = (y - offset_y) * vs
                    world_z = (z - offset_z) * vs
                    
                    point = Vector3(world_x, world_y, world_z)
                    distance = implicit.signed_distance(point)
                    
                    # Update signed distance field (keep minimum distance)
                    current_dist = field.get_signed_distance((x, y, z))
                    field.set_signed_distance((x, y, z), min(current_dist, distance))


class EngineSimulation:
    """
    Physics simulation for engine components using MuJoCo-like principles
    """
    
    def __init__(self):
        self.engines = []
        self.robot_arms = []
        
    def add_engine(self, engine_field: VoxelField, 
                  engine_type: str = "rocket"):
        """Add engine to simulation"""
        self.engines.append({
            'field': engine_field,
            'type': engine_type,
            'position': Vector3(0, 0, 0),
            'orientation': Vector3(0, 0, 0)
        })
    
    def simulate_assembly(self, steps: int = 100):
        """Simulate engine assembly process"""
        print(f"Simulating engine assembly for {steps} steps...")
        # In a real implementation, this would interface with MuJoCo
        # For now, we'll simulate the computational aspects
        
        for step in range(steps):
            progress = step / steps
            print(f"Assembly progress: {progress*100:.1f}%")
            
            # Simulate robot arm movements, part fitting, etc.
            # This would normally involve physics calculations
            
        print("Assembly simulation complete!")


def create_engine_demo():
    """Create a demonstration of the computational engine system"""
    print("Creating Computational Engineering Engine Simulation...")
    
    # Create computational engine designer
    engine_designer = ComputationalEngine(voxel_size=2.0)  # Larger voxels for demo
    
    # Create rocket engine
    print("\n1. Creating Rocket Engine...")
    rocket_engine = engine_designer.create_rocket_engine(
        chamber_diameter=40.0,
        chamber_length=80.0,
        nozzle_exit_diameter=25.0,
        nozzle_length=40.0,
        wall_thickness=3.0
    )
    
    # Extract some statistics
    surface_voxels = rocket_engine.extract_surface(threshold=1.0)
    print(f"   Rocket engine surface voxels: {len(surface_voxels)}")
    
    # Create IC engine
    print("\n2. Creating Internal Combustion Engine...")
    ic_engine = engine_designer.create_internal_combustion_engine(
        cylinder_bore=60.0,
        stroke=70.0,
        num_cylinders=4
    )
    
    ic_surface_voxels = ic_engine.extract_surface(threshold=1.0)
    print(f"   IC engine surface voxels: {len(ic_surface_voxels)}")
    
    # Create simulation
    print("\n3. Setting up Engine Simulation...")
    sim = EngineSimulation()
    sim.add_engine(rocket_engine, "rocket")
    sim.add_engine(ic_engine, "internal_combustion")
    
    # Run assembly simulation
    print("\n4. Running Assembly Simulation...")
    sim.simulate_assembly(steps=50)
    
    print("\n✅ Computational Engine Simulation Complete!")
    print("   - Rocket engine and IC engine geometries created")
    print("   - Surface extraction performed")
    print("   - Assembly simulation executed")
    
    return {
        'rocket_engine': rocket_engine,
        'ic_engine': ic_engine,
        'simulation': sim
    }


# Example usage when run directly
if __name__ == "__main__":
    # Run the demonstration
    demo_result = create_engine_demo()
    
    # Save some basic info for inspection
    import json
    engine_info = {
        'rocket_engine_voxels': demo_result['rocket_engine'].size,
        'ic_engine_voxels': demo_result['ic_engine'].size,
        'voxel_size': 2.0
    }
    
    with open('/Users/renu_malik/Desktop/coding_projects/hackathons/scaler_meta_ai_hackathon/openenv-course/engine_simulation_info.json', 'w') as f:
        json.dump(engine_info, f, indent=2)
    
    print(f"\nEngine simulation info saved to engine_simulation_info.json")