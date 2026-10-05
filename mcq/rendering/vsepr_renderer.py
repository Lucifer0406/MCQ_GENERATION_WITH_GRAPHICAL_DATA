"""VSEPR molecular geometry renderer — generates 2D drawings with stereochemical notation (wedges & dashes).

Supports all NCERT VSEPR geometries based on steric number (SN 2 to 6):
  • SN 2: Linear (flat 2D)
  • SN 3: Trigonal planar, Bent (flat 2D, lone pairs)
  • SN 4: Tetrahedral, Trigonal pyramidal, Bent (stereochemical wedges & dashes)
  • SN 5: Trigonal bipyramidal, Seesaw, T-shaped, Linear (axial + equatorial wedges/dashes)
  • SN 6: Octahedral, Square pyramidal, Square planar (stereochemical square-plane wedges/dashes)
"""

from __future__ import annotations

import html
import math
import re
import streamlit.components.v1 as components

# ── Geometry Metadata Dictionary ─────────────────────────────────────────────
_VSEPR_REGISTRY: dict[str, dict] = {
    # SN = 2 (Linear)
    "becl2": {"central": "Be", "ligands": ["Cl", "Cl"], "geometry": "linear", "sn": 2, "lp": 0, "angle": "180°", "name": "Beryllium Chloride"},
    "bef2":  {"central": "Be", "ligands": ["F", "F"],   "geometry": "linear", "sn": 2, "lp": 0, "angle": "180°", "name": "Beryllium Fluoride"},
    "co2":   {"central": "C",  "ligands": ["O", "O"],   "geometry": "linear", "sn": 2, "lp": 0, "angle": "180°", "name": "Carbon Dioxide"},
    "hcn":   {"central": "C",  "ligands": ["H", "N"],   "geometry": "linear", "sn": 2, "lp": 0, "angle": "180°", "name": "Hydrogen Cyanide"},
    "c2h2":  {"central": "C",  "ligands": ["H", "C"],   "geometry": "linear", "sn": 2, "lp": 0, "angle": "180°", "name": "Acetylene"},

    # SN = 3 (Trigonal Planar, Bent)
    "bf3":   {"central": "B",  "ligands": ["F", "F", "F"],   "geometry": "trigonal_planar", "sn": 3, "lp": 0, "angle": "120°", "name": "Boron Trifluoride"},
    "bcl3":  {"central": "B",  "ligands": ["Cl", "Cl", "Cl"], "geometry": "trigonal_planar", "sn": 3, "lp": 0, "angle": "120°", "name": "Boron Trichloride"},
    "alcl3": {"central": "Al", "ligands": ["Cl", "Cl", "Cl"], "geometry": "trigonal_planar", "sn": 3, "lp": 0, "angle": "120°", "name": "Aluminium Chloride"},
    "so3":   {"central": "S",  "ligands": ["O", "O", "O"],   "geometry": "trigonal_planar", "sn": 3, "lp": 0, "angle": "120°", "name": "Sulfur Trioxide"},
    "so2":   {"central": "S",  "ligands": ["O", "O"],       "geometry": "bent_sn3",        "sn": 3, "lp": 1, "angle": "119.5°", "name": "Sulfur Dioxide"},
    "o3":    {"central": "O",  "ligands": ["O", "O"],       "geometry": "bent_sn3",        "sn": 3, "lp": 1, "angle": "117°", "name": "Ozone"},
    "no2-":  {"central": "N",  "ligands": ["O", "O"],       "geometry": "bent_sn3",        "sn": 3, "lp": 1, "angle": "115°", "name": "Nitrite Ion"},

    # SN = 4 (Tetrahedral, Trigonal Pyramidal, Bent)
    "ch4":   {"central": "C",  "ligands": ["H", "H", "H", "H"],     "geometry": "tetrahedral",        "sn": 4, "lp": 0, "angle": "109.5°", "name": "Methane"},
    "ccl4":  {"central": "C",  "ligands": ["Cl", "Cl", "Cl", "Cl"], "geometry": "tetrahedral",        "sn": 4, "lp": 0, "angle": "109.5°", "name": "Carbon Tetrachloride"},
    "cf4":   {"central": "C",  "ligands": ["F", "F", "F", "F"],     "geometry": "tetrahedral",        "sn": 4, "lp": 0, "angle": "109.5°", "name": "Carbon Tetrafluoride"},
    "sih4":  {"central": "Si", "ligands": ["H", "H", "H", "H"],     "geometry": "tetrahedral",        "sn": 4, "lp": 0, "angle": "109.5°", "name": "Silane"},
    "sicl4": {"central": "Si", "ligands": ["Cl", "Cl", "Cl", "Cl"], "geometry": "tetrahedral",        "sn": 4, "lp": 0, "angle": "109.5°", "name": "Silicon Tetrachloride"},
    "nh4+":  {"central": "N",  "ligands": ["H", "H", "H", "H"],     "geometry": "tetrahedral",        "sn": 4, "lp": 0, "angle": "109.5°", "name": "Ammonium Ion"},

    "nh3":   {"central": "N",  "ligands": ["H", "H", "H"],     "geometry": "trigonal_pyramidal", "sn": 4, "lp": 1, "angle": "107°", "name": "Ammonia"},
    "nf3":   {"central": "N",  "ligands": ["F", "F", "F"],     "geometry": "trigonal_pyramidal", "sn": 4, "lp": 1, "angle": "102.5°", "name": "Nitrogen Trifluoride"},
    "pcl3":  {"central": "P",  "ligands": ["Cl", "Cl", "Cl"], "geometry": "trigonal_pyramidal", "sn": 4, "lp": 1, "angle": "100°", "name": "Phosphorus Trichloride"},
    "h3o+":  {"central": "O",  "ligands": ["H", "H", "H"],     "geometry": "trigonal_pyramidal", "sn": 4, "lp": 1, "angle": "107°", "name": "Hydronium Ion"},

    "h2o":   {"central": "O",  "ligands": ["H", "H"],   "geometry": "bent_sn4", "sn": 4, "lp": 2, "angle": "104.5°", "name": "Water"},
    "h2s":   {"central": "S",  "ligands": ["H", "H"],   "geometry": "bent_sn4", "sn": 4, "lp": 2, "angle": "92°", "name": "Hydrogen Sulfide"},
    "of2":   {"central": "O",  "ligands": ["F", "F"],   "geometry": "bent_sn4", "sn": 4, "lp": 2, "angle": "103°", "name": "Oxygen Difluoride"},
    "scl2":  {"central": "S",  "ligands": ["Cl", "Cl"], "geometry": "bent_sn4", "sn": 4, "lp": 2, "angle": "103°", "name": "Sulfur Dichloride"},

    # SN = 5 (Trigonal Bipyramidal, Seesaw, T-shaped, Linear)
    "pcl5":  {"central": "P",  "ligands": ["Cl", "Cl", "Cl", "Cl", "Cl"], "geometry": "trigonal_bipyramidal", "sn": 5, "lp": 0, "angle": "90° / 120°", "name": "Phosphorus Pentachloride"},
    "pf5":   {"central": "P",  "ligands": ["F", "F", "F", "F", "F"],       "geometry": "trigonal_bipyramidal", "sn": 5, "lp": 0, "angle": "90° / 120°", "name": "Phosphorus Pentafluoride"},
    "asf5":  {"central": "As", "ligands": ["F", "F", "F", "F", "F"],       "geometry": "trigonal_bipyramidal", "sn": 5, "lp": 0, "angle": "90° / 120°", "name": "Arsenic Pentafluoride"},

    "sf4":   {"central": "S",  "ligands": ["F", "F", "F", "F"], "geometry": "seesaw",   "sn": 5, "lp": 1, "angle": "102° / 173°", "name": "Sulfur Tetrafluoride"},
    "clf3":  {"central": "Cl", "ligands": ["F", "F", "F"],       "geometry": "t_shaped", "sn": 5, "lp": 2, "angle": "87.5°", "name": "Chlorine Trifluoride"},
    "brf3":  {"central": "Br", "ligands": ["F", "F", "F"],       "geometry": "t_shaped", "sn": 5, "lp": 2, "angle": "86°", "name": "Bromine Trifluoride"},
    "xef2":  {"central": "Xe", "ligands": ["F", "F"],           "geometry": "linear_sn5", "sn": 5, "lp": 3, "angle": "180°", "name": "Xenon Difluoride"},
    "i3-":   {"central": "I",  "ligands": ["I", "I"],           "geometry": "linear_sn5", "sn": 5, "lp": 3, "angle": "180°", "name": "Triiodide Ion"},

    # SN = 6 (Octahedral, Square Pyramidal, Square Planar)
    "sf6":   {"central": "S",  "ligands": ["F", "F", "F", "F", "F", "F"], "geometry": "octahedral",        "sn": 6, "lp": 0, "angle": "90°", "name": "Sulfur Hexafluoride"},
    "tef6":  {"central": "Te", "ligands": ["F", "F", "F", "F", "F", "F"], "geometry": "octahedral",        "sn": 6, "lp": 0, "angle": "90°", "name": "Tellurium Hexafluoride"},
    "pf6-":  {"central": "P",  "ligands": ["F", "F", "F", "F", "F", "F"], "geometry": "octahedral",        "sn": 6, "lp": 0, "angle": "90°", "name": "Hexafluorophosphate Ion"},

    "brf5":  {"central": "Br", "ligands": ["F", "F", "F", "F", "F"],     "geometry": "square_pyramidal",  "sn": 6, "lp": 1, "angle": "84.8°", "name": "Bromine Pentafluoride"},
    "if5":   {"central": "I",  "ligands": ["F", "F", "F", "F", "F"],     "geometry": "square_pyramidal",  "sn": 6, "lp": 1, "angle": "82°", "name": "Iodine Pentafluoride"},

    "xef4":  {"central": "Xe", "ligands": ["F", "F", "F", "F"],           "geometry": "square_planar",     "sn": 6, "lp": 2, "angle": "90°", "name": "Xenon Tetrafluoride"},

    # Generic VSEPR Formulas
    "ab2":   {"central": "A", "ligands": ["B", "B"], "geometry": "linear", "sn": 2, "lp": 0, "angle": "180°", "name": "Linear (AB₂)"},
    "ab3":   {"central": "A", "ligands": ["B", "B", "B"], "geometry": "trigonal_planar", "sn": 3, "lp": 0, "angle": "120°", "name": "Trigonal Planar (AB₃)"},
    "ab2e":  {"central": "A", "ligands": ["B", "B"], "geometry": "bent_sn3", "sn": 3, "lp": 1, "angle": "<120°", "name": "Bent (AB₂E)"},
    "ab4":   {"central": "A", "ligands": ["B", "B", "B", "B"], "geometry": "tetrahedral", "sn": 4, "lp": 0, "angle": "109.5°", "name": "Tetrahedral (AB₄)"},
    "ab3e":  {"central": "A", "ligands": ["B", "B", "B"], "geometry": "trigonal_pyramidal", "sn": 4, "lp": 1, "angle": "<109.5°", "name": "Trigonal Pyramidal (AB₃E)"},
    "ab2e2": {"central": "A", "ligands": ["B", "B"], "geometry": "bent_sn4", "sn": 4, "lp": 2, "angle": "<109.5°", "name": "Bent (AB₂E₂)"},
    "ab5":   {"central": "A", "ligands": ["B", "B", "B", "B", "B"], "geometry": "trigonal_bipyramidal", "sn": 5, "lp": 0, "angle": "90° / 120°", "name": "Trigonal Bipyramidal (AB₅)"},
    "ab4e":  {"central": "A", "ligands": ["B", "B", "B", "B"], "geometry": "seesaw", "sn": 5, "lp": 1, "angle": "<90° / <120°", "name": "Seesaw (AB₄E)"},
    "ab3e2": {"central": "A", "ligands": ["B", "B", "B"], "geometry": "t_shaped", "sn": 5, "lp": 2, "angle": "<90°", "name": "T-shaped (AB₃E₂)"},
    "ab2e3": {"central": "A", "ligands": ["B", "B"], "geometry": "linear_sn5", "sn": 5, "lp": 3, "angle": "180°", "name": "Linear (AB₂E₃)"},
    "ab6":   {"central": "A", "ligands": ["B", "B", "B", "B", "B", "B"], "geometry": "octahedral", "sn": 6, "lp": 0, "angle": "90°", "name": "Octahedral (AB₆)"},
    "ab5e":  {"central": "A", "ligands": ["B", "B", "B", "B", "B"], "geometry": "square_pyramidal", "sn": 6, "lp": 1, "angle": "<90°", "name": "Square Pyramidal (AB₅E)"},
    "ab4e2": {"central": "A", "ligands": ["B", "B", "B", "B"], "geometry": "square_planar", "sn": 6, "lp": 2, "angle": "90°", "name": "Square Planar (AB₄E₂)"},
}


def _clean_key(text: str) -> str:
    """Normalize formula or name to match dictionary keys."""
    t = text.lower().strip()
    t = re.sub(r'[\s\-_]', '', t)
    t = t.replace('molecule', '').replace('geometry', '').replace('structure', '')
    return t


def is_vsepr_molecule(smiles: str, name: str | None = None) -> bool:
    """Check if input corresponds to a known VSEPR molecular geometry."""
    candidates = [_clean_key(smiles)]
    if name:
        candidates.append(_clean_key(name))
    for c in candidates:
        if c in _VSEPR_REGISTRY:
            return True
        if c.startswith("ab") and any(c.startswith(k) for k in ("ab2", "ab3", "ab4", "ab5", "ab6")):
            return True
    return False


# ── SVG Vector Primitives ───────────────────────────────────────────────────

def _draw_solid_line(x1: float, y1: float, x2: float, y2: float, offset: float = 14) -> str:
    """Normal in-plane bond line with atom clearance."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return ""
    ux, uy = dx / L, dy / L
    sx, sy = x1 + offset * ux, y1 + offset * uy
    ex, ey = x2 - offset * ux, y2 - offset * uy
    return f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#2c2825" stroke-width="2.2" stroke-linecap="round" />'


def _draw_wedge(x1: float, y1: float, x2: float, y2: float, width: float = 7.5, offset: float = 14) -> str:
    """Solid stereochemical wedge bond (projecting forward out of the page)."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return ""
    ux, uy = dx / L, dy / L
    px, py = -uy, ux  # perpendicular normal

    tip_x, tip_y = x1 + offset * ux, y1 + offset * uy
    base_center_x, base_center_y = x2 - offset * ux, y2 - offset * uy

    p1_x = base_center_x + width * px
    p1_y = base_center_y + width * py
    p2_x = base_center_x - width * px
    p2_y = base_center_y - width * py

    return f'<polygon points="{tip_x:.1f},{tip_y:.1f} {p1_x:.1f},{p1_y:.1f} {p2_x:.1f},{p2_y:.1f}" fill="#2c2825" />'


def _draw_dash(x1: float, y1: float, x2: float, y2: float, n_dashes: int = 5, w_max: float = 7.5, offset: float = 14) -> str:
    """Hashed dashed wedge bond (projecting backward into the page)."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return ""
    ux, uy = dx / L, dy / L
    px, py = -uy, ux

    lines = []
    for i in range(n_dashes):
        frac = 0.28 + 0.52 * (i / max(1, n_dashes - 1))
        mx = x1 + frac * dx
        my = y1 + frac * dy
        w = 2.0 + (w_max - 2.0) * (i / max(1, n_dashes - 1))
        p1 = (mx + w * px, my + w * py)
        p2 = (mx - w * px, my - w * py)
        lines.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" stroke="#2c2825" stroke-width="1.8" />')
    return '\n  '.join(lines)


def _draw_lone_pair_lobe(cx: float, cy: float, angle_deg: float, dist: float = 28.0) -> str:
    """NCERT-style lone pair orbital lobe with electron pair dots."""
    rad = math.radians(angle_deg)
    lx = cx + dist * math.cos(rad)
    ly = cy + dist * math.sin(rad)
    px, py = -math.sin(rad), math.cos(rad)
    d1 = (lx + 3.8 * px, ly + 3.8 * py)
    d2 = (lx - 3.8 * px, ly - 3.8 * py)

    lobe = f'<ellipse cx="{lx:.1f}" cy="{ly:.1f}" rx="14" ry="8" transform="rotate({angle_deg} {lx:.1f} {ly:.1f})" fill="rgba(139,26,26,0.12)" stroke="#8b1a1a" stroke-width="1.2" stroke-dasharray="2.5,2.5" />'
    dot1 = f'<circle cx="{d1[0]:.1f}" cy="{d1[1]:.1f}" r="2" fill="#8b1a1a" />'
    dot2 = f'<circle cx="{d2[0]:.1f}" cy="{d2[1]:.1f}" r="2" fill="#8b1a1a" />'
    return f"{lobe}\n  {dot1}\n  {dot2}"


def _draw_atom(x: float, y: float, symbol: str, is_central: bool = False) -> str:
    """Render atom label text."""
    size = 18 if is_central else 15
    weight = 700 if is_central else 600
    color = "#8b1a1a" if is_central else "#1e1a18"
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="central" font-family="\'EB Garamond\', Georgia, serif" font-size="{size}" font-weight="{weight}" fill="{color}">{symbol}</text>'


# ── Geometry SVG Builder ────────────────────────────────────────────────────

def generate_vsepr_svg(info: dict) -> str:
    """Generate complete SVG markup for a VSEPR geometry."""
    cx, cy = 170.0, 95.0
    R = 52.0
    geom = info["geometry"]
    central = info["central"]
    ligands = info["ligands"]
    name = info.get("name", "")
    angle = info.get("angle", "")
    sn = info.get("sn", 4)
    lp = info.get("lp", 0)

    elements: list[str] = []

    # 1. Linear (SN = 2, 0 LP)
    if geom == "linear":
        l_x = ligands[0] if len(ligands) > 0 else "B"
        r_x = ligands[1] if len(ligands) > 1 else "B"
        elements.append(_draw_solid_line(cx, cy, cx - R, cy))
        elements.append(_draw_solid_line(cx, cy, cx + R, cy))
        elements.append(_draw_atom(cx - R, cy, l_x))
        elements.append(_draw_atom(cx + R, cy, r_x))
        # 180° arc
        elements.append(f'<path d="M {cx-22:.1f},{cy:.1f} A 22 22 0 0 1 {cx+22:.1f},{cy:.1f}" fill="none" stroke="#a08060" stroke-width="1.2" stroke-dasharray="2,2" />')

    # 2. Trigonal Planar (SN = 3, 0 LP)
    elif geom == "trigonal_planar":
        p_top = (cx, cy - R)
        p_bl = (cx - R * math.cos(math.radians(30)), cy + R * math.sin(math.radians(30)))
        p_br = (cx + R * math.cos(math.radians(30)), cy + R * math.sin(math.radians(30)))
        pts = [p_top, p_bl, p_br]
        for i, pt in enumerate(pts):
            sym = ligands[i] if i < len(ligands) else "B"
            elements.append(_draw_solid_line(cx, cy, pt[0], pt[1]))
            elements.append(_draw_atom(pt[0], pt[1], sym))

    # 3. Bent (SN = 3, 1 LP)
    elif geom == "bent_sn3":
        elements.append(_draw_lone_pair_lobe(cx, cy, -90))
        p_bl = (cx - R * math.cos(math.radians(30)), cy + R * math.sin(math.radians(30)))
        p_br = (cx + R * math.cos(math.radians(30)), cy + R * math.sin(math.radians(30)))
        pts = [p_bl, p_br]
        for i, pt in enumerate(pts):
            sym = ligands[i] if i < len(ligands) else "B"
            elements.append(_draw_solid_line(cx, cy, pt[0], pt[1]))
            elements.append(_draw_atom(pt[0], pt[1], sym))

    # 4. Tetrahedral (SN = 4, 0 LP) — Stereochemical wedges & dashes
    elif geom == "tetrahedral":
        p_top = (cx, cy - R)
        p_left = (cx - R * math.cos(math.radians(18)), cy + R * math.sin(math.radians(18)))
        p_wedge = (cx + R * math.cos(math.radians(65)), cy + R * math.sin(math.radians(65)))
        p_dash = (cx + R * math.cos(math.radians(12)), cy + R * math.sin(math.radians(12)) - 4)

        elements.append(_draw_solid_line(cx, cy, p_top[0], p_top[1]))
        elements.append(_draw_solid_line(cx, cy, p_left[0], p_left[1]))
        elements.append(_draw_wedge(cx, cy, p_wedge[0], p_wedge[1], width=8))
        elements.append(_draw_dash(cx, cy, p_dash[0], p_dash[1], n_dashes=5, w_max=7.5))

        elements.append(_draw_atom(p_top[0], p_top[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_left[0], p_left[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_wedge[0], p_wedge[1], ligands[2] if len(ligands) > 2 else "B"))
        elements.append(_draw_atom(p_dash[0], p_dash[1], ligands[3] if len(ligands) > 3 else "B"))

    # 5. Trigonal Pyramidal (SN = 4, 1 LP)
    elif geom == "trigonal_pyramidal":
        elements.append(_draw_lone_pair_lobe(cx, cy, -90))
        p_left = (cx - 0.82 * R, cy + 0.65 * R)
        p_wedge = (cx - 0.12 * R, cy + 0.92 * R)
        p_dash = (cx + 0.78 * R, cy + 0.65 * R)

        elements.append(_draw_solid_line(cx, cy, p_left[0], p_left[1]))
        elements.append(_draw_wedge(cx, cy, p_wedge[0], p_wedge[1], width=8))
        elements.append(_draw_dash(cx, cy, p_dash[0], p_dash[1], n_dashes=5, w_max=7.5))

        elements.append(_draw_atom(p_left[0], p_left[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_wedge[0], p_wedge[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_dash[0], p_dash[1], ligands[2] if len(ligands) > 2 else "B"))

    # 6. Bent (SN = 4, 2 LP)
    elif geom == "bent_sn4":
        elements.append(_draw_lone_pair_lobe(cx, cy, -135))
        elements.append(_draw_lone_pair_lobe(cx, cy, -45))
        p_bl = (cx - 0.78 * R, cy + 0.68 * R)
        p_br = (cx + 0.78 * R, cy + 0.68 * R)

        elements.append(_draw_solid_line(cx, cy, p_bl[0], p_bl[1]))
        elements.append(_draw_solid_line(cx, cy, p_br[0], p_br[1]))

        elements.append(_draw_atom(p_bl[0], p_bl[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_br[0], p_br[1], ligands[1] if len(ligands) > 1 else "B"))

    # 7. Trigonal Bipyramidal (SN = 5, 0 LP) — Axial + Equatorial wedges/dashes
    elif geom == "trigonal_bipyramidal":
        # 2 Axial bonds (longer)
        p_ax_up = (cx, cy - R - 10)
        p_ax_dn = (cx, cy + R + 10)
        # 3 Equatorial bonds
        p_eq_left = (cx - R, cy)
        p_eq_wedge = (cx + 0.72 * R, cy + 0.42 * R)
        p_eq_dash = (cx + 0.72 * R, cy - 0.42 * R)

        elements.append(_draw_solid_line(cx, cy, p_ax_up[0], p_ax_up[1]))
        elements.append(_draw_solid_line(cx, cy, p_ax_dn[0], p_ax_dn[1]))
        elements.append(_draw_solid_line(cx, cy, p_eq_left[0], p_eq_left[1]))
        elements.append(_draw_wedge(cx, cy, p_eq_wedge[0], p_eq_wedge[1], width=8))
        elements.append(_draw_dash(cx, cy, p_eq_dash[0], p_eq_dash[1], n_dashes=5, w_max=7.5))

        elements.append(_draw_atom(p_ax_up[0], p_ax_up[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_ax_dn[0], p_ax_dn[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_eq_left[0], p_eq_left[1], ligands[2] if len(ligands) > 2 else "B"))
        elements.append(_draw_atom(p_eq_wedge[0], p_eq_wedge[1], ligands[3] if len(ligands) > 3 else "B"))
        elements.append(_draw_atom(p_eq_dash[0], p_eq_dash[1], ligands[4] if len(ligands) > 4 else "B"))

    # 8. Seesaw (SN = 5, 1 LP)
    elif geom == "seesaw":
        p_ax_up = (cx, cy - R - 8)
        p_ax_dn = (cx, cy + R + 8)
        elements.append(_draw_solid_line(cx, cy, p_ax_up[0], p_ax_up[1]))
        elements.append(_draw_solid_line(cx, cy, p_ax_dn[0], p_ax_dn[1]))
        elements.append(_draw_lone_pair_lobe(cx, cy, 180))

        p_eq_wedge = (cx + 0.72 * R, cy + 0.42 * R)
        p_eq_dash = (cx + 0.72 * R, cy - 0.42 * R)
        elements.append(_draw_wedge(cx, cy, p_eq_wedge[0], p_eq_wedge[1], width=8))
        elements.append(_draw_dash(cx, cy, p_eq_dash[0], p_eq_dash[1], n_dashes=5, w_max=7.5))

        elements.append(_draw_atom(p_ax_up[0], p_ax_up[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_ax_dn[0], p_ax_dn[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_eq_wedge[0], p_eq_wedge[1], ligands[2] if len(ligands) > 2 else "B"))
        elements.append(_draw_atom(p_eq_dash[0], p_eq_dash[1], ligands[3] if len(ligands) > 3 else "B"))

    # 9. T-shaped (SN = 5, 2 LP)
    elif geom == "t_shaped":
        p_ax_up = (cx, cy - R - 8)
        p_ax_dn = (cx, cy + R + 8)
        p_eq_r = (cx + R, cy)
        elements.append(_draw_solid_line(cx, cy, p_ax_up[0], p_ax_up[1]))
        elements.append(_draw_solid_line(cx, cy, p_ax_dn[0], p_ax_dn[1]))
        elements.append(_draw_solid_line(cx, cy, p_eq_r[0], p_eq_r[1]))
        elements.append(_draw_lone_pair_lobe(cx, cy, 140))
        elements.append(_draw_lone_pair_lobe(cx, cy, 220))

        elements.append(_draw_atom(p_ax_up[0], p_ax_up[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_ax_dn[0], p_ax_dn[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_eq_r[0], p_eq_r[1], ligands[2] if len(ligands) > 2 else "B"))

    # 10. Linear from SN = 5 (SN = 5, 3 LP)
    elif geom == "linear_sn5":
        p_ax_up = (cx, cy - R - 8)
        p_ax_dn = (cx, cy + R + 8)
        elements.append(_draw_solid_line(cx, cy, p_ax_up[0], p_ax_up[1]))
        elements.append(_draw_solid_line(cx, cy, p_ax_dn[0], p_ax_dn[1]))
        elements.append(_draw_lone_pair_lobe(cx, cy, 0))
        elements.append(_draw_lone_pair_lobe(cx, cy, 120))
        elements.append(_draw_lone_pair_lobe(cx, cy, 240))

        elements.append(_draw_atom(p_ax_up[0], p_ax_up[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_ax_dn[0], p_ax_dn[1], ligands[1] if len(ligands) > 1 else "B"))

    # 11. Octahedral (SN = 6, 0 LP) — 2 Axial + 4 Equatorial square wedges/dashes
    elif geom == "octahedral":
        p_ax_up = (cx, cy - R - 10)
        p_ax_dn = (cx, cy + R + 10)
        p_bl = (cx - 0.70 * R, cy + 0.38 * R)
        p_br = (cx + 0.70 * R, cy + 0.38 * R)
        p_tl = (cx - 0.70 * R, cy - 0.38 * R)
        p_tr = (cx + 0.70 * R, cy - 0.38 * R)

        elements.append(_draw_solid_line(cx, cy, p_ax_up[0], p_ax_up[1]))
        elements.append(_draw_solid_line(cx, cy, p_ax_dn[0], p_ax_dn[1]))
        elements.append(_draw_dash(cx, cy, p_tl[0], p_tl[1], n_dashes=5, w_max=7.5))
        elements.append(_draw_dash(cx, cy, p_tr[0], p_tr[1], n_dashes=5, w_max=7.5))
        elements.append(_draw_wedge(cx, cy, p_bl[0], p_bl[1], width=8))
        elements.append(_draw_wedge(cx, cy, p_br[0], p_br[1], width=8))

        elements.append(_draw_atom(p_ax_up[0], p_ax_up[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_ax_dn[0], p_ax_dn[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_tl[0], p_tl[1], ligands[2] if len(ligands) > 2 else "B"))
        elements.append(_draw_atom(p_tr[0], p_tr[1], ligands[3] if len(ligands) > 3 else "B"))
        elements.append(_draw_atom(p_bl[0], p_bl[1], ligands[4] if len(ligands) > 4 else "B"))
        elements.append(_draw_atom(p_br[0], p_br[1], ligands[5] if len(ligands) > 5 else "B"))

    # 12. Square Pyramidal (SN = 6, 1 LP)
    elif geom == "square_pyramidal":
        p_ax_up = (cx, cy - R - 10)
        p_bl = (cx - 0.70 * R, cy + 0.38 * R)
        p_br = (cx + 0.70 * R, cy + 0.38 * R)
        p_tl = (cx - 0.70 * R, cy - 0.38 * R)
        p_tr = (cx + 0.70 * R, cy - 0.38 * R)

        elements.append(_draw_solid_line(cx, cy, p_ax_up[0], p_ax_up[1]))
        elements.append(_draw_dash(cx, cy, p_tl[0], p_tl[1], n_dashes=5, w_max=7.5))
        elements.append(_draw_dash(cx, cy, p_tr[0], p_tr[1], n_dashes=5, w_max=7.5))
        elements.append(_draw_wedge(cx, cy, p_bl[0], p_bl[1], width=8))
        elements.append(_draw_wedge(cx, cy, p_br[0], p_br[1], width=8))
        elements.append(_draw_lone_pair_lobe(cx, cy, 90))

        elements.append(_draw_atom(p_ax_up[0], p_ax_up[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_tl[0], p_tl[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_tr[0], p_tr[1], ligands[2] if len(ligands) > 2 else "B"))
        elements.append(_draw_atom(p_bl[0], p_bl[1], ligands[3] if len(ligands) > 3 else "B"))
        elements.append(_draw_atom(p_br[0], p_br[1], ligands[4] if len(ligands) > 4 else "B"))

    # 13. Square Planar (SN = 6, 2 LP)
    elif geom == "square_planar":
        p_bl = (cx - 0.70 * R, cy + 0.38 * R)
        p_br = (cx + 0.70 * R, cy + 0.38 * R)
        p_tl = (cx - 0.70 * R, cy - 0.38 * R)
        p_tr = (cx + 0.70 * R, cy - 0.38 * R)

        elements.append(_draw_lone_pair_lobe(cx, cy, -90))
        elements.append(_draw_lone_pair_lobe(cx, cy, 90))
        elements.append(_draw_dash(cx, cy, p_tl[0], p_tl[1], n_dashes=5, w_max=7.5))
        elements.append(_draw_dash(cx, cy, p_tr[0], p_tr[1], n_dashes=5, w_max=7.5))
        elements.append(_draw_wedge(cx, cy, p_bl[0], p_bl[1], width=8))
        elements.append(_draw_wedge(cx, cy, p_br[0], p_br[1], width=8))

        elements.append(_draw_atom(p_tl[0], p_tl[1], ligands[0] if len(ligands) > 0 else "B"))
        elements.append(_draw_atom(p_tr[0], p_tr[1], ligands[1] if len(ligands) > 1 else "B"))
        elements.append(_draw_atom(p_bl[0], p_bl[1], ligands[2] if len(ligands) > 2 else "B"))
        elements.append(_draw_atom(p_br[0], p_br[1], ligands[3] if len(ligands) > 3 else "B"))

    # Central atom rendered on top
    elements.append(_draw_atom(cx, cy, central, is_central=True))

    # Caption / Details below
    geom_title = geom.replace('_', ' ').replace('sn3', '').replace('sn4', '').replace('sn5', '').title().strip()
    caption_text = f"{name or central} • {geom_title} (SN={sn}, LP={lp})"
    if angle:
        caption_text += f" • {angle}"

    elements.append(
        f'<text x="170" y="205" text-anchor="middle" font-family="\'Inter\', sans-serif" font-size="11.5" font-weight="500" fill="#6b6460">{html.escape(caption_text)}</text>'
    )

    svg_content = "\n  ".join(elements)
    return f"""<svg width="340" height="220" viewBox="0 0 340 220" xmlns="http://www.w3.org/2000/svg">
  <!-- VSEPR 2D Stereochemical Diagram -->
  <rect width="100%" height="100%" fill="transparent" />
  {svg_content}
</svg>"""


# ── Streamlit Renderer ───────────────────────────────────────────────────────

def render_vsepr(smiles: str, name: str | None = None, height: int = 240) -> None:
    """Render a VSEPR molecular diagram with stereochemical notation in Streamlit."""
    k1 = _clean_key(smiles)
    k2 = _clean_key(name or "")
    
    info = None
    if k1 in _VSEPR_REGISTRY:
        info = _VSEPR_REGISTRY[k1]
    elif k2 in _VSEPR_REGISTRY:
        info = _VSEPR_REGISTRY[k2]
    else:
        # Fallback for generic ABn patterns
        for key in _VSEPR_REGISTRY:
            if k1.startswith(key) or k2.startswith(key):
                info = _VSEPR_REGISTRY[key]
                break

    if not info:
        info = _VSEPR_REGISTRY["ch4"]  # sensible default

    svg_html = generate_vsepr_svg(info)

    html_code = f"""<!DOCTYPE html><html>
<head><meta charset="UTF-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:transparent;display:flex;align-items:center;justify-content:center;height:{height}px}}
  svg{{max-width:100%}}
</style>
</head>
<body>
{svg_html}
</body></html>"""

    components.html(html_code, height=height, scrolling=False)
