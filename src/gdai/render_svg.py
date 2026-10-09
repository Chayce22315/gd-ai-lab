"""Render a decoNET semantic plan as a self-contained SVG concept preview.

This creates an art-direction preview, not Geometry Dash editor objects or a .gmd file.
"""
from __future__ import annotations
import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _color(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and _HEX.fullmatch(value) else fallback


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number == number and abs(number) != float("inf") else fallback


def _text(value: Any, fallback: str) -> str:
    value = str(value if value is not None else fallback)
    return html.escape(value[:160], quote=True)


def render_deco_plan_to_svg(plan: dict[str, Any], width: int = 1200, height: int = 420) -> str:
    """Convert an AI decoration plan into a safe, self-contained SVG scene preview."""
    if width < 320 or height < 160:
        raise ValueError("preview dimensions must be at least 320x160")
    palette = plan.get("palette") if isinstance(plan.get("palette"), dict) else {}
    bg = _color(palette.get("bg"), "#071426")
    primary = _color(palette.get("primary"), "#22D3EE")
    secondary = _color(palette.get("secondary"), "#6366F1")
    accent = _color(palette.get("accent"), "#F472B6")
    hazard = _color(palette.get("hazard"), "#F97316")
    title_raw = str(plan.get("theme", "decoNET concept preview"))[:160]
    mood_raw = str(plan.get("mood", "art direction preview"))[:160]
    title = html.escape(title_raw, quote=True)
    mood = html.escape(mood_raw, quote=True)
    title_caps = html.escape(title_raw.upper(), quote=True)
    mood_caps = html.escape(mood_raw.upper(), quote=True)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{title} | decoNET concept preview</title>',
        f'<desc id="desc">{mood}. Abstract SVG preview generated from a semantic decoration plan; not a Geometry Dash level export.</desc>',
        '<defs>',
        f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{bg}"/><stop offset="1" stop-color="#030611"/></linearGradient>',
        f'<linearGradient id="beam" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{primary}" stop-opacity=".52"/><stop offset="1" stop-color="{primary}" stop-opacity="0"/></linearGradient>',
        f'<linearGradient id="floor" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{secondary}" stop-opacity=".06"/><stop offset=".5" stop-color="{primary}" stop-opacity=".28"/><stop offset="1" stop-color="{accent}" stop-opacity=".08"/></linearGradient>',
        f'<filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        f'<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="{primary}" stroke-opacity=".10" stroke-width="1"/></pattern>',
        '</defs>',
        '<rect width="100%" height="100%" fill="url(#bg)"/>',
        '<rect width="100%" height="100%" fill="url(#grid)"/>',
        f'<text x="28" y="32" fill="#F8FAFC" font-family="Segoe UI,Arial,sans-serif" font-size="18" font-weight="700" letter-spacing="2">{title_caps}</text>',
        f'<text x="29" y="53" fill="{primary}" font-family="Segoe UI,Arial,sans-serif" font-size="10" letter-spacing="3">{mood_caps}</text>',
    ]

    sections = plan.get("sections", [])
    if not isinstance(sections, list):
        sections = []
    scene_width = width - 48
    origin_x = 24
    base_y = height * 0.80
    endpoints = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        candidate = section.get("x") or section.get("x_range")
        if isinstance(candidate, list) and len(candidate) >= 2:
            endpoints.append(_number(candidate[1], 520))
    world_max = max([520.0, *endpoints, 1.0])
    section_fills = [secondary, primary, accent, secondary]
    motifs = []

    for idx, section in enumerate(sections[:8]):
        if not isinstance(section, dict):
            continue
        x_range = section.get("x") or section.get("x_range") or [idx * world_max / max(1, len(sections)), (idx + 1) * world_max / max(1, len(sections))]
        if not isinstance(x_range, list) or len(x_range) < 2:
            continue
        x1 = origin_x + max(0, min(world_max, _number(x_range[0]))) / world_max * scene_width
        x2 = origin_x + max(0, min(world_max, _number(x_range[1], world_max))) / world_max * scene_width
        if x2 <= x1:
            continue
        color = section_fills[idx % len(section_fills)]
        name = section.get("n", section.get("name", f"section {idx + 1}"))
        motif = section.get("m", section.get("motif", "geometric frames"))
        lighting = section.get("l", section.get("lighting", "soft rim light"))
        transition = section.get("t", section.get("transition", "carry the motif forward"))
        center = (x1 + x2) / 2
        section_width = x2 - x1
        parts.append(f'<rect x="{x1:.2f}" y="70" width="{section_width:.2f}" height="{height - 100:.2f}" fill="{color}" fill-opacity=".035"/>')
        parts.append(f'<line x1="{x1:.2f}" y1="76" x2="{x1:.2f}" y2="{base_y:.2f}" stroke="{color}" stroke-opacity=".36" stroke-width="1"/>')
        parts.append(f'<text x="{x1 + 8:.2f}" y="76" fill="{color}" fill-opacity=".92" font-family="Segoe UI,Arial,sans-serif" font-size="10" letter-spacing="2">{html.escape(str(name).upper()[:160], quote=True)}</text>')
        # A pair of nested angular arches as a repeatable style motif.
        arch_w = max(34.0, min(104.0, section_width * .58))
        arch_h = max(34.0, min(100.0, height * .24))
        arch_top = base_y - arch_h
        for layer, inset, alpha in [(0, 0, .72), (1, 10, .38), (2, 20, .19)]:
            aw = arch_w - inset * 2
            if aw > 8:
                ax = center - aw / 2
                ay = arch_top + inset * .3
                parts.append(f'<path d="M {ax:.2f} {base_y:.2f} L {ax:.2f} {ay + 17:.2f} L {center:.2f} {ay:.2f} L {ax + aw:.2f} {ay + 17:.2f} L {ax + aw:.2f} {base_y:.2f}" fill="none" stroke="{color}" stroke-opacity="{alpha}" stroke-width="2" filter="url(#glow)"/>')
        # Motif-specific accents are deliberately generic vector forms.
        for j in range(4):
            xx = x1 + section_width * (j + .5) / 4
            yy = base_y - 18 - ((j * 13 + idx * 7) % max(24, int(height * .19)))
            if any(word in str(motif).lower() for word in ("hex", "ring", "prism", "crystal")):
                r = 8 + (j % 2) * 4
                pts = " ".join(f"{xx + r * math.cos(k * math.pi / 3):.2f},{yy + r * math.sin(k * math.pi / 3):.2f}" for k in range(6))
                parts.append(f'<polygon points="{pts}" fill="none" stroke="{primary}" stroke-opacity=".55" stroke-width="1.3"/>')
            else:
                parts.append(f'<path d="M {xx - 9:.2f} {yy + 8:.2f} L {xx:.2f} {yy - 8:.2f} L {xx + 9:.2f} {yy + 8:.2f}" fill="none" stroke="{primary}" stroke-opacity=".48" stroke-width="1.4"/>')
            motifs.append((xx, yy))
        # Thin vertical lighting shaft, with sanitized palette color only.
        parts.append(f'<rect x="{center - 1:.2f}" y="88" width="2" height="{max(18, base_y - 102):.2f}" fill="url(#beam)" opacity=".56"/>')
        parts.append(f'<text x="{center:.2f}" y="{base_y + 22:.2f}" text-anchor="middle" fill="#E2E8F0" fill-opacity=".52" font-family="Segoe UI,Arial,sans-serif" font-size="8">{_text(lighting, "lighting")}</text>')
        if idx % 2 == 1:
            for j in range(3):
                hx = x1 + section_width * (.34 + j * .16)
                parts.append(f'<path d="M {hx - 5:.2f} {base_y - 3:.2f} L {hx:.2f} {base_y - 13:.2f} L {hx + 5:.2f} {base_y - 3:.2f}" fill="none" stroke="{hazard}" stroke-opacity=".8" stroke-width="1.8"/>')
        parts.append(f'<title>{_text(transition, "transition")}</title>')

    # Plan anchors are abstract, not guaranteed GD editor coordinates.
    objects = plan.get("decorative_objects", [])
    if isinstance(objects, list):
        color_by_role = {"primary": primary, "secondary": secondary, "accent": accent}
        for obj in objects[:40]:
            if not isinstance(obj, dict):
                continue
            xy = obj.get("xy")
            if not (isinstance(xy, list) and len(xy) >= 2):
                anchor = obj.get("anchor")
                if isinstance(anchor, dict):
                    xy = [anchor.get("x", 0), anchor.get("y", 0)]
            if not (isinstance(xy, list) and len(xy) >= 2):
                continue
            x = origin_x + max(0, min(world_max, _number(xy[0]))) / world_max * scene_width
            y = base_y - max(-60, min(180, _number(xy[1]))) / 180 * height * .54
            color = color_by_role.get(obj.get("color", obj.get("palette_role", "primary")), primary)
            role = _text(obj.get("role", "decorative object"), "decorative object")
            parts.append(f'<g filter="url(#glow)"><circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="{color}" fill-opacity=".12"/><path d="M {x - 12:.2f} {y:.2f} L {x:.2f} {y - 12:.2f} L {x + 12:.2f} {y:.2f} L {x:.2f} {y + 12:.2f} Z" fill="none" stroke="{color}" stroke-width="2"/></g>')
            parts.append(f'<title>{role}</title>')

    # Lower rail and floor glow provide a strong composition baseline.
    parts.extend([
        f'<rect x="0" y="{base_y:.2f}" width="{width}" height="{height - base_y:.2f}" fill="url(#floor)"/>',
        f'<path d="M 0 {base_y:.2f} L {width} {base_y:.2f}" stroke="{primary}" stroke-opacity=".82" stroke-width="2" filter="url(#glow)"/>',
        f'<path d="M 0 {base_y + 7:.2f} L {width} {base_y + 7:.2f}" stroke="{secondary}" stroke-opacity=".28" stroke-width="1"/>',
        f'<text x="{width - 20}" y="{height - 14}" text-anchor="end" fill="#94A3B8" font-family="Segoe UI,Arial,sans-serif" font-size="9" letter-spacing="1.5">SVG CONCEPT PREVIEW · NOT A .GMD EXPORT</text>',
        '</svg>',
    ])
    svg = "\n".join(parts)
    # Catch malformed XML before writing a file.
    ET.fromstring(svg)
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="JSON decoNET plan")
    parser.add_argument("--output", required=True, type=Path, help="output SVG path")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=420)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("plan JSON must be a top-level object")
    svg = render_deco_plan_to_svg(plan, args.width, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"SVG preview written: {args.output.resolve()} ({len(svg)} chars)")


if __name__ == "__main__":
    main()
