import xml.etree.ElementTree as ET
import pytest

from gdai.render_svg import render_deco_plan_to_svg


def test_deconet_plan_renders_safe_svg_preview():
    plan = {
        "theme": "danger <zone> & neon",
        "mood": "eerie & electric",
        "palette": {"bg": "#071426", "primary": "#22D3EE", "secondary": "#6366F1", "accent": "#F472B6", "hazard": "#F97316"},
        "sections": [
            {"n": "intro", "x": [0, 120], "m": "hexagonal frames", "l": "cyan rim", "t": "carry the shapes"},
            {"n": "build & rise", "x": [120, 260], "m": "ring clusters", "l": "soft cones", "t": "palette flips"},
            {"n": "drop", "x": [260, 420], "m": "broken circuit traces", "l": "amber pulses", "t": "foreground wipe"},
            {"n": "outro", "x": [420, 520], "m": "layered arches", "l": "cool haze", "t": "leave negative space"},
        ],
        "decorative_objects": [
            {"role": "hero centerpiece", "xy": [400, 120], "color": "accent", "lookup_required": True},
        ],
    }
    svg = render_deco_plan_to_svg(plan)
    root = ET.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "danger &lt;zone&gt; &amp; neon" in svg  # Untrusted strings are escaped before embedding in XML.
    assert "SVG CONCEPT PREVIEW" in svg
    assert "#F97316" in svg


def test_svg_falls_back_from_untrusted_color_and_rejects_tiny_canvas():
    svg = render_deco_plan_to_svg({"theme": "x", "palette": {"bg": "url(javascript:bad)"}, "sections": []})
    assert 'stop-color="#071426"' in svg
    with pytest.raises(ValueError):
        render_deco_plan_to_svg({}, width=100, height=100)
