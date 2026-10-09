"""Expand compact model output into easier-to-consume descriptive structures.

The neural networks train on compact JSON to fit a byte-token context window.
This module expands shorthand into named fields for downstream tools. It does
not resolve object IDs, change coordinates into Geometry Dash editor units, or
claim that any plan is playable.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any


def _rename(mapping: dict[str, Any], short: str, long: str) -> None:
    if short in mapping and long not in mapping:
        mapping[long] = mapping.pop(short)


def normalize_deconet(value: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(value)
    for section in out.get("sections", []) if isinstance(out.get("sections"), list) else []:
        if not isinstance(section, dict):
            continue
        for short, long in [("n","name"),("x","x_range"),("m","motif"),("l","lighting"),("t","transition")]:
            _rename(section, short, long)
    for obj in out.get("decorative_objects", []) if isinstance(out.get("decorative_objects"), list) else []:
        if not isinstance(obj, dict):
            continue
        xy = obj.pop("xy", None)
        if isinstance(xy, list) and len(xy) >= 2:
            obj.setdefault("anchor", {"x": xy[0], "y": xy[1]})
        _rename(obj,"color","palette_role")
        _rename(obj,"lookup_required","requires_catalog_lookup")
    out["coordinates_note"] = "abstract planning anchors; convert via a tested coordinate mapping before GD serialization"
    return out


def normalize_gdcore(value: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(value)
    task = out.get("task_type")
    spawn = out.get("spawn_safe_zone")
    if isinstance(spawn, dict):
        _rename(spawn,"x","x_range")
        _rename(spawn,"clear","clear_overhead")
        _rename(spawn,"floor","stable_ground")
    if isinstance(out.get("rooms"), list):
        for room in out["rooms"]:
            if not isinstance(room, dict):
                continue
            _rename(room,"p","platforms")
            _rename(room,"h","hazards")
            _rename(room,"x","x_range")
            if isinstance(room.get("platforms"), list):
                room["platforms"] = [
                    {"x1": p[0], "x2": p[1], "y": p[2]} if isinstance(p,list) and len(p)>=3 else p
                    for p in room["platforms"]
                ]
            if isinstance(room.get("hazards"), list):
                room["hazards"] = [
                    {"kind": h[0], "x": h[1], "y": h[2], "requires_catalog_lookup": True}
                    if isinstance(h,list) and len(h)>=3 else h
                    for h in room["hazards"]
                ]
    coins = out.pop("optional_coins", None)
    if isinstance(coins,list):
        out["collectibles"] = [{"kind":"coin route marker","x":c[0],"y":c[1],"optional":True,"requires_catalog_lookup":True} for c in coins if isinstance(c,list) and len(c)>=2]
    if task == "platformer_plan" and isinstance(out.get("mechanics"),list):
        out["mechanics"] = [
            {"name":m,"status":"planned","object_ids":"requires verified catalog lookup","trigger_setup":"needs verification"}
            if isinstance(m,str) else m for m in out["mechanics"]
        ]
    elif task == "trigger_plan" and isinstance(out.get("id_ledger"),list):
        out["id_ledger"] = [
            {"symbol":entry[0],"type":entry[1],"numeric_id":entry[2],"status":entry[3]}
            if isinstance(entry,list) and len(entry)>=4 else entry for entry in out["id_ledger"]
        ]
    if task in {"title_screen","shop_ui"}:
        out["coordinate_system"] = "normalized 0..1 rectangles; convert to target screen size during rendering"
    out["implementation_note"] = "design plan only; verify catalog fields, serializer output, and behavior in Geometry Dash 2.2081"
    return out


def normalize_output(model_name: str, value: dict[str, Any]) -> dict[str, Any]:
    if model_name == "deconet":
        return normalize_deconet(value)
    if model_name == "gdcore":
        return normalize_gdcore(value)
    raise ValueError(f"unknown model: {model_name}")
