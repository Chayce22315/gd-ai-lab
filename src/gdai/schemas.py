"""Lightweight output checks; these validate structure, not gameplay correctness."""
from __future__ import annotations
from typing import Any


def validate_deconet(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["top-level output must be a JSON object"]
    if value.get("schema_version") != "decoNET-plan-0.1":
        errors.append("schema_version must be decoNET-plan-0.1")
    palette = value.get("palette")
    if not isinstance(palette, dict) or not all(key in palette for key in ("bg", "primary", "secondary", "accent", "hazard")):
        errors.append("palette needs bg, primary, secondary, accent, and hazard colors")
    if not isinstance(value.get("sections"), list) or len(value.get("sections", [])) < 3:
        errors.append("sections must contain at least three section plans")
    if not isinstance(value.get("decorative_objects"), list):
        errors.append("decorative_objects must be a list")
    else:
        for i, obj in enumerate(value["decorative_objects"]):
            if not isinstance(obj, dict) or not obj.get("role"):
                errors.append(f"decorative_objects[{i}] needs a semantic role")
            elif obj.get("lookup_required") is not True:
                errors.append(f"decorative_objects[{i}] must require verified catalog lookup")
    return errors


def validate_gdcore(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["top-level output must be a JSON object"]
    if value.get("schema_version") not in {"gdCORE-plan-0.1", "gdCORE-trigger-plan-0.1", "gdCORE-ui-plan-0.1"}:
        errors.append("unknown or missing schema_version")
    task = value.get("task_type")
    if task == "platformer_plan":
        if value.get("mode") != "platformer":
            errors.append("platformer_plan must set mode=platformer")
        if not isinstance(value.get("rooms"), list) or len(value.get("rooms", [])) < 2:
            errors.append("platformer_plan needs at least two rooms")
        if not isinstance(value.get("spawn_safe_zone"), dict):
            errors.append("platformer_plan needs spawn_safe_zone")
    elif task == "classic_layout_plan":
        if not isinstance(value.get("sections"), list) or len(value.get("sections", [])) < 3:
            errors.append("classic_layout_plan needs at least three sections")
    elif task == "trigger_plan":
        if not isinstance(value.get("unknowns_to_verify"), list):
            errors.append("trigger_plan needs unknowns_to_verify")
        if not isinstance(value.get("id_ledger"), list):
            errors.append("trigger_plan needs id_ledger")
    elif task in {"title_screen", "shop_ui"}:
        if not isinstance(value.get("elements"), list) or len(value.get("elements", [])) < 2:
            errors.append("UI plans need at least two elements")
    else:
        errors.append(f"unsupported task_type: {task!r}")
    return errors


def validate_output(model_name: str, value: Any) -> list[str]:
    if model_name == "deconet":
        return validate_deconet(value)
    if model_name == "gdcore":
        return validate_gdcore(value)
    return [f"unknown model name: {model_name}"]
