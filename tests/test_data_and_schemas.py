import json
from gdai.dataset_builder import build
from gdai.schemas import validate_deconet, validate_gdcore


def test_dataset_builder_outputs_valid_jsonl(tmp_path):
    counts = build(tmp_path, samples_per_model=100, seed=123)
    assert counts["deconet_train"] == 90
    assert counts["gdcore_val"] == 10
    rows = [json.loads(line) for line in (tmp_path / "deconet_train.jsonl").read_text().splitlines()]
    assert max(len(row["prompt"].encode()) + len(row["response"].encode()) + 4 for row in rows) <= 1536
    assert rows[0]["prompt"].startswith("task=decoration_plan")
    assert validate_deconet(json.loads(rows[0]["response"])) == []
    core_rows = [json.loads(line) for line in (tmp_path / "gdcore_train.jsonl").read_text().splitlines()]
    assert all(validate_gdcore(json.loads(row["response"])) == [] for row in core_rows)


def test_decoration_validator_rejects_unverified_object_ids():
    sample = {
        "schema_version":"decoNET-plan-0.1", "palette":{"colors":[]},
        "sections":[{}, {}, {}],
        "decorative_objects":[{"role":"glow outline","lookup_required":False}],
    }
    assert any("catalog lookup" in error for error in validate_deconet(sample))


def test_normalizer_expands_compact_model_fields():
    from gdai.normalize import normalize_deconet, normalize_gdcore
    deco = normalize_deconet({
        "sections":[{"n":"intro","x":[0,120],"m":"rings","l":"cyan rim","t":"wipe"}],
        "decorative_objects":[{"role":"frame","xy":[80,65],"color":"primary","lookup_required":True}],
    })
    assert deco["sections"][0]["name"] == "intro"
    assert deco["sections"][0]["x_range"] == [0,120]
    assert deco["decorative_objects"][0]["anchor"] == {"x":80,"y":65}
    core = normalize_gdcore({
        "task_type":"platformer_plan",
        "spawn_safe_zone":{"x":[0,60],"clear":True,"floor":True},
        "rooms":[{"id":"r1","x":[0,150],"p":[[12,52,30]],"h":[["spike",95,30]]}],
        "optional_coins":[[78,75]],"mechanics":["double jump"],
    })
    assert core["rooms"][0]["platforms"][0] == {"x1":12,"x2":52,"y":30}
    assert core["spawn_safe_zone"]["clear_overhead"] is True
    assert core["collectibles"][0]["optional"] is True
