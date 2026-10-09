"""Build reproducible synthetic training data from explicit GD-specific rules.

These examples are generated from human-authored templates; they are NOT
claimed to be real rated levels, scraped data, or a complete trigger catalog.
Use them to smoke-test training and then add carefully reviewed examples.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any

THEMES = [
    "storm forest", "abandoned laboratory", "underwater archive", "crystal cavern",
    "neon city", "solar temple", "frozen observatory", "mechanical garden",
    "volcanic refinery", "dreamlike library", "moonlit ruins", "bio-luminescent swamp",
    "retro arcade", "gravity reactor", "cloud palace", "deep space station",
    "clockwork cathedral", "overgrown subway", "glass desert", "electric greenhouse",
]
MOODS = ["tense", "mysterious", "triumphant", "dreamlike", "melancholic", "energetic", "eerie", "playful", "majestic", "industrial"]
INTENSITIES = ["restrained", "balanced", "dense at the drop", "high contrast", "slow-burn", "rapid and rhythmic"]
DIFFICULTIES = ["easy", "normal", "hard", "harder", "insane", "demon-intended"]
PALETTE_PREFS = ["cyan and indigo", "orange and graphite", "violet and mint", "crimson and gold", "ice blue and white", "acid green and black", "pink and navy", "amber and teal", "monochrome with one accent", "deep blue and coral"]
MOTIFS = ["hexagonal frames", "broken circuit traces", "floating shards", "layered arches", "angular vines", "glass prisms", "ring clusters", "rune-like linework", "segmented beams", "mechanical petals", "pixel-grid panels", "spiral rails"]
LIGHTING = ["edge-lit cyan", "soft volumetric cones", "pulsing amber accents", "subtle rim light", "crossed magenta beams", "faint cool haze", "thin white highlights", "alternating warm and cool pools"]
TRANSITIONS = ["motif fragments drift into the next section", "a one-beat palette inversion signals the change", "foreground silhouettes wipe from left to right", "light intensity rises before the drop", "thin outlines connect across the transition", "background layers parallax apart", "a brief negative-space bar resets visual density"]
OBJECT_ROLES = ["foreground frame", "glow outline", "background silhouette", "particle cluster", "floor accent", "portal surround", "hero centerpiece", "rhythm marker", "depth layer", "route sign", "secret-room marker", "transition shard"]

PALETTES = [
    {"name":"stormglass", "colors":[("background","#071426"),("primary","#22D3EE"),("secondary","#6366F1"),("accent","#F472B6"),("hazard","#F97316")]},
    {"name":"emberforge", "colors":[("background","#160D0B"),("primary","#FB923C"),("secondary","#9F1239"),("accent","#FDE047"),("hazard","#F43F5E")]},
    {"name":"deepgarden", "colors":[("background","#061A19"),("primary","#2DD4BF"),("secondary","#4D7C0F"),("accent","#C084FC"),("hazard","#FB7185")]},
    {"name":"royalstatic", "colors":[("background","#100B25"),("primary","#A78BFA"),("secondary","#2563EB"),("accent","#F0ABFC"),("hazard","#F59E0B")]},
    {"name":"glacierlight", "colors":[("background","#091827"),("primary","#BAE6FD"),("secondary","#38BDF8"),("accent","#F8FAFC"),("hazard","#F97316")]},
    {"name":"signalnoise", "colors":[("background","#09090B"),("primary","#A3E635"),("secondary","#27272A"),("accent","#F0FD4A"),("hazard","#FB7185")]},
    {"name":"coralnight", "colors":[("background","#111827"),("primary","#FB7185"),("secondary","#312E81"),("accent","#FDBA74"),("hazard","#FDE047")]},
    {"name":"sunkenrelic", "colors":[("background","#082F49"),("primary","#2DD4BF"),("secondary","#0F766E"),("accent","#FDE68A"),("hazard","#F97316")]},
]


def compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def make_deco_example(rng: random.Random, index: int) -> dict[str, str]:
    theme = rng.choice(THEMES)
    mood = rng.choice(MOODS)
    intensity = rng.choice(INTENSITIES)
    palette_pref = rng.choice(PALETTE_PREFS)
    difficulty = rng.choice(DIFFICULTIES)
    palette = rng.choice(PALETTES)
    sections = []
    for name, x1, x2 in [("intro",0,120),("build",120,260),("drop",260,420),("outro",420,520)]:
        sections.append({"n":name,"x":[x1,x2],"m":rng.choice(MOTIFS),"l":rng.choice(["cyan rim","soft cones","amber pulses","magenta beams","cool haze","white accents"]),"t":rng.choice(["motif shifts","palette flip","foreground wipe","light rises","outline carries","layers split","negative space"])})
    objects = []
    for role, xy in [("foreground frame",[80,65]),("glow outline",[250,95]),("hero centerpiece",[400,120])]:
        objects.append({"role":role if rng.random()<0.55 else rng.choice(OBJECT_ROLES),"xy":[xy[0]+rng.choice([-8,0,8]),xy[1]+rng.choice([-10,0,10]),],"layer":rng.choice(["background","midground","foreground"]),"color":rng.choice(["primary","secondary","accent"]),"lookup_required":True})
    color_map={role:hex_value for role,hex_value in palette["colors"]}
    response = {
        "schema_version":"decoNET-plan-0.1","theme":theme,"mood":mood,"intensity":intensity,
        "palette":{"name":palette["name"],"bg":color_map["background"],"primary":color_map["primary"],"secondary":color_map["secondary"],"accent":color_map["accent"],"hazard":color_map["hazard"]},
        "global_motif":rng.choice(MOTIFS),"depth_layers":["distant silhouettes","midground structures","foreground frames"],
        "sections":sections,"decorative_objects":objects,"readability_rules":["high hazard contrast","never hide route edges","density peaks at drop"],
    }
    request = f"{theme}; mood={mood}; intensity={intensity}; palette={palette_pref}; difficulty={difficulty}. Plan 4 sections, depth, hero moment, and readable route."
    prompt = "task=decoration_plan\nrequest=" + request + "\nJSON; no guessed numeric IDs."
    return {"id":f"deconet-{index:06d}","prompt":prompt,"response":compact(response),"task_type":"decoration_plan"}

def _platformer_response(rng: random.Random, theme: str, difficulty: str, mechanics: list[str], index: int) -> dict[str, Any]:
    rooms = []
    for i in range(4):
        base_x = i * 150
        y = rng.choice([30,45,60])
        platforms = [[base_x+12,base_x+52,y],[base_x+66,base_x+91,y+rng.choice([15,30])],[base_x+108,base_x+140,y]]
        hazards = [] if i == 0 else [[rng.choice(["spike","gap","timed hazard"]),base_x+95,y]]
        rooms.append({"id":f"r{i+1}","x":[base_x,base_x+150],"p":platforms,"h":hazards})
    collectibles = [[78,105],[228,75],[378,105],[528,90]]
    return {
        "schema_version":"gdCORE-plan-0.1","task_type":"platformer_plan","title":f"{theme.title()} Run {index % 97:02d}","mode":"platformer","difficulty":difficulty,
        "spawn_safe_zone":{"x":[0,60],"clear":True,"floor":True},"rooms":rooms,"optional_coins":collectibles,
        "mechanics":mechanics[:3],"mechanic_status":"planned; IDs and trigger fields need catalog verification",
        "trigger_ledger":[{"family":rng.choice(["Group activation","Item Edit / Item Compare","Spawn trigger","Checkpoint behavior"]),"ids":"allocate uniquely","status":"verify 2.2081"}],
        "validation_checks":["safe spawn","reachable landings","optional secrets","check ID collisions","import/play-test"],
    }

def _classic_response(rng: random.Random, theme: str, difficulty: str, index: int) -> dict[str, Any]:
    patterns = ["short jump / long landing", "orb / safe platform", "high-low route", "syncopation / recovery", "portal / readable pattern"]
    sections=[]
    for i, name in enumerate(["intro","build","drop","outro"]):
        sections.append({"name":name,"x":[i*120,(i+1)*120],"goal":rng.choice(["teach rhythm","increase challenge","main payoff","resolve phrase"]),"pattern":rng.choice(patterns),"difficulty":difficulty if i<3 else "recovery"})
    return {"schema_version":"gdCORE-plan-0.1","task_type":"classic_layout_plan","title":f"{theme.title()} Pulse {index % 97:02d}","mode":"classic","difficulty":difficulty,"sections":sections,"rules":["show hazards early","avoid filler","add recovery space","use catalog objects"],"validation_checks":["readable start","plausible jumps","sync section changes","clear finish","import-test"]}

def _trigger_response(rng: random.Random, trigger_topic: str) -> dict[str, Any]:
    known = [
        "Group IDs target object groups.",
        "Item Edit changes values; Item Compare checks conditions.",
        "Item Persistence affects values after death.",
        "Trigger order/source/ID reuse affect behavior.",
        "SFX IDs and fields require versioned verification.",
    ]
    return {"schema_version":"gdCORE-trigger-plan-0.1","task_type":"trigger_plan","topic":trigger_topic,"goal":"testable plan; don't guess undocumented settings","setup_steps":["find trigger in verified 2.2081 catalog","allocate symbolic IDs and check collisions","record activation source/order","implement one mechanic","test play, death/reset, and repeat"],"known_facts":rng.sample(known,k=3),"id_ledger":[["GROUP_PLATFORM_A","group",None,"allocate in editor"],["ITEM_JUMP_COUNT","item",None,"allocate in editor"]],"unknowns_to_verify":["object/field names","allowed values/defaults","reset/persistence","channel/spawn requirements"],"safety_rules":["never invent numeric IDs","test before calling implemented"]}

def _ui_response(rng: random.Random, theme: str, ui_kind: str, index: int) -> dict[str, Any]:
    if ui_kind == "shop_ui":
        elements = [
            {"id":"currency","rect":[0.04,0.04,0.34,0.09],"role":"show balances","action":"none"},
            {"id":"preview","rect":[0.28,0.20,0.44,0.43],"role":"item silhouette","action":"preview"},
            {"id":"grid","rect":[0.05,0.66,0.90,0.22],"role":"browse items","action":"select"},
            {"id":"purchase","rect":[0.66,0.49,0.27,0.10],"role":"cost and confirm","action":"check balance/ownership"},
        ]
    else:
        elements = [
            {"id":"logo","rect":[0.24,0.08,0.52,0.18],"role":"game identity","action":"none"},
            {"id":"play","rect":[0.34,0.43,0.32,0.12],"role":"primary action","action":"start level"},
            {"id":"settings","rect":[0.08,0.78,0.18,0.10],"role":"settings","action":"open settings"},
            {"id":"collection","rect":[0.72,0.78,0.20,0.10],"role":"secondary nav","action":"open collection/shop"},
        ]
    return {"schema_version":"gdCORE-ui-plan-0.1","task_type":ui_kind,"theme":theme,"composition":"clear hierarchy; one primary action; consistent margins","palette":{"bg":"dark neutral","primary":"bright focal","secondary":"support","accent":"rare highlight","danger":"warning contrast"},"elements":elements,"motion":["animate focus changes","short eased transitions","respect reduced motion"],"rules":["show focus state","confirm purchases","labels stay legible"],"serialization":"plan only; map elements to verified objects or an external UI host"}

def make_core_example(rng: random.Random, index: int) -> dict[str, str]:
    theme = rng.choice(THEMES)
    difficulty = rng.choice(DIFFICULTIES)
    kind = rng.choices(["platformer","classic","trigger_plan","title_screen","shop_ui"],[42,18,20,10,10],k=1)[0]
    if kind == "platformer":
        mechanics = rng.sample(["double-jump challenge","moving platform route","secret coin branch","checkpoint after a hard room","timed gate","vertical shaft","switch-activated bridge","one-way drop route","optional skill shortcut"], k=3)
        prompt = f"task=platformer; theme={theme}; difficulty={difficulty}; mechanics={', '.join(mechanics)}; 4 rooms, safe spawn; JSON only; do not guess IDs."
        output = _platformer_response(rng,theme,difficulty,mechanics,index)
    elif kind == "classic":
        prompt = f"task=classic_layout; theme={theme}; difficulty={difficulty}; intro/build/drop/outro, readable rhythms; JSON only."
        output = _classic_response(rng,theme,difficulty,index)
    elif kind == "trigger_plan":
        topic = rng.choice(["a double-jump limit", "a switch opening a bridge", "a platform moving and returning", "a coin counter gate", "a checkpoint reset", "an SFX cue", "a trigger activation channel", "a timed room lock"])
        prompt = f"task=trigger_plan; GD 2.2081 plan for {topic}; label unknown fields; do not guess IDs; JSON only."
        output = _trigger_response(rng,topic)
    elif kind == "shop_ui":
        prompt = f"task=shop_ui; {theme} GD shop: currency, preview, purchase confirmation, compact layout; JSON only."
        output = _ui_response(rng,theme,"shop_ui",index)
    else:
        prompt = f"task=title_screen; {theme} GD title: play, settings, collection navigation; JSON only."
        output = _ui_response(rng,theme,"title_screen",index)
    return {"id":f"gdcore-{index:06d}","prompt":prompt,"response":compact(output),"task_type":kind}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=True,separators=(",",":"))+"\n")


def build(output_dir: Path, samples_per_model: int = 2400, seed: int = 42, val_fraction: float = 0.1) -> dict[str, int]:
    if samples_per_model < 100:
        raise ValueError("use at least 100 samples per model for a meaningful smoke experiment")
    if not 0.05 <= val_fraction <= 0.3:
        raise ValueError("val_fraction must be between 0.05 and 0.30")
    counts = {}
    for name, maker, model_seed in [("deconet",make_deco_example,seed),("gdcore",make_core_example,seed+99991)]:
        rng = random.Random(model_seed)
        rows = [maker(rng, i) for i in range(samples_per_model)]
        rng.shuffle(rows)
        split = max(1,int(len(rows)*(1-val_fraction)))
        train, val = rows[:split], rows[split:]
        _write_jsonl(output_dir/f"{name}_train.jsonl",train)
        _write_jsonl(output_dir/f"{name}_val.jsonl",val)
        counts[f"{name}_train"] = len(train)
        counts[f"{name}_val"] = len(val)
    (output_dir/"manifest.json").write_text(json.dumps({"generator":"gd-ai-lab dataset_builder 0.1.0","seed":seed,"samples_per_model":samples_per_model,"counts":counts,"warning":"Synthetic starter data generated from hand-written templates. Not human ratings or a real GD level corpus."},indent=2),encoding="utf-8")
    return counts


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default="data/generated",type=Path)
    parser.add_argument("--samples-per-model",default=2400,type=int)
    parser.add_argument("--seed",default=42,type=int)
    parser.add_argument("--val-fraction",default=0.1,type=float)
    args=parser.parse_args()
    counts=build(args.output_dir,args.samples_per_model,args.seed,args.val_fraction)
    print(json.dumps({"output_dir":str(args.output_dir),"counts":counts},indent=2))

if __name__ == "__main__":
    main()
