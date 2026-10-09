# Geometry Dash 2.2081 grounding notes (starter, not exhaustive)

This file is intentionally conservative. It is not a complete object database and must not be treated as an authoritative list of object IDs or exact trigger fields.

## Concepts used by the starter training data

- Group IDs are used to organize objects and let triggers target groups.
- Item IDs act as integer-like values for trigger logic. Item Edit changes item values; Item Compare checks conditions; Item Persistence affects whether values persist after death.
- Timer-related trigger IDs are distinct from item IDs and should be tracked separately.
- Spawn-trigger activation, trigger order, channels, and ID remapping can affect whether a mechanic fires.
- SFX-related triggers and fields must use IDs confirmed against the installed Geometry Dash version.
- Area / Enter, collision, camera, shader, audio, transition, and manipulation systems have version-specific fields and interactions.
- A natural-language trigger plan is not an implemented trigger. It must be serialized to a real GD object and tested in Geometry Dash 2.2081.

## Required behavior for gdCORE

1. Do not fabricate numeric object IDs, SFX IDs, parameter names, enum values, or defaults.
2. Represent unresolved object references symbolically until the verified object catalog resolves them.
3. Maintain an ID ledger for groups, items, timers, channels, effects, and SFX; check collisions and unbound references.
4. Each mechanic must state activation source, trigger order, expected result, reset/death behavior, and how it will be tested.
5. Distinguish `known`, `needs_catalog_lookup`, and `import_tested` claims.
6. Do not state that any generated level is playable or featured-worthy until tested in-game.

## Starter references to verify and extend

- Geometry Dash Wiki, Triggers: https://geometrydash.wiki.gg/wiki/Triggers
- GDCreatorSchool, using IDs: https://github.com/GDCreatorSchool/gdcs2/blob/main/content/docs/guides/the-editor/using-ids.md
- GDCreatorSchool, Item Edit / Compare / Persistence: https://github.com/GDCreatorSchool/gdcs2/blob/main/content/docs/guides/triggers-1/item-edit-comp-pers.md
- GDCreatorSchool, Enter triggers: https://github.com/GDCreatorSchool/gdcs2/blob/main/content/docs/guides/triggers-1/enter-triggers.md
- GDCreatorSchool, spawn remap properties: https://github.com/GDCreatorSchool/gdcs2/blob/main/content/docs/guides/triggers-2/using-remap-properties.md

Before adding exact numeric IDs or properties, inspect a versioned, authoritative source and test a minimal object in the target game version. Record the source URL and verification status in a future machine-readable catalog.
