# shardwright-export

Converts a Shardwright `.fracphys` (FlatBuffers) into the `shardwright-stress/1` JSON that
`stress-ref::shardwright` imports (and every tool reading scene packs, through
`scene_pack::import`). Standalone, outside the workspace: it builds against a Shardwright
checkout (the path dependencies assume `Shardwright` beside `PhysX`; edit `Cargo.toml`
otherwise) and pins `flatbuffers` to Shardwright's version.

```bash
cd tools/shardwright-export
CARGO_TARGET_DIR=/tmp/shard-export cargo build --release
/tmp/shard-export/release/shard-export two_storey_building.fracphys two_storey_L2.json 2
```
