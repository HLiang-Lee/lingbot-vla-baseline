#!/usr/bin/env python3
"""Rewrite a packed data index so episode paths live under VLA_DATA_ROOT."""
import json, os, pathlib, sys

root = pathlib.Path(os.environ["VLA_DATA_ROOT"]).resolve()
index = root / "data_index.json"
if not index.exists():
    sys.exit(f"Place data_index.json in {root} before retargeting.")
doc = json.loads(index.read_text())
episodes = 0
for task in doc["tasks"]:
    name = task["task"]
    task.pop("zip", None)
    task["archive"] = f"{name}/aloha-agilex_clean_50.zip"
    for episode in task["episodes"]:
        stem = f"episode{episode['episode']}"
        episode["path"] = str(root / name / f"{stem}.hdf5")
        episode["rgb_path"] = str(root / name / f"{stem}.rgb")
        episode["arrays_path"] = str(root / name / f"{stem}.npz")
        episodes += 1
tmp = index.with_suffix(".json.tmp")
tmp.write_text(json.dumps(doc))
tmp.replace(index)
print(json.dumps({"index": str(index), "episodes": episodes, "tasks": len(doc["tasks"])}))
