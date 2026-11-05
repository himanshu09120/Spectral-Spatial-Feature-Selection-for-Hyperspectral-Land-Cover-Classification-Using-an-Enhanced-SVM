"""Check real-data class coverage and MNF pairs without fitting classifiers."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_scene
from src.evaluation import write_json
from src.preprocessing import make_split, training_noise_pairs


def main():
    config = json.loads((ROOT / "config/config.json").read_text())
    cube, labels, _ = load_scene(config["dataset"], ROOT)
    records = []
    for protocol in ("pixel", "spatial_block"):
        for seed in config["seeds"]:
            try:
                split = make_split(labels, config, seed, protocol)
                pairs = training_noise_pairs(cube, split["train"], seed=seed,
                    min_pairs=config["mnf"]["min_noise_pairs"], max_pairs=config["mnf"]["max_noise_pairs"])
                records.append(dict(protocol=protocol, seed=seed, status="feasible",
                    partitions={k: len(v) for k, v in split.items()}, noise_pairs=len(pairs)))
            except ValueError as error:
                records.append(dict(protocol=protocol, seed=seed, status="infeasible", reason=str(error)))
    write_json(ROOT / config["output_dir"] / "protocol_feasibility.json", records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
