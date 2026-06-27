import json
from pathlib import Path

from utils.results import normalize_rig_name, sorted_rigs


def rig_config_path(rigs_root, rig_name):
    rigs_root = Path(rigs_root)
    normalized = normalize_rig_name(rig_name)
    candidates = [
        rigs_root / f"{normalized}.json",
        rigs_root / f"{normalized.lower()}.json",
        rigs_root / f"{normalized.upper()}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_rig_configs(rigs_root, required_rigs):
    rigs_root = Path(rigs_root)
    if not rigs_root.exists():
        raise FileNotFoundError(
            f"Rig config directory does not exist: {rigs_root}. "
            "Pass --rigs-root pointing to JSON rig calibration files."
        )

    configs = {}
    missing = []
    for rig in sorted_rigs(required_rigs):
        path = rig_config_path(rigs_root, rig)
        if path is None:
            missing.append(rig)
            continue
        with path.open() as handle:
            configs[rig] = json.load(handle)

    if missing:
        raise FileNotFoundError(
            f"Missing rig config JSON files in {rigs_root}: {', '.join(missing)}"
        )
    return configs

