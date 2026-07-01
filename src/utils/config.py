import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str = "configs/settings.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_nested(config: Dict, *keys, default=None) -> Any:
    for key in keys:
        if not isinstance(config, dict):
            return default
        config = config.get(key, default)
    return config
