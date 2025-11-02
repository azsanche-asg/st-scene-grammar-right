import yaml, hashlib, json
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def cfg_hash(cfg: Dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]
