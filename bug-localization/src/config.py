# src/config.py
#
# ROLE 3 owns this file (03_ROLE_data_and_evaluation.md §1). Written here on day 1
# because Role 1 cannot run without it; the implementation follows that spec exactly.
import os

import yaml
from dotenv import load_dotenv

# config.yaml lives at the repo root, one level above src/. Resolving it relative to
# this file means the run scripts work from any working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()                                 # pulls API keys into env
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, p in cfg["paths"].items():
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(_ROOT, p))
            cfg["paths"][key] = p
        os.makedirs(p, exist_ok=True)
    return cfg
