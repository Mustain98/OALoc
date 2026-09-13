from src.data.loader import load_dataset
from src.config import load_config
import sys

cfg = load_config()
insts = load_dataset(cfg["dataset"], cfg["split"], 300)
inst = next((i for i in insts if i.instance_id == "astropy__astropy-14365"), None)
if inst:
    print("Gold files:", inst.gold_files)
    print("Gold functions:", inst.gold_functions)
    print("Problem statement:\n", inst.problem_statement[:500])
else:
    print("Not found")
