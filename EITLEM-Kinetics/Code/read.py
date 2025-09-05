# quick_check_labels.py
import json, numpy as np
with open("/workspace/EITLEM-Kinetics/Data/concat_train_dataset_final_latest.json","r") as f:
    d=json.load(f)
vals=[float(v["Value"]) for v in d.values()]
print("count:",len(vals),
      "min:",np.min(vals),"p5:",np.percentile(vals,5),
      "median:",np.median(vals),
      "p95:",np.percentile(vals,95),"max:",np.max(vals))
