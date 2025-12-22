#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path
from arb_sim import ArbHarnessRunner

from typing import Any, Dict, List


NAME = "6_A_rate"
THREADS = 8

REPO_ROOT = Path(__file__).resolve().parent.parent.parent / 'curve-sim-cpp'

runner = ArbHarnessRunner(repo_root=REPO_ROOT, real="double")
runner.build()

pool_config_path = Path(__file__).resolve().parent / "run_data" / f"config-{NAME}.json"
with open(pool_config_path, "r") as f:
    cfg = json.load(f)
candles_path = Path(cfg["meta"]["datafile"])

out_json_path = Path(__file__).resolve().parent / "run_data" / f"run-{NAME}.json"

ts = datetime.now()
raw = runner.run(
    pool_config_path,
    candles_path,
    out_json_path,
    threads=THREADS,
    dustswapfreq=600,
    apy_period_days=1,
    apy_period_cap=30,
    save_actions=False,
    detailed_log=False,
    candle_filter=None
)

runs_raw: List[Dict[str, Any]] = raw.get("runs", [])
print(f"Time taken: {(datetime.now() - ts).total_seconds()} seconds")


# Derive x/y and base_pool from pool_config meta
def get_meta(conf: Dict[str, Any]):
    meta = conf.get("meta", {}) if isinstance(conf, dict) else {}
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    x_name = (grid.get("X") or {}).get("name") if isinstance(grid, dict) else None
    y_name = (grid.get("Y") or {}).get("name") if isinstance(grid, dict) else None
    base_pool = meta.get("base_pool") if isinstance(meta, dict) else None
    return x_name, y_name, base_pool


x_name, y_name, base_pool_meta = get_meta(cfg)


# Derive base_pool from actual pools if meta missing
def pools_list():
    if isinstance(cfg, dict) and "pools" in cfg:
        return cfg["pools"]
    elif isinstance(cfg, dict) and "pool" in cfg:
        return [{"pool": cfg["pool"], "costs": cfg.get("costs", {})}]
    return []


plist = list(pools_list())
base_pool: Dict[str, Any] = {}
if not base_pool_meta:

    def to_strish(v):
        if isinstance(v, list):
            return [str(x) for x in v]
        return str(v)

    if plist:
        keys = set(plist[0].get("pool", {}).keys())
        for e in plist[1:]:
            keys &= set(e.get("pool", {}).keys())
        for k in sorted(keys):
            if k == x_name or k == y_name:
                continue
            vals = [to_strish(e.get("pool", {}).get(k)) for e in plist]
            if all(v == vals[0] for v in vals):
                base_pool[k] = vals[0]
else:
    base_pool = base_pool_meta

# Enrich runs with x/y keys/values
enriched_runs: List[Dict[str, Any]] = []
total_trades = 0
for rr in runs_raw:
    pool_obj = rr.get("params", {}).get("pool", {})
    xv = str(pool_obj.get(x_name)) if x_name and x_name in pool_obj else None
    yv = str(pool_obj.get(y_name)) if y_name and y_name in pool_obj else None
    # Accumulate total trades for metadata (no duplicate field in result)
    result_obj = rr.get("result", {}) or {}
    try:
        total_trades += int(result_obj.get("trades", 0))
    except Exception:
        pass

    enriched = {
        "x_key": x_name,
        "x_val": xv,
        "y_key": y_name,
        "y_val": yv,
        "result": result_obj,
        "final_state": rr.get("final_state", {}),
    }
    if "actions" in rr:
        enriched["actions"] = rr.get("actions")
    if "states" in rr:
        enriched["states"] = rr.get("states")
    enriched_runs.append(enriched)

agg = {
    "metadata": {
        "candles_file": str(candles_path),
        "threads": THREADS,
        "base_pool": base_pool,
        "grid": cfg.get("meta", {}).get("grid") if isinstance(cfg, dict) else None,
        "candles_read_ms": raw.get("metadata", {}).get("candles_read_ms"),
        "exec_ms": raw.get("metadata", {}).get("exec_ms"),
        "total_trades": total_trades,
    },
    "runs": enriched_runs,
}
with open(out_json_path, "w") as f:
    json.dump(agg, f, indent=2)
print(f"\n✓ Wrote aggregated run: {out_json_path}")
