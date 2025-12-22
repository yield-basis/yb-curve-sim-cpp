#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

NAME = "4_rate_fg"
METRIC = "tw_capped_apy_net"
X_SCALE = 1.0
Y_SCALE = 1e-18


out_json_path = Path(__file__).resolve().parent / "run_data" / f"run-{NAME}.json"
with open(out_json_path, 'r') as f:
    data = json.load(f)
x_name = data['metadata']['grid']['X']['name']
x_size = data['metadata']['grid']['X']['n']
y_name = data['metadata']['grid']['Y']['name']
y_size = data['metadata']['grid']['Y']['n']

x_values = sorted(set([float(r["x_val"]) * X_SCALE for r in data["runs"]]))
y_values = sorted(set([float(r["y_val"]) * Y_SCALE for r in data["runs"]]))

Z = np.zeros((y_size, x_size))

for r in data['runs']:
    x = float(r["x_val"]) * X_SCALE
    y = float(r["y_val"]) * Y_SCALE
    Z[y_values.index(y), x_values.index(x)] = float(r['result'][METRIC])

fig, ax = plt.subplots()
plt.yscale('log')
plt.xscale('log')
im = ax.pcolormesh(x_values, y_values, Z, cmap=plt.get_cmap('jet'))
im.set_edgecolor('face')
cbar = fig.colorbar(im, ax=ax)

plt.title(NAME)
plt.xlabel(x_name)
plt.ylabel(y_name)
cbar.set_label(METRIC, rotation=270, labelpad=15)
plt.tight_layout()

plot_path = Path(__file__).resolve().parent / "plots" / f"result-{NAME}.png"
plt.savefig(plot_path)

plt.show()
