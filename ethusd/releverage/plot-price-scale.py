#!/usr/bin/env python3

import pylab
import lzma
import json
from pathlib import Path


NAME = "8_one"
STEP_SIZE = 1000

detailed_name = Path(__file__).parent.parent / 'run_data' / f'run-{NAME}-detailed.json.xz'
png_path = Path(__file__).parent.parent / 'plots' / f'pscale-{NAME}.png'


t = []
price_scale = []
price = []

with lzma.open(detailed_name) as f:
    data = json.load(f)

for d in data[::STEP_SIZE]:
    t.append(d['t'])
    price_scale.append(d['price_scale'])
    price.append(d['close'])

pylab.plot(t, price_scale, c="gray", label="price_scale")
pylab.plot(t, price, c="black", label="spot price")

pylab.legend()
pylab.tight_layout()

pylab.savefig(png_path)
pylab.show()
