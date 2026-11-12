"""Replay native frames through the actual pressure decision. Labels never enter it.
Usage: python tools/dial_gate_v2.py RUN_DIRECTORY
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_pump import dial_reader as dr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
rows = [json.loads(line) for line in (args.run / 'telemetry.jsonl').read_text().splitlines()]
episodes, current = [], []
for row in rows:
    if row.get('reset'):
        if current:
            episodes.append(current)
        current = []
    elif 't' in row:
        current.append(row)
if current:
    episodes.append(current)
out = {}
for e, labels in enumerate(episodes):
    frames = sorted(args.run.glob(f'frames/*/scene-0-e{e}_top_cam_*.npy'))
    rule = dr.PressureStop()
    records, unreadable, first = [], 0, None
    for i, frame in enumerate(frames):
        # Original motion schedule; all compression segments have identical timing.
        # Time controls the motion phase only, never the requested pressure.
        t = (i + 1) * .05
        pumping = 5.8 <= t < 41.8 and (t - 5.8) % 3.6 < 1.5
        stop, estimate = rule.update(np.load(frame), pumping)
        truth = labels[min(i, len(labels) - 1)]['pressure_kpa']
        unreadable += estimate is None
        if stop and first is None:
            first = dict(frame=i, pressure_kpa=truth, estimate_kpa=estimate)
        records.append((truth, estimate, stop))
    result = dict(frames=len(frames), unreadable=unreadable, first_stop=first)
    for name, low, high in [('below', -1, 54), ('band', 54, 66.001), ('above', 66.001, 1000)]:
        subset = [r for r in records if low <= r[0] < high]
        result[name] = dict(frames=len(subset), stop_decisions=sum(r[2] for r in subset))
    out[f'e{e}'] = result
print(json.dumps(out, indent=2))
