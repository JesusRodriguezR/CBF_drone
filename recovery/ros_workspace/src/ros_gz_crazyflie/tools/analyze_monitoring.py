#!/usr/bin/env python3
"""Summarize spatially associated monitoring events and write SVG plots."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def value(row, key):
    try:
        x = float(row.get(key, ''))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def plot(path, series, title, xlabel, ylabel, y_range=None):
    width, height, left, top, right, bottom = 1000, 620, 90, 60, 30, 80
    points = [p for _, vals in series for p in vals] or [(0, 0), (1, 1)]
    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = y_range or (min(ys), max(ys))
    if xmin == xmax: xmax = xmin + 1
    if ymin == ymax: ymin, ymax = ymin - 1, ymax + 1
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="500" y="30" text-anchor="middle" font-size="20">{title}</text>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
           f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
           f'<text x="500" y="605" text-anchor="middle">{xlabel}</text>',
           f'<text x="18" y="310" text-anchor="middle" transform="rotate(-90 18 310)">{ylabel}</text>']
    colors = ('#145cff', '#e32626', '#159447', '#a23bb8', '#ff7f0e')
    for i, (label, vals) in enumerate(series):
        if not vals: continue
        coords = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in vals)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{colors[i % len(colors)]}" stroke-width="2"/>')
        out.append(f'<text x="{left + 15 + i * 170}" y="48" fill="{colors[i % len(colors)]}">{label}</text>')
    out.append('</svg>')
    path.write_text('\n'.join(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    events = defaultdict(list)
    for row in rows:
        event = row.get('monitoring_event_id', '0')
        if event != '0' and row.get('observation_active') == '1': events[event].append(row)
    summary = []
    distance_series, validity_series = [], []
    for event, event_rows in sorted(events.items(), key=lambda item: int(item[0])):
        detected = [r for r in event_rows if r.get('detection_valid') == '1']
        estimates = [value(r, 'estimated_wire_distance_m') for r in detected]
        estimates = [x for x in estimates if x is not None]
        gt = [value(r, 'wire_gt_distance_m') for r in detected]
        gt = [x for x in gt if x is not None]
        errors = [value(r, 'distance_error_m') for r in detected]
        errors = [x for x in errors if x is not None]
        aoi = [value(r, 'age_of_information_ms') for r in event_rows]
        aoi = [x for x in aoi if x is not None]
        conf = [value(r, 'confidence') for r in detected]
        conf = [x for x in conf if x is not None]
        times = [value(r, 'processing_timestamp_s') for r in event_rows]
        times = [x for x in times if x is not None]
        summary.append({'monitoring_event_id': event, 'monitoring_point_id': event_rows[0].get('monitoring_point_id', ''),
                        'aisle_id': event_rows[0].get('aisle_id', ''), 'crop_section_id': event_rows[0].get('crop_section_id', ''),
                        'frames_acquired': len(event_rows), 'detection_rate': len(detected) / len(event_rows),
                        'mean_distance_m': sum(estimates) / len(estimates) if estimates else None,
                        'median_distance_m': sorted(estimates)[len(estimates)//2] if estimates else None,
                        'ground_truth_distance_m': sum(gt) / len(gt) if gt else None,
                        'mae_m': sum(abs(x) for x in errors) / len(errors) if errors else None,
                        'rmse_m': math.sqrt(sum(x*x for x in errors) / len(errors)) if errors else None,
                        'mean_confidence': sum(conf) / len(conf) if conf else None,
                        'mean_aoi_ms': sum(aoi) / len(aoi) if aoi else None,
                        'observation_duration_s': max(times) - min(times) if len(times) > 1 else 0.0})
        distance_series.append((event + ' estimate', list(zip(range(len(estimates)), estimates))))
        gt_values = [value(r, 'wire_gt_distance_m') for r in detected]
        gt_values = [x for x in gt_values if x is not None]
        distance_series.append((event + ' ground truth', list(zip(range(len(gt_values)), gt_values))))
        validity_series.append((event, [(i, int(r.get('detection_valid') == '1')) for i, r in enumerate(event_rows)]))
    if not summary: raise SystemExit('No observation frames found (check observation_active and log path)')
    with (output / 'monitoring_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    (output / 'monitoring_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(output / 'distance_vs_ground_truth_by_event.svg', distance_series, 'Tutor-wire distance during monitoring events', 'frame index within event', 'estimated distance [m]')
    plot(output / 'detection_validity_by_event.svg', validity_series, 'Detection validity during monitoring events', 'frame index within event', 'valid', (0, 1))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
