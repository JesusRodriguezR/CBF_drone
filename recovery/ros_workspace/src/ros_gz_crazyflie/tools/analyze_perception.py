#!/usr/bin/env python3
"""Generate self-contained SVG plots and summary metrics from a perception CSV."""

import argparse
import csv
import json
import math
from pathlib import Path


def number(row, key):
    try:
        value = float(row.get(key, ''))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def paired(rows, x_key, y_key, valid_only=False):
    pairs = []
    for row in rows:
        if valid_only and row.get('detection_valid') != '1':
            continue
        x, y = number(row, x_key), number(row, y_key)
        if x is not None and y is not None:
            pairs.append((x, y))
    return pairs


def svg_plot(path, series, title, xlabel, ylabel, scatter=False, y_range=None):
    width, height = 1000, 620
    left, right, top, bottom = 90, 30, 60, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    all_points = [point for _, values in series for point in values]
    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 1.0)]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = y_range if y_range else (min(ys), max(ys))
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle">{xlabel}</text>',
        f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.1f})">{ylabel}</text>',
    ]
    colors = ('#145cff', '#e32626', '#159447', '#a23bb8')
    for index, (label, points) in enumerate(series):
        if not points:
            continue
        color = colors[index % len(colors)]
        if scatter:
            for x, y in points:
                parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        else:
            coords = ' '.join(f'{sx(x):.2f},{sy(y):.2f}' for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left + 12 + index * 150}" y="{top - 12}" fill="{color}">{label}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def presentation_chart(parts, x, y, width, height, title, series, y_range=None, scatter=False):
    """Append a compact chart to the 16:9 presentation figure."""
    pad = 34
    points = [point for _, values in series for point in values]
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = y_range if y_range else (min(ys), max(ys))
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    def sx(value):
        return x + pad + (value - xmin) / (xmax - xmin) * (width - 2 * pad)
    def sy(value):
        return y + height - pad - (value - ymin) / (ymax - ymin) * (height - 2 * pad)
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="#f5f7fb" stroke="#d6dbe5"/>')
    parts.append(f'<text x="{x + 18}" y="{y + 25}" font-size="16" font-weight="bold">{title}</text>')
    parts.append(f'<line x1="{x + pad}" y1="{y + height - pad}" x2="{x + width - pad}" y2="{y + height - pad}" stroke="#778"/>')
    parts.append(f'<line x1="{x + pad}" y1="{y + pad}" x2="{x + pad}" y2="{y + height - pad}" stroke="#778"/>')
    colors = ('#145cff', '#e32626', '#159447')
    for index, (label, values) in enumerate(series):
        if not values:
            continue
        color = colors[index % len(colors)]
        if scatter:
            for xv, yv in values[::max(1, len(values) // 250)]:
                parts.append(f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="2" fill="{color}"/>')
        else:
            coords = ' '.join(f'{sx(xv):.1f},{sy(yv):.1f}' for xv, yv in values[::max(1, len(values) // 400)])
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{x + width - 130 + index * 55}" y="{y + 25}" fill="{color}" font-size="11">{label}</text>')


def presentation_svg(path, summary, valid, aoi, errors, aoi_error):
    width, height = 1600, 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect width="100%" height="110" fill="#14213d"/>',
        '<text x="55" y="52" fill="white" font-size="30" font-weight="bold">Passive Camera Perception Characterization</text>',
        '<text x="55" y="86" fill="#c9d5ee" font-size="18">Crazyflie greenhouse simulation · baseline single-frame tutor-wire detector · perception is not connected to control</text>',
    ]
    cards = [
        ('Detection rate', f"{summary['detection_rate'] * 100:.1f}%", '#159447'),
        ('Mean AoI', f"{summary['mean_aoi_ms']:.1f} ms" if summary['mean_aoi_ms'] is not None else 'n/a', '#145cff'),
        ('Distance MAE', f"{summary['distance_mae_m']:.2f} m" if summary['distance_mae_m'] is not None else 'n/a', '#e32626'),
        ('Observations', str(summary['observations']), '#a23bb8'),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 55 + index * 375
        parts.extend([
            f'<rect x="{x}" y="135" width="340" height="100" rx="12" fill="#f5f7fb" stroke="#d6dbe5"/>',
            f'<rect x="{x}" y="135" width="8" height="100" rx="4" fill="{color}"/>',
            f'<text x="{x + 25}" y="165" fill="#596273" font-size="17">{label}</text>',
            f'<text x="{x + 25}" y="210" fill="#14213d" font-size="32" font-weight="bold">{value}</text>',
        ])
    presentation_chart(parts, 55, 270, 720, 260, 'Detection validity', [('valid', valid)], y_range=(0, 1))
    presentation_chart(parts, 825, 270, 720, 260, 'Camera age of information', [('AoI [ms]', aoi)])
    presentation_chart(parts, 55, 555, 720, 260, 'Absolute perception error',
                       [('distance error [m]', [(x, abs(y)) for x, y in errors])])
    presentation_chart(parts, 825, 555, 720, 260, 'AoI vs absolute error',
                       [('observations', [(x, abs(y)) for x, y in aoi_error])], scatter=True)
    parts.extend([
        '<text x="55" y="850" fill="#596273" font-size="16">Interpretation: baseline frames are arriving reliably under the no-impairment condition; the simple detector currently has high detection recall but large distance error.</text>',
        '</svg>',
    ])
    path.write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Perception CSV log')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--impairment-log', help='Optional impairment CSV for drop/rejection counts')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit('The perception log contains no observations')

    valid = [(index, 1 if row.get('detection_valid') == '1' else 0)
             for index, row in enumerate(rows)]
    valid_rows = [row for row in rows if row.get('detection_valid') == '1']
    gt_dist = paired(rows, 'frame_sequence_id', 'wire_gt_distance_m', True)
    est_dist = paired(rows, 'frame_sequence_id', 'estimated_wire_distance_m', True)
    errors = [number(row, 'distance_error_m') for row in valid_rows]
    errors = [value for value in errors if value is not None]
    aoi = paired(rows, 'processing_timestamp_s', 'age_of_information_ms')
    error_t = paired(valid_rows, 'processing_timestamp_s', 'distance_error_m')
    aoi_error = paired(valid_rows, 'age_of_information_ms', 'distance_error_m')
    confidence_error = paired(valid_rows, 'confidence', 'distance_error_m')

    svg_plot(output / 'detection_success_vs_time.svg', [('valid', valid)],
             'Detection success vs time', 'observation index', 'detection valid', y_range=(0, 1))
    svg_plot(output / 'distance_vs_ground_truth.svg',
             [('ground truth', gt_dist), ('estimate', est_dist)],
             'Estimated wire distance vs ground truth', 'observation index', 'distance [m]')
    svg_plot(output / 'perception_error_vs_time.svg', [('error', error_t)],
             'Perception error vs time', 'processing time [s]', 'distance error [m]')
    svg_plot(output / 'aoi_vs_time.svg', [('AoI', aoi)],
             'Camera age of information vs time', 'processing time [s]', 'AoI [ms]')
    svg_plot(output / 'aoi_vs_perception_error.svg',
             [('absolute error', [(x, abs(y)) for x, y in aoi_error])],
             'AoI vs perception error', 'AoI [ms]', 'absolute distance error [m]', scatter=True)
    svg_plot(output / 'confidence_vs_error.svg',
             [('absolute error', [(x, abs(y)) for x, y in confidence_error])],
             'Confidence vs perception error', 'confidence', 'absolute distance error [m]', scatter=True)

    total = len(rows)
    detected = sum(value for _, value in valid)
    mean_aoi = [y for _, y in aoi]
    impairment_dropped = None
    if args.impairment_log:
        with Path(args.impairment_log).open(newline='') as handle:
            impairment_rows = list(csv.DictReader(handle))
        impairment_dropped = sum(row.get('delivered') != 'True' for row in impairment_rows)
    summary = {
        'observations': total,
        'detection_rate': detected / total if total else 0.0,
        'missed_detection_rate': 1.0 - detected / total if total else 0.0,
        'distance_mae_m': sum(abs(value) for value in errors) / len(errors) if errors else None,
        'distance_rmse_m': math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None,
        'mean_aoi_ms': sum(mean_aoi) / len(mean_aoi) if mean_aoi else None,
        'max_aoi_ms': max(mean_aoi) if mean_aoi else None,
        'dropped_or_rejected_frames': sum(row.get('frame_accepted') != '1' for row in rows),
        'impairment_dropped_or_rejected_frames': impairment_dropped,
        'valid_distance_pairs': len(gt_dist),
    }
    presentation_svg(output / 'presentation_summary.svg', summary, valid, aoi, error_t, aoi_error)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'value'])
        writer.writerows(summary.items())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
