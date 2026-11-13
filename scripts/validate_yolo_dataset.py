"""Audit YOLO detection labels, image integrity and exact split overlap."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


def audit(root):
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / 'data.yaml').read_text())
    names = cfg['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    errors, warnings, splits = [], [], {}
    if cfg.get('nc', len(names)) != len(names):
        errors.append('nc differs from names length')
    hashes = defaultdict(list)
    for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        image_dir, label_dir = root / folder / 'images', root / folder / 'labels'
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f'{split}: missing images/labels directory')
            continue
        configured = (root / cfg[split]).resolve()
        if configured != image_dir:
            errors.append(f'{split}: YAML resolves to {configured}, expected {image_dir}')
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        counts, sizes = Counter(), Counter()
        empty, missing, boxes, duplicate_rows = 0, 0, 0, 0
        stems = Counter(p.stem for p in images)
        for stem, count in stems.items():
            if count > 1:
                errors.append(f'{split}: multiple images share label stem {stem}')
        for image in images:
            relative = image.relative_to(root).as_posix()
            try:
                with Image.open(image) as im:
                    im.load()
                    width, height = im.size
                    sizes[f'{width}x{height}'] += 1
            except Exception as exc:
                errors.append(f'{relative}: unreadable image: {exc}')
                continue
            hashes[hashlib.sha256(image.read_bytes()).hexdigest()].append(relative)
            label = label_dir / (image.stem + '.txt')
            if not label.exists():
                missing += 1
                errors.append(f'{relative}: missing label (review before treating as background)')
                continue
            lines = [s for s in label.read_text().splitlines() if s.strip()]
            empty += not lines
            seen = set()
            for line_no, line in enumerate(lines, 1):
                location = f'{label.relative_to(root)}:{line_no}'
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError('expected class xc yc w h (5 fields)')
                    cls, x, y, w, h = map(float, fields)
                    if not all(math.isfinite(v) for v in [cls, x, y, w, h]):
                        raise ValueError('non-finite value')
                    if not cls.is_integer() or not 0 <= cls < len(names):
                        raise ValueError('invalid class ID')
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError('invalid normalized xywh')
                    if x-w/2 < -1e-5 or y-h/2 < -1e-5 or x+w/2 > 1+1e-5 or y+h/2 > 1+1e-5:
                        raise ValueError('box extends outside image')
                    row = (cls, x, y, w, h)
                    duplicate_rows += row in seen
                    seen.add(row)
                    counts[names[int(cls)]] += 1
                    boxes += 1
                except ValueError as exc:
                    errors.append(f'{location}: {exc}')
        orphan = [str(p.relative_to(root)) for p in label_dir.glob('*.txt') if p.stem not in stems]
        errors.extend(f'orphan label: {p}' for p in orphan)
        if duplicate_rows:
            warnings.append(f'{split}: {duplicate_rows} duplicate annotation rows')
        if empty:
            warnings.append(f'{split}: {empty} empty labels; manually verify background images')
        splits[split] = dict(images=len(images), boxes=boxes, class_counts=dict(counts),
                             empty_labels=empty, missing_labels=missing, image_sizes=dict(sizes),
                             duplicate_label_rows=duplicate_rows)
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split = [paths for paths in duplicates if len({p.split('/')[0] for p in paths}) > 1]
    if cross_split:
        warnings.append(f'{len(cross_split)} exact-image hash groups cross dataset splits')
    return dict(names=names, splits=splits, errors=errors, warnings=warnings,
                exact_duplicate_groups=duplicates, cross_split_duplicates=cross_split,
                scope='Format/integrity only; no semantic label or near-duplicate verification')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--report', type=Path, default=Path('reports/dataset_audit.json'))
    args = p.parse_args()
    result = audit(args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'exact_duplicate_groups', 'cross_split_duplicates'}}, indent=2, ensure_ascii=False))
    print('Full report:', args.report)
    raise SystemExit(bool(result['errors']))
