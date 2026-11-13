"""Resumable YOLO detection: snapshots/<camera_id>/YYYYmmdd_HHMMSS.jpg.

One process per GPU, one SQLite output per shard. No network side effects.
"""
import argparse
import csv
import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LOG = logging.getLogger(__name__)
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def image_key(path, root, timezone):
    relative = path.relative_to(root).as_posix()
    timestamp = datetime.strptime(path.stem, '%Y%m%d_%H%M%S').replace(
        tzinfo=ZoneInfo(timezone)).isoformat()
    return relative, path.parent.name, timestamp


def shard_for(key, shards):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % shards


def open_database(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('''CREATE TABLE IF NOT EXISTS images (
        path TEXT PRIMARY KEY, camera_id TEXT, timestamp TEXT,
        status TEXT, total INTEGER, counts TEXT, error TEXT,
        size INTEGER, mtime_ns INTEGER)''')
    value = json.dumps(config, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    if old and old[0] != value:
        db.close()
        raise ValueError('Output configuration differs. Use a new --output directory.')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('config', ?)", (value,))
    db.commit()
    return db


def export_csv(db, destination, names):
    """Stream an atomic CSV export; errors have blank counts, not zero vehicles."""
    destination = Path(destination)
    temporary = destination.with_suffix('.csv.tmp')
    ids = sorted(names)
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'camera_id', 'timestamp', 'status', 'total_vehicles'] +
                        [f'count_{i}_{names[i]}' for i in ids] + ['error'])
        for row in db.execute('SELECT path,camera_id,timestamp,status,total,counts,error '
                              'FROM images ORDER BY camera_id,timestamp,path'):
            counts = json.loads(row[5]) if row[5] else {}
            writer.writerow(list(row[:5]) +
                            [counts.get(str(i), 0) if row[3] == 'ok' else '' for i in ids] +
                            [row[6]])
    temporary.replace(destination)


def process_batch(model, records, db, args, class_ids, device):
    from PIL import Image
    images, valid = [], []
    for record in records:
        path, key, camera, timestamp, size, mtime = record
        try:
            with Image.open(path) as image:
                image.load()
                images.append(image.convert('RGB'))
            valid.append(record)
        except (OSError, ValueError) as exc:
            db.execute('INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?)',
                       (key, camera, timestamp, 'error', None, None,
                        f'{type(exc).__name__}: {exc}', size, mtime))
    if valid:
        try:
            results = list(model.predict(source=images, device=device, imgsz=args.imgsz,
                conf=args.conf, iou=args.iou, classes=class_ids, max_det=args.max_det,
                half=args.half, verbose=False, save=False, stream=True))
        except Exception:
            db.rollback()
            # Never silently mark OOM/runtime failures as corrupt input images.
            raise
        if len(results) != len(valid):
            db.rollback()
            raise RuntimeError('YOLO result count differs from input count')
        for record, result in zip(valid, results):
            _, key, camera, timestamp, size, mtime = record
            if result.boxes is None:
                raise RuntimeError('Expected detection boxes; check model task')
            counts = Counter(int(x) for x in result.boxes.cls.cpu().tolist())
            db.execute('INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?)',
                       (key, camera, timestamp, 'ok', sum(counts.values()),
                        json.dumps(counts), None, size, mtime))
    db.commit()
    return len(valid)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Directory containing camera folders')
    p.add_argument('--weights', type=Path, default=Path('src/models/yolo/best.pt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='auto', help='auto (GPU required), 0, 1, or explicit cpu')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--half', action='store_true', help='FP16 on CUDA only')
    p.add_argument('--classes', type=int, nargs='+', required=True,
                   help='Vehicle class IDs from your custom model, NOT assumed COCO IDs')
    p.add_argument('--timezone', default='Asia/Ho_Chi_Minh', help='Timezone of filename timestamps')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--limit', type=int, help='Max pending images this invocation, useful for benchmarks')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--export-only', action='store_true', help='Export committed results without inference')
    return p


def main():
    args = parser().parse_args()
    if args.batch < 1 or args.imgsz < 1 or args.max_det < 1:
        raise ValueError('batch, imgsz and max-det must be positive')
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Require 0 <= shard-index < num-shards')
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError('conf and iou must be between 0 and 1')
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')
    if not args.source.is_dir() or not args.weights.is_file():
        raise ValueError('source directory or local weights missing')
    ZoneInfo(args.timezone)
    import torch
    import ultralytics
    from ultralytics import YOLO
    device = args.device
    if device == 'auto':
        if not torch.cuda.is_available() and not args.export_only:
            raise RuntimeError('No CUDA GPU. Enable cloud GPU or explicitly pass --device cpu.')
        device = '0'
    if args.half and device == 'cpu':
        raise ValueError('--half requires CUDA')
    model = YOLO(str(args.weights))
    if model.task != 'detect':
        raise ValueError('Expected a detection model')
    names = {int(k): v for k, v in model.names.items()}
    class_ids = sorted(set(args.classes))
    if not set(class_ids) <= names.keys():
        raise ValueError(f'Unknown class IDs. Model classes: {names}')
    selected = {i: names[i] for i in class_ids}
    LOG.info('Vehicle classes: %s; device=%s', selected, device)
    with args.weights.open('rb') as weights_file:
        weights_hash = hashlib.file_digest(weights_file, 'sha256').hexdigest()
    config = dict(weights_sha256=weights_hash,
                  classes=selected, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  max_det=args.max_det, half=args.half, timezone=args.timezone,
                  num_shards=args.num_shards, shard_index=args.shard_index,
                  ultralytics=ultralytics.__version__, torch=torch.__version__, roi='full_frame')
    output = args.output
    db = open_database(output / 'counts.sqlite', config)
    (output / 'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    started, completed, scheduled, discovered = time.monotonic(), 0, 0, 0
    try:
        if not args.export_only:
            batch = []
            for path in args.source.rglob('*'):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                discovered += 1
                key, camera, timestamp = image_key(path, args.source, args.timezone)
                if shard_for(key, args.num_shards) != args.shard_index:
                    continue
                stat = path.stat()
                old = db.execute('SELECT status,size,mtime_ns FROM images WHERE path=?', (key,)).fetchone()
                if old and (old[1], old[2]) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError(f'Input changed since previous run: {key}; use a new output')
                if old and (old[0] == 'ok' or not args.retry_errors):
                    continue
                batch.append((path, key, camera, timestamp, stat.st_size, stat.st_mtime_ns))
                scheduled += 1
                if len(batch) == args.batch:
                    completed += process_batch(model, batch, db, args, class_ids, device)
                    batch = []
                    LOG.info('Processed %d valid images (%.2f images/s)', completed,
                             completed / max(time.monotonic() - started, .001))
                if args.limit and scheduled >= args.limit:
                    break
            if batch:
                completed += process_batch(model, batch, db, args, class_ids, device)
            if not discovered:
                raise ValueError('No images found under source')
    finally:
        db.rollback()
        export_csv(db, output / 'counts.csv', selected)
        LOG.info('Committed results: %s', db.execute(
            'SELECT status,count(*) FROM images GROUP BY status').fetchall())
        db.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
