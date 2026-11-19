"""Count selected vehicle classes inside approved camera ROIs (Python >= 3.11).

Supports local camera folders or Hugging Face raw images/embedded-image Parquet.
Run with --help. Only local detection weights are accepted; no automatic uploads.
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import shutil
import sqlite3
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw

LOG = logging.getLogger(__name__)
EXTENSIONS = {'.jpg', '.jpeg', '.png'}
COUNT_RULE = 'bottom_center_inside_or_on_boundary_of_polygon_union'
EPS = 1e-9


def on_segment(p, a, b):
    cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
    return (abs(cross) <= EPS and
            min(a[0], b[0])-EPS <= p[0] <= max(a[0], b[0])+EPS and
            min(a[1], b[1])-EPS <= p[1] <= max(a[1], b[1])+EPS)


def inside_polygon(point, points):
    """Ray casting for concave polygons, including edges and vertices."""
    x, y = point
    inside = False
    for a, b in zip(points, points[1:]+points[:1]):
        if on_segment(point, a, b):
            return True
        if (a[1] > y) != (b[1] > y):
            if x < a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1]):
                inside = not inside
    return inside


def validate_polygon(points):
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError('ROI polygon requires at least three vertices')
    if any(not isinstance(p, list) or len(p) != 2 or
           any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in points):
        raise ValueError('ROI coordinates must be finite normalized [x,y] pairs')
    edges = list(zip(points, points[1:]+points[:1]))
    if any(a == b for a, b in edges):
        raise ValueError('ROI polygon has a zero-length edge')
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in edges))/2
    if area <= .0001:
        raise ValueError('ROI polygon is degenerate')
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    for i, (a, b) in enumerate(edges):
        for j in range(i+1, len(edges)):
            if j == i+1 or (i == 0 and j == len(edges)-1):
                continue
            c, d = edges[j]
            if (orient(a,b,c)*orient(a,b,d) < 0 and orient(c,d,a)*orient(c,d,b) < 0
                    or on_segment(c,a,b) or on_segment(d,a,b)
                    or on_segment(a,c,d) or on_segment(b,c,d)):
                raise ValueError('ROI polygon self-intersects')


def load_rois(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get('schema_version') != 1 or data.get('count_rule') != COUNT_RULE:
        raise ValueError('Unsupported ROI schema or count_rule')
    if data.get('review_status') != 'approved' or data.get('missing_cameras'):
        raise ValueError('ROI file must be fully approved with no missing cameras')
    cameras = {}
    for camera in data['cameras']:
        key = camera['camera_id']
        if not isinstance(key, str) or not key or key in cameras:
            raise ValueError('ROI camera IDs must be unique nonempty strings')
        if camera.get('review_status') != 'approved' or not camera['regions']:
            raise ValueError(f'Camera {key} has no approved ROI')
        size = camera['reference_size']
        if (len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) or v <= 0 for v in size)):
            raise ValueError(f'Invalid reference size: {key}')
        for region in camera['regions']:
            validate_polygon(region['points_normalized'])
        cameras[key] = camera
    if not cameras:
        raise ValueError('ROI file contains no cameras')
    return cameras, hashlib.sha256(raw).hexdigest(), data


def identity(row, path, args):
    path = Path(str(path or '').replace('\\', '/'))
    camera_keys = [args.camera_column] if args.camera_column else ['camera_id','cam_id','camera']
    time_keys = [args.timestamp_column] if args.timestamp_column else ['timestamp','datetime','time']
    camera = next((row[k] for k in camera_keys if row.get(k) is not None), path.parent.name)
    timestamp = next((row[k] for k in time_keys if row.get(k) is not None), path.stem)
    if not camera or isinstance(camera, (dict, list)):
        raise ValueError(f'Cannot identify camera for {path}')
    if isinstance(timestamp, datetime):
        dt = timestamp
    else:
        try:
            dt = datetime.strptime(str(timestamp), '%Y%m%d_%H%M%S')
        except ValueError:
            dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(args.timezone))
    return str(camera), dt.astimezone(ZoneInfo(args.timezone)).isoformat()


def record(repo, revision, source, index, camera, timestamp, content, fingerprint=None):
    key = hashlib.sha256(json.dumps([repo, revision, source, index]).encode()).hexdigest()
    return dict(id=key, repo=repo, revision=revision, source=source, row_index=index,
                camera_id=camera, timestamp=timestamp, content=content, fingerprint=fingerprint)


def pending(db, item, args):
    if int(item['id'], 16) % args.num_shards != args.shard_index:
        return False
    old = db.execute('SELECT status,fingerprint FROM counts WHERE id=?', (item['id'],)).fetchone()
    if old and old[1] != item['fingerprint']:
        raise ValueError(f"Input changed: {item['source']}; use a new --output")
    return not old or (old[0] == 'error' and args.retry_errors)


def local_records(args, db):
    found = False
    for path in args.source.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        found = True
        camera, timestamp = identity({}, path.relative_to(args.source), args)
        stat = path.stat()
        item = record('', '', path.relative_to(args.source).as_posix(), 0, camera,
                      timestamp, path, f'{stat.st_size}:{stat.st_mtime_ns}')
        if pending(db, item, args):
            yield item
    if not found:
        raise ValueError('No images found under --source')


def parquet_records(local, repo, revision, source, args, db):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(local)
    if args.image_column not in pf.schema_arrow.names:
        raise ValueError(f'{source}: missing column {args.image_column}')
    index = 0
    for batch in pf.iter_batches(batch_size=args.batch):
        for row in batch.to_pylist():
            image = row[args.image_column]
            path = row.get(args.path_column or 'path') or row.get('file_name') or row.get('filename')
            if isinstance(image, dict):
                path = path or image.get('path')
                content = image.get('bytes')
            elif isinstance(image, (bytes, type(None))):
                content = image
            else:
                raise ValueError('Parquet image must contain embedded bytes')
            camera, timestamp = identity(row, path, args)
            item = record(repo, revision, source, index, camera, timestamp, content)
            if pending(db, item, args):
                yield item
            index += 1
    # A partial final batch remains unmarked and is safely replayed on resume.
    saved = db.execute('SELECT count(*) FROM counts WHERE repo=? AND revision=? AND source=?',
                       (repo,revision,source)).fetchone()[0]
    if args.num_shards == 1 and saved == index:
        key = 'completed:'+json.dumps([repo,revision,source])
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,'true'))
        db.commit()


def hf_records(args, db):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    for repo in args.hf_repos:
        key = 'revision:'+repo
        old = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        revision = old[0] if old else api.repo_info(repo, repo_type='dataset').sha
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (key, revision))
        db.commit()
        files = sorted(api.list_repo_files(repo, repo_type='dataset', revision=revision))
        parquet = [p for p in files if p.endswith('.parquet')]
        raw = [p for p in files if Path(p).suffix.lower() in EXTENSIONS]
        use_parquet = args.format == 'parquet' or (args.format == 'auto' and bool(parquet))
        LOG.info('%s @ %s: %s', repo, revision, 'parquet' if use_parquet else 'raw')
        if not (parquet if use_parquet else raw):
            raise ValueError(f'{repo}: no files for the requested format')
        if use_parquet:
            for source in parquet:
                complete_key = 'completed:'+json.dumps([repo,revision,source])
                if not args.retry_errors and db.execute(
                        'SELECT 1 FROM metadata WHERE key=?',(complete_key,)).fetchone():
                    continue
                with tempfile.TemporaryDirectory(prefix='yolo-roi-parquet-') as tmp:
                    local = hf_hub_download(repo, source, repo_type='dataset',
                                            revision=revision, local_dir=tmp)
                    yield from parquet_records(local, repo, revision, source, args, db)
        else:
            for start in range(0, len(raw), args.batch):
                chunk = []
                for source in raw[start:start+args.batch]:
                    camera, timestamp = identity({}, source, args)
                    item = record(repo, revision, source, 0, camera, timestamp, None)
                    if pending(db, item, args):
                        chunk.append(item)
                with tempfile.TemporaryDirectory(prefix='yolo-roi-raw-') as tmp:
                    def download(item):
                        local = hf_hub_download(repo, item['source'], repo_type='dataset',
                                                revision=revision, local_dir=tmp)
                        return Path(local).read_bytes()
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        for item, content in zip(chunk, pool.map(download, chunk)):
                            item['content'] = content
                            yield item


def open_database(output, config):
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / 'counts.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('''CREATE TABLE IF NOT EXISTS counts (
        id TEXT PRIMARY KEY, repo TEXT, revision TEXT, source TEXT, row_index INTEGER,
        camera_id TEXT, timestamp TEXT, fingerprint TEXT, image_sha256 TEXT,
        status TEXT, total INTEGER, counts TEXT, error TEXT)''')
    encoded = json.dumps(config, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    if old and old[0] != encoded:
        db.close()
        raise ValueError('Configuration changed: use a new --output directory')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('config',?)", (encoded,))
    db.commit()
    (output / 'run.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n')
    return db


def save_count(db, item, digest, status, counts=None, error=None):
    fields = [item[k] for k in ('id','repo','revision','source','row_index',
                                'camera_id','timestamp','fingerprint')]
    db.execute('INSERT OR REPLACE INTO counts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
               fields+[digest, status, sum(counts.values()) if counts is not None else None,
                       json.dumps(counts) if counts is not None else None, error])


def export_csv(db, output, names, roi_hash):
    target = output / 'counts.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','repo','revision','source','row_index','camera_id','timestamp',
                         'status','total_vehicles']+[f'count_{k}_{v}' for k,v in names.items()]+
                        ['image_sha256','roi_sha256','error'])
        query = '''SELECT id,repo,revision,source,row_index,camera_id,timestamp,status,
                   total,counts,image_sha256,error FROM counts ORDER BY camera_id,timestamp,id'''
        for row in db.execute(query):
            counts = json.loads(row[9]) if row[9] else {}
            writer.writerow(list(row[:9])+[counts.get(str(k),0) if row[7]=='ok' else ''
                                           for k in names]+[row[10],roi_hash,row[11]])
    temporary.replace(target)


def backup_checkpoint(output, destination):
    """Snapshot committed SQLite rows, including while inference is running.

    SQLite's backup API reads a consistent snapshot through a separate connection.
    Stage destination files before replacement so an interrupted copy never
    truncates the previous checkpoint database. One backup writer per destination.
    """
    output, destination = Path(output), Path(destination)
    if not (output / 'counts.sqlite').is_file():
        return False
    if (output.resolve().is_relative_to(destination.resolve()) or
            destination.resolve().is_relative_to(output.resolve())):
        raise ValueError('Checkpoint destination must be separate from the output directory')
    config = json.loads((output / 'run.json').read_text())
    previous_config = destination / 'run.json'
    if previous_config.is_file() and json.loads(previous_config.read_text()) != config:
        raise ValueError('Backup configuration differs; choose another run/destination')
    with tempfile.TemporaryDirectory(prefix='yolo-roi-checkpoint-') as tmp:
        snapshot = Path(tmp)
        source_uri = (output / 'counts.sqlite').resolve().as_uri()+'?mode=ro'
        with closing(sqlite3.connect(source_uri, uri=True, timeout=60)) as source:
            with closing(sqlite3.connect(snapshot / 'counts.sqlite')) as target:
                source.backup(target)
                saved = target.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
                if not saved or json.loads(saved[0]) != config:
                    raise ValueError('Database and run.json configurations differ')
                export_csv(target, snapshot,
                           {int(k): v for k,v in config['names'].items()}, config['roi_sha256'])
        for filename in ('run.json','roi.json'):
            shutil.copy2(output / filename, snapshot / filename)
        destination.mkdir(parents=True, exist_ok=True)

        def publish(source_path, target_path):
            # Same destination filesystem for replace, including mounted Drive.
            with tempfile.NamedTemporaryFile(dir=target_path.parent, prefix='.'+target_path.name,
                                             suffix='.tmp', delete=False) as f:
                staged = Path(f.name)
            try:
                shutil.copy2(source_path, staged)
                staged.replace(target_path)
            finally:
                staged.unlink(missing_ok=True)

        for filename in ('run.json','roi.json','counts.csv','counts.sqlite'):
            publish(snapshot / filename, destination / filename)
        for preview in (output / 'previews').glob('*.jpg'):
            target_dir = destination / 'previews'
            target_dir.mkdir(exist_ok=True)
            publish(preview, target_dir / preview.name)
    LOG.info('Checkpoint backed up: %s', destination)
    return True


class PeriodicCheckpoint:
    """Back up on a background thread; the inference connection stays on its thread."""

    def __init__(self, output, destination, interval_seconds=3600):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError('Checkpoint interval must be positive and finite')
        self.output = Path(output)
        self.destination = Path(destination)
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = None

    def _worker(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                backup_checkpoint(self.output, self.destination)
            except Exception:
                LOG.exception('Periodic checkpoint failed; retrying at the next interval')

    def start(self):
        if self._thread is not None:
            raise RuntimeError('Periodic checkpoint worker already started')
        self._thread = threading.Thread(target=self._worker, name='roi-checkpoint', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            # Wait for any active backup before the caller makes a final backup.
            self._thread.join()


def filter_boxes(result, camera, class_ids):
    if result.boxes is None:
        raise RuntimeError('Expected detection boxes')
    height, width = result.orig_shape
    rw, rh = camera['reference_size']
    if width <= 0 or height <= 0 or abs(width*rh/(height*rw)-1) > .01:
        raise ValueError(f"Image aspect ratio differs from ROI reference: {camera['camera_id']}")
    boxes = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    if len(boxes) != len(classes):
        raise RuntimeError('Detection coordinates/classes mismatch')
    kept, counts = [], Counter()
    for box, class_id in zip(boxes, classes):
        class_id = int(class_id)
        if class_id not in class_ids:
            continue
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise RuntimeError('Invalid detection coordinates')
        x1, y1, x2, y2 = box
        point = ((x1+x2)/(2*width), y2/height)
        if any(inside_polygon(point, r['points_normalized']) for r in camera['regions']):
            counts[class_id] += 1
            kept.append((box, class_id))
    return counts, kept


def save_preview(image, camera, kept, counts, names, target):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for region in camera['regions']:
        points = [(x*image.width,y*image.height) for x,y in region['points_normalized']]
        draw.line(points+points[:1], fill='#36e8ae', width=2)
    for box, class_id in kept:
        draw.rectangle(box, outline='#ffca62', width=2)
        x1,y1,x2,y2 = box
        x,y = (x1+x2)/2,y2
        draw.ellipse((x-2,y-2,x+2,y+2), fill='#ffca62')
        draw.text((x1,y1), str(class_id), fill='#ffca62')
    label = f'ROI total: {sum(counts.values())} | '+', '.join(
        f'{k} {v}: {counts[k]}' for k,v in names.items())
    draw.rectangle((0,0,image.width,30), fill='black')
    draw.text((4,4), label, fill='white')
    image.save(target, quality=95)


def process_batch(model, records, db, args, rois, names, device):
    images, good = [], []
    for item in records:
        camera = rois.get(item['camera_id'])
        if camera is None:
            raise ValueError(f"No approved ROI for camera {item['camera_id']}")
        try:
            content = item['content']
            if isinstance(content, Path):
                stat = content.stat()
                if f'{stat.st_size}:{stat.st_mtime_ns}' != item['fingerprint']:
                    raise RuntimeError(f'Input changed while reading: {content}')
                content = content.read_bytes()
            if not content:
                raise ValueError('Missing embedded image bytes')
            digest = hashlib.sha256(content).hexdigest()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                images.append(image.convert('RGB'))
            good.append((item, digest))
        except (OSError, ValueError) as exc:
            save_count(db, item, None, 'error', error=f'{type(exc).__name__}: {exc}')
    if images:
        results = list(model.predict(source=images, device=device, imgsz=args.imgsz,
            conf=args.conf, iou=args.iou, max_det=args.max_det, classes=list(names),
            half=args.half, stream=True, verbose=False, save=False))
        if len(results) != len(good):
            raise RuntimeError('YOLO result count differs from input count')
        preview_dir = args.output / 'previews'
        preview_dir.mkdir(exist_ok=True)
        for image, result, (item,digest) in zip(images, results, good):
            if tuple(result.orig_shape) != (image.height,image.width):
                raise RuntimeError('YOLO original shape differs from source image')
            counts, kept = filter_boxes(result, rois[item['camera_id']], names)
            save_count(db, item, digest, 'ok', counts)
            if args.previews_saved < args.preview_count:
                save_preview(image, rois[item['camera_id']], kept, counts, names,
                             preview_dir / f"{item['id']}.jpg")
                args.previews_saved += 1
    db.commit()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', type=Path, help='Local image directory with camera subfolders')
    source.add_argument('--hf-repos', nargs='+', help='Hugging Face dataset repo IDs')
    p.add_argument('--roi', type=Path, required=True, help='Approved camera ROI JSON')
    p.add_argument('--weights', type=Path, required=True, help='Local detection .pt weights')
    p.add_argument('--classes', type=int, nargs='+', required=True, help='Vehicle IDs from model.names')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='auto', help='auto requires CUDA; use cpu explicitly for CPU')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--iou', type=float, default=.7)
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--half', action='store_true')
    p.add_argument('--timezone', default='Asia/Ho_Chi_Minh')
    p.add_argument('--limit', type=int, help='Maximum new/retried images; omit for all')
    p.add_argument('--preview-count', type=int, default=28)
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--format', choices=['auto','raw','parquet'], default='auto', help='HF format')
    p.add_argument('--workers', type=int, default=4, help='HF raw download workers')
    p.add_argument('--image-column', default='image')
    p.add_argument('--path-column')
    p.add_argument('--camera-column')
    p.add_argument('--timestamp-column')
    return p


def run(args):
    if any(v < 1 for v in (args.batch,args.imgsz,args.max_det,args.workers,args.num_shards)):
        raise ValueError('batch, imgsz, max-det, workers and num-shards must be positive')
    if (not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or args.preview_count < 0
            or not 0 <= args.shard_index < args.num_shards
            or (args.limit is not None and args.limit < 1)):
        raise ValueError('Invalid threshold, preview count, shard index or limit')
    if not args.weights.is_file() or (args.source is not None and not args.source.is_dir()):
        raise ValueError('Local weights or source directory missing')
    if args.hf_repos and len(set(args.hf_repos)) != len(args.hf_repos):
        raise ValueError('Duplicate HF repos')
    ZoneInfo(args.timezone)
    rois, roi_hash, roi_data = load_rois(args.roi)
    import torch
    import ultralytics
    from ultralytics import YOLO
    device = args.device
    if device == 'auto':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: enable GPU or pass --device cpu')
        device = '0'
    if args.half and (device == 'cpu' or device == 'mps'):
        raise ValueError('--half requires CUDA')
    model = YOLO(str(args.weights))
    if model.task != 'detect':
        raise ValueError('Detection weights required')
    all_names = {int(k):v for k,v in model.names.items()}
    if not set(args.classes) <= all_names.keys():
        raise ValueError(f'Invalid vehicle classes; model mapping: {all_names}')
    names = {k:all_names[k] for k in sorted(set(args.classes))}
    with args.weights.open('rb') as f:
        weights_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    config = dict(engine_version=1, weights_sha256=weights_hash, roi_sha256=roi_hash,
                  count_rule=COUNT_RULE, names=names,
                  source=str(args.source.resolve()) if args.source else None,
                  hf_repos=args.hf_repos, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  max_det=args.max_det, half=args.half, timezone=args.timezone,
                  num_shards=args.num_shards, shard_index=args.shard_index, format=args.format,
                  image_column=args.image_column, path_column=args.path_column,
                  camera_column=args.camera_column, timestamp_column=args.timestamp_column,
                  torch=torch.__version__, ultralytics=ultralytics.__version__)
    db = open_database(args.output, config)
    (args.output / 'roi.json').write_text(json.dumps(roi_data,indent=2,ensure_ascii=False)+'\n')
    args.previews_saved = sum(1 for _ in (args.output / 'previews').glob('*.jpg'))
    records = local_records(args,db) if args.source else hf_records(args,db)
    processed, batch = 0, []
    LOG.info('Vehicle classes: %s; ROI cameras: %d; device: %s', names,len(rois),device)
    try:
        for item in records:
            if item['camera_id'] not in rois:
                raise ValueError(f"No approved ROI for camera {item['camera_id']}")
            batch.append(item)
            processed += 1
            if len(batch) == args.batch:
                process_batch(model,batch,db,args,rois,names,device)
                batch.clear()
                LOG.info('Processed %d new/retried images', processed)
            if args.limit and processed >= args.limit:
                break
        if batch:
            process_batch(model,batch,db,args,rois,names,device)
    finally:
        records.close()
        db.rollback()
        export_csv(db,args.output,names,roi_hash)
        LOG.info('Saved %s: %s',args.output,db.execute(
            'SELECT status,count(*) FROM counts GROUP BY status').fetchall())
        db.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(parser().parse_args())
