"""Bounded-memory Hugging Face YOLO inference, SQLite resume, raw images/Parquet."""
import csv
import hashlib
import io
import json
import sqlite3
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image


def identity(row, path, timezone, camera_column=None, timestamp_column=None):
    camera_keys = [camera_column] if camera_column else ['camera_id', 'cam_id', 'camera']
    time_keys = [timestamp_column] if timestamp_column else ['timestamp', 'datetime', 'time']
    camera = next((row[k] for k in camera_keys if row.get(k) is not None), None)
    timestamp = next((row[k] for k in time_keys if row.get(k) is not None), None)
    path = Path(str(path or '').replace('\\', '/'))
    if camera is None and path.parent.name:
        camera = path.parent.name
    if timestamp is None:
        timestamp = path.stem
    if camera is None or isinstance(camera, (dict, list)):
        raise ValueError('Cannot identify camera. Configure CAMERA_COLUMN/PATH_COLUMN.')
    if isinstance(timestamp, datetime):
        dt = timestamp
    else:
        value = str(timestamp)
        try:
            dt = datetime.strptime(value, '%Y%m%d_%H%M%S')
        except ValueError:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone))
    return str(camera), dt.isoformat()


def setup_db(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / 'counts.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS completed (repo TEXT, revision TEXT, source TEXT, PRIMARY KEY(repo,revision,source))')
    db.execute('''CREATE TABLE IF NOT EXISTS counts (
        id TEXT PRIMARY KEY, repo TEXT, revision TEXT, source TEXT, row_index INTEGER,
        camera_id TEXT, timestamp TEXT, status TEXT, total INTEGER, counts TEXT,
        image_sha256 TEXT, error TEXT)''')
    encoded = json.dumps(config, sort_keys=True)
    old = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
    if old and old[0] != encoded:
        raise ValueError('Configuration changed: choose a new OUTPUT directory.')
    db.execute("INSERT OR IGNORE INTO meta VALUES ('config',?)", (encoded,))
    db.commit()
    (output / 'run.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    return db


def export(db, output, names):
    target = Path(output) / 'counts.csv'
    tmp = target.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','repo','revision','source','row_index','camera_id','timestamp',
                         'status','total_vehicles'] + [f'count_{k}_{v}' for k,v in names.items()] +
                        ['image_sha256','error'])
        for row in db.execute('SELECT * FROM counts ORDER BY repo,source,row_index'):
            counts = json.loads(row[9]) if row[9] else {}
            writer.writerow(list(row[:9]) +
                [counts.get(str(k), 0) if row[7]=='ok' else '' for k in names] + list(row[10:]))
    tmp.replace(target)


def parquet_records(local, filename, config):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(local)
    keys = pf.schema_arrow.names
    image_col = config.get('image_column', 'image')
    if image_col not in keys:
        raise ValueError(f'{filename}: missing image column {image_col}; columns={keys}')
    print('Parquet schema:', pf.schema_arrow, flush=True)
    index = 0
    for batch in pf.iter_batches(batch_size=config['batch']):
        for row in batch.to_pylist():
            image = row[image_col]
            path = row.get(config.get('path_column') or 'path') or row.get('file_name') or row.get('filename')
            if isinstance(image, dict):
                path = path or image.get('path')
                content = image.get('bytes')
            elif isinstance(image, bytes):
                content = image
            else:
                raise ValueError('Parquet image must contain embedded bytes (HF Image struct supported).')
            # Metadata errors are fatal: never silently lose camera/timestamp.
            camera, timestamp = identity(row, path, config['timezone'],
                config.get('camera_column'), config.get('timestamp_column'))
            yield index, camera, timestamp, content
            index += 1


def run(config):
    if config['batch'] < 1 or config['workers'] < 1:
        raise ValueError('batch and workers must be positive')
    if config['limit'] is not None and config['limit'] < 1:
        raise ValueError('limit must be positive or None')
    if config.get('format', 'auto') not in {'auto', 'raw', 'parquet'}:
        raise ValueError('format must be auto/raw/parquet')
    from huggingface_hub import HfApi, hf_hub_download
    import torch
    import ultralytics
    from ultralytics import YOLO
    import os
    assert torch.cuda.is_available(), 'Enable GPU; CPU fallback is not automatic.'
    weights = Path(config['weights'])
    assert weights.is_file(), f'Weights missing: {weights}'
    with weights.open('rb') as f:
        weight_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    model = YOLO(str(weights))
    if model.task != 'detect':
        raise ValueError('Detection weights required')
    names = {int(k):v for k,v in model.names.items()}
    classes = sorted(config['classes'])
    if not classes or not set(classes) <= names.keys():
        raise ValueError(f'Invalid classes; model mapping={names}')
    names = {k:names[k] for k in classes}
    settings = {k:v for k,v in config.items() if k not in {'weights','output','batch','limit','preview_count','workers'}}
    settings.update(weights_sha256=weight_hash, names=names,
                    ultralytics=ultralytics.__version__, torch=torch.__version__)
    output = Path(config['output'])
    db = setup_db(output, settings)
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    pending, processed = [], 0
    preview_dir = output / 'previews'
    preview_dir.mkdir(exist_ok=True)
    previews = len(list(preview_dir.glob('*.jpg')))

    def flush():
        nonlocal previews
        if not pending:
            return
        images, good = [], []
        for key, repo, revision, source, index, camera, timestamp, content in pending:
            prefix = (key,repo,revision,source,index,camera,timestamp)
            try:
                if not content:
                    raise ValueError('Missing embedded image bytes')
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    images.append(image.convert('RGB'))
                good.append((prefix,hashlib.sha256(content).hexdigest()))
            except (OSError, ValueError) as exc:
                db.execute('INSERT OR REPLACE INTO counts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    prefix+('error',None,None,None,f'{type(exc).__name__}: {exc}'))
        if images:
            results = list(model.predict(images, device=0, imgsz=config['imgsz'],
                conf=config['conf'], iou=config['iou'], max_det=config['max_det'],
                classes=classes, half=config['half'], stream=True, verbose=False, save=False))
            if len(results)!=len(good):
                raise RuntimeError('YOLO result count mismatch')
            for result,(prefix,digest) in zip(results,good):
                counts = Counter(int(i) for i in result.boxes.cls.cpu().tolist())
                db.execute('INSERT OR REPLACE INTO counts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    prefix+('ok',sum(counts.values()),json.dumps(counts),digest,None))
                if previews < config['preview_count']:
                    Image.fromarray(result.plot()[..., ::-1]).save(preview_dir / f'{prefix[0]}.jpg')
                    previews += 1
        db.commit()
        pending.clear()

    def accept(repo, revision, source, index, camera, timestamp, content):
        nonlocal processed
        key = hashlib.sha256(json.dumps([repo,revision,source,index]).encode()).hexdigest()
        if db.execute('SELECT 1 FROM counts WHERE id=?',(key,)).fetchone():
            return
        pending.append((key,repo,revision,source,index,camera,timestamp,content))
        processed += 1
        if len(pending)>=config['batch']:
            flush()
            if processed % (config['batch'] * 100) == 0:
                print(f'Processed {processed} new images',flush=True)

    try:
        for repo in config['repos']:
            pin_key = 'revision:'+repo
            old = db.execute('SELECT value FROM meta WHERE key=?',(pin_key,)).fetchone()
            revision = old[0] if old else api.repo_info(repo,repo_type='dataset').sha
            db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)',(pin_key,revision))
            db.commit()
            files = sorted(api.list_repo_files(repo,repo_type='dataset',revision=revision))
            parquet = [p for p in files if p.endswith('.parquet')]
            raw = [p for p in files if Path(p).suffix.lower() in {'.jpg','.jpeg','.png'}]
            # Prevent double counting if repo exposes both formats.
            mode = config.get('format','auto')
            use_parquet = mode=='parquet' or (mode=='auto' and bool(parquet))
            print(repo,revision,'parquet' if use_parquet else 'raw',flush=True)
            if use_parquet:
                if not parquet: raise ValueError(f'{repo}: no Parquet found')
                for source in parquet:
                    if db.execute('SELECT 1 FROM completed WHERE repo=? AND revision=? AND source=?', (repo,revision,source)).fetchone():
                        continue
                    with tempfile.TemporaryDirectory(prefix='yolo-shard-') as tmp:
                        local = hf_hub_download(repo,source,repo_type='dataset',revision=revision,local_dir=tmp)
                        for index,camera,timestamp,content in parquet_records(local,source,config):
                            accept(repo,revision,source,index,camera,timestamp,content)
                            if config['limit'] and processed>=config['limit']: break
                        flush()
                        if not config['limit'] or processed<config['limit']:
                            db.execute('INSERT OR IGNORE INTO completed VALUES (?,?,?)',(repo,revision,source))
                            db.commit()
                    if config['limit'] and processed>=config['limit']: break
            else:
                if not raw: raise ValueError(f'{repo}: no raw images; check format/archives')
                # Download only one bounded group at a time, then remove its scratch directory.
                for start in range(0,len(raw),config['batch']):
                    chunk = raw[start:start+config['batch']]
                    chunk = [p for p in chunk if not db.execute('SELECT 1 FROM counts WHERE id=?',(
                        hashlib.sha256(json.dumps([repo,revision,p,0]).encode()).hexdigest(),)).fetchone()]
                    if config['limit']: chunk=chunk[:max(0,config['limit']-processed)]
                    with tempfile.TemporaryDirectory(prefix='yolo-raw-') as tmp:
                        def download(source):
                            local=hf_hub_download(repo,source,repo_type='dataset',revision=revision,local_dir=tmp)
                            return Path(local).read_bytes()
                        with ThreadPoolExecutor(max_workers=config['workers']) as pool:
                            for source,content in zip(chunk,pool.map(download,chunk)):
                                camera,timestamp=identity({},source,config['timezone'])
                                accept(repo,revision,source,0,camera,timestamp,content)
                        flush()
                    if config['limit'] and processed>=config['limit']: break
            export(db,output,names)
            if config['limit'] and processed>=config['limit']: break
    finally:
        db.rollback()  # An interrupted/uncommitted batch is processed again on resume.
        export(db,output,names)
        print('Saved:',output,db.execute('SELECT status,count(*) FROM counts GROUP BY status').fetchall())
        db.close()
