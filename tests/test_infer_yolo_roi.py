import csv
import io
import json
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from scripts import infer_yolo_roi as engine


def tensor(values):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))


class FakeModel:
    task = 'detect'
    names = {0: '2-wheel vehicles', 1: 'bus', 2: 'car', 3: 'truck'}
    calls = 0

    def __init__(self, weights):
        pass

    def predict(self, source, **kwargs):
        FakeModel.calls += 1
        assert kwargs['classes'] == [0,1,2,3]
        results = []
        for image in source:
            # Two inside the overlapping union, one on an edge, two outside.
            boxes = [[40,10,60,50], [10,60,30,80], [80,10,100,50],
                     [40,30,60,95], [45,20,55,60]]
            results.append(SimpleNamespace(orig_shape=(image.height,image.width),
                boxes=SimpleNamespace(xyxy=tensor(boxes), cls=tensor([0,1,0,2,3]))))
        return results


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'snapshots'
        self.camera = self.source / 'cam1'
        self.camera.mkdir(parents=True)
        self.weights = self.root / 'best.pt'
        self.weights.write_bytes(b'test weights')
        self.roi = self.root / 'roi.json'
        self.roi_data = dict(schema_version=1, review_status='approved',
            count_rule=engine.COUNT_RULE, missing_cameras=[], cameras=[dict(
                camera_id='cam1', review_status='approved', reference_size=[100,100],
                regions=[dict(points_normalized=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]),
                         dict(points_normalized=[[.4,.4],[.7,.4],[.7,.7],[.4,.7]])])])
        self.write_roi()
        FakeModel.calls = 0
        self.modules = {
            'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), __version__='test'),
            'ultralytics': SimpleNamespace(YOLO=FakeModel, __version__='test'),
        }

    def write_roi(self):
        self.roi.write_text(json.dumps(self.roi_data))

    def args(self, *extra, hf=False):
        source = ['--hf-repos','owner/repo'] if hf else ['--source',str(self.source)]
        return engine.parser().parse_args(source+[
            '--weights',str(self.weights), '--roi',str(self.roi),
            '--output',str(self.root/'output'), '--classes','0','1','2','3',
            '--batch','2', '--preview-count','1', *extra])

    def image(self, name='20261114_120000.jpg'):
        path = self.camera / name
        Image.new('RGB',(100,100),'white').save(path)
        return path

    def rows(self):
        with (self.root/'output/counts.csv').open() as f:
            return list(csv.DictReader(f))

    def run_engine(self, args):
        with patch.dict(sys.modules,self.modules):
            engine.run(args)

    def test_union_bottom_center_and_csv_resume(self):
        self.image()
        (self.camera/'20261114_120013.jpg').write_bytes(b'broken')
        self.run_engine(self.args())
        rows = self.rows()
        self.assertEqual(rows[0]['total_vehicles'],'3')
        self.assertEqual([rows[0][f'count_{i}_{name}'] for i,name in FakeModel.names.items()],
                         ['1','1','0','1'])
        self.assertEqual(rows[1]['status'],'error')
        self.assertEqual(rows[1]['total_vehicles'],'')
        self.assertEqual(rows[1]['count_0_2-wheel vehicles'],'')
        self.assertEqual(len(rows[0]['roi_sha256']),64)
        self.assertEqual(len(rows[0]['image_sha256']),64)
        previews = list((self.root/'output/previews').glob('*.jpg'))
        self.assertEqual(len(previews),1)
        with Image.open(previews[0]) as image:
            self.assertEqual(image.size,(100,100))
        self.assertEqual(json.loads((self.root/'output/roi.json').read_text()),self.roi_data)
        self.run_engine(self.args('--batch','1'))
        self.assertEqual(FakeModel.calls,1)
        self.assertEqual(len(self.rows()),2)

    def test_empty_detections_zero_and_concave_polygon(self):
        self.image()
        empty = SimpleNamespace(orig_shape=(100,100),
            boxes=SimpleNamespace(xyxy=tensor([]),cls=tensor([])))
        with patch.object(FakeModel,'predict',return_value=[empty]):
            self.run_engine(self.args())
        row = self.rows()[0]
        self.assertEqual(row['total_vehicles'],'0')
        self.assertTrue(all(row[k]=='0' for k in row if k.startswith('count_')))
        concave = [[0,0],[1,0],[1,.4],[.4,.4],[.4,1],[0,1]]
        self.assertTrue(engine.inside_polygon((.2,.8),concave))
        self.assertFalse(engine.inside_polygon((.8,.8),concave))
        self.assertTrue(engine.inside_polygon((.4,.8),concave))
        self.assertTrue(engine.inside_polygon((0,0),concave))

    def test_approved_config_and_self_intersection_required(self):
        for edit in ['global','camera','geometry']:
            with self.subTest(edit=edit):
                data = json.loads(json.dumps(self.roi_data))
                if edit=='global':
                    data['review_status']='draft'
                elif edit=='camera':
                    data['cameras'][0]['review_status']='draft'
                else:
                    data['cameras'][0]['regions'][0]['points_normalized'] = [
                        [0,0],[1,1],[0,1],[1,0]]
                self.roi.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    engine.load_rois(self.roi)

    def test_repository_approved_rois_are_valid(self):
        path = Path(__file__).parents[1]/'configs/camera_rois.reviewed.json'
        cameras, _, _ = engine.load_rois(path)
        self.assertEqual(len(cameras),28)

    def test_unknown_camera_stops_without_counts(self):
        other = self.source/'unknown'
        other.mkdir()
        Image.new('RGB',(100,100)).save(other/'20261114_120000.jpg')
        with self.assertRaisesRegex(ValueError,'No approved ROI'):
            self.run_engine(self.args())
        self.assertEqual(self.rows(),[])
        self.assertEqual(FakeModel.calls,0)

    def test_changed_input_weights_threshold_and_roi_reject_resume(self):
        image = self.image()
        self.run_engine(self.args())
        with self.assertRaisesRegex(ValueError,'Configuration changed'):
            self.run_engine(self.args('--conf','.3'))
        self.weights.write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError,'Configuration changed'):
            self.run_engine(self.args())
        self.weights.write_bytes(b'test weights')
        self.roi_data['cameras'][0]['regions'][0]['points_normalized'][0]=[.1,.1]
        self.write_roi()
        with self.assertRaisesRegex(ValueError,'Configuration changed'):
            self.run_engine(self.args())
        self.roi_data['cameras'][0]['regions'][0]['points_normalized'][0]=[.2,.2]
        self.write_roi()
        image.write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError,'Input changed'):
            self.run_engine(self.args())

    def test_wrong_aspect_ratio_rolls_back(self):
        self.image()
        Image.new('RGB',(200,100)).save(self.camera/'20261114_120013.jpg')
        with self.assertRaisesRegex(ValueError,'aspect ratio'):
            self.run_engine(self.args())
        self.assertEqual(self.rows(),[])

    def test_runtime_failure_rolls_back_entire_batch(self):
        self.image()
        (self.camera/'20261114_120013.jpg').write_bytes(b'broken')
        with patch.object(FakeModel,'predict',side_effect=RuntimeError('OOM')):
            with self.assertRaisesRegex(RuntimeError,'OOM'):
                self.run_engine(self.args())
        self.assertEqual(self.rows(),[])
        self.run_engine(self.args())
        self.assertEqual(len(self.rows()),2)

    def test_local_limit_resume_and_error_retry(self):
        for second in ['00','13','26']:
            self.image(f'20261114_1200{second}.jpg')
        self.run_engine(self.args('--limit','1'))
        self.assertEqual(len(self.rows()),1)
        self.run_engine(self.args())
        self.assertEqual(len(self.rows()),3)
        with closing(sqlite3.connect(self.root/'output/counts.sqlite')) as db:
            db.execute("UPDATE counts SET status='error',total=NULL,counts=NULL")
            db.commit()
        self.run_engine(self.args('--retry-errors'))
        self.assertTrue(all(r['status']=='ok' for r in self.rows()))

    def test_disjoint_shards_cover_source(self):
        for second in ['00','13','26','39']:
            self.image(f'20261114_1200{second}.jpg')
        ids = []
        for shard in [0,1]:
            args = self.args('--num-shards','2','--shard-index',str(shard))
            args.output = self.root/f'shard-{shard}'
            self.run_engine(args)
            with (args.output/'counts.csv').open() as f:
                ids.append({r['id'] for r in csv.DictReader(f)})
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual(len(ids[0] | ids[1]),4)

    def test_empty_source_rejected(self):
        with self.assertRaisesRegex(ValueError,'No images found'):
            self.run_engine(self.args())

    def test_live_backup_only_committed_rows_and_resume(self):
        self.image()
        self.run_engine(self.args())
        output = self.root/'output'
        backup = self.root/'drive-backup'
        with closing(sqlite3.connect(output/'counts.sqlite')) as writer:
            item = engine.record('', '', 'cam1/not_committed.jpg', 0, 'cam1',
                                 '2026-11-14T12:00:13+07:00', None)
            engine.save_count(writer,item,None,'ok',{0:99})
            self.assertTrue(writer.in_transaction)
            self.assertTrue(engine.backup_checkpoint(output,backup))
            self.assertTrue(writer.in_transaction)
            with closing(sqlite3.connect(backup/'counts.sqlite')) as saved:
                self.assertEqual(saved.execute('SELECT total FROM counts').fetchall(),[(3,)])
            writer.rollback()
        with (backup/'counts.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['total_vehicles'] for r in rows],['3'])
        self.assertTrue((backup/'previews').is_dir())
        restored = self.root/'restored'
        shutil.copytree(backup,restored)
        args = self.args()
        args.output = restored
        self.run_engine(args)
        self.assertEqual(FakeModel.calls,1)
        self.assertFalse(list(backup.rglob('*.tmp')))

    def test_failed_backup_keeps_previous_database(self):
        self.image()
        self.run_engine(self.args())
        output = self.root/'output'
        backup = self.root/'drive-backup'
        engine.backup_checkpoint(output,backup)
        previous = (backup/'counts.sqlite').read_bytes()
        self.image('20261114_120013.jpg')
        self.run_engine(self.args())
        copy = shutil.copy2
        def failing_copy(source,target,*args,**kwargs):
            if Path(source).name=='counts.sqlite':
                raise OSError('Drive copy interrupted')
            return copy(source,target,*args,**kwargs)
        with patch.object(engine.shutil,'copy2',side_effect=failing_copy):
            with self.assertRaisesRegex(OSError,'Drive copy interrupted'):
                engine.backup_checkpoint(output,backup)
        self.assertEqual((backup/'counts.sqlite').read_bytes(),previous)
        self.assertFalse(list(backup.rglob('*.tmp')))
        engine.backup_checkpoint(output,backup)
        with closing(sqlite3.connect(backup/'counts.sqlite')) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM counts').fetchone()[0],2)
        (backup/'run.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Backup configuration differs'):
            engine.backup_checkpoint(output,backup)

    def test_hourly_schedule_retries_backup_failures(self):
        periodic = engine.PeriodicCheckpoint(self.root/'output',self.root/'backup')
        self.assertEqual(periodic.interval_seconds,3600)
        periodic._stop = SimpleNamespace(wait=Mock(side_effect=[False,False,True]))
        with patch.object(engine,'backup_checkpoint',side_effect=[OSError('Drive offline'),True]) as backup:
            with self.assertLogs(engine.LOG,level='ERROR') as messages:
                periodic._worker()
        self.assertEqual(backup.call_count,2)
        self.assertEqual([c.args for c in periodic._stop.wait.call_args_list],[(3600,)]*3)
        self.assertIn('Periodic checkpoint failed',messages.output[0])

    def test_periodic_thread_stops_before_final_backup(self):
        called = threading.Event()
        periodic = engine.PeriodicCheckpoint(self.root/'output',self.root/'backup',.01)
        with patch.object(engine,'backup_checkpoint',side_effect=lambda *args:called.set()):
            periodic.start()
            try:
                self.assertTrue(called.wait(timeout=2))
            finally:
                periodic.stop()
        self.assertFalse(periodic._thread.is_alive())
        for interval in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                engine.PeriodicCheckpoint(self.root/'output',self.root/'backup',interval)

    def test_hf_raw_and_parquet_limit_resume(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        stream = io.BytesIO()
        Image.new('RGB',(100,100)).save(stream,format='JPEG')
        content = stream.getvalue()
        shard = self.root/'source.parquet'
        pq.write_table(pa.Table.from_pylist([
            {'image':{'bytes':content,'path':'cam1/20261114_120000.jpg'}},
            {'image':{'bytes':b'broken','path':'cam1/20261114_120013.jpg'}},
            {'image':{'bytes':content,'path':'cam1/20261114_120026.jpg'}},
        ]),shard)
        for mode in ['raw','parquet']:
            with self.subTest(mode=mode):
                output = self.root/'output'
                if output.exists():
                    shutil.rmtree(output)
                revisions = []
                class API:
                    def __init__(self,**kwargs): pass
                    def repo_info(self,*args,**kwargs): return SimpleNamespace(sha='commit1')
                    def list_repo_files(self,*args,**kwargs):
                        revisions.append(kwargs['revision'])
                        return ['data/train.parquet'] if mode=='parquet' else [
                            f'cam1/20261114_1200{s}.jpg' for s in ['00','13','26']]
                def download(repo,source,**kwargs):
                    target = Path(kwargs['local_dir'])/Path(source).name
                    if mode=='parquet':
                        shutil.copy2(shard,target)
                    else:
                        target.write_bytes(b'broken' if '120013' in source else content)
                    return str(target)
                self.modules['huggingface_hub'] = SimpleNamespace(
                    HfApi=API,hf_hub_download=download)
                self.run_engine(self.args('--limit','1',hf=True))
                self.assertEqual(len(self.rows()),1)
                self.run_engine(self.args(hf=True))
                rows = self.rows()
                self.assertEqual([r['status'] for r in rows],['ok','error','ok'])
                self.assertEqual([r['total_vehicles'] for r in rows],['3','','3'])
                calls = FakeModel.calls
                self.run_engine(self.args(hf=True))
                self.assertEqual(FakeModel.calls,calls)
                self.assertEqual(revisions,['commit1']*3)


if __name__ == '__main__':
    unittest.main()
