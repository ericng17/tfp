import importlib.util
import io
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image

spec=importlib.util.spec_from_file_location('engine',Path(__file__).parents[1]/'scripts/infer_hf_yolo.py')
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


class CountingTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(engine.identity({},'snapshots/cam1/20261114_120000.jpg','Asia/Ho_Chi_Minh'),
                         ('cam1','2026-11-14T12:00:00+07:00'))
        with self.assertRaises(ValueError): engine.identity({},'bad.jpg','UTC')

    def test_parquet_resume_export_and_configuration_guard(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stream=io.BytesIO()
            Image.new('RGB',(16,16)).save(stream,format='JPEG')
            content=stream.getvalue()
            shard=root/'source.parquet'
            pq.write_table(pa.Table.from_pylist([
                {'image':{'bytes':content,'path':'cam1/20261114_120000.jpg'}},
                {'image':{'bytes':b'broken','path':'cam1/20261114_120013.jpg'}},
            ]),shard)
            class API:
                def __init__(self,**kwargs):pass
                def repo_info(self,*args,**kwargs): return SimpleNamespace(sha='commit1')
                def list_repo_files(self,*args,**kwargs): return ['data/train.parquet']
            def download(repo,source,**kwargs):
                target=Path(kwargs['local_dir'])/'part.parquet'
                shutil.copy2(shard,target)
                return str(target)
            class Model:
                names={0:'motorcycle',1:'car'}
                task='detect'
                calls=0
                def __init__(self,*args):pass
                def predict(self,images,**kwargs):
                    Model.calls+=1
                    return [SimpleNamespace(boxes=SimpleNamespace(cls=SimpleNamespace(
                        cpu=lambda:SimpleNamespace(tolist=lambda:[0,0,1])))) for _ in images]
            modules={
                'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),__version__='test'),
                'ultralytics':SimpleNamespace(YOLO=Model,__version__='test'),
                'huggingface_hub':SimpleNamespace(HfApi=API,hf_hub_download=download),
            }
            weights=root/'best.pt';weights.write_bytes(b'weights')
            config=dict(repos=['owner/repo'],weights=str(weights),output=str(root/'output'),
                classes=[0,1],batch=2,imgsz=1024,conf=.25,iou=.7,max_det=1000,half=False,
                timezone='UTC',limit=None,preview_count=0,workers=1,image_column='image')
            with patch.dict(sys.modules,modules):
                engine.run(config)
                engine.run(config)
                self.assertEqual(Model.calls,1)
                with sqlite3.connect(root/'output/counts.sqlite') as db:
                    rows=db.execute('SELECT status,total FROM counts ORDER BY row_index').fetchall()
                    self.assertEqual(rows,[('ok',3),('error',None)])
                    self.assertEqual(db.execute('SELECT count(*) FROM completed').fetchone()[0],1)
                self.assertTrue((root/'output/counts.csv').is_file())
                with self.assertRaises(ValueError): engine.run(dict(config,conf=.1))


if __name__=='__main__':unittest.main()
