
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo, revision, dataset, need_data, need_weights, result_path = sys.argv[1:]
info = HfApi().dataset_info(repo, revision=revision)
sha = info.sha
available = {f.rfilename for f in info.siblings}
files = {}
if need_data == '1':
    files['data'] = {'luad': 'LUAD-HistoSeg.zip', 'bcss': 'BCSS-WSSS.zip'}[dataset]
if need_weights == '1':
    files['weights'] = 'ilsvrc-cls_rna-a1_cls1000_ep-0001.params'
result = {'repo_id': repo, 'revision': sha}
for key, name in files.items():
    if name not in available:
        raise FileNotFoundError(f'{name} missing from {repo}@{sha}')
    result[key] = hf_hub_download(repo_id=repo, repo_type='dataset', filename=name,
        revision=sha, local_dir='/tmp/sshr-hf/' + sha)
Path(result_path).write_text(json.dumps(result))
