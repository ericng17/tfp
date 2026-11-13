"""Validate/package YOLO dataset; upload only when --upload is supplied."""
import argparse
import json
import os
import zipfile
from pathlib import Path
from validate_yolo_dataset import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('finetune_data'))
    p.add_argument('--repo-id', default='ericng17/hcm-traffic-yolo-v20')
    p.add_argument('--output', type=Path, default=Path('/tmp/tfp-yolo-upload'))
    p.add_argument('--upload', action='store_true')
    args = p.parse_args()
    result = audit(args.source)
    if result['errors']:
        raise SystemExit(f'Validation failed: {result["errors"][:10]}')
    print('Warnings:', result['warnings'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'dataset_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    archive = args.output / 'dataset.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for path in sorted(args.source.rglob('*')):
            if path.is_file() and (path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.txt', '.yaml', '.yml'}):
                z.write(path, path.relative_to(args.source).as_posix())
    (args.output / 'README.md').write_text('''---
license: cc-by-4.0
task_categories:
- object-detection
---
# HCM Traffic Detection v2 — YOLO export v20

Source: https://universe.roboflow.com/giaothongdetectai/hcm-traffic-detection-v2/dataset/20
Original dataset credited to GiaoThongDetectAI, exported via Roboflow.
See original README.dataset.txt and README.roboflow.txt inside dataset.zip.
License: CC BY 4.0. This repository preserves the exported annotations and split.
Modification: data.yaml split paths corrected to be relative to dataset root.
Classes: 0=2-wheel vehicles, 1=bus, 2=car, 3=truck.

Download dataset.zip, extract, and set an absolute `path` in data.yaml before training.
Audit checks format/integrity only; it does not certify annotation accuracy,
completeness or absence of near-duplicate leakage. See dataset_audit.json.
''')
    print('Prepared:', args.output, 'ZIP bytes:', archive.stat().st_size)
    if args.upload:
        from huggingface_hub import HfApi
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get('HF_TOKEN'):
            raise SystemExit('Set HF_TOKEN environment variable or .env (write permission required).')
        api = HfApi(token=os.environ['HF_TOKEN'])
        api.create_repo(args.repo_id, repo_type='dataset', private=True, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type='dataset', folder_path=args.output,
                          allow_patterns=['dataset.zip', 'dataset_audit.json', 'README.md'],
                          commit_message='Upload audited YOLO v20 export with portable split paths')
        print('Uploaded: https://huggingface.co/datasets/' + args.repo_id)
    else:
        print('Not uploaded. Add --upload to publish to your private dataset repo.')


if __name__ == '__main__':
    main()
