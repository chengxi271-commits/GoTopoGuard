"""Fit the PRICAI R020 full-data diagnostic scorers for local inference.

These are newly fitted inference models, not archived checkpoints or held-out
evaluation outputs. Use reproduce.py for the paper's evaluation protocols.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','GOTOPOGUARD_MODEL_N_JOBS'):
    os.environ.setdefault(key,'1')

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data/pricai2026')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/models')
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('guard_models',ROOT/'experiments/scripts/guard_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=a.data_root.resolve()/'experiments/results'
    config=argparse.Namespace(feature_rows=str(data/'R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl'),
        r014_budget_features=str(data/'R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl'),
        d3_positions=str(data/'R008_ADVERSARIAL_REPLAY_SPLIT/metadata.jsonl'),
        random_state=19019,toporisk_fpr_cap=.05,utility_penalty=1.5)
    scorer,metadata=module.train_guard_scorers(config)
    import joblib,numpy,sklearn
    a.output.mkdir(parents=True,exist_ok=True)
    payload=dict(vars(scorer),feature_names=module.TOPO_FEATURES)
    joblib.dump(payload,a.output/'pricai_scorers.joblib')
    metadata.update({'kind':'new full-data fit for inference; not held-out evaluation',
        'python':sys.version.split()[0],'numpy':numpy.__version__,'scikit-learn':sklearn.__version__,
        'random_state':19019,'feature_names':module.TOPO_FEATURES})
    (a.output/'training.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
