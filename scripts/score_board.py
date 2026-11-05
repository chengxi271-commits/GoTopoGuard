"""Score a board with trusted, locally fitted PRICAI model files."""
import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('board',type=Path,help='JSON array of B/W/. row strings')
    p.add_argument('--model',type=Path,default=ROOT/'reproduced/models/pricai_scorers.joblib')
    a=p.parse_args()
    # Joblib uses pickle. Load only files you created or otherwise trust.
    bundle=joblib.load(a.model)
    spec=importlib.util.spec_from_file_location('r011',ROOT/'experiments/scripts/r011_detection_baseline_sweep.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=json.loads(a.board.read_text(encoding='utf-8'))
    m.TOPO.parse_board(rows)
    if len(rows)!=19 or any(len(row)!=19 for row in rows):
        p.error('The fitted scorer uses 19x19 boards; the topology-only API accepts smaller examples.')
    f=m.feature_row_from_board(position_id='input',board_rows=rows,label=0,dataset='input',family='input')
    x=np.array([[float(f[k]) for k in bundle['feature_names']]])
    risk=float(bundle['toporisk_model'].predict_proba(x)[0,1])
    correction=float(bundle['utility_corr_model'].predict_proba(x)[0,1])
    regression=float(bundle['utility_reg_model'].predict_proba(x)[0,1])
    utility=correction-bundle['utility_penalty']*regression
    print(json.dumps({'toporisk':risk,'toporisk_threshold':bundle['toporisk_threshold'],
        'toporisk_triggered':risk>=bundle['toporisk_threshold'],
        'utility':utility,'utility_threshold':bundle['utility_threshold'],
        'utility_triggered':utility>=bundle['utility_threshold'],
        'interpretation':'diagnostic model outputs, not calibrated failure probabilities'},indent=2))

if __name__=='__main__':main()
