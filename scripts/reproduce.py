"""Reproduce PRICAI analyses from released measurements, without running Go engines."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

# Set before loading numerical libraries. Users can explicitly choose larger values.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','GOTOPOGUARD_MODEL_N_JOBS','GOTOPOGUARD_GROUP_N_JOBS'):
    os.environ.setdefault(key,'1')

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'experiments/scripts'

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True,choices=['summary','detection','guard-s09','guard-sweep','early-warning','operating-points','normal-aware','controls','significance','stress-records'])
    p.add_argument('--data-root',type=Path,default=ROOT/'data/pricai2026',help='Directory where the evidence archive was extracted')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced')
    args=p.parse_args(argv)
    data=args.data_root.resolve()/'experiments/results';out=args.output.resolve()/args.suite;out.mkdir(parents=True,exist_ok=True)
    if args.suite=='summary':
        return load('verify_paper',ROOT/'scripts/verify_paper.py').main(['--output',str(out)])
    features=data/'R011_DETECTION_BASELINE_SWEEP/feature_rows.jsonl'
    d3=data/'R008_ADVERSARIAL_REPLAY_SPLIT/metadata.jsonl'
    if not features.exists():
        p.error('Evidence inputs missing. Run scripts/fetch_artifacts.py, or extract the release evidence ZIP into --data-root.')
    common=['--feature-rows',str(features)]
    budget=data/'R014_GUARD_BUDGET_QUERIES/position_budget_features.jsonl'
    if args.suite=='detection':
        module=load('r011',SCRIPTS/'r011_detection_baseline_sweep.py')
        rows=read_jsonl(features)
        metrics=module.run_family_heldout(rows,include_search_baselines=True,include_metamorphic_baselines=True)
        module.write_jsonl(out/'family_heldout_metrics.jsonl',metrics)
        write(out/'run.json',{'rows':len(rows),'metric_rows':len(metrics),'mode':'refit on released measured features'})
    elif args.suite in {'guard-s09','guard-sweep'}:
        configs='S09_b3_extra_p1p5' if args.suite=='guard-s09' else 'all'
        cmd=[sys.executable,str(SCRIPTS/'r014c_utility_guard_model_sweep.py'),*common,'--budget-features',str(budget),'--d3-positions',str(d3),'--configs',configs,'--output-root',str(out),'--fail-on-incomplete']
        subprocess.run(cmd,cwd=ROOT,check=True)
    elif args.suite=='early-warning':
        cmd=[sys.executable,str(SCRIPTS/'r012_lead_time_evaluation.py'),*common,'--metrics',str(data/'R011_DETECTION_BASELINE_SWEEP/family_heldout_metrics.jsonl'),'--d3-positions',str(d3),'--detector-baseline','B3_topology_gbdt','--output-root',str(out),'--fail-on-incomplete']
        subprocess.run(cmd,cwd=ROOT,check=True)
    elif args.suite=='operating-points':
        cmd=[sys.executable,str(SCRIPTS/'r013_operating_point_tradeoff.py'),*common,'--d1-positions',str(data/'R006_NATURAL_HUMAN_SPLIT/metadata.jsonl'),'--d2-positions',str(data/'R007_NATURAL_ENGINE_SPLIT/metadata.jsonl'),'--d3-positions',str(d3),'--output-root',str(out),'--fail-on-incomplete']
        subprocess.run(cmd,cwd=ROOT,check=True)
    elif args.suite=='normal-aware':
        # This original script uses the same input switches documented by --help.
        module=load('r015b',SCRIPTS/'r015b_normal_aware_utility_guard.py')
        a=module.build_arg_parser().parse_args([])
        for name,value in vars(a).copy().items():
            if isinstance(value,str) and value.startswith('experiments/results/'):
                value=str(args.data_root.resolve()/value)
                if name in {'d1_positions','d2_positions','d3_positions'}:
                    value=str(Path(value).with_name('metadata.jsonl'))
                setattr(a,name,value)
        a.output_root=str(out)
        result=module.run(a)
        if result:raise SystemExit(result)
    elif args.suite in {'controls','significance','stress-records'}:
        module=load('r022',ROOT/'analysis/r022_controls.py')
        module.ROOT=args.data_root.resolve();module.OUT=out
        if args.suite=='controls':
            rows=read_jsonl(data/'R022_STRONG_ACCEPT_AUDIT/enriched_feature_rows.jsonl')
            summary={}
            for name,fn in [('strict_control_family',module.family_heldout_metrics),('strict_control_matched',module.matched_family_metrics),('mechanism_holdout',module.mechanism_holdout)]:
                records,summarized=fn(rows);module.write_jsonl(out/(name+'_metrics.jsonl'),records);summary[name]=summarized
            write(out/'summary.json',summary)
        elif args.suite=='significance':
            write(out/'guard_significance.json',module.guard_significance())
        else:
            write(out/'score_sensitivity.json',module.summarize_score_sensitivity())
    print(json.dumps({'suite':args.suite,'output':str(out),'status':'completed'}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
