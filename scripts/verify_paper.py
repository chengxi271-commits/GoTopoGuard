"""Check recorded PRICAI measurements against the camera-ready's printed numbers.

This is an independent arithmetic/readback check, not a refit or an engine rerun.
It uses only the Python standard library and the compact records in this repo.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMPACT=ROOT/'artifacts/compact'

def read(rel):
    return json.loads((COMPACT/rel).read_text(encoding='utf-8'))

def rows(rel):
    return [json.loads(line) for line in (COMPACT/rel).read_text(encoding='utf-8').splitlines() if line.strip()]

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced/summary')
    args=parser.parse_args(argv);checks=[]
    def check(name, value, printed, digits, source):
        actual=f'{value:.{digits}f}';expected=f'{printed:.{digits}f}'
        checks.append({'quantity':name,'value':value,'paper':expected,'recomputed':actual,'matches':actual==expected,'source':source})

    source='R011_DETECTION_BASELINE_SWEEP/family_heldout_metrics.jsonl'
    detections=rows(source)
    expected={
        'B0_prior':(.5000,.1243,1.0000),
        'B1_board_proxy_logreg':(.9308,.7530,.1490),
        'B1_engine_confidence_logreg':(.7085,.5190,.5539),
        'B2_search_brittleness_logreg':(.6950,.1642,.4508),
        'B3_topology_gbdt':(.9577,.9044,.0834),
        'B5_topology_plus_search_plus_metamorphic_gbdt':(.9554,.9250,.0849)}
    for baseline, targets in expected.items():
        selected=[r for r in detections if r['baseline']==baseline and r['status']=='done']
        check(baseline+'/folds',len(selected),13,0,source)
        for key,target in zip(['auroc','auprc','fpr_at_recall_90'],targets):
            check(baseline+'/'+key,statistics.fmean(r[key] for r in selected),target,4,source)

    for filename,targets in [
        ('strict_control_family_metrics.jsonl',{'strict_controls':(.9587,.9258),'strict_controls_plus_topology':(.9572,.9150)}),
        ('strict_control_matched_metrics.jsonl',{'strict_controls':(.9456,.9501),'strict_controls_plus_topology':(.9238,.9326)})]:
        source='R022_STRONG_ACCEPT_AUDIT/'+filename
        for model,target in targets.items():
            selected=[r for r in rows(source) if r['model']==model]
            for key,expected_value in zip(['auroc','auprc'],target):
                check(filename+'/'+model+'/'+key,statistics.fmean(r[key] for r in selected),expected_value,4,source)
    source='R022_STRONG_ACCEPT_AUDIT/mechanism_holdout_metrics.jsonl'
    for task,model,a,b in [('holdout_d3_public_attack','board_proxy',.9306,.3087),('holdout_d3_public_attack','b3_board_plus_topology',.9578,.6103),('holdout_d5_generator','board_proxy',.0239,.1399),('holdout_d5_generator','b3_board_plus_topology',.0472,.1360)]:
        record=next(r for r in rows(source) if r['task']==task and r['model']==model)
        check(task+'/'+model+'/auroc',record['auroc'],a,4,source)
        check(task+'/'+model+'/auprc',record['auprc'],b,4,source)

    source='R022_STRONG_ACCEPT_AUDIT/guard_significance_pairs.jsonl';pairs=rows(source)
    u=[r['uniform_correct'] for r in pairs];m=[r['method_correct'] for r in pairs];n=len(pairs)
    n10=sum(a==1 and b==0 for a,b in zip(m,u));n01=sum(a==0 and b==1 for a,b in zip(m,u))
    probability=min(1.,2*sum(math.comb(n10+n01,i)*.5**(n10+n01) for i in range(min(n10,n01)+1)))
    for key,value,target in [('paired_positions',n,1144),('uniform_correct',sum(u),964),('S09_correct',sum(m),969),('S09_only',n10,20),('uniform_only',n01,15)]:check(key,value,target,0,source)
    check('McNemar_exact_p',probability,.4996,4,source)
    rng=random.Random(19019);deltas=[]
    for _ in range(5000):
        indices=[rng.randrange(n) for _ in range(n)]
        deltas.append(sum(m[i]-u[i] for i in indices)/n)
    deltas.sort();check('bootstrap_CI_low',deltas[125],-.0052,4,source);check('bootstrap_CI_high',deltas[4875],.0149,4,source)
    source='R014C_UTILITY_GUARD_MODEL_SWEEP/r014c_utility_guard_model_sweep_report.json';g=read(source)
    for key,target,digits in [('uniform_oracle_agreement_rate',.8427,4),('best_sweep_oracle_agreement_rate',.8470,4),('best_sweep_mean_visits',875.0,1),('best_sweep_trigger_rate',.806,3)]:check(key,g[key],target,digits,source)
    source='R015B_NORMAL_AWARE_UTILITY_GUARD/r015b_normal_aware_utility_guard_report.json';g=read(source)
    for key,target,digits in [('D3+D5_best_agreement',.8435,4),('D3+D5_best_mean_visits',854.2,1),('D1+D2_best_agreement',.7845,4),('D1+D2_best_mean_visits',896.9,1)]:check(key,g[key],target,digits,source)

    source='R012_LEAD_TIME_EVALUATION/lead_time_records.jsonl';lead=rows(source)
    for recall in [.8,.9]:
        rs=[r for r in lead if r['target_recall']==recall]
        check(f'recall_{recall}/games',len(rs),20,0,source)
        check(f'recall_{recall}/warned',sum(r['warning_detected'] for r in rs),20,0,source)
        check(f'recall_{recall}/mean_lead',statistics.fmean(r['earliest_warning_lead_moves'] for r in rs),80,1,source)
    source='R013_OPERATING_POINT_TRADEOFF/r013_operating_point_tradeoff_report.json';op=read(source)['summary']['B3_topology_gbdt']['0.05']
    for key,target,digits in [('warning_detected_rate',1.,4),('mean_earliest_warning_lead_moves',73.2,1),('mean_test_natural_trigger_rate',.0668,4)]:check('FPR05/'+key,op[key],target,digits,source)
    source='R016_BUDGET_GENERALIZATION_ANALYSIS/budget_stability_summary.jsonl'
    for name,pct,score,win in [('ALL',17.49,.336,.0056),('D3',25.74,.243,.0001)]:
        row=next(r for r in rows(source) if r['name']==name)
        check(name+'/budget_disagree_percent',100*row['disagreement_rate_1024_8192'],pct,2,source)
        check(name+'/budget_score_movement',row['mean_score_abs_delta_1024_8192'],score,3,source)
        check(name+'/budget_win_movement',row['mean_winrate_abs_delta_1024_8192'],win,4,source)
    source='R017_NETWORK_GENERALIZATION_ANALYSIS/network_stability_summary.jsonl'
    for name,pct,score,win in [('b18',27.38,.820,.0155),('b40',29.43,1.088,.0217)]:
        row=next(r for r in rows(source) if r['name']==name)
        check(name+'/disagree_percent',100*row['top_move_disagreement_rate_vs_b28_1024'],pct,2,source)
        check(name+'/score_movement',row['mean_score_abs_delta_vs_b28_1024'],score,3,source)
        check(name+'/win_movement',row['mean_winrate_abs_delta_vs_b28_1024'],win,4,source)
    source='R020_CROSS_ENGINE_TOPO_ANALYSIS/r020_toporisk_transfer_summary.jsonl'
    for name,pct in [('leela-zero',80.46),('elf-opengo',91.04)]:
        row=next(r for r in rows(source) if r['name']==name and r['group']=='engine' and r['guard']=='toporisk_b3_fpr05')
        check(name+'/disagree_percent',100*row['disagree_1024_all'],pct,2,source)
    source='R020_CROSS_ENGINE_TOPO_ANALYSIS/r020_toporisk_engine_pair_summary.jsonl'
    check('Leela_vs_ELF_percent',100*rows(source)[0]['top_move_disagreement'],87.73,2,source)
    source='R019R1_MAIN_CYCLIC_FULL_1000GPM_T128/summary.json';attack=read(source)
    check('attack_games',len(attack['rows']),1002,0,source)
    check('attack_wins',sum(r['adversary_win'] for r in attack['rows']),984,0,source)
    check('attack_win_percent',100*sum(r['adversary_win'] for r in attack['rows'])/len(attack['rows']),98.2,1,source)
    report={'mode':'recorded measurement arithmetic; not new experiments','checks':checks,'passed':sum(r['matches'] for r in checks),'total':len(checks)}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'paper_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (args.output/'paper_checks.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    print(json.dumps({'passed':report['passed'],'total':report['total'],'mismatches':[r for r in checks if not r['matches']]}))
    return 0 if all(r['matches'] for r in checks) else 1

if __name__=='__main__':
    raise SystemExit(main())
