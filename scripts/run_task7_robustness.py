"""Task 7 reoptimization robustness: pooled displacement and product exclusion."""
from __future__ import annotations
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from dynamic_promotion_planning.policy import build_schedule_system, run_three_policy_grid, schedule_signature, load_pickle, save_pickle

EXCLUDE='3800001611'

def pooled_displacement_draws(base: pd.DataFrame, pooled: pd.DataFrame, products: list[str]):
    """Keep product ε/γ bootstrap rows; attach same-bootstrap pooled ψ/r draws."""
    pooled=pooled[['bootstrap_id','displacement_strength','inventory_persistence','draw_weight']].copy()
    out={}
    for upc in products:
        left=base.loc[base.upc.astype(str).eq(upc)].copy()
        if left.bootstrap_id.duplicated().any(): raise AssertionError('Expected one contemporaneous product draw per bootstrap id.')
        joined=left.merge(pooled,on='bootstrap_id',how='inner',suffixes=('_product','_pooled'),validate='one_to_many')
        if not np.isclose(joined.draw_weight.sum(),1): raise AssertionError('Pooled weights must sum to one per product.')
        out[upc]={
            'epsilon':joined.price_elasticity.to_numpy(float), 'gamma':joined.promotion_lift_log.to_numpy(float),
            'psi':joined.displacement_strength_pooled.to_numpy(float), 'r':joined.inventory_persistence_pooled.to_numpy(float),
            'weights':joined.draw_weight.to_numpy(float),
            'base_demand':joined.base_demand.to_numpy(float), 'regular_price':joined.regular_price.to_numpy(float), 'unit_cost':joined.unit_cost.to_numpy(float)}
    return out

def transition_summary(schedules, capacities):
    rows=[]
    aliases={'myopic':'piM','naive_dynamic':'piN','dynamic':'piD'}
    for b in capacities:
        entries=sorted((share, s) for (share, cap),s in schedules.items() if cap==b)
        for key,label in aliases.items():
            signatures=[schedule_signature(s[key]) for _,s in entries]
            switches=[share for (share,_),before,after in zip(entries[1:],signatures,signatures[1:]) if before!=after]
            rows.append({'capacity':b,'policy':label,'transition_count':len(switches),'transition_locations':'|'.join(f'{x:.2f}' for x in switches)})
    return pd.DataFrame(rows)

def summarize(name, results, schedules, capacities):
    rows=[]; transitions=transition_summary(schedules,capacities)
    for b,group in results.groupby('capacity',observed=True):
        peak=group.loc[group.delta_total.idxmax()]
        counts=transitions.loc[transitions.capacity.eq(b)].set_index('policy')
        rows.append({'specification':name,'capacity':b,'peak_delta_total':peak.delta_total,'peak_lambda':peak.reimbursement_share,
                     'peak_delta_plan':peak.delta_plan,'peak_delta_disp':peak.delta_disp,
                     'displacement_share_at_peak':peak.delta_disp/peak.delta_total if peak.delta_total else np.nan,
                     'piM_transitions':counts.loc['piM','transition_count'],'piN_transitions':counts.loc['piN','transition_count'],'piD_transitions':counts.loc['piD','transition_count'],
                     'piM_transition_locations':counts.loc['piM','transition_locations'],'piN_transition_locations':counts.loc['piN','transition_locations'],'piD_transition_locations':counts.loc['piD','transition_locations']})
    full=results.copy(); full.insert(0,'specification',name)
    return pd.DataFrame(rows),full

def optimize(draws, profiles, actions, planning, grid, capacities):
    d=build_schedule_system(draws,profiles,actions,planning,grid)
    n=build_schedule_system(draws,profiles,actions,planning,grid,add_new_promotion_displacement=False)
    return run_three_policy_grid(d,n,draws,profiles,actions,grid,capacities,compute_second_best=False)

def main():
    started = time.monotonic()
    artifact=load_pickle(ROOT/'artifacts/policy/policy_optimization.pkl')
    products=list(artifact['products']); grid=list(artifact['reimbursement_grid']); capacities=list(artifact['capacities'])
    planning=artifact['schedule_system']['planning']; profiles=artifact['weekly_profiles']; actions=artifact['action_sets']
    main_run={'results':artifact['policy_results'],'schedules':artifact['three_policy_schedules']}
    base=pd.read_pickle(ROOT/'artifacts/calibration/product_behavioral_bootstrap.pkl'); pooled=pd.read_pickle(ROOT/'artifacts/calibration/pooled_behavioral_draws.pkl')
    print("Task 7: optimizing pooled-displacement specification...", flush=True)
    pooled_run=optimize(pooled_displacement_draws(base,pooled,products),profiles,actions,planning,grid,capacities)
    print(f"Task 7: pooled-displacement complete ({time.monotonic() - started:.1f}s).", flush=True)
    keep=[p for p in products if p!=EXCLUDE]
    print("Task 7: optimizing exclude-3800001611 specification...", flush=True)
    exclude_run=optimize({p:artifact['draws_by_product'][p] for p in keep},{p:profiles[p] for p in keep},{p:actions[p] for p in keep},planning,grid,capacities)
    print(f"Task 7: exclusion complete ({time.monotonic() - started:.1f}s).", flush=True)
    outputs=[summarize('main',main_run['results'],main_run['schedules'],capacities),summarize('pooled_displacement',pooled_run['results'],pooled_run['schedules'],capacities),summarize('exclude_3800001611',exclude_run['results'],exclude_run['schedules'],capacities)]
    tables=ROOT/'results/final/tables'; tables.mkdir(parents=True,exist_ok=True)
    pd.concat([x[0] for x in outputs],ignore_index=True).to_csv(tables/'task7_robustness_comparison.csv',index=False)
    pd.concat([x[1] for x in outputs],ignore_index=True).to_csv(tables/'task7_robustness_full_grid.csv',index=False)
    save_pickle({'pooled_displacement': pooled_run, 'exclude_3800001611': exclude_run}, ROOT/'artifacts/policy/task7_robustness_runs.pkl')
    print(pd.concat([x[0] for x in outputs],ignore_index=True).to_string(index=False))
    print(f"Task 7: finished ({time.monotonic() - started:.1f}s).", flush=True)

if __name__=='__main__': main()
