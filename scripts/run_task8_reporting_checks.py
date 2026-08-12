"""Task 8 denominators and fixed-tail replay checks; no reoptimization."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from dynamic_promotion_planning.policy import load_pickle,evaluate_schedule_map,schedule_signature
from scripts.run_task7_robustness import pooled_displacement_draws,EXCLUDE

def prefix(profile,n):
    return {p:{k:np.asarray(v)[:n] for k,v in values.items()} for p,values in profile.items()}
def extend(profile,n):
    out={}
    for p,values in profile.items():
        out[p]={}
        for k,v in values.items():
            a=np.asarray(v); out[p][k]=np.pad(a[:min(len(a),n)],(0,max(0,n-len(a))),mode='edge')
    return out
def specs(include_robustness: bool = True):
    a=load_pickle(ROOT/'artifacts/policy/policy_optimization.pkl'); r=load_pickle(ROOT/'artifacts/policy/task7_robustness_runs.pkl')
    base=pd.read_pickle(ROOT/'artifacts/calibration/product_behavioral_bootstrap.pkl');pool=pd.read_pickle(ROOT/'artifacts/calibration/pooled_behavioral_draws.pkl')
    products=list(a['products']); common={'planning':a['schedule_system']['planning'],'profiles':a['weekly_profiles'],'actions':a['action_sets']}
    output = {'main':dict(results=a['policy_results'],schedules=a['three_policy_schedules'],draws=a['draws_by_product'],**common)}
    if not include_robustness:
        return output
    output.update({
      'pooled_displacement':dict(results=r['pooled_displacement']['results'],schedules=r['pooled_displacement']['schedules'],draws=pooled_displacement_draws(base,pool,products),**common),
      'exclude_3800001611':dict(results=r['exclude_3800001611']['results'],schedules=r['exclude_3800001611']['schedules'],draws={p:a['draws_by_product'][p] for p in products if p!=EXCLUDE},profiles={p:a['weekly_profiles'][p] for p in products if p!=EXCLUDE},actions={p:a['action_sets'][p] for p in products if p!=EXCLUDE},planning=a['schedule_system']['planning'])})
    return output
def values(schedule,draws,profiles,planning,share):
    return {k:evaluate_schedule_map(schedule[k],draws,profiles,planning,share)['total_profit'] for k in ['myopic','naive_dynamic','dynamic']}
def main(include_robustness: bool = True):
    grid_rows=[];tail_rows=[]
    for name,s in specs(include_robustness).items():
      p36=s['planning']; p12=replace(p36,washout_horizon=0); p52=replace(p36,washout_horizon=52)
      prof12=prefix(s['profiles'],12);prof52=extend(s['profiles'],p52.evaluation_horizon)
      for row in s['results'].itertuples(index=False):
        share=float(row.reimbursement_share);b=int(row.capacity); schedule=s['schedules'][(round(share,8),b)]
        h=values(schedule,s['draws'],prof12,p12,share); denom=h['myopic']
        dp=float(row.delta_plan);dd=float(row.delta_disp);dt=float(row.delta_total)
        if not np.isclose(dp+dd,dt,atol=1e-8):raise AssertionError('Three-policy components fail to add.')
        grid_rows.append({'specification':name,'capacity':b,'reimbursement_share':share,'delta_plan':dp,'delta_disp':dd,'delta_total':dt,'myopic_planning_profit':denom,'delta_plan_pct':100*dp/denom,'delta_disp_pct':100*dd/denom,'delta_total_pct':100*dt/denom})
        if name=='main':
          v36=values(schedule,s['draws'],s['profiles'],p36,share);v52=values(schedule,s['draws'],prof52,p52,share)
          d36=np.array([v36['naive_dynamic']-v36['myopic'],v36['dynamic']-v36['naive_dynamic'],v36['dynamic']-v36['myopic']]);d52=np.array([v52['naive_dynamic']-v52['myopic'],v52['dynamic']-v52['naive_dynamic'],v52['dynamic']-v52['myopic']])
          ranks36=tuple(sorted(v36,key=v36.get,reverse=True));ranks52=tuple(sorted(v52,key=v52.get,reverse=True))
          tail_rows.append({'capacity':b,'reimbursement_share':share,'max_residual_I_w36':max(max(evaluate_schedule_map(schedule[k],s['draws'],s['profiles'],p36,share)['terminal_state'].values()) for k in schedule),'max_residual_I_w52':max(max(evaluate_schedule_map(schedule[k],s['draws'],prof52,p52,share)['terminal_state'].values()) for k in schedule),'delta_plan_w36':d36[0],'delta_disp_w36':d36[1],'delta_total_w36':d36[2],'delta_plan_w52':d52[0],'delta_disp_w52':d52[1],'delta_total_w52':d52[2],'delta_plan_abs_change':abs(d52[0]-d36[0]),'delta_disp_abs_change':abs(d52[1]-d36[1]),'delta_total_abs_change':abs(d52[2]-d36[2]),'delta_plan_rel_change':abs(d52[0]-d36[0])/max(abs(d36[0]),1e-12),'delta_disp_rel_change':abs(d52[1]-d36[1])/max(abs(d36[1]),1e-12),'delta_total_rel_change':abs(d52[2]-d36[2])/max(abs(d36[2]),1e-12),'ranking_changed':ranks36!=ranks52})
    grid=pd.DataFrame(grid_rows);tail=pd.DataFrame(tail_rows)
    summary=[]
    transitions=pd.read_csv(ROOT/'results/final/tables/task7_robustness_comparison.csv') if include_robustness else None
    for (name,b),g in grid.groupby(['specification','capacity'],observed=True):
      peak=g.loc[g.delta_total.idxmax()]
      if transitions is None:
        schedules=[s['schedules'][(round(float(x.reimbursement_share),8),int(b))] for x in g.itertuples(index=False)]
        counts={key:sum(schedule_signature(left[key]) != schedule_signature(right[key]) for left,right in zip(schedules,schedules[1:])) for key in ['myopic','naive_dynamic','dynamic']}
        transition_values={'piM_transitions':counts['myopic'],'piN_transitions':counts['naive_dynamic'],'piD_transitions':counts['dynamic']}
      else:
        tr=transitions.loc[(transitions.specification==name)&(transitions.capacity==b)].iloc[0]
        transition_values={'piM_transitions':tr.piM_transitions,'piN_transitions':tr.piN_transitions,'piD_transitions':tr.piD_transitions}
      summary.append({'specification':name,'capacity':b,'peak_delta_total':peak.delta_total,'peak_delta_total_pct':peak.delta_total_pct,'peak_lambda':peak.reimbursement_share,'peak_delta_plan':peak.delta_plan,'peak_delta_disp':peak.delta_disp,'displacement_share':peak.delta_disp/peak.delta_total,**transition_values})
    out=ROOT/'results/final/tables';grid.to_csv(out/'task8_three_policy_reporting_grid.csv',index=False);pd.DataFrame(summary).to_csv(out/'task8_development_summary.csv',index=False);tail.to_csv(out/'task8_main_terminal_tail_full_grid.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False));print(tail.agg({'max_residual_I_w36':'max','max_residual_I_w52':'max','delta_plan_abs_change':'max','delta_disp_abs_change':'max','delta_total_abs_change':'max','delta_plan_rel_change':'max','delta_disp_rel_change':'max','delta_total_rel_change':'max','ranking_changed':'sum'}))
if __name__=='__main__':main()
