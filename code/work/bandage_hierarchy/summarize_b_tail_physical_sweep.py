"""Audit proxy mass/friction sensitivity and support-removal counterfactual."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
cases=[(8.,1.,False),(.5,1.,False),(1.,1.,False),(1.5,1.,False),
       (2.,1.,False),(2.2,1.,False),(2.5,1.,False),(3.,1.,False),
       (2.,2.,False),(8.,2.,False),(20.,.5,False),(2.,1.,True)]
rows=[]
for mass_g,mu,no_rest in cases:
    suffix='' if (mass_g,mu)==(8.,1.) else f'_mass{mass_g:g}g_mu{mu:g}'
    name=f'joint_b_tail_dynamic_proxy_grasp_pilot_7mm_tip_pinch_rank0{suffix}_gap{"_no_rest" if no_rest else ""}.json'
    trial=json.loads((OUT/name).read_text())
    contact=trial['contact_fractions']['opposing']
    feet=trial['contact_fractions']['two_feet']
    drift=trial['max_tail_center_drift_m']
    pen=trial['max_b_nonfloor_penetration_m']
    passed=(trial['error'] is None and trial['duration_s']>=.499 and
            feet>=.99 and contact>=.9 and drift<=.005 and pen<.003)
    rows.append({'mass_g':mass_g,'friction':mu,'has_rest':not no_rest,
                 'source':name,'opposing_contact_fraction':contact,
                 'two_foot_support_fraction':feet,'tail_drift_m':drift,
                 'max_nonfloor_penetration_m':pen,'strict_proxy_grasp_pass':passed})
report={'scope':'Rigid tail proxy only; parameter sensitivity is not a cloth or hardware validation',
        'passed_count':sum(row['strict_proxy_grasp_pass'] for row in rows),
        'case_count':len(rows),'cases':rows}
(OUT/'joint_b_tail_physical_sweep.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
