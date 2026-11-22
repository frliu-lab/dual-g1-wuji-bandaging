"""Compare state-triggered fingertip reacquisition without mistaking rest support for grasp."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
cases=[(8.,w,False) for w in (1,2,5,8,10,12,15,25,100)]
cases += [(2.,w,False) for w in (5,8,10)]
cases += [(8.,10,True)]
rows=[]
for mass_g,weight,no_rest in cases:
    mass_suffix='_mass2g_mu1' if mass_g==2. else ''
    name=(f'joint_b_tail_dynamic_proxy_grasp_pilot_7mm_tip_pinch_rank0{mass_suffix}'
          f'_gap_reacquire_w{weight}p0_fixed_world{"_no_rest" if no_rest else ""}.json')
    trial=json.loads((OUT/name).read_text())
    fractions=trial['contact_fractions']
    passed=(trial['error'] is None and trial['duration_s']>=.499 and
            fractions['two_feet']>=.99 and fractions['opposing']>=.9 and
            trial['max_tail_center_drift_m']<=.005 and
            trial['max_b_nonfloor_penetration_m']<.003)
    rows.append({'mass_g':mass_g,'tip_weight':weight,'has_rest':not no_rest,
                 'source':name,'opposing_contact_fraction':fractions['opposing'],
                 'tail_drift_m':trial['max_tail_center_drift_m'],
                 'two_foot_support_fraction':fractions['two_feet'],
                 'strict_proxy_grasp_pass':passed})
report={'scope':'Rigid tail proxy and conditional world-offset fingertip task only; no flexible bandage or hardware validation',
        'passed_count':sum(r['strict_proxy_grasp_pass'] for r in rows),
        'trial_count':len(rows),'trials':rows}
(OUT/'joint_b_tail_reacquire_sweep.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
