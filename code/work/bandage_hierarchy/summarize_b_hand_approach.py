"""Compare B-left dynamic pregrasp contact and actuator tracking across diagnostic runs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
rows=[]
for path in sorted(OUT.glob('joint_support_pinch_staged_wbc_report*bhanddiag*.json')):
    d=json.loads(path.read_text(encoding='utf8'))
    frames=[f for f in d['trace'] if f.get('b_hand_diagnostic')]
    if not frames:continue
    first={digit:next(({'time_s':f['time_s'],
                        'geoms':sorted({c['spool_geom'] for c in f['b_contact_details']
                                        if c['digit']==digit})}
                       for f in frames if any(c['digit']==digit for c in f['b_contact_details'])),None)
           for digit in ('thumb','index')}
    errors={}
    saturated={}
    for f in frames:
        diag=f['b_hand_diagnostic']
        for name,value in diag['joint_error_rad'].items():
            if name.startswith(('g1_b/wuji_left/l_thumb_','g1_b/wuji_left/l_index_finger_')):
                errors[name]=max(errors.get(name,0.),abs(value))
        for name,margin in diag['actuator_margin_fraction'].items():
            if margin<=.02:saturated[name]=saturated.get(name,0)+1
    rows.append({'file':path.name,'simulated_time_s':d['simulated_time_s'],
                 'planned_duration_s':d['planned_duration_s'],
                 'clear_pregrasp_spool':d.get('clear_pregrasp_spool',False),
                 'pregrasp_clearance_m':d.get('pregrasp_clearance_m',0.),
                 'first_digit_contact':first,
                 'max_thumb_index_joint_error_rad':dict(sorted(errors.items(),key=lambda x:-x[1])[:8]),
                 'actuator_saturation_frames':dict(sorted(saturated.items(),key=lambda x:-x[1])[:10]),
                 'max_spool_displacement_m':d['max_spool_center_displacement_m'],
                 'b_side_both_control_steps':d.get('b_handle_side_both_control_steps',0),
                 'b_opposed_s':d['longest_b_left_opposed_spool_contact_s'],
                 'a_support_valid_final':d['online_support_quality_final']['valid'],
                 'errors':d['errors']})
report={'scope':'Dynamic B-left approach diagnostics; prototype handle only, no bandaging claim',
        'run_count':len(rows),'runs':rows}
(OUT/'joint_spool_handle_approach_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'run_count':len(rows),'first_contacts':[(r['first_digit_contact'],r['max_spool_displacement_m']) for r in rows]}))
