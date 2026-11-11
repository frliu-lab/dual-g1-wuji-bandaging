"""Audit simultaneous support and pinch WBC runs against a joint subtask gate."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
rows=[]
for path in sorted(OUT.glob('joint_support_pinch*_wbc_report*.json')):
    report=json.loads(path.read_text(encoding='utf8'))
    full=report['simulated_time_s']>=report['planned_duration_s']-.02
    support=report['online_support_quality_final']['valid']
    opposed=report['longest_opposed_contact_s']>=2.
    static_pad=report['max_pad_center_displacement_m']<.005
    valid=bool(full and support and opposed and static_pad and
               report['foot_support_fraction']>=.999 and
               report['deep_nonfloor_robot_contact_control_steps']==0 and
               report['finite_state'] and not report['errors'])
    rows.append({'file':path.name,'staged_pregrasp':report.get('staged_pregrasp',False),
                 'analytic_grip_specialist':report.get('analytic_grip_specialist',False),
                 'duration_s':report['simulated_time_s'],'full_duration':full,
                 'support_valid_final':support,'peak_support_force_n':report['peak_a_left_forearm_force_n'],
                 'last_support_palm_share':report['online_support_quality_final']['window_palm_impulse_share'],
                 'last_support_finger_share':report['online_support_quality_final']['window_finger_impulse_share'],
                 'longest_opposed_contact_s':report['longest_opposed_contact_s'],
                 'max_pad_displacement_m':report['max_pad_center_displacement_m'],
                 'deep_contact_steps':report['deep_nonfloor_robot_contact_control_steps'],
                 'router_mode_counts':report.get('router_mode_counts'),
                 'errors':report['errors'],'joint_subtask_gate_pass':valid})
summary={'scope':'Simulation-only simultaneous A-left support and A-right unloaded pinch; B robot stands. Gate is provisional, not bandaging or hardware certification.',
         'gate':{'full_duration':True,'support_online_final_valid':True,
                 'longest_opposed_contact_s_min':2.,'max_pad_displacement_m_exclusive':.005,
                 'foot_support_fraction_min':.999,'deep_contact_steps_max':0,'finite_error_free':True},
         'cases':rows,'passing_cases':sum(row['joint_subtask_gate_pass'] for row in rows)}
(OUT/'joint_support_pinch_sweep_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps({'cases':len(rows),'passing_cases':summary['passing_cases'],
                  'best_opposed_contact_s':max((r['longest_opposed_contact_s'] for r in rows),default=0.)}))
