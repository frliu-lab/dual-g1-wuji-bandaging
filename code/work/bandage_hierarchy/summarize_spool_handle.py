"""Audit optional prototype handle geometry and integrated double-G1 rollouts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
static = []
for path in sorted(OUT.glob('joint_spool_grasp_static*handle*screen.json')):
    d = json.loads(path.read_text(encoding='utf8'))
    static.append({'file': path.name, 'radius_m': d.get('handle_radius_m'),
                   'clearance_m': d.get('handle_clearance_m'),
                   'target_z_m': d['goals_m'][0][2],
                   'max_tip_error_m': max(d['tip_errors_m']),
                   'forces_n': d['forces_n'], 'normal_cosine': d['normal_cosine'],
                   'deep_contact_count': len(d['deep_contacts']),
                   'strict_static_endpoint': d['strict_static_endpoint']})
dynamic = []
for path in sorted(OUT.glob('joint_support_pinch_staged_wbc_report*settled_handle*.json')):
    d = json.loads(path.read_text(encoding='utf8'))
    dynamic.append({'file': path.name,
                    'simulated_time_s': d['simulated_time_s'],
                    'support_valid_final': d['online_support_quality_final']['valid'],
                    'a_pad_opposed_s': d['longest_opposed_contact_s'],
                    'b_handle_opposed_s': d['longest_b_left_opposed_spool_contact_s'],
                    'b_thumb_peak_n': d['peak_b_left_thumb_force_n'],
                    'b_index_peak_n': d['peak_b_left_index_force_n'],
                    'deep_control_steps': d['deep_nonfloor_robot_contact_control_steps'],
                    'both_feet_fraction': d['foot_support_fraction'],
                    'max_spool_displacement_m': d['max_spool_center_displacement_m'],
                    'errors': d['errors']})
report = {'scope': 'Prototype spool handle research branch; fixture not in official scene or real hardware',
          'static_count': len(static), 'strict_static_count': sum(r['strict_static_endpoint'] for r in static),
          'dynamic_count': len(dynamic),
          'joint_dynamic_pass_count': sum(r['support_valid_final'] and r['a_pad_opposed_s']>=1 and
                                          r['b_handle_opposed_s']>=1 and r['b_thumb_peak_n']<=2 and
                                          r['b_index_peak_n']<=2 and r['deep_control_steps']==0 and
                                          r['both_feet_fraction']==1 and not r['errors'] for r in dynamic),
          'static': static, 'dynamic': dynamic}
(OUT / 'joint_spool_handle_summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({k: report[k] for k in ('static_count','strict_static_count','dynamic_count','joint_dynamic_pass_count')}))
