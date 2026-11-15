"""Audit whether a small raised spool holder fixes the B-left contact geometry."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
rows = []
for path in sorted(OUT.glob('joint_spool_grasp_static*pedestal_screen.json')):
    d = json.loads(path.read_text(encoding='utf8'))
    rows.append({
        'file': path.name,
        'grasp_axis': d['grasp_axis'],
        'b_root_x_m': d['b_root_x_m'],
        'b_root_y_m': d['b_root_y_m'],
        'target_z_m': d['goals_m'][0][2],
        'max_tip_goal_error_m': max(d['tip_errors_m']),
        'normal_cosine': d['normal_cosine'],
        'forces_n': d['forces_n'],
        'deep_contact_count': len(d['deep_contacts']),
        'strict_static_endpoint': d['strict_static_endpoint'],
    })
report = {
    'scope': 'Static raised-spool pedestal prototype; no dynamic or bandaging claim',
    'candidate_count': len(rows),
    'strict_static_count': sum(r['strict_static_endpoint'] for r in rows),
    'candidates': rows,
}
(OUT / 'joint_spool_pedestal_summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({'candidate_count': report['candidate_count'], 'strict_static_count': report['strict_static_count']}))
