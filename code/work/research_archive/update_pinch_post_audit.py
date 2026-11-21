"""Summarize local pinch-post pilots without promoting them to task success."""
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'outputs/wuji_multiagent/bandage_v0'
AUDIT = REPORTS / 'joint_dressing_pinch_post_pilot_audit.json'
TAGS = (
    'post_force_model_check', 'post_late_transport35', 'post_padpose_weak',
    'post_xy_force30', 'post_xy_force10', 'post_xy_force10_up04',
    'post_early_unload',
    'post_contact_gate',
    'post_gate_zplus2', 'post_gate_zplus2_xy30',
    'post_adaptive_z', 'post_adaptive_z06', 'post_adaptive_z06_side8',
    'post_gate_wrist002', 'post_gate_index50_force02', 'post_gate_dipmin0',
    'post_tipfirst_candidate3', 'post_moving_tipfirst_ik',
    'post_moving_tipfirst_early',
    'post_ik_gap_diagnosis', 'post_ik_frame_fix', 'post_framefix_lowforce',
    'post_framefix_lowforce_zplus2', 'post_tip_task1000',
    'post_tip_task_up15_xy20', 'post_unload3mm', 'post_unload3mm_xyearly',
)


def pad_robot_force(frame):
    return sum(value for name, value in frame.get('actual_pad_up_force_by_body_n', {}).items()
               if name.startswith(('g1_a/', 'g1_b/')))


def distal_contact_frames(frames, digit):
    prefix = 'g1_a/wuji_right/r_' + digit
    return sum(any(any(body.startswith(prefix + suffix) for body in contact['bodies'])
                   and contact['normal_force_n'] >= 0.04
                   for suffix in ('_distal', '_tip_sensor_frame')
                   for contact in frame['pad_contacts']) for frame in frames)


def main():
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    reports = {data.get('tag'): (path, data)
               for path in REPORTS.glob('joint_dressing_pilot_*.json')
               if (data := json.loads(path.read_text(encoding='utf-8'))).get('tag') in TAGS}
    if set(reports) != set(TAGS):
        raise RuntimeError(f'Missing reports: {set(TAGS) - set(reports)}')
    audit['trials'] = [trial for trial in audit['trials'] if trial['name'] not in TAGS]
    for tag in TAGS:
        path, data = reports[tag]
        tail = data['trace'][-25:]
        audit['trials'].append({
            'name': tag,
            'report': path.name,
            'completed': data['completed'],
            'steps': len(data['trace']),
            'pad_center_lift_m': data['final_pad_lift_m'],
            'peak_pad_center_lift_m': max(f['pad_lift_m'] for f in data['trace']),
            'max_pad_lateral_shift_m': max(f['pad_lateral_shift_m'] for f in data['trace']),
            'median_last25_predicted_up_n': median(f['predicted_dressing_up_force_n'] for f in tail),
            'median_last25_actual_robot_up_n': median(pad_robot_force(f) for f in tail),
            'median_last25_actual_a_right_up_n': median(
                sum(v for body, v in f.get('actual_pad_up_force_by_body_n', {}).items()
                    if body.startswith('g1_a/wuji_right/')) for f in tail),
            'last25_stage_steps': sum(any('dressing_stage' in c['geoms'] for c in f['pad_contacts']) for f in tail),
            'last25_index_distal_contact_steps': distal_contact_frames(tail, 'index_finger'),
            'last25_thumb_distal_contact_steps': distal_contact_frames(tail, 'thumb'),
            'transport_started': data.get('transport_started'),
            'max_ready_streak': max((f.get('transport_gate', {}).get('consecutive_ready_frames', 0)
                                     for f in data['trace']), default=0),
            'errors': data['errors'],
        })
    audit['finding'] = ('Thirty-nine isolated pinch-post pilots do not establish a repeatable pickup. '
                        'The online IK initially mixed current and reference robot frames, producing '
                        'roughly 34 mm reference fingertip error; correcting it reduced the error to '
                        'about 0.3 mm. A WBC Cartesian fingertip task then achieved 25/25 terminal '
                        'index distal contacts. A 3 mm unloading stage briefly raised the pad center '
                        'about 5.9 mm with some zero stage-load frames, but a full-pad corner audit '
                        'found peak conservative underside clearance only 0.15 mm and zero frames '
                        'above 2 mm. Lateral drift reached 33 mm. Earlier lateral feedback reduced drift '
                        'to about 24 mm but lost the unloading effect. The measured-contact gate '
                        'never authorized transport. This diagnostic prop is not the '
                        'original dressing geometry and does not establish physical feasibility.')
    audit['pass'] = False
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{len(audit["trials"])} pilots; no pickup pass')


if __name__ == '__main__':
    main()
