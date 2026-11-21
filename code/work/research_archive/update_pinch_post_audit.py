"""Summarize local pinch-post pilots without promoting them to task success."""
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'outputs/wuji_multiagent/bandage_v0'
AUDIT = ROOT / 'research_archive/media/bandage_v0/joint_dressing_pinch_post_pilot_audit.json'
TAGS = (
    'post_force_model_check', 'post_late_transport35', 'post_padpose_weak',
    'post_xy_force30', 'post_xy_force10', 'post_xy_force10_up04',
)


def pad_robot_force(frame):
    return sum(value for name, value in frame.get('actual_pad_up_force_by_body_n', {}).items()
               if name.startswith(('g1_a/', 'g1_b/')))


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
            'max_pad_lateral_shift_m': max(f['pad_lateral_shift_m'] for f in data['trace']),
            'median_last25_predicted_up_n': median(f['predicted_dressing_up_force_n'] for f in tail),
            'median_last25_actual_robot_up_n': median(pad_robot_force(f) for f in tail),
            'last25_stage_steps': sum(any('dressing_stage' in c['geoms'] for c in f['pad_contacts']) for f in tail),
            'errors': data['errors'],
        })
    audit['finding'] = ('Eighteen isolated pinch-post pilots fail pickup. World-frame force logging '
                        'confirms some predicted upward grip is realized, but the pad slides laterally '
                        'or remains supported by the stage. Lateral-force feedback limits drift to about '
                        '30 mm in one run, without clearing the stage. This diagnostic prop is not the '
                        'original dressing geometry and does not establish physical feasibility.')
    audit['pass'] = False
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{len(audit["trials"])} pilots; no pickup pass')


if __name__ == '__main__':
    main()
