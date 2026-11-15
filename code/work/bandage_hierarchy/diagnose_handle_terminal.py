"""Compare closed-loop B-left finger positions with the prototype handle goal."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
MODEL = OUT / 'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_spoolz0p702_handle_variant.xml'
model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)
rows = []
for state in sorted(OUT.glob('joint_support_pinch_staged_wbc_report*settled_handle*final_state.npz')):
    saved = np.load(state)
    ref_data = mujoco.MjData(model)
    ref_data.qpos[:] = saved['reference_qpos']
    mujoco.mj_forward(model, ref_data)
    data.qpos[:] = saved['qpos']
    mujoco.mj_forward(model, data)
    center = data.xpos[model.body('bandage_spool').id]
    rot = data.xmat[model.body('bandage_spool').id].reshape(3, 3)
    tip_rows = {}
    for digit in ('thumb', 'index'):
        name = f'g1_b/wuji_left/l_{digit if digit == "thumb" else "index_finger"}_tip_sensor_frame'
        tip = data.xpos[model.body(name).id]
        ref_tip = ref_data.xpos[model.body(name).id]
        local = rot.T @ (tip - center)
        tip_rows[digit] = {
            'world_position_m': tip.tolist(),
            'reference_position_m': ref_tip.tolist(),
            'position_error_m': float(np.linalg.norm(tip-ref_tip)),
            'spool_local_position_m': local.tolist(),
            'radial_distance_to_handle_axis_m': float(np.linalg.norm(local[:2])),
        }
    wrist = model.body('g1_b/wuji_left/l_wrist').id
    joints = [j for j in range(model.njnt) if model.joint(j).name.startswith((
        'g1_b/left_shoulder', 'g1_b/left_elbow', 'g1_b/left_wrist',
        'g1_b/wuji_left/l_thumb_', 'g1_b/wuji_left/l_index_finger_'))]
    joint_errors = sorted(({'joint': model.joint(j).name,
                            'error_rad': float(abs(saved['qpos'][model.jnt_qposadr[j]]-
                                                   saved['reference_qpos'][model.jnt_qposadr[j]]))}
                           for j in joints), key=lambda r: -r['error_rad'])
    rows.append({'state_file': state.name, 'spool_center_m': center.tolist(), 'tips': tip_rows,
                 'b_wrist_position_error_m': float(np.linalg.norm(data.xpos[wrist]-ref_data.xpos[wrist])),
                 'worst_b_joint_errors': joint_errors[:8]})
report = {'scope': 'Terminal kinematic diagnostic for three prototype-handle closed-loop runs; no grasp claim',
          'run_count': len(rows), 'handle_radius_m': .012, 'runs': rows}
(OUT / 'joint_spool_handle_terminal_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({'run_count': len(rows), 'radii_m': [r['tips']['index']['radial_distance_to_handle_axis_m'] for r in rows]}))
