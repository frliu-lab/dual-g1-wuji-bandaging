"""Single-frame inverse-dynamics feasibility screen for two static handle endpoints."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
sys.path.insert(0, str(ROOT / 'work/wuji_multiagent'))
from joint_wbc import JointWBC

def run(stem, candidate_suffix=''):
    m = mujoco.MjModel.from_xml_path(str(OUT / f'{stem}_variant.xml'))
    d = mujoco.MjData(m)
    d.qpos[:] = np.load(OUT / f'{stem}{candidate_suffix}_candidate.npz')['qpos']
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    b_contacts=[]
    for ci,c in enumerate(d.contact):
        names=(m.body(m.geom_bodyid[c.geom1]).name,m.body(m.geom_bodyid[c.geom2]).name)
        if not any(n.startswith('g1_b/wuji_left/') for n in names):continue
        wrench=np.zeros(6)
        mujoco.mj_contactForce(m,d,ci,wrench)
        b_contacts.append({'bodies':names,'geoms':(m.geom(c.geom1).name,m.geom(c.geom2).name),
                           'distance_m':float(c.dist),'normal_force_n':max(0.,float(wrench[0]))})
    palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
             'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
    def contact_filter(mm, c):
        return any(mm.body(mm.geom_bodyid[int(g)]).name.startswith(('g1_a/', 'g1_b/'))
                   for g in c.geom if g >= 0)
    w = JointWBC(m, d.qpos.copy(), palm_names=palms, contact_filter=contact_filter,
                 palm_rotation_weights=[10., 10., 10., 10.],
                 palm_position_weights=[1000., 1500., 1000., 1000.])
    w.forearm_contact_force_limit_n = 12.
    w.forearm_finger_force_limit_n = 2.
    w.support_palm_force_goal_n = 11.5
    try:
        _, qp, tau = w.solve(d)
        low, high = w.actuator_limits(d)
        b = []
        for i in range(m.nu):
            name = m.actuator(i).name
            if name.startswith('g1_b/') and ('left_' in name or 'wuji_left/' in name):
                margin = min(float(tau[i]-low[i]), float(high[i]-tau[i]))
                half_range = max(1e-12, float(high[i]-low[i])/2)
                b.append({'actuator': name, 'torque_nm': float(tau[i]),
                          'lower_nm': float(low[i]), 'upper_nm': float(high[i]),
                          'limit_margin_nm': margin,
                          'limit_margin_fraction_of_half_range': margin/half_range})
        return {'stem': stem+candidate_suffix, 'qp_feasible': True, 'qp': qp,
                'b_hand_contacts':b_contacts,
                'b_actuator_margin_min_fraction': min(r['limit_margin_fraction_of_half_range'] for r in b),
                'b_actuator_margin_10_percent_pass': all(r['limit_margin_fraction_of_half_range']>=.1 for r in b),
                'b_smallest_actuator_margins': sorted(b, key=lambda r:r['limit_margin_nm'])[:10]}
    except Exception as exc:
        return {'stem': stem+candidate_suffix, 'qp_feasible': False, 'error': f'{type(exc).__name__}: {exc}'}

stems = [
    ('joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle',''),
    ('joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle','_middle_retracted'),
    ('joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyaw20p0_spoolz0p702_handle',''),
]
report = {'scope': 'Single static frame WBC inverse-dynamics feasibility only; no trajectory or hardware claim',
          'cases': [run(s, suffix) for s, suffix in stems]}
(OUT / 'joint_spool_handle_equilibrium_screen.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({'cases': [{'stem': r['stem'], 'qp_feasible': r['qp_feasible'],
                             'error': r.get('error')} for r in report['cases']]}))
