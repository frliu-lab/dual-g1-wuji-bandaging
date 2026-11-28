"""Compare flex-tab relaxation with robots externally frozen; diagnostic only."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
SCENE=OUT/'joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml'
PRIOR=OUT/'joint_b_right_loaded_pinch_static_refine_radius2mm.json'
candidate=json.loads(PRIOR.read_text())['top_passing_clearance_force'][0]
for release in (False,True):
    m=mujoco.MjModel.from_xml_path(str(SCENE))
    m.flex_radius[:]=.002
    d=mujoco.MjData(m)
    q=np.load(OUT/'joint_b_pinchtab_rrt_path.npz')['qpos'][-1]
    d.qpos[:167]=q[:167]
    pad=np.load(OUT/'joint_b_flex_a_thumb_side_contact_candidates.npz')['qpos'][0]
    for j in range(m.njnt):
        if m.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_',
                                       'g1_a/wuji_right/r_index_finger_')):
            adr=int(m.jnt_qposadr[j]);d.qpos[adr]=pad[adr]
    for name in ('g1_b/wuji_right/r_middle_finger_mcp_flex',
                 'g1_b/wuji_right/r_middle_finger_pip',
                 'g1_b/wuji_right/r_middle_finger_dip'):
        d.qpos[m.joint(name).qposadr[0]]+=.5
    names=json.loads(PRIOR.read_text())['joint_names']
    for name,value in zip(names,candidate['joint_targets_rad']):
        d.qpos[m.joint(name).qposadr[0]]=value
    mujoco.mj_forward(m,d)
    frozen=d.qpos[:167].copy()
    first_flex_dof=int(m.jnt_dofadr[next(j for j in range(m.njnt)
        if int(m.jnt_qposadr[j])>=167)])
    fixture_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,
        f'bandage_tab_fixture_{i}') for i in (m.nflexvert-2,m.nflexvert-1)]
    edge0=d.flexvert_xpos[-2:].mean(axis=0).copy()
    trace=[];max_pen=0.
    for step in range(round(.5/m.opt.timestep)):
        d.qpos[:167]=frozen;d.qvel[:first_flex_dof]=0.
        if release and d.time>=.1:
            d.eq_active[fixture_ids]=0
        mujoco.mj_step(m,d)
        max_pen=max(max_pen,max((max(0.,-float(c.dist)) for c in d.contact
            if any(g>=0 and m.body(m.geom_bodyid[g]).name.startswith('g1_b/') for g in c.geom)
            and not any(g>=0 and m.geom(g).name=='floor' for g in c.geom)),default=0.))
        if (step+1)%round(.02/m.opt.timestep):continue
        force={'thumb':0.,'index':0.}
        for ci,c in enumerate(d.contact):
            if not any(f>=0 for f in c.flex):continue
            bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
            for digit in force:
                if any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies):
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                    force[digit]+=max(0.,float(wrench[0]))
        trace.append({'time_s':float(d.time),
                      'edge_drift_m':float(np.linalg.norm(d.flexvert_xpos[-2:].mean(axis=0)-edge0)),
                      'digit_normal_force_n':force})
    report={'scope':'External fixture freezes robot, objects and forearm at seeded qpos; only flex dynamics run; not a robot rollout',
            'fixture_released_at_s':.1 if release else None,
            'max_b_nonfloor_penetration_m':max_pen,'trace':trace,
            'final_edge_drift_m':trace[-1]['edge_drift_m']}
    target=OUT/f'joint_b_loaded_pinchtab_passive_release_{int(release)}.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
