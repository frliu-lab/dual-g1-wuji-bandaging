"""Static A-left Wuji forearm support screen on the training manikin."""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from scene_evidence import hand_contacts

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--a-root-x',type=float,default=-.70)
parser.add_argument('--hand-curl-rad',type=float,default=0.)
parser.add_argument('--endpoint',choices=('index','wrist'),default='index')
parser.add_argument('--forward-arm',action='store_true')
args=parser.parse_args()
scene=ET.parse(OUT/'scene.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos',f'{args.a_root_x} 0 .8')
if args.forward_arm:
    for g in scene.iter('geom'):
        if g.get('name')=='manikin_upper_arm':g.set('fromto','0 -.20 1.09 -.12 -.45 .90')
        if g.get('name')=='manikin_forearm':g.set('fromto','-.12 -.45 .90 .16 -.60 .86')
m=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
d=mujoco.MjData(m)
base=np.load(OUT/'reach_ik_seed.npz')['qpos'].copy()
base[m.joint('g1_a/floating_base').qposadr[0]]=args.a_root_x
if args.hand_curl_rad:
    for j in range(m.njnt):
        n=m.joint(j).name
        if not n.startswith('g1_a/wuji_left/') or 'abd' in n:continue
        if any(n.endswith(s) for s in ('_mcp_flex','_pip','_dip','_cmc_flex','_mcp','_ip')):
            a=int(m.jnt_qposadr[j]);low,high=m.jnt_range[j]
            base[a]=np.clip(base[a]+args.hand_curl_rad,low+.03,high-.03)
name=(f'forearm_support_root_{str(args.a_root_x).replace(".","p").replace("-","m")}'
      +f'_curl{str(args.hand_curl_rad).replace(".","p")}' if args.hand_curl_rad else
      f'forearm_support_root_{str(args.a_root_x).replace(".","p").replace("-","m")}')
name += '_wrist' if args.endpoint=='wrist' else ''
name += '_forwardarm' if args.forward_arm else ''
tip=m.body('g1_a/wuji_left/l_wrist' if args.endpoint=='wrist'
           else 'g1_a/wuji_left/l_index_finger_tip_sensor_frame').id
js=[j for j in range(m.njnt)
    if m.joint(j).name.startswith('g1_a/left_')
    and any(s in m.joint(j).name for s in ('shoulder','elbow','wrist'))
    and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
qa=np.array([m.jnt_qposadr[j] for j in js],dtype=int)
va=np.array([m.jnt_dofadr[j] for j in js],dtype=int)
results=[];qposes=[]
xs=(-.26,-.22,-.18,-.14) if args.endpoint=='wrist' else (-.16,-.12,-.08,-.04)
zs=(.78,.80,.82,.84) if args.endpoint=='wrist' else (.82,.84,.86,.88)
for x in xs:
    ys=(-.12,-.08,-.04,0.,.04) if args.forward_arm else (0.,.04,.08,.12,.16)
    for y in ys:
        for z in zs:
            d.qpos[:]=base
            goal=np.array([x,y,z])
            for _ in range(160):
                mujoco.mj_forward(m,d)
                error=goal-d.xpos[tip]
                if np.linalg.norm(error)<.0005:break
                jac=np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jac,None,d.xpos[tip],tip)
                J=jac[:,va]
                dq=J.T@np.linalg.solve(J@J.T+.02**2*np.eye(3),error)
                d.qpos[qa]+=np.clip(dq,-.03,.03)
                for j,a in zip(js,qa):
                    if m.jnt_limited[j]:
                        low,high=m.jnt_range[j]
                        d.qpos[a]=np.clip(d.qpos[a],low+.03,high-.03)
            mujoco.mj_forward(m,d)
            deep=[];forearm=[]
            for i,c in enumerate(d.contact):
                if c.geom1<0 or c.geom2<0:continue
                names=[m.geom(g).name for g in (c.geom1,c.geom2)]
                bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
                if 'floor' in names or not any(b.startswith(('g1_a/','g1_b/')) for b in bodies):continue
                if c.dist<-.003:deep.append({'geoms':names,'bodies':bodies,'distance_m':float(c.dist)})
                if 'manikin_forearm' in names and any(b.startswith('g1_a/wuji_left/') for b in bodies):
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench)
                    forearm.append({'body':next(b for b in bodies if b.startswith('g1_a/')),
                                    'distance_m':float(c.dist),'normal_force_n':float(wrench[0])})
            results.append({'goal_m':goal.tolist(),
                            'achieved_m':d.xpos[tip].tolist(),
                            'tip_error_m':float(np.linalg.norm(goal-d.xpos[tip])),
                            'max_arm_joint_change_rad':float(np.max(np.abs(d.qpos[qa]-base[qa]))),
                            'deep_contacts':deep,'forearm_contacts':forearm,
                            'role_evidence':hand_contacts(m,d)[0]})
            qposes.append(d.qpos.copy())
eligible=[i for i,r in enumerate(results)
          if r['forearm_contacts'] and not r['deep_contacts'] and r['tip_error_m']<.03
          and (args.endpoint!='wrist' or any(c['body'].endswith('/l_wrist') for c in r['forearm_contacts']))]
selected=min(eligible,key=lambda i:results[i]['tip_error_m']+.01*results[i]['max_arm_joint_change_rad']) if eligible else None
path_bad=[]
if selected is not None:
    qend=qposes[selected]
    for u in np.linspace(0,1,101):
        d.qpos[:]=(1-u)*base+u*qend
        mujoco.mj_forward(m,d)
        for c in d.contact:
            if c.geom1<0 or c.geom2<0 or c.dist>=-.003:continue
            names=[m.geom(g).name for g in (c.geom1,c.geom2)]
            bodies=[m.body(m.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
            if 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                path_bad.append({'u':float(u),'bodies':bodies,'distance_m':float(c.dist)})
    np.savez_compressed(OUT/f'{name}_seed.npz',initial_qpos=base,qpos=qend)
report={'scope':'Static A-left support-hand geometric scan; static normal force may be zero; no closed-loop force control or clinical safety claim',
        'a_root_x_m':args.a_root_x,
        'hand_curl_rad':args.hand_curl_rad,
        'endpoint':args.endpoint,
        'forward_arm_variant':args.forward_arm,
        'candidates':results,'eligible_count':len(eligible),'selected_candidate_index':selected,
        'selected_direct_path_deep_contacts':len(path_bad),
        'selected_direct_path_worst_contacts':sorted(path_bad,key=lambda r:r['distance_m'])[:8]}
(OUT/f'{name}_scan.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'candidates':len(results),'eligible':len(eligible),'selected':selected,
                  'selected_goal':results[selected]['goal_m'] if selected is not None else None,
                  'direct_path_deep_contacts':len(path_bad)}))
