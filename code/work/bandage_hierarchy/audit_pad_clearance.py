"""Audit full pad underside clearance and contact during joint pickup rollouts."""
from pathlib import Path
import argparse,json
import mujoco,numpy as np

p=argparse.ArgumentParser();p.add_argument('stem');args=p.parse_args()
root=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0'
m=mujoco.MjModel.from_xml_path(str(root/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
q=np.load(root/f'{args.stem}_qpos.npz')['qpos'];report=json.loads((root/f'{args.stem}.json').read_text())
d=mujoco.MjData(m);stage=m.geom('dressing_stage').id;pad=m.body('dressing_pad').id
stage_top=m.geom_pos[stage,2]+m.geom_size[stage,2]
corners=np.array([[x,y,-.008] for x in (-.045,.045) for y in (-.03,.03)])
rows=[]
for i in range(len(q)):
 d.qpos[:]=q[i];mujoco.mj_forward(m,d)
 r=d.xmat[pad].reshape(3,3)
 zs=d.xpos[pad,2]+corners@r[2,:]
 rows.append({'time_s':float((i+1)*.02),'min_underside_clearance_m':float(zs.min()-stage_top),
              'pad_z_axis_world':r[:,2].tolist(),
              'stage_contact':any(c['bodies']==['world','dressing_pad'] for c in report['trace'][i]['pad_contacts'])})
end=rows[-25:]
print(json.dumps({'stem':args.stem,'stage_top_m':float(stage_top),
 'last_25_min_clearance_m':min(r['min_underside_clearance_m'] for r in end),
 'last_25_stage_contact_steps':sum(r['stage_contact'] for r in end),
 'last_pad_z_axis_world':end[-1]['pad_z_axis_world']},indent=2))
