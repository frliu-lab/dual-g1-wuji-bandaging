"""Audit full pad underside clearance and contact during joint pickup rollouts."""
from pathlib import Path
import argparse,json
import mujoco,numpy as np

p=argparse.ArgumentParser();p.add_argument('stem');p.add_argument('--output');args=p.parse_args()
root=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0'
q=np.load(root/f'{args.stem}_qpos.npz')['qpos'];report=json.loads((root/f'{args.stem}.json').read_text())
scene=('joint_spool_grasp_dressing_pinch_post_variant.xml' if report.get('pinch_post_scene') else
       'joint_spool_grasp_dressing_pull_tab_variant.xml' if report.get('pull_tab_scene') else
       'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml')
m=mujoco.MjModel.from_xml_path(str(root/scene))
d=mujoco.MjData(m);stage=m.geom('dressing_stage').id;pad=m.body('dressing_pad').id
stage_top=m.geom_pos[stage,2]+m.geom_size[stage,2]
pad_size=m.geom_size[m.geom('dressing_pad_contact').id]
corners=np.array([[x,y,-pad_size[2]] for x in (-pad_size[0],pad_size[0])
                  for y in (-pad_size[1],pad_size[1])])
rows=[]
for i in range(len(q)):
 d.qpos[:]=q[i];mujoco.mj_forward(m,d)
 r=d.xmat[pad].reshape(3,3)
 zs=d.xpos[pad,2]+corners@r[2,:]
 contacts=report['trace'][i]['pad_contacts']
 rows.append({'time_s':float((i+1)*.02),'min_underside_clearance_m':float(zs.min()-stage_top),
              'pad_z_axis_world':r[:,2].tolist(),
              'stage_contact':any('dressing_stage' in c['geoms'] or 'equipment_tray' in c['geoms']
                                  for c in contacts),
              'pad_lateral_shift_m':report['trace'][i]['pad_lateral_shift_m']})
end=rows[-25:]
clear=[r for r in rows if r['min_underside_clearance_m']>.002 and not r['stage_contact']]
result={'scope':'Conservative full-pad underside clearance above the stage top',
 'scene':scene,'stem':args.stem,'stage_top_m':float(stage_top),
 'last_25_min_clearance_m':min(r['min_underside_clearance_m'] for r in end),
 'last_25_stage_contact_steps':sum(r['stage_contact'] for r in end),
 'peak_min_underside_clearance_m':max(r['min_underside_clearance_m'] for r in rows),
 'clearance_above_2mm_no_stage_frames':len(clear),
 'clearance_above_2mm_no_stage_under_20mm_drift_frames':sum(r['pad_lateral_shift_m']<.02 for r in clear),
 'last_pad_z_axis_world':end[-1]['pad_z_axis_world']}
if args.output:(root/args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
