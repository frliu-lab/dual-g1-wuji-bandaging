"""Render the verified joint subskill from recorded physical simulation states."""
import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser()
parser.add_argument('--lift-pass',action='store_true',help='Render verified 10 mm pickup after joint grasp')
parser.add_argument('--lift-target-m',type=float,default=.01,choices=(.01,.02))
args=parser.parse_args()
candidates=[]
for p in OUT.glob('joint_support_pinch_staged_wbc_report_spool_settled_handle_*.json'):
    d=json.loads(p.read_text(encoding='utf8'))
    if (d.get('record_qpos') and d.get('b_handle_force_cap_n')==1.5 and d.get('b_hold_closure_m')==.002
        and (d.get('b_lift_after_grasp_m')==args.lift_target_m and d.get('b_lift_feedback') and
             d.get('support_wrist_hold_offset_rad')==[0.,0.,-.16])==args.lift_pass):
        candidates.append((p,d))
if not candidates:raise RuntimeError('No recorded grasp run')
source,report=max(candidates,key=lambda item:item[0].stat().st_mtime)
if (report['errors'] or not report['online_support_quality_final']['valid'] or
    report['longest_b_left_opposed_spool_contact_s']<1. or
    report['peak_b_left_thumb_force_n']>2. or report['peak_b_left_index_force_n']>2. or
    report['max_spool_center_displacement_m']>(args.lift_target_m+.004 if args.lift_pass else .005) or
    report['deep_nonfloor_robot_contact_control_steps'] or report['foot_support_fraction']<1.):
    raise RuntimeError('Recorded run does not pass the joint grasp quality gate')
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
if args.lift_pass:
    spool=model.body('bandage_spool').id
    check=mujoco.MjData(model)
    positions=[]
    for frame in report['trace']:
        check.qpos[:]=np.asarray(frame['qpos']);mujoco.mj_forward(model,check)
        positions.append(check.xpos[spool].copy())
    positions=np.asarray(positions)
    lift_start=positions[max(0,len(positions)-100)]
    lifted=float(positions[-1,2]-lift_start[2])
    lateral=float(np.max(np.linalg.norm(positions[-100:,:2]-lift_start[:2],axis=1)))
    if not .7*args.lift_target_m<=lifted<=1.2*args.lift_target_m or lateral>.005:
        raise RuntimeError(f'Pickup gate failed: lift {lifted:.4f} m, lateral {lateral:.4f} m')
model.vis.global_.offwidth=960;model.vis.global_.offheight=540
data=mujoco.MjData(model)
camera=mujoco.MjvCamera()
camera.lookat[:]=[0.,-.16,.78]
camera.distance=2.15
camera.azimuth=130
camera.elevation=-19
ffmpeg=ROOT/'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
name=('joint_handle_pickup_20mm_joint_pass' if args.lift_target_m==.02 else
      'joint_handle_pickup_first_joint_pass') if args.lift_pass else 'joint_handle_grasp_first_joint_pass'
target=OUT/f'{name}_overview.mp4'
trajectory=OUT/f'{name}_qpos.npz'
np.savez_compressed(trajectory,qpos=np.asarray([f['qpos'] for f in report['trace']]),
                    time_s=np.asarray([f['time_s'] for f in report['trace']]))
proc=subprocess.Popen([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
                       '-video_size','960x540','-framerate','10','-i','pipe:0','-c:v','libx264',
                       '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],
                      stdin=subprocess.PIPE)
frames=report['trace'][::5]
try:
    with mujoco.Renderer(model,height=540,width=960) as renderer:
        for frame in frames:
            if frame['qpos'] is None:raise RuntimeError('Missing recorded qpos')
            data.qpos[:]=np.asarray(frame['qpos'])
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            proc.stdin.write(renderer.render().tobytes())
finally:
    proc.stdin.close()
if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
manifest={'scope':'Simulation replay of joint forearm support, dressing pinch, and prototype-handle pickup; no wrapping or hardware claim' if args.lift_pass else 'Simulation replay of joint forearm support, dressing pinch, and prototype-handle grasp; no wrapping or hardware claim',
          'source_report':source.name,'video':target.name,'trajectory':trajectory.name,
          'state_hz':50,'frames':len(frames),'fps':10,
          'duration_s':len(frames)/10,'b_grasp_opposed_s':report['longest_b_left_opposed_spool_contact_s'],
          'b_thumb_peak_n':report['peak_b_left_thumb_force_n'],
          'b_index_peak_n':report['peak_b_left_index_force_n'],
          'spool_drift_m':report['max_spool_center_displacement_m'],
          'spool_lift_m':lifted if args.lift_pass else 0.,
          'spool_lateral_shift_m':lateral if args.lift_pass else None,
          'support_quality_final':report['online_support_quality_final']}
(OUT/f'{name}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps(manifest))
