"""Render a continuous state replay of verified 15.7 s route plus pad pickup pilot.

The controller is restarted at the identical qpos/qvel checkpoint. This is a
simulation demonstration, not a pass for wrapping, hardware, or spool pickup.
"""
import json
import subprocess
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
parser=argparse.ArgumentParser();parser.add_argument('--closeup',action='store_true')
parser.add_argument('--sustained',action='store_true',help='Use 3 s pilot with verified final half-second hold')
args=parser.parse_args()
SOURCE=OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_65e2b2086889.json'
PILOT_STEM=('joint_dressing_pickup_checkpoint_lift0.04_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_motionend2.5_smooth_xshift-0.075_syaw-0.22_at1.5_palmshift0,0,0.02_from75fdd3a3b4b1'
            if args.sustained else
            'joint_dressing_pickup_checkpoint_lift0.035_force0.7_iclose0_ipre-1_side0_iw10_up0.12_start1.65_xshift-0.065_syaw-0.22_at1.5_from75fdd3a3b4b1')
route=json.loads(SOURCE.read_text(encoding='utf8'))
pilot=json.loads((OUT/f'{PILOT_STEM}.json').read_text(encoding='utf8'))
segment=route['trace'][:785]
pickup=np.load(OUT/f'{PILOT_STEM}_qpos.npz')
prior=np.asarray([r['qpos'] for r in segment])
checkpoint=np.load(OUT/'joint_support_pinch_staged_wbc_report_spool_settled_handle_75fdd3a3b4b1_final_state.npz')
if not np.array_equal(prior[-1],checkpoint['qpos']):raise RuntimeError('Route/checkpoint state mismatch')
if not pilot['completed'] or len(pickup['qpos'])!=(150 if args.sustained else 125):raise RuntimeError('Incomplete pickup pilot')
if not pilot['final_support_quality']['valid'] or not pilot['all_feet_supported'] or pilot['any_deep_robot_contact']:
    raise RuntimeError('Pickup support/feet/collision gate failed')
if pilot['final_pad_lift_m']<.025 or pilot['max_pad_lateral_shift_m']>.075:
    raise RuntimeError('Pickup movement gate failed')
if args.sustained:
    tail=[r for r in pilot['trace'] if r['time_s']>=2.7]
    if len(tail)<14 or not all(r['support_valid'] for r in tail):raise RuntimeError('Sustained support gate failed')
    if any(any('dressing_stage' in c['geoms'] for c in r['pad_contacts']) for r in tail):
        raise RuntimeError('Sustained dressing stage clearance failed')
    for r in tail:
        for digit in ('thumb','index_finger'):
            force=sum(c['normal_force_n'] for c in r['pad_contacts']
                      if any(f'g1_a/wuji_right/r_{digit}' in b for b in c['bodies']))
            if not .05<=force<=2.:raise RuntimeError(f'Sustained {digit} grip gate failed')
states=np.vstack([prior,pickup['qpos']])
times=np.arange(1,len(states)+1,dtype=float)/50.
name=('joint_dressing_pickup_sustained_half_second_demo' if args.sustained else
      'joint_dressing_pickup_checkpoint_continuous_demo')
np.savez_compressed(OUT/f'{name}_qpos.npz',qpos=states,time_s=times)
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'))
model.vis.global_.offwidth=960;model.vis.global_.offheight=540
data=mujoco.MjData(model)
camera=mujoco.MjvCamera();camera.lookat[:]=([-.25,-.38,.73] if args.closeup else [0.,-.16,.78])
camera.distance=.95 if args.closeup else 2.15
camera.azimuth=130;camera.elevation=-19
ffmpeg=ROOT/'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
video=OUT/f'{name}{"_hand_closeup" if args.closeup else ""}.mp4'
proc=subprocess.Popen([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo',
    '-pixel_format','rgb24','-video_size','960x540','-framerate','10','-i','pipe:0',
    '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
    stdin=subprocess.PIPE)
try:
    with mujoco.Renderer(model,height=540,width=960) as renderer:
        for q in states[::5]:
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            proc.stdin.write(renderer.render().tobytes())
finally:
    proc.stdin.close()
if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
manifest={'scope':f'Continuous state replay of 15.7 s double-G1 grasp route and {3.0 if args.sustained else 2.5} s dressing pickup from identical checkpoint; controller restarted at checkpoint',
          'source_route':SOURCE.name,'source_pilot':f'{PILOT_STEM}.json',
          'video':video.name,'trajectory':f'{name}_qpos.npz','state_hz':50,'video_fps':10,
          'camera':'hand closeup' if args.closeup else 'overview',
          'duration_s':float(len(states)/50),'frames':int(len(states)/5),
          'checkpoint_qpos_exact_match':True,'pad_lift_m':pilot['final_pad_lift_m'],
          'pad_lateral_shift_m':pilot['max_pad_lateral_shift_m'],
          'support_quality_final':pilot['final_support_quality'],
          'spool_lifted':False,'wrap_performed':False,'hardware_validated':False,
          'strict_continuous_dressing_stage_clearance':False,
          'final_0p3s_dressing_stage_clearance':bool(args.sustained),
          'sustained_final_0p3s_contact_and_support_gate':bool(args.sustained)}
(OUT/f'{name}{"_hand_closeup" if args.closeup else ""}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps(manifest))
