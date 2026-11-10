"""Render close views of the isolated A-left static forearm contact candidate."""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
scene=ET.parse(OUT/'dynamic_forearm_variant.xml').getroot()
scene.find(".//body[@name='g1_a/pelvis']").set('pos','-.3 0 .8')
model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
data=mujoco.MjData(model)
data.qpos[:]=np.load(OUT/'dynamic_palm_support_search_root_m0p3_pinky0p5_candidate.npz')['qpos']
mujoco.mj_forward(model,data)
forearm_axis=data.xmat[model.body('training_forearm_dynamic').id].reshape(3,3)@np.array([.34,-.17,-.04])
forearm_axis/=np.linalg.norm(forearm_axis)
wrist=data.xpos[model.body('g1_a/wuji_left/l_wrist').id]
index_tip=data.xpos[model.body('g1_a/wuji_left/l_index_finger_tip_sensor_frame').id]
finger_axis=index_tip-wrist;finger_axis/=np.linalg.norm(finger_axis)
print('Finger-to-forearm alignment cosine:',float(finger_axis@forearm_axis))
model.vis.global_.offwidth=1000;model.vis.global_.offheight=750
opt=mujoco.MjvOption();opt.geomgroup[2]=0
opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=0
ffmpeg=ROOT/'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
with mujoco.Renderer(model,height=750,width=1000) as renderer:
    for name,azimuth,elevation in [('front',90,-15),('side',25,-15),('top',90,-60)]:
        camera=mujoco.MjvCamera();camera.lookat[:]=[.02,.075,.83]
        camera.distance=.68;camera.azimuth=azimuth;camera.elevation=elevation
        renderer.update_scene(data,camera=camera,scene_option=opt)
        subprocess.run([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo',
                        '-pixel_format','rgb24','-video_size','1000x750','-i','pipe:0',
                        '-frames:v','1',str(OUT/f'dynamic_palm_candidate_{name}.png')],
                       input=renderer.render().tobytes(),check=True)
print('Rendered three diagnostic views')
