"""Render the static four-hand precontact IK seed for visual inspection."""
from pathlib import Path
import subprocess
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/wuji_multiagent/bandage_v0'
model = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
data = mujoco.MjData(model)
seed = np.load(OUT / 'reach_ik_seed.npz')
data.qpos[:] = seed['qpos']
mujoco.mj_forward(model, data)
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 720
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, .80]
cam.distance = 2.8
cam.azimuth = 90
cam.elevation = -18
opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
opt.geomgroup[2] = 0
ffmpeg = ROOT / 'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
with mujoco.Renderer(model, height=720, width=1280) as renderer:
    renderer.update_scene(data, camera=cam, scene_option=opt)
    subprocess.run([str(ffmpeg), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgb24', '-video_size', '1280x720', '-i', 'pipe:0',
                    '-frames:v', '1', str(OUT / 'reach_ik_seed_preview.png')],
                   input=renderer.render().tobytes(), check=True)
print(OUT / 'reach_ik_seed_preview.png')
