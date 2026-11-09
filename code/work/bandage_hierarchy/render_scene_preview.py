"""Render the static task scene for geometry inspection; no action is implied."""
from pathlib import Path
import subprocess
import mujoco

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / 'outputs/wuji_multiagent/bandage_v0'
m = mujoco.MjModel.from_xml_path(str(p / 'scene.xml'))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
m.vis.global_.offwidth = 1280
m.vis.global_.offheight = 720
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, .80]
cam.distance = 3.4
cam.azimuth = 90
cam.elevation = -18
opt = mujoco.MjvOption()
opt.geomgroup[2] = 0
ffmpeg = ROOT / 'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
with mujoco.Renderer(m, height=720, width=1280) as renderer:
    renderer.update_scene(d, camera=cam, scene_option=opt)
    subprocess.run([str(ffmpeg), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                    '-pixel_format', 'rgb24', '-video_size', '1280x720', '-i', 'pipe:0',
                    '-frames:v', '1', str(p / 'scene_preview.png')],
                   input=renderer.render().tobytes(), check=True)
print(p / 'scene_preview.png')
