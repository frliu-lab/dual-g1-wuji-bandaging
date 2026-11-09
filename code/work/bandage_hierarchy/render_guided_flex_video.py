"""Render a slowed diagnostic of the kinematic guide and prewrapped strip.

The simulation lasts 0.5 s; the 2 s video is 4x slow motion for inspection.
"""
from pathlib import Path
import subprocess
import numpy as np
import mujoco

from guided_flex_pilot import make_guided_xml, OUT

ROOT = Path(__file__).resolve().parents[2]
model = mujoco.MjModel.from_xml_string(make_guided_xml())
data = mujoco.MjData(model)
guide_id = model.body('guide_proxy').mocapid[0]
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 360
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, .12]
cam.distance = .55
cam.azimuth = 105
cam.elevation = -18
ffmpeg = ROOT / 'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
target = OUT / 'guided_flex_pilot_4x_slow.mp4'
samples = set(np.linspace(0, 999, 60).round().astype(int))
proc = subprocess.Popen([str(ffmpeg), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                         '-pixel_format', 'rgb24', '-video_size', '640x360', '-framerate', '30',
                         '-i', 'pipe:0', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                         '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
try:
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        for step in range(1000):
            time = step*model.opt.timestep
            u = min(1., max(0., (time-.05)/.40))
            theta = .5*np.pi*(3*u*u-2*u*u*u)
            data.mocap_pos[guide_id] = [.055, .049*np.cos(theta), .12+.049*np.sin(theta)]
            mujoco.mj_step(model, data)
            if step in samples:
                renderer.update_scene(data, camera=cam)
                proc.stdin.write(renderer.render().tobytes())
finally:
    proc.stdin.close()
if proc.wait() != 0:
    raise RuntimeError('Diagnostic video rendering failed')
print(target)
