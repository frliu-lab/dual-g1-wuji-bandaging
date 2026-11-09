"""Prewrapped flexible strip around a static forearm proxy, without robots.

The experiment asks whether a deformable two-turn strip can initialize and
simulate without severe penetration. It is not an executed bandaging action.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
import mujoco

OUT = Path(__file__).resolve().parents[2] / 'outputs/wuji_multiagent/bandage_v0'


def make_xml(n=33, radius=.049, width=.018, pitch=.11):
    points = []
    triangles = []
    for i in range(n):
        u = i/(n-1)
        theta = 4*np.pi*u
        for side in (-.5, .5):
            points.extend((pitch*(u-.5)+width*side,
                           radius*np.cos(theta), .12+radius*np.sin(theta)))
        if i:
            a = 2*(i-1)
            b = 2*i
            triangles.extend((a, a+1, b, a+1, b+1, b))
    p = ' '.join(f'{x:.7f}' for x in points)
    e = ' '.join(map(str, triangles))
    return f'''<mujoco model="prewrapped_flex_pilot">
      <option timestep=".0005" iterations="100"/>
      <worldbody>
        <geom name="forearm_proxy" type="capsule" fromto="-.16 0 .12 .16 0 .12"
              size=".048" friction=".8 .005 .0001" rgba=".64 .71 .76 1"/>
        <geom name="floor" type="plane" pos="0 0 -.05" size="1 1 .01" rgba=".12 .15 .19 1"/>
        <flexcomp name="bandage_strip" type="direct" dim="2" mass=".025"
                  radius=".002" rgba=".95 .9 .75 1" point="{p}" element="{e}">
          <pin id="0 1 64 65"/>
          <edge equality="true" damping=".2"/>
          <contact solref=".0015 1" friction=".8 .005 .0001" selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xml = make_xml()
    (OUT / 'helix_flex_pilot.xml').write_text(xml, encoding='utf-8')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    forearm = model.geom('forearm_proxy').id
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, 'bandage_strip')
    contact_samples = 0
    max_penetration = 0.
    finite = True
    for _ in range(round(.25/model.opt.timestep)):
        mujoco.mj_step(model, data)
        finite &= bool(np.all(np.isfinite(data.qpos)))
        for c in data.contact:
            if forearm in c.geom and flex_id in c.flex:
                contact_samples += 1
                max_penetration = max(max_penetration, max(0., -c.dist))
    vertices = np.asarray(data.flexvert_xpos)[:66].reshape(33, 2, 3)
    centers = vertices.mean(axis=1)
    radial = np.hypot(centers[:, 1], centers[:, 2]-.12)
    near_arm = (.047 <= radial) & (radial <= .060)
    angle = np.mod(np.arctan2(centers[:, 2]-.12, centers[:, 1]), 2*np.pi)
    bins = np.floor(angle/(2*np.pi)*12).astype(int).clip(0, 11)
    turn_coverage = [len(set(bins[lo:hi][near_arm[lo:hi]]))/12 for lo, hi in ((0, 17), (16, 33))]
    report = {
        'scope': 'Prewrapped deformable strip around fixed forearm; no robot motion or force calibration',
        'flex_vertices': model.nflexvert,
        'forearm_contact_samples': contact_samples,
        'maximum_forearm_penetration_m': max_penetration,
        'near_arm_centerline_fraction': float(near_arm.mean()),
        'angular_coverage_fraction_per_turn_12_bins': turn_coverage,
        'finite_after_quarter_second': finite,
    }
    (OUT / 'helix_flex_pilot_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, .12]
    cam.distance = .55
    cam.azimuth = 105
    cam.elevation = -18
    ffmpeg = Path(__file__).resolve().parents[2] / 'work/retarget_env/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    with mujoco.Renderer(model, height=540, width=960) as renderer:
        renderer.update_scene(data, camera=cam)
        subprocess.run([str(ffmpeg), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                        '-pixel_format', 'rgb24', '-video_size', '960x540', '-i', 'pipe:0',
                        '-frames:v', '1', str(OUT / 'helix_flex_pilot_preview.png')],
                       input=renderer.render().tobytes(), check=True)
    print(json.dumps(report))
    if model.nflex != 1 or not finite or min(turn_coverage) < .75:
        raise RuntimeError('Prewrapped flexible strip pilot failed')


if __name__ == '__main__':
    main()
