"""Move a proxy guide holding the free end of a prewrapped flexible strip.

This tests a kinematic hand proxy, not grasping with Wuji fingers.
"""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from helix_flex_pilot import make_xml, OUT


def make_guided_xml():
    root = ET.fromstring(make_xml())
    root.set('model', 'guided_prewrapped_flex_pilot')
    world = root.find('worldbody')
    world.find('flexcomp/pin').set('id', '0 1')
    guide = ET.SubElement(world, 'body', {'name': 'guide_proxy', 'mocap': 'true',
                                         'pos': '.055 .049 .12'})
    ET.SubElement(guide, 'geom', {'name': 'guide_proxy_visual', 'type': 'sphere',
                                  'size': '.006', 'rgba': '.15 .5 .95 .8',
                                  'contype': '0', 'conaffinity': '0'})
    eq = ET.SubElement(root, 'equality')
    for vertex, x in ((64, .046), (65, .064)):
        ET.SubElement(eq, 'connect', {'name': f'guide_strip_{vertex}',
                                     'body1': f'bandage_strip_{vertex}',
                                     'body2': 'guide_proxy',
                                     'anchor': f'{x} .049 .12',
                                     'solref': '.005 1'})
    return ET.tostring(root, encoding='unicode')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xml = make_guided_xml()
    (OUT / 'guided_flex_pilot.xml').write_text(xml, encoding='utf-8')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    forearm = model.geom('forearm_proxy').id
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, 'bandage_strip')
    guide_id = model.body('guide_proxy').mocapid[0]
    guide_eq_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                     f'guide_strip_{vertex}') for vertex in (64, 65)]
    contact_samples = 0
    max_penetration = 0.
    max_constraint_reaction = [0., 0.]
    finite = True
    for step in range(round(.5/model.opt.timestep)):
        time = step*model.opt.timestep
        u = min(1., max(0., (time-.05)/.40))
        theta = .5*np.pi*(3*u*u-2*u*u*u)
        data.mocap_pos[guide_id] = [.055, .049*np.cos(theta), .12+.049*np.sin(theta)]
        mujoco.mj_step(model, data)
        finite &= bool(np.all(np.isfinite(data.qpos)))
        for c in data.contact:
            if forearm in c.geom and flex_id in c.flex:
                contact_samples += 1
                max_penetration = max(max_penetration, max(0., -c.dist))
        for j, eq_id in enumerate(guide_eq_ids):
            rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq_id)
            if np.any(rows):
                max_constraint_reaction[j] = max(max_constraint_reaction[j],
                                                 float(np.linalg.norm(data.efc_force[rows])))
    vertices = np.asarray(data.flexvert_xpos)[:66].reshape(33, 2, 3)
    centers = vertices.mean(axis=1)
    radial = np.hypot(centers[:, 1], centers[:, 2]-.12)
    near_arm = (.047 <= radial) & (radial <= .065)
    angle = np.mod(np.arctan2(centers[:, 2]-.12, centers[:, 1]), 2*np.pi)
    bins = np.floor(angle/(2*np.pi)*12).astype(int).clip(0, 11)
    coverage = [len(set(bins[lo:hi][near_arm[lo:hi]]))/12 for lo, hi in ((0, 17), (16, 33))]
    endpoint_error = float(np.linalg.norm(centers[-1]-data.mocap_pos[guide_id]))
    report = {
        'scope': 'Prewrapped strip with free end held by moving kinematic guide, no robot grasp or wrapping',
        'guide_angular_motion_deg': 90.,
        'guide_chord_displacement_m': float(.049*np.sqrt(2)),
        'endpoint_tracking_error_m': endpoint_error,
        'angular_coverage_fraction_per_turn_12_bins': coverage,
        'near_arm_centerline_fraction': float(near_arm.mean()),
        'forearm_contact_samples': contact_samples,
        'maximum_forearm_penetration_m': max_penetration,
        'guide_constraint_reaction_max_N_per_edge_vertex': max_constraint_reaction,
        'finite_after_half_second': finite,
    }
    (OUT / 'guided_flex_pilot_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
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
                        '-frames:v', '1', str(OUT / 'guided_flex_pilot_preview.png')],
                       input=renderer.render().tobytes(), check=True)
    print(json.dumps(report))
    if not finite or endpoint_error > .01 or min(coverage) < .75:
        raise RuntimeError('Guided flexible strip pilot failed')


if __name__ == '__main__':
    main()
