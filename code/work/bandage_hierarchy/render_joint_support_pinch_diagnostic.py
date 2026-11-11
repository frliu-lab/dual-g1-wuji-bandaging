"""Render static target and failed dynamic endpoint for visual audit."""
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wuji_multiagent/bandage_v0'
model=mujoco.MjModel.from_xml_path(str(OUT/'joint_support_pinch_variant.xml'))
model.vis.global_.offwidth=1200
model.vis.global_.offheight=800
renderer=mujoco.Renderer(model,height=800,width=1200)
data=mujoco.MjData(model)
visual=mujoco.MjvOption()
visual.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=0
visual.geomgroup[2]=0
states={
    'static_target':np.load(OUT/'joint_support_pinch_static_candidate.npz')['qpos'],
    'dynamic_endpoint':np.load(OUT/'joint_support_pinch_staged_wbc_report_final_state.npz')['qpos'],
}
for label,qpos in states.items():
    data.qpos[:]=qpos
    mujoco.mj_forward(model,data)
    for view,azimuth in (('front',90),('oblique',135)):
        camera=mujoco.MjvCamera()
        camera.lookat[:]=[-.22,-.30,.82]
        camera.distance=1.3
        camera.azimuth=azimuth
        camera.elevation=-18
        renderer.update_scene(data,camera=camera,scene_option=visual)
        iio.imwrite(OUT/f'joint_support_pinch_{label}_{view}.png',renderer.render())
renderer.close()
print('rendered static target and dynamic endpoint from saved states')
