"""Make an isolated bandage-pad variant with a physical pinch tab."""
from pathlib import Path
import mujoco

root=Path(__file__).resolve().parents[2]/'outputs/wuji_multiagent/bandage_v0'
src=root/'joint_spool_grasp_static_z0p78_axisx_root0p55_rooty0p1_rootyawm30p0_spoolz0p702_handle_variant.xml'
dst=root/'joint_spool_grasp_dressing_pull_tab_variant.xml'
text=src.read_text(encoding='utf-8')
old='<geom name="dressing_pad_contact" type="box" size=".045 .03 .008" mass=".012" rgba=".95 .95 .92 1" friction="1 .005 .0001" />'
new=old+'<geom name="dressing_pull_tab" type="box" pos="0 0 .018" size=".012 .004 .014" mass=".001" rgba=".82 .89 .96 1" friction="1.2 .005 .0001" />'
if text.count(old)!=1:raise RuntimeError('Expected one dressing pad geom')
dst.write_text(text.replace(old,new),encoding='utf-8')
m=mujoco.MjModel.from_xml_path(str(dst))
assert m.geom('dressing_pull_tab').id>=0
print(dst,'nq',m.nq,'nv',m.nv,'nu',m.nu)
