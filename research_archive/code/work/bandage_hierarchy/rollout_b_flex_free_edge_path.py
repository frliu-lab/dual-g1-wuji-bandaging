"""Closed-loop dual-G1 WBC test of the collision-screened free-edge approach."""
import json
import hashlib
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/wuji_multiagent'))
from joint_wbc import JointWBC
from flex_contact_jac import flex_contact_jacobian

OUT = ROOT/'outputs/wuji_multiagent/bandage_v0'
parser = argparse.ArgumentParser()
parser.add_argument('--track-edge', action='store_true')
parser.add_argument('--edge-damping', type=float, default=.2)
parser.add_argument('--pinchtab',action='store_true')
parser.add_argument('--extended-tab',action='store_true')
parser.add_argument('--start-at-goal',action='store_true')
parser.add_argument('--duration-s',type=float,default=1.)
parser.add_argument('--flex-tip-force-n',type=float,default=None)
parser.add_argument('--b-right-force-digits',choices=('both','thumb','index'),default='both')
parser.add_argument('--b-right-position-control',action='store_true')
parser.add_argument('--b-right-loaded-pinch-seed',action='store_true')
parser.add_argument('--b-right-loaded-pinch-seed-rank',type=int,default=-1)
parser.add_argument('--reinforced-hem',action='store_true')
parser.add_argument('--forearm-anchored-bandage',action='store_true')
parser.add_argument('--body-parented-bandage',action='store_true')
parser.add_argument('--body-parented-refaligned-bandage',action='store_true')
parser.add_argument('--b-right-hem-seed-rank',type=int,default=0)
parser.add_argument('--hem-force-n',type=float,default=None)
parser.add_argument('--hem-radius-mm',type=float,default=2.5)
parser.add_argument('--b-right-digit-material-anchor',action='store_true')
parser.add_argument('--flex-contact-radius-mm',type=float,default=1.)
parser.add_argument('--b-right-initial-index-pip-offset-rad',type=float,default=0.)
parser.add_argument('--measured-tip-force-n',type=float,default=None)
parser.add_argument('--pinch-bias-preset',choices=('balanced_a','balanced_b'),default=None)
parser.add_argument('--pinch-bias-until-s',type=float,default=None)
parser.add_argument('--one-step-hand-search',action='store_true')
parser.add_argument('--one-step-hand-search-until-s',type=float,default=None)
parser.add_argument('--hand-search-horizon-s',type=float,default=.02)
parser.add_argument('--hand-search-edge-weight',type=float,default=.3)
parser.add_argument('--hand-search-force-weight',type=float,default=0.)
parser.add_argument('--hand-search-trajectory-force-weight',type=float,default=0.)
parser.add_argument('--hand-search-penetration-weight',type=float,default=10.)
parser.add_argument('--hand-accel-cap',type=float,default=200.)
parser.add_argument('--wbc-constraint-tolerance',type=float,default=2e-5)
parser.add_argument('--disable-dressing-pad-contact',action='store_true')
parser.add_argument('--dressing-pad-solref',type=float,default=None)
parser.add_argument('--a-pad-side-grasp-seed',action='store_true')
parser.add_argument('--stage-release-s',type=float,default=None)
parser.add_argument('--a-pad-normal-goal-n',type=float,default=None)
parser.add_argument('--a-pad-force-control',action='store_true')
parser.add_argument('--release-equipment-tray-with-stage',action='store_true')
parser.add_argument('--b-middle-retract-mode',choices=('flex','abd'),default=None)
parser.add_argument('--b-spool-normal-goal-n',type=float,default=None)
parser.add_argument('--b-spool-force-control',action='store_true')
parser.add_argument('--b-left-lift-mm',type=float,default=0.)
parser.add_argument('--b-spool-one-step-search',action='store_true')
parser.add_argument('--b-spool-search-horizon-s',type=float,default=.04)
parser.add_argument('--b-spool-search-bias-rad',type=float,default=.05)
parser.add_argument('--b-spool-search-post-release-bias-rad',type=float,default=None)
parser.add_argument('--b-spool-search-switch-s',type=float,default=None)
parser.add_argument('--b-spool-search-support-weight',type=float,default=0.)
parser.add_argument('--a-left-support-goal-n',type=float,default=None)
parser.add_argument('--a-left-palm-weight',type=float,default=1000.)
parser.add_argument('--a-left-follow-forearm',action='store_true')
parser.add_argument('--a-left-palm-contact-servo',action='store_true')
parser.add_argument('--a-left-servo-max-step-mm',type=float,default=.5)
parser.add_argument('--a-left-servo-max-offset-mm',type=float,default=20.)
parser.add_argument('--a-left-servo-gain',type=float,default=.5)
parser.add_argument('--a-left-servo-lookahead-s',type=float,default=0.)
parser.add_argument('--a-left-palm-preload-mm',type=float,default=0.)
parser.add_argument('--a-left-near-elbow-seed',action='store_true')
parser.add_argument('--a-left-palm-only-seed',action='store_true')
parser.add_argument('--a-left-follow-max-offset-mm',type=float,default=2.)
parser.add_argument('--a-left-follow-max-step-mm',type=float,default=.2)
parser.add_argument('--training-elbow-stiffness',type=float,default=None)
parser.add_argument('--training-elbow-damping',type=float,default=None)
parser.add_argument('--training-elbow-wbc-weight',type=float,default=None)
parser.add_argument('--training-elbow-wbc-kp',type=float,default=None)
parser.add_argument('--training-elbow-range-rad',type=float,default=.35)
parser.add_argument('--flex-friction',type=float,default=None)
parser.add_argument('--wrist-weight',type=float,default=None)
parser.add_argument('--local-physics-correction',action='store_true')
parser.add_argument('--stabilize-other-digits',action='store_true')
parser.add_argument('--stabilize-all-digits',action='store_true')
parser.add_argument('--material-tip-follow',action='store_true')
parser.add_argument('--forearm-grasp-hold-weight', type=float, default=0.)
parser.add_argument('--relative-grasp-jacobian', action='store_true')
parser.add_argument('--tip-closure-mm',type=float,default=0.)
parser.add_argument('--tip-weight',type=float,default=1000.)
parser.add_argument('--flex-measured-force-prior-weight',type=float,default=0.)
parser.add_argument('--releasable-fixture',action='store_true')
parser.add_argument('--gate-fixture-release-on-loaded-pinch',action='store_true')
parser.add_argument('--fixture-release-s',type=float,default=None)
parser.add_argument('--pinch-closure-mm',type=float,default=0.)
parser.add_argument('--b-right-palm-weight',type=float,default=1000.)
args = parser.parse_args()
if args.relative_grasp_jacobian and not args.forearm_grasp_hold_weight:
    parser.error('Relative grasp Jacobian requires a forearm grasp hold weight')
if not np.isfinite(args.forearm_grasp_hold_weight) or args.forearm_grasp_hold_weight < 0:
    parser.error('Forearm grasp hold weight must be finite and nonnegative')
if args.forearm_grasp_hold_weight and (not args.b_right_digit_material_anchor or args.track_edge or args.material_tip_follow):
    parser.error('Forearm grasp hold needs digit anchors and exclusive control of their position targets')
if args.extended_tab and (not args.pinchtab or not args.start_at_goal):
    parser.error('--extended-tab requires --pinchtab --start-at-goal; the existing path has a different flex state')
if args.fixture_release_s is not None and not args.releasable_fixture:
    parser.error('--fixture-release-s requires --releasable-fixture')
if args.gate_fixture_release_on_loaded_pinch and args.fixture_release_s is None:
    parser.error('--gate-fixture-release-on-loaded-pinch requires --fixture-release-s')
if args.measured_tip_force_n is not None and (args.measured_tip_force_n<=0 or not args.pinchtab):
    parser.error('--measured-tip-force-n requires a positive force and --pinchtab')
if not np.isfinite(args.b_right_initial_index_pip_offset_rad) or abs(args.b_right_initial_index_pip_offset_rad)>.15:
    parser.error('--b-right-initial-index-pip-offset-rad must be finite and within ±0.15')
if not np.isfinite(args.pinch_closure_mm) or not 0<=args.pinch_closure_mm<=2.:
    parser.error('--pinch-closure-mm must be finite and in [0,2]')
if not np.isfinite(args.flex_contact_radius_mm) or not 1<=args.flex_contact_radius_mm<=3:
    parser.error('--flex-contact-radius-mm must be finite and in [1,3]')
if args.b_right_loaded_pinch_seed and (not args.extended_tab or args.flex_contact_radius_mm not in (1.5,2.)):
    parser.error('--b-right-loaded-pinch-seed requires extended tab and 1.5 or 2 mm flex contact radius')
if args.b_right_loaded_pinch_seed_rank < -2 or args.b_right_loaded_pinch_seed_rank > 29 or (
        args.b_right_loaded_pinch_seed_rank != -1 and not args.b_right_loaded_pinch_seed):
    parser.error('--b-right-loaded-pinch-seed-rank requires loaded seed and rank -2 or 0..29')
if args.b_right_digit_material_anchor and (not args.b_right_loaded_pinch_seed or args.material_tip_follow):
    if not args.reinforced_hem or args.material_tip_follow:
        parser.error('--b-right-digit-material-anchor requires a loaded or hem seed and excludes tip following')
if args.reinforced_hem and (not args.extended_tab or not args.releasable_fixture or
                            not args.b_right_digit_material_anchor or args.b_right_loaded_pinch_seed):
    parser.error('--reinforced-hem requires extended tab, releasable fixture and digit anchor; excludes loaded flex seed')
if args.forearm_anchored_bandage and not args.reinforced_hem:
    parser.error('--forearm-anchored-bandage requires reinforced hem')
if args.body_parented_bandage and (not args.reinforced_hem or args.forearm_anchored_bandage):
    parser.error('--body-parented-bandage requires reinforced hem and excludes equality forearm anchor')
if args.body_parented_refaligned_bandage and (not args.reinforced_hem or args.forearm_anchored_bandage or args.body_parented_bandage):
    parser.error('--body-parented-refaligned-bandage requires reinforced hem and excludes other forearm anchors')
if not 0<=args.b_right_hem_seed_rank<=29:
    parser.error('--b-right-hem-seed-rank must be in [0,29]')
if args.hem_force_n is not None and (not args.reinforced_hem or args.hem_force_n<=0 or
                                     not np.isfinite(args.hem_force_n)):
    parser.error('--hem-force-n requires reinforced hem and a positive finite target')
if not np.isfinite(args.hem_radius_mm) or not 1.5<=args.hem_radius_mm<=2.5 or (
        not args.reinforced_hem and args.hem_radius_mm!=2.5):
    parser.error('--hem-radius-mm requires reinforced hem and must be in [1.5,2.5]')
if args.flex_friction is not None and (not np.isfinite(args.flex_friction) or args.flex_friction<=0):
    parser.error('--flex-friction must be positive and finite')
if args.pinch_bias_until_s is not None and (args.pinch_bias_preset is None or args.pinch_bias_until_s<0):
    parser.error('--pinch-bias-until-s requires a preset and nonnegative time')
if args.one_step_hand_search and (not args.extended_tab or args.pinch_bias_preset or args.measured_tip_force_n):
    parser.error('--one-step-hand-search requires --extended-tab and excludes other hand corrections')
if args.one_step_hand_search_until_s is not None and (
        not args.one_step_hand_search or args.one_step_hand_search_until_s<=0 or
        not np.isfinite(args.one_step_hand_search_until_s)):
    parser.error('--one-step-hand-search-until-s requires hand search and must be positive and finite')
if args.hand_search_horizon_s<.02 or args.hand_search_horizon_s>.1:
    parser.error('--hand-search-horizon-s must be between 0.02 and 0.1')
if args.hand_search_edge_weight<0 or not np.isfinite(args.hand_search_edge_weight):
    parser.error('--hand-search-edge-weight must be finite and nonnegative')
if args.hand_search_trajectory_force_weight<0 or not np.isfinite(args.hand_search_trajectory_force_weight):
    parser.error('--hand-search-trajectory-force-weight must be finite and nonnegative')
if args.hand_search_penetration_weight<0 or not np.isfinite(args.hand_search_penetration_weight):
    parser.error('--hand-search-penetration-weight must be finite and nonnegative')
if args.hand_accel_cap<200 or args.hand_accel_cap>1000 or not np.isfinite(args.hand_accel_cap):
    parser.error('--hand-accel-cap must be finite and between 200 and 1000')
if not 0<args.wbc_constraint_tolerance<=1e-4 or not np.isfinite(args.wbc_constraint_tolerance):
    parser.error('--wbc-constraint-tolerance must be finite and in (0, 1e-4]')
if args.dressing_pad_solref is not None and (args.dressing_pad_solref<.02 or args.dressing_pad_solref>.2):
    parser.error('--dressing-pad-solref must be between 0.02 and 0.2 s')
if args.a_pad_side_grasp_seed and not args.extended_tab:
    parser.error('--a-pad-side-grasp-seed requires --extended-tab')
if args.a_left_near_elbow_seed and args.a_left_palm_only_seed:
    parser.error('Choose one A-left support seed')
if args.a_left_follow_forearm and args.a_left_palm_contact_servo:
    parser.error('Choose one A-left palm target controller')
if args.a_left_palm_preload_mm and (args.a_left_follow_forearm or args.a_left_palm_contact_servo):
    parser.error('Palm preload is a standalone target controller in this experiment')
if args.stage_release_s is not None and args.stage_release_s<0:
    parser.error('--stage-release-s must be nonnegative')
if args.a_pad_normal_goal_n is not None and (args.a_pad_normal_goal_n<0 or not np.isfinite(args.a_pad_normal_goal_n)):
    parser.error('--a-pad-normal-goal-n must be finite and nonnegative')
if args.a_pad_force_control and (not args.stabilize_other_digits or args.a_pad_normal_goal_n is None):
    parser.error('--a-pad-force-control requires --stabilize-other-digits and a pad force goal')
if args.b_right_position_control and not args.stabilize_other_digits:
    parser.error('--b-right-position-control requires --stabilize-other-digits')
if args.release_equipment_tray_with_stage and args.stage_release_s is None:
    parser.error('--release-equipment-tray-with-stage requires --stage-release-s')
if args.b_middle_retract_mode and not (args.stabilize_other_digits or args.stabilize_all_digits):
    parser.error('--b-middle-retract-mode requires hand position control')
if args.b_spool_normal_goal_n is not None and (args.b_spool_normal_goal_n<0 or not np.isfinite(args.b_spool_normal_goal_n)):
    parser.error('--b-spool-normal-goal-n must be finite and nonnegative')
if args.b_spool_force_control and (not args.stabilize_other_digits or args.b_spool_normal_goal_n is None):
    parser.error('--b-spool-force-control requires --stabilize-other-digits and a spool force goal')
if args.b_left_lift_mm<0 or args.b_left_lift_mm>30 or not np.isfinite(args.b_left_lift_mm):
    parser.error('--b-left-lift-mm must be finite and between 0 and 30')
if not .02<=args.b_spool_search_horizon_s<=.1 or not np.isfinite(args.b_spool_search_horizon_s):
    parser.error('--b-spool-search-horizon-s must be finite and between 0.02 and 0.1')
if not 0<args.b_spool_search_bias_rad<=.2 or not np.isfinite(args.b_spool_search_bias_rad):
    parser.error('--b-spool-search-bias-rad must be finite and between 0 and 0.2')
if args.b_spool_search_post_release_bias_rad is not None and (
        not 0<args.b_spool_search_post_release_bias_rad<=.2 or
        not np.isfinite(args.b_spool_search_post_release_bias_rad) or args.stage_release_s is None or
        not args.b_spool_one_step_search):
    parser.error('--b-spool-search-post-release-bias-rad requires spool search and stage release, and must be in (0, 0.2]')
if args.b_spool_search_switch_s is not None and (
        args.b_spool_search_post_release_bias_rad is None or args.b_spool_search_switch_s<0 or
        not np.isfinite(args.b_spool_search_switch_s)):
    parser.error('--b-spool-search-switch-s requires post-release bias and must be nonnegative')
if args.b_spool_search_support_weight<0 or not np.isfinite(args.b_spool_search_support_weight):
    parser.error('--b-spool-search-support-weight must be finite and nonnegative')
if args.a_left_support_goal_n is not None and (args.a_left_support_goal_n<0 or not np.isfinite(args.a_left_support_goal_n)):
    parser.error('--a-left-support-goal-n must be finite and nonnegative')
if args.a_left_palm_weight<=0 or not np.isfinite(args.a_left_palm_weight):
    parser.error('--a-left-palm-weight must be positive and finite')
if args.a_left_follow_max_offset_mm<0 or not np.isfinite(args.a_left_follow_max_offset_mm):
    parser.error('--a-left-follow-max-offset-mm must be finite and nonnegative')
if args.a_left_follow_max_step_mm<0 or not np.isfinite(args.a_left_follow_max_step_mm):
    parser.error('--a-left-follow-max-step-mm must be finite and nonnegative')
if (not 0<args.a_left_servo_max_step_mm<=5 or not np.isfinite(args.a_left_servo_max_step_mm) or
        not 0<args.a_left_servo_max_offset_mm<=50 or not np.isfinite(args.a_left_servo_max_offset_mm)):
    parser.error('A-left palm servo step/offset must be finite and within (0,5]/(0,50] mm')
if not 0<args.a_left_servo_gain<=5 or not np.isfinite(args.a_left_servo_gain):
    parser.error('--a-left-servo-gain must be finite and in (0,5]')
if not 0<=args.a_left_servo_lookahead_s<=.1 or not np.isfinite(args.a_left_servo_lookahead_s):
    parser.error('--a-left-servo-lookahead-s must be finite and in [0,0.1]')
if not 0<=args.a_left_palm_preload_mm<=10 or not np.isfinite(args.a_left_palm_preload_mm):
    parser.error('--a-left-palm-preload-mm must be finite and in [0,10]')
for value,name in ((args.training_elbow_stiffness,'--training-elbow-stiffness'),
                   (args.training_elbow_damping,'--training-elbow-damping')):
    if value is not None and (value<0 or not np.isfinite(value)):
        parser.error(f'{name} must be finite and nonnegative')
if not np.isfinite(args.training_elbow_range_rad) or not .35<=args.training_elbow_range_rad<=1.:
    parser.error('--training-elbow-range-rad must be finite and in [0.35,1]')
for value,name in ((args.training_elbow_wbc_weight,'--training-elbow-wbc-weight'),
                   (args.training_elbow_wbc_kp,'--training-elbow-wbc-kp')):
    if value is not None and (value<=0 or not np.isfinite(value)):
        parser.error(f'{name} must be positive and finite')
suffix = f'_damping{args.edge_damping:g}' if args.edge_damping != .2 else ''
SCENE = OUT/(('joint_dual_g1_pinchtab_extended20mm_releasable_fixture.xml' if args.releasable_fixture else 'joint_dual_g1_pinchtab_extended20mm.xml') if args.extended_tab else
             'joint_dual_g1_pinchtab_releasable_fixture.xml' if args.releasable_fixture else
             f'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5_pinchtab_bulged.xml'
             if args.pinchtab else
             f'joint_dual_g1_prewrapped_flex_scene_world_anchored_radius49mm_phase225deg_thickness1mm_tab1.5{suffix}.xml')
if args.reinforced_hem:
    SCENE=OUT/('joint_dual_g1_pinchtab_extended20mm_reinforced_hem_body_parented_refaligned.xml'
               if args.body_parented_refaligned_bandage else
               'joint_dual_g1_pinchtab_extended20mm_reinforced_hem_body_parented.xml'
               if args.body_parented_bandage else
               'joint_dual_g1_pinchtab_extended20mm_reinforced_hem_forearm_anchor.xml'
               if args.forearm_anchored_bandage else
               'joint_dual_g1_pinchtab_extended20mm_reinforced_hem.xml')
PATH = OUT/('joint_b_pinchtab_rrt_path.npz' if args.pinchtab else
            'joint_b_flex_free_edge_rrt_path.npz')
m = mujoco.MjModel.from_xml_path(str(SCENE))
hem_body_id=m.body('bandage_reinforced_hem').id if args.reinforced_hem else None
if args.reinforced_hem:
    m.geom_size[m.geom('bandage_reinforced_hem_grasp').id,0]=args.hem_radius_mm/1000
def bandage_contact(c):
    return any(f>=0 for f in c.flex) or (args.reinforced_hem and any(
        g>=0 and m.geom(int(g)).name=='bandage_reinforced_hem_grasp' for g in c.geom))
m.flex_radius[:]=args.flex_contact_radius_mm/1000
elbow_jid=m.joint('training_elbow_hinge').id
m.jnt_range[elbow_jid]=[-args.training_elbow_range_rad,args.training_elbow_range_rad]
if args.training_elbow_stiffness is not None:
    m.jnt_stiffness[elbow_jid]=args.training_elbow_stiffness
if args.training_elbow_damping is not None:
    m.dof_damping[m.jnt_dofadr[elbow_jid]]=args.training_elbow_damping
if args.disable_dressing_pad_contact:
    pad_geom=m.geom('dressing_pad_contact').id
    m.geom_contype[pad_geom]=0;m.geom_conaffinity[pad_geom]=0
if args.dressing_pad_solref is not None:
    m.geom_solref[m.geom('dressing_pad_contact').id,0]=args.dressing_pad_solref
if args.flex_friction is not None:
    m.flex_friction[:,0]=args.flex_friction
d = mujoco.MjData(m)
path_file = np.load(PATH)
waypoints = path_file['qpos']
joint_names = [str(s) for s in path_file['joint_names']]
qa = np.array([m.joint(s).qposadr[0] for s in joint_names])
va = np.array([m.joint(s).dofadr[0] for s in joint_names])
wrist_joint=m.joint('g1_b/right_wrist_roll_joint').id
wrist_qadr=int(m.jnt_qposadr[wrist_joint]);wrist_vadr=int(m.jnt_dofadr[wrist_joint])
wrist_actuator=int(np.flatnonzero(m.actuator_trnid[:,0]==wrist_joint)[0])
source_model=(mujoco.MjModel.from_xml_path(str(OUT/'joint_dual_g1_pinchtab_extended20mm_reinforced_hem.xml'))
              if args.body_parented_bandage or args.body_parented_refaligned_bandage else None)
if args.extended_tab and source_model is not None:
    source_state=source_model.qpos0.copy();source_state[:167]=waypoints[-1,:167]
    for j in range(source_model.njnt):
        name=source_model.joint(j).name
        old_adr=int(source_model.jnt_qposadr[j])
        if not name or old_adr>=167:continue
        width=(7 if source_model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else
               4 if source_model.jnt_type[j]==mujoco.mjtJoint.mjJNT_BALL else 1)
        adr=int(m.joint(name).qposadr[0])
        d.qpos[adr:adr+width]=source_state[old_adr:old_adr+width]
elif args.extended_tab:
    d.qpos[:167] = waypoints[-1,:167]
else:
    d.qpos[:] = waypoints[-1 if args.start_at_goal else 0]
if args.a_pad_side_grasp_seed:
    seed=np.load(OUT/'joint_b_flex_a_thumb_side_contact_candidates.npz')['qpos'][0]
    for j in range(m.njnt):
        if m.joint(j).name.startswith(('g1_a/wuji_right/r_thumb_','g1_a/wuji_right/r_index_finger_')):
            adr=int(m.jnt_qposadr[j])
            old_adr=int(source_model.joint(m.joint(j).name).qposadr[0]) if source_model is not None else adr
            d.qpos[adr]=seed[old_adr]
if args.a_left_near_elbow_seed or args.a_left_palm_only_seed:
    seed_name=('joint_a_left_near_elbow_support_palm_only_candidate.npz' if args.a_left_palm_only_seed
               else 'joint_a_left_near_elbow_support_candidate.npz')
    seed=np.load(OUT/seed_name)
    for name in seed['joint_names']:
        adr=int(m.joint(str(name)).qposadr[0]);d.qpos[adr]=seed['qpos'][adr]
if args.b_right_initial_index_pip_offset_rad:
    joint=m.joint('g1_b/wuji_right/r_index_finger_pip')
    adr=int(joint.qposadr[0]);lo,hi=m.jnt_range[joint.id]
    d.qpos[adr]=np.clip(d.qpos[adr]+args.b_right_initial_index_pip_offset_rad,lo+.01,hi-.01)
if args.b_right_loaded_pinch_seed:
    candidate=json.loads((OUT/f'joint_b_right_loaded_pinch_static_refine_radius{args.flex_contact_radius_mm:g}mm.json').read_text())
    hand_seed=(candidate['best_passing'] if args.b_right_loaded_pinch_seed_rank==-2 else
               candidate['best_safe'] if args.b_right_loaded_pinch_seed_rank==-1 else
               candidate['top_passing_clearance_force'][args.b_right_loaded_pinch_seed_rank])
    for name in ('g1_b/wuji_right/r_middle_finger_mcp_flex',
                 'g1_b/wuji_right/r_middle_finger_pip',
                 'g1_b/wuji_right/r_middle_finger_dip'):
        joint=m.joint(name)
        adr=int(joint.qposadr[0]);lo,hi=m.jnt_range[joint.id]
        d.qpos[adr]=np.clip(d.qpos[adr]+.5,lo+.01,hi-.01)
    for name,value in zip(candidate['joint_names'],hand_seed['joint_targets_rad']):
        d.qpos[m.joint(name).qposadr[0]]=value
if args.reinforced_hem:
    candidate=json.loads((OUT/'joint_b_reinforced_hem_static_grasp_screen.json').read_text())
    hand_seed=candidate['top_passing'][args.b_right_hem_seed_rank]
    for name in ('g1_b/wuji_right/r_middle_finger_mcp_flex',
                 'g1_b/wuji_right/r_middle_finger_pip',
                 'g1_b/wuji_right/r_middle_finger_dip'):
        d.qpos[m.joint(name).qposadr[0]]+=.5
    for name,value in zip(candidate['joint_names'],hand_seed['joint_targets_rad']):
        d.qpos[m.joint(name).qposadr[0]]=value
mujoco.mj_forward(m, d)
tip_names = ('g1_b/wuji_right/r_thumb_tip_sensor_frame',
             'g1_b/wuji_right/r_index_finger_tip_sensor_frame')
tip_ids=[m.body(name).id for name in tip_names]
pinch_actuators=[i for i in range(m.nu) if m.actuator(i).name.startswith('g1_b/wuji_right/r_') and
                 any(token in m.actuator(i).name for token in ('THJ','FFJ'))]
thumb_tip_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==tip_ids[0]]
index_tip_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==tip_ids[1]]
initial_tip_positions=np.array([d.xpos[bid].copy() for bid in tip_ids])
pad_body_id=m.body('dressing_pad').id
initial_pad_position=d.xpos[pad_body_id].copy()
spool_body_id=m.body('bandage_spool').id
b_left_wrist_body_id=m.body('g1_b/wuji_left/l_wrist').id
initial_spool_position=d.xpos[spool_body_id].copy()
initial_spool_wrist_relative=(d.xpos[spool_body_id]-d.xpos[b_left_wrist_body_id]).copy()
stage_geom_id=m.geom('dressing_stage').id
tray_geom_id=m.geom('equipment_tray').id

palms = ('g1_a/wuji_left/l_wrist', 'g1_a/wuji_right/r_wrist',
         'g1_b/wuji_left/l_wrist', 'g1_b/wuji_right/r_wrist')
def relevant_contact(model, c):
    return any(f >= 0 for f in c.flex) or any(
        model.body(model.geom_bodyid[g]).name.startswith(('g1_a/', 'g1_b/'))
        for g in c.geom if g >= 0)

wbc = JointWBC(m, d.qpos.copy(), palm_names=palms, contact_filter=relevant_contact,
               palm_position_weights=[args.a_left_palm_weight,1000.,1000.,args.b_right_palm_weight])
wbc.backend = 'DIRECT'
wbc.hand_acceleration_cap_rad_s2=args.hand_accel_cap
wbc.max_qp_constraint_violation=args.wbc_constraint_tolerance
if args.training_elbow_wbc_weight is not None:
    wbc.weights[m.jnt_dofadr[elbow_jid]]=args.training_elbow_wbc_weight
if args.training_elbow_wbc_kp is not None:
    wbc.kp[m.jnt_dofadr[elbow_jid]]=args.training_elbow_wbc_kp
wbc.flex_tip_normal_goal_n=args.flex_tip_force_n
wbc.hem_contact_normal_goal_n=args.hem_force_n
wbc.dressing_contact_normal_goal_n=args.a_pad_normal_goal_n
wbc.spool_contact_normal_goal_n=args.b_spool_normal_goal_n
spool_pinch_actuators=[i for i in range(m.nu) if m.actuator(i).name.startswith('g1_b/wuji_left/l_') and
                       any(token in m.actuator(i).name for token in ('THJ','FFJ'))]
wbc.support_palm_force_goal_n=args.a_left_support_goal_n
initial_b_left_wrist_world=d.xpos[m.body('g1_b/wuji_left/l_wrist').id].copy()
if args.a_left_follow_forearm or args.a_left_palm_contact_servo or args.a_left_palm_preload_mm:
    forearm_bid=m.body('training_forearm_dynamic').id
    support_wrist_bid=m.body('g1_a/wuji_left/l_wrist').id
    initial_support_local=d.xmat[forearm_bid].reshape(3,3).T @ (
        d.xpos[support_wrist_bid]-d.xpos[forearm_bid])
    initial_support_world=d.xpos[support_wrist_bid].copy()
    support_target_world=initial_support_world.copy()
if args.a_left_palm_contact_servo or args.a_left_palm_preload_mm:
    support_forearm_geom=m.geom('manikin_forearm').id
    support_palm_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==support_wrist_bid and m.geom_contype[g]]
    if not support_palm_geoms:raise RuntimeError('No collidable A-left palm geometry')
if args.a_left_palm_preload_mm:
    preload_candidates=[]
    for geom in support_palm_geoms:
        points=np.zeros(6)
        gap=float(mujoco.mj_geomDistance(m,d,geom,support_forearm_geom,1.,points))
        preload_candidates.append((gap,points))
    preload_gap,preload_points=min(preload_candidates,key=lambda item:item[0])
    preload_direction=(preload_points[3:]-preload_points[:3])*(1. if preload_gap>=0 else -1.)
    preload_direction/=max(float(np.linalg.norm(preload_direction)),1e-9)
wbc.flex_measured_force_prior_weight=args.flex_measured_force_prior_weight
if args.stabilize_other_digits or args.stabilize_all_digits:
    wbc.hand_position_mode=True
    for i in wbc.hand_indices:
        wbc.hand_commands[i]=d.qpos[wbc.act_qadr[i]]
    if args.b_middle_retract_mode:
        token='MFJ0' if args.b_middle_retract_mode=='flex' else 'MFJ1'
        actuator=m.actuator(f'g1_b/wuji_left/l_{token}').id
        amount=-.05 if args.b_middle_retract_mode=='flex' else .05
        wbc.hand_commands[actuator]=np.clip(wbc.hand_commands[actuator]+amount,*m.actuator_ctrlrange[actuator])
    if args.stabilize_other_digits:
        wbc.force_control_indices=(set() if args.b_right_position_control else
            {int(i) for i in wbc.hand_indices
             if m.actuator(int(i)).name.startswith('g1_b/wuji_right/r_') and
             any(token in m.actuator(int(i)).name for token in (
                 ('THJ','FFJ') if args.b_right_force_digits=='both' else
                 ('THJ',) if args.b_right_force_digits=='thumb' else ('FFJ',)))})
        if args.a_pad_force_control:
            wbc.force_control_indices.update(int(i) for i in wbc.hand_indices
                if m.actuator(int(i)).name.startswith('g1_a/wuji_right/r_') and
                any(token in m.actuator(int(i)).name for token in ('THJ','FFJ')))
        if args.b_spool_force_control:
            wbc.force_control_indices.update(int(i) for i in wbc.hand_indices
                if m.actuator(int(i)).name.startswith('g1_b/wuji_left/l_') and
                any(token in m.actuator(int(i)).name for token in ('THJ','FFJ')))
    if args.pinch_closure_mm:
        if not (args.stabilize_all_digits or args.b_right_position_control):
            parser.error('--pinch-closure-mm requires --stabilize-all-digits or --b-right-position-control')
        thumb_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==m.body(tip_names[0]).id]
        index_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==m.body(tip_names[1]).id]
        pinch_actuators=[int(i) for i in wbc.hand_indices
                         if m.actuator(int(i)).name.startswith('g1_b/wuji_right/r_') and
                         any(token in m.actuator(int(i)).name for token in ('THJ','FFJ'))]
        original_q=d.qpos.copy()
        def surface_gap():
            mujoco.mj_forward(m,d)
            return min(float(mujoco.mj_geomDistance(m,d,a,b,.1,np.zeros(6)))
                       for a in thumb_geoms for b in index_geoms)
        gap0=surface_gap()
        gradient=[]
        for i in pinch_actuators:
            adr=wbc.act_qadr[i]
            d.qpos[adr]+=1e-4
            gradient.append((surface_gap()-gap0)/1e-4)
            d.qpos[adr]=original_q[adr]
        gradient=np.asarray(gradient)
        delta=-args.pinch_closure_mm/1000*gradient/max(float(gradient@gradient),1e-12)
        for i,move in zip(pinch_actuators,delta):
            low,high=m.actuator_ctrlrange[i]
            wbc.hand_commands[i]=np.clip(wbc.hand_commands[i]+np.clip(move,-.15,.15),low,high)
        d.qpos[:]=original_q;mujoco.mj_forward(m,d)
if args.wrist_weight is not None:
    wbc.weights[wrist_vadr]=args.wrist_weight
interval = round(.02/m.opt.timestep)
fixture_vertices=tuple(range(m.nflexvert-2,m.nflexvert)) if args.extended_tab else (64,65)
fixture_ids=([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,f'bandage_tab_fixture_{i}') for i in fixture_vertices]
             if args.releasable_fixture else [])
duration = args.duration_s
initial_edge = d.flexvert_xpos[list(fixture_vertices)].mean(axis=0).copy()
grasp_bodies = ['g1_b/wuji_right/r_wrist', *(['bandage_reinforced_hem'] if args.reinforced_hem else [])]
grasp_origins = {name: d.body(name).xpos.copy() for name in grasp_bodies}
edge_forearm_bid=m.body('training_forearm_dynamic').id
initial_edge_forearm_local=(d.xmat[edge_forearm_bid].reshape(3,3).T @
    (initial_edge-d.xpos[edge_forearm_bid]))
initial_row28 = d.flexvert_xpos[56:58].mean(axis=0).copy()
screen = json.loads((OUT/'joint_b_flex_free_edge_reach_screen_all_digits_staged_tab1.5_indexrow28_offset20p0_cap2p5.json').read_text())
tip_target = np.array(screen['results'][0]['target_m'])
if args.track_edge:
    wbc.tip_position_weight = 1000.
material_targets={}
initial_material_offsets={}
def material_point(data,verts,weights):
    return data.xpos[hem_body_id] if verts is None else weights@data.flexvert_xpos[verts]
if args.material_tip_follow and not args.start_at_goal:
    parser.error('Material-point fingertip following currently requires --start-at-goal')
if args.material_tip_follow:
    wbc.tip_position_weight=args.tip_weight
if args.pinchtab and args.start_at_goal:
    for digit,tip_name in [('thumb',tip_names[0]),('index',tip_names[1])]:
        contacts=[]
        for ci,c in enumerate(d.contact):
            hem_contact=(args.reinforced_hem and any(g>=0 and
                m.geom(int(g)).name=='bandage_reinforced_hem_grasp' for g in c.geom))
            if not (hem_contact or any(f>=0 for f in c.flex)) or c.dist>=.003:continue
            bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
            candidates=([b for b in bodies if b.startswith(f'g1_b/wuji_right/r_{digit}_')]
                        if args.b_right_digit_material_anchor else [b for b in bodies if b==tip_name])
            if not candidates:continue
            if hem_contact:
                verts,weights,point=None,None,d.xpos[hem_body_id].copy()
            else:
                side=0 if c.flex[0]>=0 else 1
                _,_,verts,weights=flex_contact_jacobian(m,d,c,side)
                point=weights@d.flexvert_xpos[verts]
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
            contacts.append((max(0.,float(wrench[0])),float(c.dist),candidates[0],verts,weights,point))
        if not contacts:
            raise RuntimeError(f'No initial {digit} flex contact for material target')
        _,_,anchor_name,verts,weights,point=(max(contacts,key=lambda x:x[0])
            if args.b_right_digit_material_anchor else min(contacts,key=lambda x:x[1]))
        offset=d.xpos[m.body(anchor_name).id]-point
        initial_material_offsets[anchor_name]=offset.copy()
        target_offset=offset*max(0.,1.-args.tip_closure_mm/1000/max(np.linalg.norm(offset),1e-9))
        material_targets[anchor_name]=(verts,weights,target_offset)
if args.forearm_grasp_hold_weight:
    forearm_rotation = d.xmat[edge_forearm_bid].reshape(3, 3)
    grasp_local = {name: forearm_rotation.T @ (d.xpos[m.body(name).id] - d.xpos[edge_forearm_bid])
                   for name in material_targets}
    if len(grasp_local) != 2:
        raise ValueError('Forearm grasp hold requires two distinct digit anchors')
    wbc.tip_position_weight = args.forearm_grasp_hold_weight
    if args.relative_grasp_jacobian:
        wbc.tip_reference_body_ids = dict.fromkeys(grasp_local, edge_forearm_bid)
max_pen = 0.0
max_a_pen=0.0
worst_b_contact=None
worst_a_contact=None
worst_b_qpos=None
max_tip_error = 0.0
feet_count = 0
flex_contact_count = 0
opposed_contact_count = 0
a_pad_tip_pair_count=0
a_pad_segment_pair_count=0
a_pad_opposed_segment_count=0
a_left_forearm_support_count=0
a_left_forearm_palm_contact_count=0
a_left_forearm_palm_loaded_count=0
a_left_forearm_palm_normal_impulse_ns=0.
a_left_forearm_max_palm_contact_gap_s=0.
a_left_forearm_current_palm_contact_gap_s=0.
a_left_forearm_contact_body_counts={}
a_left_forearm_normal_impulse_ns=0.
a_left_forearm_peak_single_contact_n=0.
b_left_spool_pair_count=0
b_left_spool_opposed_count=0
b_left_spool_opposed_loaded_count=0
b_left_spool_normal_impulse_ns={'thumb':0.,'index':0.}
spool_world_contact_after_release_count=0
max_spool_displacement=0.
max_post_stage_release_spool_displacement=0.
max_post_stage_release_spool_drop=0.
max_post_stage_release_spool_wrist_relative_drift=0.
stage_release_spool_position=None
stage_release_spool_wrist_relative=None
max_pad_displacement=0.
stage_release_pad_position=None
max_post_stage_release_pad_displacement=0.
pad_stage_contact_count=0
pad_world_contact_after_release_count=0
pad_world_contact_geoms_after_release=set()
tip_normal_impulse_ns={'thumb':0.,'index':0.}
tip_loaded_ge0p5n_steps=0
digit_loaded_ge0p5n_steps=0
digit_normal_impulse_ns={'thumb':0.,'index':0.}
tip_peak_single_contact_normal_force_n={'thumb':0.,'index':0.}
tip_shear_impulse_ns={'thumb':0.,'index':0.}
edge_drift = 0.0
edge_forearm_relative_drift=0.
max_hand_joint_speed=0.
post_release_offsets=None
loaded_pinch_continuous_s=0.
actual_fixture_release_s=None
post_release_edge=None
max_post_release_material_slip=0.
max_post_release_edge_drift=0.
solves = 0
error = None
trace = []
last_palm_gap=None
previous_palm_gap=None
scratch=mujoco.MjData(m) if (args.one_step_hand_search or args.b_spool_one_step_search) else None

def ref_at(u):
    scaled = np.clip(u, 0, 1)*(len(waypoints)-1)
    i = min(len(waypoints)-2, int(scaled))
    t = scaled-i
    return waypoints[i]*(1-t)+waypoints[i+1]*t

try:
    for k in range(round(duration/.02)):
        progress = (k+1)/round(duration/.02)
        smooth = 1. if args.start_at_goal else progress*progress*(3-2*progress)
        if not args.extended_tab:
            wbc.qref[:] = ref_at(smooth)
        if args.b_left_lift_mm:
            lift_progress=np.clip((d.time-.1)/.2,0.,1.)
            lift_progress=lift_progress*lift_progress*(3.-2.*lift_progress)
            wbc.palm_world_position_targets['g1_b/wuji_left/l_wrist']=(
                initial_b_left_wrist_world+np.array([0.,0.,args.b_left_lift_mm/1000*lift_progress]))
        if args.a_left_palm_preload_mm:
            preload_progress=np.clip((d.time-.1)/.08,0.,1.)
            preload_progress=preload_progress*preload_progress*(3.-2.*preload_progress)
            wbc.palm_world_position_targets['g1_a/wuji_left/l_wrist']=(
                initial_support_world+preload_direction*args.a_left_palm_preload_mm/1000*preload_progress)
        if args.a_left_follow_forearm:
            desired=d.xpos[forearm_bid]+d.xmat[forearm_bid].reshape(3,3)@initial_support_local
            delta=desired-initial_support_world
            distance=float(np.linalg.norm(delta))
            if distance>args.a_left_follow_max_offset_mm/1000:
                delta*=args.a_left_follow_max_offset_mm/1000/distance
            next_delta=initial_support_world+delta-support_target_world
            next_distance=float(np.linalg.norm(next_delta))
            if next_distance>args.a_left_follow_max_step_mm/1000:
                next_delta*=args.a_left_follow_max_step_mm/1000/next_distance
            support_target_world+=next_delta
            wbc.palm_world_position_targets['g1_a/wuji_left/l_wrist']=support_target_world.copy()
        if args.a_left_palm_contact_servo:
            candidate_gaps=[]
            for geom in support_palm_geoms:
                points=np.zeros(6)
                gap=float(mujoco.mj_geomDistance(m,d,geom,support_forearm_geom,1.,points))
                candidate_gaps.append((gap,points))
            palm_gap,points=min(candidate_gaps,key=lambda item:item[0])
            last_palm_gap=palm_gap
            projected_gap=(palm_gap if previous_palm_gap is None else
                           palm_gap+max(0.,palm_gap-previous_palm_gap)*
                           args.a_left_servo_lookahead_s/.02)
            previous_palm_gap=palm_gap
            if projected_gap>.0002:
                direction=(points[3:]-points[:3])*(1. if palm_gap>=0 else -1.)
                length=float(np.linalg.norm(direction))
                if length>1e-9:
                    step=min(args.a_left_servo_gain*projected_gap,args.a_left_servo_max_step_mm/1000)
                    proposed=support_target_world+step*direction/length
                    offset=proposed-initial_support_world
                    norm=float(np.linalg.norm(offset))
                    if norm>args.a_left_servo_max_offset_mm/1000:
                        offset*=args.a_left_servo_max_offset_mm/1000/norm
                    support_target_world=initial_support_world+offset
            wbc.palm_world_position_targets['g1_a/wuji_left/l_wrist']=support_target_world.copy()
        if args.track_edge:
            wbc.tip_position_targets = {
                tip_names[0]: tip_target[0] + d.flexvert_xpos[64:66].mean(axis=0)-initial_edge,
                tip_names[1]: tip_target[1] + d.flexvert_xpos[56:58].mean(axis=0)-initial_row28,
            }
        if args.material_tip_follow:
            wbc.tip_position_targets={name:material_point(d,verts,weights)+offset
                                      for name,(verts,weights,offset) in material_targets.items()}
        if args.forearm_grasp_hold_weight:
            rotation = d.xmat[edge_forearm_bid].reshape(3, 3)
            wbc.tip_position_targets = {name: d.xpos[edge_forearm_bid] + rotation @ local
                                        for name, local in grasp_local.items()}
        ctrl, meta, _ = wbc.solve(d)
        d.ctrl[:] = ctrl
        if args.pinch_bias_preset and (args.pinch_bias_until_s is None or d.time<args.pinch_bias_until_s):
            bias={'balanced_a':{'THJ1':.1,'FFJ1':.1,'FFJ3':.1},
                  'balanced_b':{'THJ1':.05,'THJ0':.05,'THJ3':-.05,'FFJ3':.05}}[args.pinch_bias_preset]
            for token,amount in bias.items():
                actuator=next(i for i in pinch_actuators if m.actuator(i).name.endswith(token))
                d.ctrl[actuator]=np.clip(d.ctrl[actuator]+amount,*m.actuator_ctrlrange[actuator])
        mujoco.mj_forward(m,d)
        local_tip_force_correction_norm=0.
        if args.measured_tip_force_n is not None:
            def measured_tip_forces():
                forces=np.zeros(2)
                for ci,c in enumerate(d.contact):
                    if not bandage_contact(c):continue
                    bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
                    for digit,body in enumerate(tip_names):
                        if body in bodies:
                            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                            forces[digit]+=max(0.,float(wrench[0]))
                return forces
            base=measured_tip_forces()
            columns=[]
            for actuator in pinch_actuators:
                old=float(d.ctrl[actuator])
                d.ctrl[actuator]=np.clip(old+.01,*m.actuator_ctrlrange[actuator])
                mujoco.mj_forward(m,d)
                columns.append((measured_tip_forces()-base)/max(float(d.ctrl[actuator]-old),1e-9))
                d.ctrl[actuator]=old
            mujoco.mj_forward(m,d)
            jac=np.asarray(columns).T
            if np.all(np.isfinite(jac)):
                delta=jac.T@np.linalg.solve(jac@jac.T+.01*np.eye(2),
                    np.maximum(0.,args.measured_tip_force_n-base))
                delta=np.clip(delta,-.05,.05)
                for actuator,adjustment in zip(pinch_actuators,delta):
                    d.ctrl[actuator]=np.clip(d.ctrl[actuator]+adjustment,*m.actuator_ctrlrange[actuator])
                local_tip_force_correction_norm=float(np.linalg.norm(delta))
                mujoco.mj_forward(m,d)
        pre_step_wrist_actual_acceleration=float(d.qacc[wrist_vadr])
        pre_step_wrist_actual_constraint_force=float(d.qfrc_constraint[wrist_vadr])
        wrist_control_correction=0.
        if args.local_physics_correction:
            base_ctrl=float(d.ctrl[wrist_actuator])
            d.ctrl[wrist_actuator]=base_ctrl+.1
            mujoco.mj_forward(m,d)
            sensitivity=(float(d.qacc[wrist_vadr])-pre_step_wrist_actual_acceleration)/.1
            if np.isfinite(sensitivity) and sensitivity>1.:
                wrist_control_correction=float(np.clip(
                    (wbc.last_predicted_acceleration[wrist_vadr]-pre_step_wrist_actual_acceleration)/sensitivity,
                    -2.,2.))
            low,high=m.actuator_ctrlrange[wrist_actuator] if m.actuator_ctrllimited[wrist_actuator] else (-1e4,1e4)
            d.ctrl[wrist_actuator]=np.clip(base_ctrl+wrist_control_correction,low,high)
            wrist_control_correction=float(d.ctrl[wrist_actuator]-base_ctrl)
            mujoco.mj_forward(m,d)
        selected_hand_candidate='baseline'
        predicted_hand_score=None
        if args.one_step_hand_search and (args.one_step_hand_search_until_s is None or d.time<args.one_step_hand_search_until_s):
            candidate_biases=[('baseline',{})]
            candidate_biases += [(f'{m.actuator(i).name}_{sign:+g}',{i:sign})
                                 for i in pinch_actuators for sign in (-.05,.05)]
            for label,amounts in (('balanced_a',{'THJ1':.1,'FFJ1':.1,'FFJ3':.1}),
                                  ('balanced_b',{'THJ1':.05,'THJ0':.05,'THJ3':-.05,'FFJ3':.05})):
                candidate_biases.append((label,{next(i for i in pinch_actuators if m.actuator(i).name.endswith(token)):amount
                                                for token,amount in amounts.items()}))
            scores=[]
            base_ctrl=d.ctrl.copy()
            for label,bias in candidate_biases:
                mujoco.mj_copyData(scratch,m,d)
                stage_contact_backup=(int(m.geom_contype[stage_geom_id]),int(m.geom_conaffinity[stage_geom_id]))
                tray_contact_backup=(int(m.geom_contype[tray_geom_id]),int(m.geom_conaffinity[tray_geom_id]))
                for i,amount in bias.items():
                    scratch.ctrl[i]=np.clip(scratch.ctrl[i]+amount,*m.actuator_ctrlrange[i])
                candidate_max_penetration=0.
                candidate_force_deficit_sum=0.
                candidate_force_steps=0
                for _ in range(round(args.hand_search_horizon_s/m.opt.timestep)):
                    if args.stage_release_s is not None and scratch.time>=args.stage_release_s:
                        m.geom_contype[stage_geom_id]=0;m.geom_conaffinity[stage_geom_id]=0
                        if args.release_equipment_tray_with_stage:
                            m.geom_contype[tray_geom_id]=0;m.geom_conaffinity[tray_geom_id]=0
                    if (args.fixture_release_s is not None and scratch.time>=args.fixture_release_s and
                            (not args.gate_fixture_release_on_loaded_pinch or
                             not any(d.eq_active[i] for i in fixture_ids))):
                        scratch.eq_active[fixture_ids]=0
                    mujoco.mj_step(m,scratch)
                    candidate_max_penetration=max(candidate_max_penetration,
                        max((max(0.,-float(c.dist)) for c in scratch.contact
                             if any(g>=0 and m.body(m.geom_bodyid[g]).name.startswith(('g1_a/','g1_b/')) for g in c.geom)
                             and not any(g>=0 and m.geom(g).name=='floor' for g in c.geom)),default=0.))
                    if args.hand_search_trajectory_force_weight:
                        predicted_forces={'thumb':0.,'index':0.}
                        for ci,c in enumerate(scratch.contact):
                            if not bandage_contact(c):continue
                            bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
                            for digit,tip in zip(('thumb','index'),tip_names):
                                contacted=(any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies)
                                           if args.b_right_digit_material_anchor else tip in bodies)
                                if contacted:
                                    wrench=np.zeros(6);mujoco.mj_contactForce(m,scratch,ci,wrench)
                                    predicted_forces[digit]+=max(0.,float(wrench[0]))
                        candidate_force_deficit_sum+=sum(max(0.,.5-force) for force in predicted_forces.values())
                        candidate_force_steps+=1
                slip=max(float(np.linalg.norm(scratch.xpos[m.body(name).id]-
                    material_point(scratch,verts,weights)-initial_material_offsets[name]))
                    for name,(verts,weights,_) in material_targets.items())
                drift=float(np.linalg.norm(scratch.flexvert_xpos[list(fixture_vertices)].mean(axis=0)-initial_edge))
                tip_forces={'thumb':0.,'index':0.}
                for ci,c in enumerate(scratch.contact):
                    if not bandage_contact(c):
                        continue
                    bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
                    for digit in ('thumb','index'):
                        if any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies):
                            wrench=np.zeros(6)
                            mujoco.mj_contactForce(m,scratch,ci,wrench)
                            tip_forces[digit]+=max(0.,float(wrench[0]))
                force_deficit=sum(max(0.,.5-force) for force in tip_forces.values())
                score=(slip+args.hand_search_edge_weight*drift+
                       args.hand_search_penetration_weight*max(0.,candidate_max_penetration-.003)+
                       args.hand_search_force_weight*force_deficit+
                       args.hand_search_trajectory_force_weight*candidate_force_deficit_sum/max(1,candidate_force_steps))
                scores.append((score,label,bias))
                m.geom_contype[stage_geom_id],m.geom_conaffinity[stage_geom_id]=stage_contact_backup
                m.geom_contype[tray_geom_id],m.geom_conaffinity[tray_geom_id]=tray_contact_backup
            predicted_hand_score,selected_hand_candidate,chosen_bias=min(scores,key=lambda x:x[0])
            d.ctrl[:]=base_ctrl
            for i,amount in chosen_bias.items():
                d.ctrl[i]=np.clip(d.ctrl[i]+amount,*m.actuator_ctrlrange[i])
            mujoco.mj_forward(m,d)
        selected_spool_candidate='baseline'
        predicted_spool_score=None
        if args.b_spool_one_step_search:
            spool_bias_magnitude=(args.b_spool_search_post_release_bias_rad
                                  if args.b_spool_search_post_release_bias_rad is not None and
                                     d.time>=(args.b_spool_search_switch_s if args.b_spool_search_switch_s is not None
                                              else args.stage_release_s) else args.b_spool_search_bias_rad)
            candidate_biases=[('baseline',{})]
            candidate_biases += [(f'{m.actuator(i).name}_{sign:+g}',{i:sign})
                                 for i in spool_pinch_actuators for sign in (-spool_bias_magnitude,spool_bias_magnitude)]
            scores=[]
            base_ctrl=d.ctrl.copy()
            for label,bias in candidate_biases:
                mujoco.mj_copyData(scratch,m,d)
                stage_contact_backup=(int(m.geom_contype[stage_geom_id]),int(m.geom_conaffinity[stage_geom_id]))
                tray_contact_backup=(int(m.geom_contype[tray_geom_id]),int(m.geom_conaffinity[tray_geom_id]))
                for i,amount in bias.items():
                    scratch.ctrl[i]=np.clip(scratch.ctrl[i]+amount,*m.actuator_ctrlrange[i])
                for _ in range(round(args.b_spool_search_horizon_s/m.opt.timestep)):
                    if args.stage_release_s is not None and scratch.time>=args.stage_release_s:
                        m.geom_contype[stage_geom_id]=0;m.geom_conaffinity[stage_geom_id]=0
                        if args.release_equipment_tray_with_stage:
                            m.geom_contype[tray_geom_id]=0;m.geom_conaffinity[tray_geom_id]=0
                    if args.fixture_release_s is not None and scratch.time>=args.fixture_release_s:
                        scratch.eq_active[fixture_ids]=0
                    mujoco.mj_step(m,scratch)
                relative=(scratch.xpos[spool_body_id]-scratch.xpos[b_left_wrist_body_id])
                target_relative=(stage_release_spool_wrist_relative if stage_release_spool_wrist_relative is not None
                                 else initial_spool_wrist_relative)
                slip=float(np.linalg.norm(relative-target_relative))
                forces={'thumb':0.,'index':0.}
                support_palm_force=0.
                candidate_max_pen=0.
                for ci,c in enumerate(scratch.contact):
                    bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom if g>=0]
                    names=[m.geom(g).name for g in c.geom if g>=0]
                    if c.dist<0 and 'floor' not in names and any(b.startswith(('g1_a/','g1_b/')) for b in bodies):
                        candidate_max_pen=max(candidate_max_pen,-float(c.dist))
                    if ('training_forearm_dynamic' in bodies and 'g1_a/wuji_left/l_wrist' in bodies
                            and c.dist<.003):
                        wrench=np.zeros(6);mujoco.mj_contactForce(m,scratch,ci,wrench)
                        support_palm_force+=max(0.,float(wrench[0]))
                    if 'bandage_spool' not in bodies or c.dist>=.003:continue
                    for digit in forces:
                        if any(b.startswith(f'g1_b/wuji_left/l_{digit}_') for b in bodies):
                            wrench=np.zeros(6);mujoco.mj_contactForce(m,scratch,ci,wrench)
                            forces[digit]+=max(0.,float(wrench[0]))
                score=(slip+.002*sum(max(0.,.5-force) for force in forces.values())+
                       args.b_spool_search_support_weight*max(0.,.5-support_palm_force)+
                       10.*max(0.,candidate_max_pen-.003))
                scores.append((score,label,bias))
                m.geom_contype[stage_geom_id],m.geom_conaffinity[stage_geom_id]=stage_contact_backup
                m.geom_contype[tray_geom_id],m.geom_conaffinity[tray_geom_id]=tray_contact_backup
            predicted_spool_score,selected_spool_candidate,chosen_bias=min(scores,key=lambda x:x[0])
            d.ctrl[:]=base_ctrl
            for i,amount in chosen_bias.items():
                d.ctrl[i]=np.clip(d.ctrl[i]+amount,*m.actuator_ctrlrange[i])
            mujoco.mj_forward(m,d)
        solves += 1
        interval_opposed = 0
        interval_a_left_support = 0
        interval_a_left_palm = 0
        interval_max_pen = 0.
        interval_max_a_pen=0.
        interval_max_a_right_hand_speed=0.
        first_step_actual_wrist_acceleration=None
        for substep in range(interval):
            if args.stage_release_s is not None and d.time>=args.stage_release_s and stage_release_pad_position is None:
                stage_release_pad_position=d.xpos[pad_body_id].copy()
                stage_release_spool_position=d.xpos[spool_body_id].copy()
                stage_release_spool_wrist_relative=(d.xpos[spool_body_id]-d.xpos[b_left_wrist_body_id]).copy()
                m.geom_contype[stage_geom_id]=0;m.geom_conaffinity[stage_geom_id]=0
                if args.release_equipment_tray_with_stage:
                    m.geom_contype[tray_geom_id]=0;m.geom_conaffinity[tray_geom_id]=0
            if args.gate_fixture_release_on_loaded_pinch and post_release_offsets is None:
                release_forces={'thumb':0.,'index':0.}
                for ci,c in enumerate(d.contact):
                    if not bandage_contact(c):continue
                    bodies=[m.body(m.geom_bodyid[int(g)]).name for g in c.geom if g>=0]
                    for digit,tip in zip(('thumb','index'),tip_names):
                        contacted=(any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies)
                                   if args.b_right_digit_material_anchor else tip in bodies)
                        if contacted:
                            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                            release_forces[digit]+=max(0.,float(wrench[0]))
                loaded_pinch_continuous_s=(loaded_pinch_continuous_s+m.opt.timestep
                    if min(release_forces.values())>=.5 else 0.)
            release_ready=(not args.gate_fixture_release_on_loaded_pinch or
                           loaded_pinch_continuous_s>=.05-1e-9)
            if (args.gate_fixture_release_on_loaded_pinch and
                    d.time>=args.fixture_release_s and not release_ready):
                raise RuntimeError('Loaded-pinch release interlock: scheduled fixture release blocked; '
                    f'thumb={release_forces["thumb"]:.3f} N index={release_forces["index"]:.3f} N, '
                    f'continuous={loaded_pinch_continuous_s:.3f} s')
            if args.fixture_release_s is not None and d.time>=args.fixture_release_s and release_ready:
                if post_release_offsets is None:
                    actual_fixture_release_s=float(d.time)
                    post_release_offsets={name:d.xpos[m.body(name).id]-material_point(d,verts,weights)
                                          for name,(verts,weights,_) in material_targets.items()}
                    post_release_edge=d.flexvert_xpos[list(fixture_vertices)].mean(axis=0).copy()
                d.eq_active[fixture_ids]=0
            mujoco.mj_step(m, d)
            if stage_release_pad_position is not None:
                max_post_stage_release_pad_displacement=max(max_post_stage_release_pad_displacement,
                    float(np.linalg.norm(d.xpos[pad_body_id]-stage_release_pad_position)))
                if stage_release_spool_position is not None:
                    max_post_stage_release_spool_displacement=max(max_post_stage_release_spool_displacement,
                        float(np.linalg.norm(d.xpos[spool_body_id]-stage_release_spool_position)))
                    max_post_stage_release_spool_drop=max(max_post_stage_release_spool_drop,
                        float(stage_release_spool_position[2]-d.xpos[spool_body_id,2]))
                    max_post_stage_release_spool_wrist_relative_drift=max(
                        max_post_stage_release_spool_wrist_relative_drift,
                        float(np.linalg.norm(d.xpos[spool_body_id]-d.xpos[b_left_wrist_body_id]-
                                             stage_release_spool_wrist_relative)))
            max_hand_joint_speed=max(max_hand_joint_speed,float(np.max(np.abs(d.qvel[wbc.act_vadr[wbc.hand_indices]]))))
            interval_max_a_right_hand_speed=max(interval_max_a_right_hand_speed,
                max((abs(float(d.qvel[wbc.act_vadr[i]])) for i in wbc.hand_indices
                     if m.actuator(int(i)).name.startswith('g1_a/wuji_right/')),default=0.))
            if post_release_offsets is not None:
                max_post_release_material_slip=max(max_post_release_material_slip,
                    max(float(np.linalg.norm(d.xpos[m.body(name).id]-material_point(d,verts,weights)-post_release_offsets[name]))
                        for name,(verts,weights,_) in material_targets.items()))
                max_post_release_edge_drift=max(max_post_release_edge_drift,float(np.linalg.norm(
                    d.flexvert_xpos[list(fixture_vertices)].mean(axis=0)-post_release_edge)))
            if substep==0:
                first_step_actual_wrist_acceleration=float(d.qacc[wrist_vadr])
            if not np.all(np.isfinite(d.qpos)):
                raise RuntimeError('Nonfinite state')
            feet = {'a': set(), 'b': set()}
            tip_step_normal_force_n={'thumb':0.,'index':0.}
            digit_step_normal_force_n={'thumb':0.,'index':0.}
            digit_normals={'thumb':[],'index':[]}
            has_forearm_flex = False
            tip_normals={'thumb':[],'index':[]}
            a_pad_tips={'thumb':False,'index':False}
            a_pad_segment_normals={'thumb':[],'index':[]}
            a_left_support_here=False
            a_left_palm_here=False
            a_left_palm_force_here=0.
            b_left_spool_normals={'thumb':[],'index':[]}
            b_left_spool_forces={'thumb':0.,'index':0.}
            for ci,c in enumerate(d.contact):
                geoms = [int(g) for g in c.geom if g >= 0]
                names = [m.geom(g).name for g in geoms]
                bodies = [m.body(m.geom_bodyid[g]).name for g in geoms]
                flex = any(f >= 0 for f in c.flex)
                if flex and 'training_forearm_dynamic' in bodies:
                    has_forearm_flex = True
                if 'training_forearm_dynamic' in bodies and any(name.startswith('g1_a/wuji_left/') for name in bodies) and c.dist<.003:
                    a_left_support_here=True
                    for body in bodies:
                        if body.startswith('g1_a/wuji_left/'):
                            a_left_forearm_contact_body_counts[body]=a_left_forearm_contact_body_counts.get(body,0)+1
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                    normal=max(0.,float(wrench[0]))
                    a_left_forearm_normal_impulse_ns+=normal*m.opt.timestep
                    a_left_forearm_peak_single_contact_n=max(a_left_forearm_peak_single_contact_n,normal)
                    if 'g1_a/wuji_left/l_wrist' in bodies:
                        a_left_palm_here=True
                        a_left_palm_force_here+=normal
                        a_left_forearm_palm_normal_impulse_ns+=normal*m.opt.timestep
                if 'bandage_spool' in bodies and c.dist<.003:
                    for digit in b_left_spool_normals:
                        if any(name.startswith(f'g1_b/wuji_left/l_{digit}_') for name in bodies):
                            sign=1. if c.geom1>=0 and m.geom(c.geom1).name=='bandage_handle_prototype' else -1.
                            b_left_spool_normals[digit].append(sign*c.frame[:3].copy())
                            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                            normal=max(0.,float(wrench[0]))
                            b_left_spool_forces[digit]+=normal
                            b_left_spool_normal_impulse_ns[digit]+=normal*m.opt.timestep
                    if stage_release_spool_position is not None and 'world' in bodies:
                        spool_world_contact_after_release_count+=1
                if 'dressing_pad' in bodies and c.dist<.003:
                    a_pad_tips['thumb'] |= 'g1_a/wuji_right/r_thumb_tip_sensor_frame' in bodies
                    a_pad_tips['index'] |= 'g1_a/wuji_right/r_index_finger_tip_sensor_frame' in bodies
                    for digit in a_pad_segment_normals:
                        if any(name.startswith(f'g1_a/wuji_right/r_{digit}_') for name in bodies):
                            sign=1. if c.geom1>=0 and m.geom(c.geom1).name=='dressing_pad_contact' else -1.
                            a_pad_segment_normals[digit].append(sign*c.frame[:3].copy())
                if 'dressing_stage' in names and 'dressing_pad' in bodies and c.dist<.003:
                    pad_stage_contact_count+=1
                if stage_release_pad_position is not None and 'dressing_pad' in bodies and 'world' in bodies and c.dist<.003:
                    pad_world_contact_after_release_count+=1
                    pad_world_contact_geoms_after_release.update(name for name in names if name!='dressing_pad_contact')
                if bandage_contact(c) and c.dist<.003:
                    for digit in ('thumb','index'):
                        if any(b.startswith(f'g1_b/wuji_right/r_{digit}_') for b in bodies):
                            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                            normal=max(0.,float(wrench[0]))
                            digit_step_normal_force_n[digit]+=normal
                            digit_normal_impulse_ns[digit]+=normal*m.opt.timestep
                            digit_normals[digit].append(c.frame[:3].copy())
                    for digit,body_name in (('thumb',tip_names[0]),('index',tip_names[1])):
                        if body_name in bodies:
                            tip_normals[digit].append(c.frame[:3].copy())
                            wrench=np.zeros(6)
                            mujoco.mj_contactForce(m,d,ci,wrench)
                            normal=max(0.,float(wrench[0]))
                            tip_step_normal_force_n[digit]+=normal
                            tip_normal_impulse_ns[digit]+=normal*m.opt.timestep
                            tip_shear_impulse_ns[digit]+=float(np.linalg.norm(wrench[1:3]))*m.opt.timestep
                            tip_peak_single_contact_normal_force_n[digit]=max(tip_peak_single_contact_normal_force_n[digit],normal)
                if c.dist < 0 and 'floor' not in names and any(n.startswith('g1_b/') for n in bodies):
                    if -float(c.dist)>max_pen:
                        worst_b_contact={'time_s':float(d.time),'distance_m':float(c.dist),
                                         'bodies':bodies,'geoms':names,'flex':list(map(int,c.flex))}
                        worst_b_qpos=d.qpos.copy()
                    max_pen = max(max_pen, -float(c.dist))
                    interval_max_pen=max(interval_max_pen,-float(c.dist))
                if c.dist < 0 and 'floor' not in names and any(n.startswith('g1_a/') for n in bodies):
                    if -float(c.dist)>max_a_pen:
                        worst_a_contact={'time_s':float(d.time),'distance_m':float(c.dist),
                                         'bodies':bodies,'geoms':names,'flex':list(map(int,c.flex))}
                    max_a_pen=max(max_a_pen,-float(c.dist))
                    interval_max_a_pen=max(interval_max_a_pen,-float(c.dist))
                if 'floor' in names:
                    for agent in ('a', 'b'):
                        for side in ('left', 'right'):
                            if any(n.startswith(f'g1_{agent}/{side}_official_foot_contact_') for n in names):
                                feet[agent].add(side)
            feet_count += all(feet[a] == {'left', 'right'} for a in ('a', 'b'))
            flex_contact_count += has_forearm_flex
            opposed_here=any(np.dot(a,b)<-.5 for a in tip_normals['thumb'] for b in tip_normals['index'])
            opposed_contact_count += opposed_here
            tip_loaded_ge0p5n_steps += (opposed_here and all(
                force>=.5 for force in tip_step_normal_force_n.values()))
            digit_opposed_here=any(float(a@b)<-.5 for a in digit_normals['thumb']
                                   for b in digit_normals['index'])
            digit_loaded_ge0p5n_steps += (digit_opposed_here and all(
                force>=.5 for force in digit_step_normal_force_n.values()))
            a_pad_tip_pair_count += all(a_pad_tips.values())
            a_pad_segment_pair_count += all(a_pad_segment_normals.values())
            a_pad_opposed_segment_count += any(float(a@b)<-.5 for a in a_pad_segment_normals['thumb']
                                                for b in a_pad_segment_normals['index'])
            a_left_forearm_support_count += a_left_support_here
            interval_a_left_support += a_left_support_here
            a_left_forearm_palm_contact_count += a_left_palm_here
            a_left_forearm_palm_loaded_count += a_left_palm_force_here>=.5
            interval_a_left_palm += a_left_palm_here
            a_left_forearm_current_palm_contact_gap_s=(
                0. if a_left_palm_here else a_left_forearm_current_palm_contact_gap_s+m.opt.timestep)
            a_left_forearm_max_palm_contact_gap_s=max(
                a_left_forearm_max_palm_contact_gap_s,a_left_forearm_current_palm_contact_gap_s)
            b_left_spool_pair_count += all(b_left_spool_normals.values())
            b_left_spool_opposed_count += any(float(a@b)<-.5 for a in b_left_spool_normals['thumb']
                                               for b in b_left_spool_normals['index'])
            b_left_spool_opposed_loaded_count += (b_left_spool_forces['thumb']>=.5 and
                                                  b_left_spool_forces['index']>=.5 and
                                                  any(float(a@b)<-.5 for a in b_left_spool_normals['thumb']
                                                      for b in b_left_spool_normals['index']))
            max_pad_displacement=max(max_pad_displacement,float(np.linalg.norm(d.xpos[pad_body_id]-initial_pad_position)))
            max_spool_displacement=max(max_spool_displacement,float(np.linalg.norm(d.xpos[spool_body_id]-initial_spool_position)))
            interval_opposed += opposed_here
            edge_drift = max(edge_drift, float(np.linalg.norm(d.flexvert_xpos[list(fixture_vertices)].mean(axis=0)-initial_edge)))
            edge_now=d.flexvert_xpos[list(fixture_vertices)].mean(axis=0)
            edge_local=d.xmat[edge_forearm_bid].reshape(3,3).T @ (edge_now-d.xpos[edge_forearm_bid])
            edge_forearm_relative_drift=max(edge_forearm_relative_drift,
                float(np.linalg.norm(edge_local-initial_edge_forearm_local)))
        tip_error = float(np.max(np.abs(d.qpos[qa]-wbc.qref[qa])))
        thumb_index_surface_gap=min(float(mujoco.mj_geomDistance(m,d,a,b,.2,np.zeros(6)))
                                    for a in thumb_tip_geoms for b in index_tip_geoms)
        tip_displacements=[float(np.linalg.norm(d.xpos[bid]-initial_tip_positions[i]))
                           for i,bid in enumerate(tip_ids)]
        material_relative_shifts={name:float(np.linalg.norm(
            d.xpos[m.body(name).id]-material_point(d,verts,weights)-initial_material_offsets[name]))
            for name,(verts,weights,_) in material_targets.items()}
        material_relative_shift_vectors={name:(d.xpos[m.body(name).id]-
            material_point(d,verts,weights)-initial_material_offsets[name]).tolist()
            for name,(verts,weights,_) in material_targets.items()}
        max_tip_error = max(max_tip_error, tip_error)
        trace.append({'time_s': float(d.time), 'path_progress': float(smooth),
                      'predicted_tip_contact_force_n':{name:float(wbc.last_flex_tip_predicted_force_n.get(tip_names[i],0.)) for i,name in enumerate(('thumb','index'))},
                      'local_tip_force_correction_norm':local_tip_force_correction_norm,
                      'one_step_hand_candidate':selected_hand_candidate,
                      'one_step_hand_predicted_score_m':predicted_hand_score,
                      'one_step_spool_candidate':selected_spool_candidate,
                      'one_step_spool_predicted_score_m':predicted_spool_score,
                      'max_joint_tracking_error_rad': tip_error,
                      'thumb_index_surface_gap_m':thumb_index_surface_gap,
                      'tip_displacement_m':tip_displacements,
                      'grasp_body_displacement_m':{name:(d.body(name).xpos-origin).tolist()
                                                   for name,origin in grasp_origins.items()},
                      'edge_displacement_m':(edge_now-initial_edge).tolist(),
                      'tip_to_material_relative_shift_m':material_relative_shifts,
                      'tip_to_material_relative_shift_vector_m':material_relative_shift_vectors,
                      'worst_joint_name':joint_names[int(np.argmax(np.abs(d.qpos[qa]-wbc.qref[qa])))],
                      'opposed_tip_contact_fraction':interval_opposed/interval,
                      'a_left_forearm_support_contact_fraction':interval_a_left_support/interval,
                      'a_left_forearm_palm_contact_fraction':interval_a_left_palm/interval,
                      'a_left_servo_measured_palm_gap_m':last_palm_gap if args.a_left_palm_contact_servo else None,
                      'a_left_servo_projected_palm_gap_m':projected_gap if args.a_left_palm_contact_servo else None,
                      'a_left_servo_target_offset_m':float(np.linalg.norm(support_target_world-initial_support_world)) if args.a_left_palm_contact_servo else None,
                      'training_elbow_angle_rad':float(d.qpos[m.joint('training_elbow_hinge').qposadr[0]]),
                      'max_b_nonfloor_penetration_m':interval_max_pen,
                      'max_a_nonfloor_penetration_m':interval_max_a_pen,
                      'max_a_right_hand_joint_speed_rad_s':interval_max_a_right_hand_speed,
                      'b_right_wrist_roll_rad':float(d.qpos[wrist_qadr]),
                      'b_right_wrist_roll_ref_rad':float(wbc.qref[wrist_qadr]),
                      'b_right_wrist_roll_velocity_rad_s':float(d.qvel[wrist_vadr]),
                      'b_right_wrist_roll_requested_torque_Nm':float(wbc.last_tau[wrist_actuator]),
                      'b_right_wrist_roll_predicted_acceleration_rad_s2':float(wbc.last_predicted_acceleration[wrist_vadr]),
                      'b_right_wrist_roll_pre_step_actual_acceleration_rad_s2':pre_step_wrist_actual_acceleration,
                      'b_right_wrist_roll_pre_step_actual_constraint_force_Nm':pre_step_wrist_actual_constraint_force,
                      'b_right_wrist_roll_predicted_contact_generalized_force_Nm':float(wbc.last_predicted_contact_generalized_force[wrist_vadr]),
                      'b_right_wrist_roll_first_step_actual_acceleration_rad_s2':first_step_actual_wrist_acceleration,
                      'b_right_wrist_roll_actual_actuator_force_Nm':float(d.actuator_force[wrist_actuator]),
                      'b_right_wrist_roll_local_control_correction':wrist_control_correction,
                      'b_right_wrist_roll_constraint_force_Nm':float(d.qfrc_constraint[wrist_vadr]),
                      'b_right_wrist_roll_bias_force_Nm':float(d.qfrc_bias[wrist_vadr]),
                      'b_right_wrist_roll_passive_force_Nm':float(d.qfrc_passive[wrist_vadr]),
                      'free_edge_drift_m': float(np.linalg.norm(d.flexvert_xpos[list(fixture_vertices)].mean(axis=0)-initial_edge)),
                      'contact_task_residual_m_s2': meta['contact_acceleration_task_residual_m_s2']})
except Exception as exc:
    error = f'{type(exc).__name__}: {exc}'
    np.savez_compressed(OUT/'joint_b_pinchtab_last_wbc_failure_state.npz',
        qpos=d.qpos,qvel=d.qvel,act=d.act,eq_active=d.eq_active,ctrl=d.ctrl,
        qref=wbc.qref,hand_commands=wbc.hand_commands,time_s=d.time)

failure_diagnostic=None
if error and hasattr(wbc,'last_numeric'):
    numeric=wbc.last_numeric
    gaps=numeric['acc_lo']-numeric['acc_hi']
    failure_diagnostic={'acceleration_bound_inversion_count':int(np.count_nonzero(gaps>1e-9)),
                        'max_acceleration_bound_inversion_rad_s2':float(max(0.,np.max(gaps))),
                        'actuator_bound_inversion_count':int(np.count_nonzero(numeric['lo']>numeric['hi']+1e-9)),
                        'contact_count':int(len(numeric['mu'])),
                        'fixed_hand_actuator_count':int(len(numeric['fixed_hand'])),
                        'inverted_dofs':[{'joint':m.joint(int(m.dof_jntid[i])).name,
                                          'velocity':float(d.qvel[i]),
                                          'acc_lo':float(numeric['acc_lo'][i]),
                                          'acc_hi':float(numeric['acc_hi'][i])}
                                         for i in np.flatnonzero(gaps>1e-9)]}

steps = max(1, round(float(d.time) / m.opt.timestep))
report = {'scope': 'Dynamic WBC tracking of a static path toward a flexible free tab; no force-closed grasp or completed wrap',
          'track_edge_tip_targets': args.track_edge,
          'pinchtab':args.pinchtab,
          'extended_tab':args.extended_tab,
          'start_at_goal':args.start_at_goal,
          'edge_damping':args.edge_damping,
          'flex_tip_force_goal_n':args.flex_tip_force_n,
          'b_right_force_digits':args.b_right_force_digits,
          'b_right_position_control':args.b_right_position_control,
          'b_right_loaded_pinch_seed_static_only':args.b_right_loaded_pinch_seed,
          'b_right_loaded_pinch_seed_rank':args.b_right_loaded_pinch_seed_rank,
          'b_right_digit_material_anchor':args.b_right_digit_material_anchor,
          'b_right_material_anchor_bodies':list(material_targets),
          'reinforced_hem_material_hypothesis':args.reinforced_hem,
          'forearm_anchored_bandage_hypothesis':args.forearm_anchored_bandage,
          'body_parented_bandage_hypothesis':args.body_parented_bandage,
          'body_parented_refaligned_bandage_hypothesis':args.body_parented_refaligned_bandage,
          'b_right_hem_seed_rank':args.b_right_hem_seed_rank if args.reinforced_hem else None,
          'hem_contact_normal_goal_n':args.hem_force_n,
          'hem_radius_mm':args.hem_radius_mm if args.reinforced_hem else None,
          'flex_contact_radius_mm':args.flex_contact_radius_mm,
          'b_right_initial_index_pip_offset_rad':args.b_right_initial_index_pip_offset_rad,
          'measured_tip_force_goal_n':args.measured_tip_force_n,
          'pinch_bias_preset':args.pinch_bias_preset,
          'pinch_bias_until_s':args.pinch_bias_until_s,
          'one_step_hand_search':args.one_step_hand_search,
          'one_step_hand_search_until_s':args.one_step_hand_search_until_s,
          'hand_search_horizon_s':args.hand_search_horizon_s if args.one_step_hand_search else None,
          'hand_search_edge_weight':args.hand_search_edge_weight if args.one_step_hand_search else None,
          'hand_search_force_weight':args.hand_search_force_weight if args.one_step_hand_search else None,
          'hand_search_trajectory_force_weight':args.hand_search_trajectory_force_weight if args.one_step_hand_search else None,
          'hand_search_penetration_weight':args.hand_search_penetration_weight if args.one_step_hand_search else None,
          'hand_acceleration_cap_rad_s2':args.hand_accel_cap,
          'wbc_constraint_tolerance':args.wbc_constraint_tolerance,
          'dressing_pad_contact_disabled_counterfactual':args.disable_dressing_pad_contact,
          'dressing_pad_solref_time_s':args.dressing_pad_solref,
          'a_pad_side_grasp_seed_static_only':args.a_pad_side_grasp_seed,
          'stage_release_s':args.stage_release_s,
          'a_pad_normal_goal_n':args.a_pad_normal_goal_n,
          'a_pad_force_control':args.a_pad_force_control,
          'release_equipment_tray_with_stage_counterfactual':args.release_equipment_tray_with_stage,
          'b_middle_retract_mode':args.b_middle_retract_mode,
          'b_spool_normal_goal_n':args.b_spool_normal_goal_n,
          'b_spool_force_control':args.b_spool_force_control,
          'b_left_lift_mm':args.b_left_lift_mm,
          'b_spool_one_step_search':args.b_spool_one_step_search,
          'b_spool_search_horizon_s':args.b_spool_search_horizon_s if args.b_spool_one_step_search else None,
          'b_spool_search_bias_rad':args.b_spool_search_bias_rad if args.b_spool_one_step_search else None,
          'b_spool_search_post_release_bias_rad':args.b_spool_search_post_release_bias_rad,
          'b_spool_search_switch_s':args.b_spool_search_switch_s,
          'b_spool_search_support_weight':args.b_spool_search_support_weight if args.b_spool_one_step_search else None,
          'a_left_support_goal_n':args.a_left_support_goal_n,
          'a_left_palm_position_weight':args.a_left_palm_weight,
          'a_left_follow_forearm':args.a_left_follow_forearm,
          'a_left_palm_contact_servo':args.a_left_palm_contact_servo,
          'a_left_servo_max_step_mm':args.a_left_servo_max_step_mm if args.a_left_palm_contact_servo else None,
          'a_left_servo_max_offset_mm':args.a_left_servo_max_offset_mm if args.a_left_palm_contact_servo else None,
          'a_left_servo_gain':args.a_left_servo_gain if args.a_left_palm_contact_servo else None,
          'a_left_servo_lookahead_s':args.a_left_servo_lookahead_s if args.a_left_palm_contact_servo else None,
          'a_left_palm_preload_mm':args.a_left_palm_preload_mm,
          'a_left_near_elbow_seed_static_only':args.a_left_near_elbow_seed,
          'a_left_palm_only_seed_static_only':args.a_left_palm_only_seed,
          'a_left_follow_max_offset_mm':args.a_left_follow_max_offset_mm if args.a_left_follow_forearm else None,
          'a_left_follow_max_step_mm':args.a_left_follow_max_step_mm if args.a_left_follow_forearm else None,
          'training_elbow_stiffness_nm_rad':float(m.jnt_stiffness[elbow_jid]),
          'training_elbow_damping_nms_rad':float(m.dof_damping[m.jnt_dofadr[elbow_jid]]),
          'training_elbow_wbc_weight':args.training_elbow_wbc_weight,
          'training_elbow_wbc_kp':args.training_elbow_wbc_kp,
          'training_elbow_range_rad':args.training_elbow_range_rad,
          'flex_sliding_friction':float(m.flex_friction[0,0]),
          'b_right_wrist_roll_acceleration_weight':args.wrist_weight,
          'local_physics_correction':args.local_physics_correction,
          'stabilize_other_digits':args.stabilize_other_digits,
          'stabilize_all_digits':args.stabilize_all_digits,
          'material_tip_follow':args.material_tip_follow,
          'tip_closure_mm':args.tip_closure_mm,
          'tip_position_weight':args.tip_weight if args.material_tip_follow else None,
          'flex_measured_force_prior_weight':args.flex_measured_force_prior_weight,
          'releasable_fixture':args.releasable_fixture,
          'fixture_release_s':args.fixture_release_s,
          'fixture_active_at_end':bool(all(d.eq_active[i] for i in fixture_ids)) if fixture_ids else None,
          'pinch_closure_command_mm':args.pinch_closure_mm,
          'b_right_palm_position_weight':args.b_right_palm_weight,
          'duration_s': float(d.time), 'planned_duration_s': duration, 'control_solves': solves,
          'error': error, 'two_robot_two_foot_fraction': feet_count/steps,
          'failure_diagnostic':failure_diagnostic,
          'flex_forearm_contact_fraction': flex_contact_count/steps,
          'opposed_b_tip_flex_contact_fraction':opposed_contact_count/steps,
          'a_pad_thumb_index_contact_fraction':a_pad_tip_pair_count/steps,
          'a_pad_thumb_index_segment_contact_fraction':a_pad_segment_pair_count/steps,
          'a_pad_opposed_segment_contact_fraction':a_pad_opposed_segment_count/steps,
          'a_left_forearm_support_contact_fraction':a_left_forearm_support_count/steps,
          'a_left_forearm_palm_contact_fraction':a_left_forearm_palm_contact_count/steps,
          'a_left_forearm_palm_loaded_ge0p5n_fraction':a_left_forearm_palm_loaded_count/steps,
          'a_left_forearm_palm_mean_normal_force_n':a_left_forearm_palm_normal_impulse_ns/max(float(d.time),1e-9),
          'a_left_forearm_max_palm_contact_gap_s':a_left_forearm_max_palm_contact_gap_s,
          'a_left_forearm_contact_body_events':a_left_forearm_contact_body_counts,
          'a_left_forearm_mean_normal_force_n':a_left_forearm_normal_impulse_ns/max(float(d.time),1e-9),
          'a_left_forearm_peak_single_contact_force_n':a_left_forearm_peak_single_contact_n,
          'final_training_elbow_angle_rad':float(d.qpos[m.joint('training_elbow_hinge').qposadr[0]]),
          'b_left_spool_thumb_index_contact_fraction':b_left_spool_pair_count/steps,
          'b_left_spool_opposed_contact_fraction':b_left_spool_opposed_count/steps,
          'b_left_spool_opposed_loaded_ge0p5n_fraction':b_left_spool_opposed_loaded_count/steps,
          'b_left_spool_mean_digit_normal_force_n':{name:impulse/max(float(d.time),1e-9) for name,impulse in b_left_spool_normal_impulse_ns.items()},
          'spool_world_contact_after_stage_release_events':spool_world_contact_after_release_count,
          'max_spool_displacement_m':max_spool_displacement,
          'max_post_stage_release_spool_displacement_m':max_post_stage_release_spool_displacement if stage_release_spool_position is not None else None,
          'max_post_stage_release_spool_drop_m':max_post_stage_release_spool_drop if stage_release_spool_position is not None else None,
          'max_post_stage_release_spool_wrist_relative_drift_m':max_post_stage_release_spool_wrist_relative_drift if stage_release_spool_position is not None else None,
          'max_dressing_pad_displacement_m':max_pad_displacement,
          'max_post_stage_release_pad_displacement_m':max_post_stage_release_pad_displacement if stage_release_pad_position is not None else None,
          'pad_stage_contact_events':pad_stage_contact_count,
          'pad_world_contact_after_release_events':pad_world_contact_after_release_count,
          'pad_world_contact_geoms_after_release':sorted(pad_world_contact_geoms_after_release),
          'max_b_nonfloor_penetration_m': max_pen,
          'max_a_nonfloor_penetration_m': max_a_pen,
          'max_either_robot_nonfloor_penetration_m': max(max_pen,max_a_pen),
          'worst_b_contact':worst_b_contact,
          'worst_a_contact':worst_a_contact,
          'max_free_edge_drift_m': edge_drift,
          'max_free_edge_forearm_relative_drift_m':edge_forearm_relative_drift,
          'max_hand_joint_speed_rad_s':max_hand_joint_speed,
          'max_post_release_material_slip_m':max_post_release_material_slip if post_release_offsets is not None else None,
          'max_post_release_edge_drift_m':max_post_release_edge_drift if post_release_edge is not None else None,
          'mean_tip_normal_force_n':{name:impulse/max(float(d.time),1e-9) for name,impulse in tip_normal_impulse_ns.items()},
          'b_right_opposed_loaded_ge0p5n_fraction':tip_loaded_ge0p5n_steps/max(steps,1),
          'b_right_digit_opposed_loaded_ge0p5n_fraction':digit_loaded_ge0p5n_steps/max(steps,1),
          'b_right_digit_mean_normal_force_n':{name:impulse/max(float(d.time),1e-9)
              for name,impulse in digit_normal_impulse_ns.items()},
          'fixture_release_gated_on_loaded_pinch':args.gate_fixture_release_on_loaded_pinch,
          'actual_fixture_release_s':actual_fixture_release_s,
          'peak_single_contact_normal_force_n':tip_peak_single_contact_normal_force_n,
          'mean_tip_shear_force_n':{name:impulse/max(float(d.time),1e-9) for name,impulse in tip_shear_impulse_ns.items()},
          'max_tip_to_material_slip_m': max((max(item.get('tip_to_material_relative_shift_m',{}).values(),default=0.) for item in trace),default=0.),
          'max_joint_tracking_error_rad': max_tip_error,
          'final_joint_target_error_rad': float(np.max(np.abs(d.qpos[qa]-waypoints[-1, qa]))),
          'trace': trace}
report['external_end_fixture_removed_for_test'] = bool(not fixture_ids or
    (args.fixture_release_s is not None and args.fixture_release_s <= duration-.2+1e-9 and
     not any(d.eq_active[i] for i in fixture_ids)))
report['a_pad_independent_hold_gate_pass']=bool(args.stage_release_s is not None and
    args.stage_release_s<=duration-.2+1e-9 and stage_release_pad_position is not None and
    a_pad_opposed_segment_count/steps>=.95 and max_post_stage_release_pad_displacement<=.003 and
    pad_world_contact_after_release_count==0 and max_a_pen<=.003 and
    error is None and d.time>=duration-1e-6)
report['b_spool_independent_hold_gate_pass']=bool(
    args.stage_release_s is not None and args.release_equipment_tray_with_stage and
    args.stage_release_s<=duration-.2+1e-9 and stage_release_spool_position is not None and
    b_left_spool_opposed_loaded_count/steps>=.95 and
    spool_world_contact_after_release_count==0 and
    max_post_stage_release_spool_drop<=.003 and
    max_post_stage_release_spool_wrist_relative_drift<=.003 and
    max(max_pen,max_a_pen)<=.003 and error is None and d.time>=duration-1e-6)
report['material_grasp_gate_pass'] = bool(not args.disable_dressing_pad_contact and
    report['external_end_fixture_removed_for_test'] and
    error is None and d.time >= duration-1e-6 and
    report['b_right_digit_opposed_loaded_ge0p5n_fraction'] >= .95 and edge_drift <= .003 and
    report['max_tip_to_material_slip_m'] <= .003 and max(max_pen,max_a_pen) <= .003)
# This is only a simultaneous contact check, not a full wrapping or hardware gate.
report['four_hand_sim_contact_gate_pass'] = bool(
    report['a_pad_independent_hold_gate_pass'] and report['material_grasp_gate_pass'] and
    a_left_forearm_palm_contact_count/steps >= .95 and
    a_left_forearm_palm_loaded_count/steps >= .95 and
    report['b_spool_independent_hold_gate_pass'] and feet_count/steps >= .95)
if worst_b_qpos is not None and args.a_pad_side_grasp_seed:
    np.savez_compressed(OUT/'joint_b_pinchtab_worst_b_penetration_state.npz',
                        qpos=worst_b_qpos,time_s=worst_b_contact['time_s'])
report_name = (('joint_b_pinchtab_dynamic_path_rollout' if args.pinchtab else
                'joint_b_flex_free_edge_dynamic_path_rollout')+
               ('_goal_hold' if args.start_at_goal else '')+
               ('_extended20mm' if args.extended_tab else '')+
               ('_track_edge' if args.track_edge else '')+
               (f'_flexforce{args.flex_tip_force_n:g}' if args.flex_tip_force_n is not None else '')+
               (f'_brightforcedigits{args.b_right_force_digits}' if args.b_right_force_digits!='both' else '')+
               ('_brightposition' if args.b_right_position_control else '')+
               ('_brightloadedpinchseed' if args.b_right_loaded_pinch_seed else '')+
               (f'_brightseedrank{args.b_right_loaded_pinch_seed_rank}' if args.b_right_loaded_pinch_seed_rank!=-1 else '')+
               ('_brightdigitanchor' if args.b_right_digit_material_anchor else '')+
               ('_reinforcedhem' if args.reinforced_hem else '')+
               ('_forearmanchor' if args.forearm_anchored_bandage else '')+
               ('_bodyparented' if args.body_parented_bandage else '')+
               ('_bodyparentedrefaligned' if args.body_parented_refaligned_bandage else '')+
               (f'_hemseedrank{args.b_right_hem_seed_rank}' if args.reinforced_hem else '')+
               (f'_hemforce{args.hem_force_n:g}' if args.hem_force_n is not None else '')+
               (f'_hemradius{args.hem_radius_mm:g}mm' if args.reinforced_hem and args.hem_radius_mm!=2.5 else '')+
               (f'_flexradius{args.flex_contact_radius_mm:g}mm' if args.flex_contact_radius_mm!=1 else '')+
               (f'_brightinitialpip{args.b_right_initial_index_pip_offset_rad:g}' if args.b_right_initial_index_pip_offset_rad else '')+
               (f'_measuredforce{args.measured_tip_force_n:g}' if args.measured_tip_force_n is not None else '')+
               (f'_pinchbias{args.pinch_bias_preset}' if args.pinch_bias_preset else '')+
               (f'_biasuntil{args.pinch_bias_until_s:g}s' if args.pinch_bias_until_s is not None else '')+
               ('_onestephandsearch' if args.one_step_hand_search else '')+
               (f'_handsearchuntil{args.one_step_hand_search_until_s:g}s' if args.one_step_hand_search_until_s is not None else '')+
               (f'_horizon{args.hand_search_horizon_s:g}s' if args.one_step_hand_search and args.hand_search_horizon_s!=.02 else '')+
               (f'_edgeweight{args.hand_search_edge_weight:g}' if args.one_step_hand_search and args.hand_search_edge_weight!=.3 else '')+
               (f'_forceweight{args.hand_search_force_weight:g}' if args.one_step_hand_search and args.hand_search_force_weight else '')+
               (f'_trajectoryforceweight{args.hand_search_trajectory_force_weight:g}' if args.one_step_hand_search and args.hand_search_trajectory_force_weight else '')+
               (f'_penweight{args.hand_search_penetration_weight:g}' if args.one_step_hand_search and args.hand_search_penetration_weight!=10. else '')+
               (f'_handaccelcap{args.hand_accel_cap:g}' if args.hand_accel_cap!=200. else '')+
               (f'_wbctol{args.wbc_constraint_tolerance:g}' if args.wbc_constraint_tolerance!=2e-5 else '')+
               ('_nopadcontact' if args.disable_dressing_pad_contact else '')+
               (f'_padsolref{args.dressing_pad_solref:g}' if args.dressing_pad_solref is not None else '')+
               ('_apadsidegraspseed' if args.a_pad_side_grasp_seed else '')+
               ('_aleftnearelbowseed' if args.a_left_near_elbow_seed else '')+
               ('_aleftpalmseed' if args.a_left_palm_only_seed else '')+
               (f'_stagerelease{args.stage_release_s:g}s' if args.stage_release_s is not None else '')+
               (f'_apadforce{args.a_pad_normal_goal_n:g}' if args.a_pad_normal_goal_n is not None else '')+
               ('_apadforcecontrol' if args.a_pad_force_control else '')+
               ('_releaseequipmenttray' if args.release_equipment_tray_with_stage else '')+
               (f'_bmiddleretract{args.b_middle_retract_mode}' if args.b_middle_retract_mode else '')+
               (f'_bspoolforce{args.b_spool_normal_goal_n:g}' if args.b_spool_normal_goal_n is not None else '')+
               ('_bspoolforcecontrol' if args.b_spool_force_control else '')+
               (f'_bleftlift{args.b_left_lift_mm:g}mm' if args.b_left_lift_mm else '')+
               ('_bspoolonestepsearch' if args.b_spool_one_step_search else '')+
               (f'_bspoolhorizon{args.b_spool_search_horizon_s:g}s' if args.b_spool_one_step_search and args.b_spool_search_horizon_s!=.04 else '')+
               (f'_bspoolbias{args.b_spool_search_bias_rad:g}' if args.b_spool_one_step_search and args.b_spool_search_bias_rad!=.05 else '')+
               (f'_bspoolpostbias{args.b_spool_search_post_release_bias_rad:g}' if args.b_spool_search_post_release_bias_rad is not None else '')+
               (f'_bspoolswitch{args.b_spool_search_switch_s:g}s' if args.b_spool_search_switch_s is not None else '')+
               (f'_bspoolsupportweight{args.b_spool_search_support_weight:g}' if args.b_spool_one_step_search and args.b_spool_search_support_weight else '')+
               (f'_aleftsupport{args.a_left_support_goal_n:g}' if args.a_left_support_goal_n is not None else '')+
               (f'_aleftpalmweight{args.a_left_palm_weight:g}' if args.a_left_palm_weight!=1000. else '')+
               (f'_aleftfollowforearm{args.a_left_follow_max_offset_mm:g}mmstep{args.a_left_follow_max_step_mm:g}mm' if args.a_left_follow_forearm else '')+
               (f'_aleftpalmservo{args.a_left_servo_max_step_mm:g}step{args.a_left_servo_max_offset_mm:g}maxgain{args.a_left_servo_gain:g}' if args.a_left_palm_contact_servo else '')+
               (f'_aleftlookahead{args.a_left_servo_lookahead_s:g}s' if args.a_left_palm_contact_servo and args.a_left_servo_lookahead_s else '')+
               (f'_aleftpreload{args.a_left_palm_preload_mm:g}mm' if args.a_left_palm_preload_mm else '')+
               (f'_elbowk{args.training_elbow_stiffness:g}' if args.training_elbow_stiffness is not None else '')+
               (f'_elbowd{args.training_elbow_damping:g}' if args.training_elbow_damping is not None else '')+
               (f'_elbowwbcweight{args.training_elbow_wbc_weight:g}' if args.training_elbow_wbc_weight is not None else '')+
               (f'_elbowwbckp{args.training_elbow_wbc_kp:g}' if args.training_elbow_wbc_kp is not None else '')+
               (f'_elbowrange{args.training_elbow_range_rad:g}' if args.training_elbow_range_rad!=.35 else '')+
               (f'_friction{args.flex_friction:g}' if args.flex_friction is not None else '')+
               (f'_wristweight{args.wrist_weight:g}' if args.wrist_weight is not None else '')+
               ('_localphysics' if args.local_physics_correction else '')+
               ('_stabilizedigits' if args.stabilize_other_digits else '')+
               ('_stabilizealldigits' if args.stabilize_all_digits else '')+
               ('_materialfollow' if args.material_tip_follow else '')+
               (f'_closure{args.tip_closure_mm:g}mm' if args.tip_closure_mm else '')+
               (f'_tipweight{args.tip_weight:g}' if args.material_tip_follow and args.tip_weight!=1000. else '')+
               (f'_forceprior{args.flex_measured_force_prior_weight:g}' if args.flex_measured_force_prior_weight else '')+
               ('_fixture' if args.releasable_fixture else '')+
               ('_fixtureloadgate' if args.gate_fixture_release_on_loaded_pinch else '')+
               (f'_release{args.fixture_release_s:g}s' if args.fixture_release_s is not None else '')+
               (f'_pinchclosure{args.pinch_closure_mm:g}mm' if args.pinch_closure_mm else '')+
               (f'_palmweight{args.b_right_palm_weight:g}' if args.b_right_palm_weight!=1000. else '')+
               suffix+'.json')
if args.forearm_grasp_hold_weight:
    report['forearm_grasp_hold_weight'] = args.forearm_grasp_hold_weight
    report_name = report_name[:-5] + f'_forearmhold{args.forearm_grasp_hold_weight:g}.json'
if args.reinforced_hem:
    seam = m.equality('bandage_hem_seam_72').id
    report['hem_seam_first_body_anchor_m'] = m.eq_data[seam,:3].tolist()
    if np.linalg.norm(m.eq_data[seam,:3]) < .02:
        report_name = report_name[:-5] + '_localseam.json'
report['run_label']=report_name
if args.relative_grasp_jacobian:
    report['relative_grasp_jacobian'] = True
    report_name = report_name[:-5] + '_relativejac.json'
    report['run_label'] = report_name
if len(report_name)>150:
    report_name=report_name[:115].rstrip('_')+'_'+hashlib.sha256(report_name.encode()).hexdigest()[:12]+'.json'
(OUT/report_name).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
