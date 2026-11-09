"""Joint inverse-dynamics WBC using measured MuJoCo contacts (simulation only).

Floating roots/object have no actuators. Contact forces are QP predictions;
MuJoCo generates actual forces. No qpos teleport, applied body force, weld,
object actuator, or constraint-force injection is used in closed-loop replay.
"""
from pathlib import Path
import json,time
import numpy as np,mujoco,cvxpy as cp
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/wuji_multiagent'

class JointWBC:
 def __init__(self,m,qref,palm_names=None,contact_filter=None):
  self.disabled_pressure_agents=set();self.backend='CLARABEL';self.force_control_indices=set();self.hard_hand_indices=set();self.world_force_targets={};self.world_force_command={};self.last_tau=np.zeros(m.nu)
  self.hand_position_mode=False;self.hand_indices=np.array([i for i in range(m.nu) if '/wuji_' in m.actuator(i).name]);self.hand_commands=np.zeros(m.nu);self.hand_tau_correction=np.zeros(m.nu);self.hand_admittance_offset=np.zeros(m.nu);self.last_pad_force_targets={}
  self.pressure_cmd={};self.pressure_time=None
  self.reference_data=mujoco.MjData(m);palm_names=palm_names or ('g1_a/wuji_right/r_wrist','g1_b/wuji_left/l_wrist');self.palm_ids=[m.body(name).id for name in palm_names];self.task_dim=6*len(self.palm_ids);self.act_qadr=np.array([m.jnt_qposadr[j] for j in m.actuator_trnid[:,0]]);self.act_vadr=np.array([m.jnt_dofadr[j] for j in m.actuator_trnid[:,0]]);self.act_gear=m.actuator_gear[:,0].copy()
  self.m=m;self.contact_filter=contact_filter;self.qref=qref.copy();self.vref=np.zeros(m.nv);self.aref=np.zeros(m.nv);self.preload_enabled=False;self.B=np.zeros((m.nv,m.nu));self.cache={}
  self.kp=np.full(m.nv,60.);self.kd=np.full(m.nv,14.);self.weights=np.ones(m.nv)
  for i in range(m.nu):self.B[m.jnt_dofadr[m.actuator_trnid[i,0]],i]=m.actuator_gear[i,0]
  for j in range(m.njnt):
   name=m.joint(j).name;dof=int(m.jnt_dofadr[j]);n=6 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
   if '/floating_base' in name:self.kp[dof:dof+n]=80.;self.kd[dof:dof+n]=18.;self.weights[dof:dof+n]=30.
   elif '/wuji_' in name:self.kp[dof]=80.;self.kd[dof]=18.;self.weights[dof]=1.
   elif name=='shared_joystick_free_base':self.kp[dof:dof+n]=400.;self.kd[dof:dof+n]=40.;self.weights[dof:dof+n]=30.
   elif name=='joystick_tilt':self.kp[dof]=80.;self.kd[dof]=18.;self.weights[dof]=10.
   elif any(token in name for token in ('shoulder','elbow','wrist')):
    passive=name.startswith(('g1_a/left_','g1_b/right_'));self.weights[dof]=10. if passive else 3.
 def actuator_limits(self,d):
  m=self.m;lo=np.full(m.nu,-1e4);hi=-lo
  for i in range(m.nu):
   if m.actuator_forcelimited[i]:lo[i],hi[i]=m.actuator_forcerange[i]
   joint=m.actuator_trnid[i,0]
   if m.jnt_actfrclimited[joint]:lo[i]=max(lo[i],m.jnt_actfrcrange[joint,0]);hi[i]=min(hi[i],m.jnt_actfrcrange[joint,1])
   if m.actuator_gainprm[i,0]!=1 or np.any(m.actuator_biasprm[i]):
    gain=m.actuator_gainprm[i,0];bias=m.actuator_biasprm[i,0]+m.actuator_biasprm[i,1]*d.actuator_length[i]+m.actuator_biasprm[i,2]*d.actuator_velocity[i]
    if m.actuator_ctrllimited[i]:
     reach=gain*m.actuator_ctrlrange[i]+bias;lo[i]=max(lo[i],min(reach));hi[i]=min(hi[i],max(reach))
  if np.any(lo>hi):raise RuntimeError('No feasible actuator force/control intersection')
  return lo,hi
 def controls(self,d,tau):
  gain=self.m.actuator_gainprm[:,0]
  bias=self.m.actuator_biasprm[:,0]+self.m.actuator_biasprm[:,1]*(self.act_gear*d.qpos[self.act_qadr])+self.m.actuator_biasprm[:,2]*(self.act_gear*d.qvel[self.act_vadr])
  ctrl=(tau-bias)/gain
  if self.hand_position_mode:ctrl[self.hand_indices]=self.hand_commands[self.hand_indices]+self.hand_tau_correction[self.hand_indices]/gain[self.hand_indices]
  for i in range(self.m.nu):
   if self.m.actuator_ctrllimited[i]:ctrl[i]=np.clip(ctrl[i],*self.m.actuator_ctrlrange[i])
  return ctrl
 def contacts(self,d):
  m=self.m;Js=[];targets=[];mus=[];identities=[];self.contact_indices=[]
  for contact_index,c in enumerate(d.contact):
   if c.dist>.001:continue
   g1,g2=int(c.geom1),int(c.geom2)
   if g1<0 or g2<0:raise RuntimeError('JointWBC does not yet support flex contacts; refusing invalid geom indexing')
   b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2]);R=c.frame.reshape(3,3)
   if self.contact_filter is not None and not self.contact_filter(m,c):continue
   j1=np.zeros((3,m.nv));j2=np.zeros_like(j1);mujoco.mj_jac(m,d,j1,None,c.pos,b1);mujoco.mj_jac(m,d,j2,None,c.pos,b2)
   J=R@(j2-j1);velocity=J@d.qvel
   # Kinematic bias acceleration of body-fixed material points.
   jd1=np.zeros_like(j1);jd2=np.zeros_like(j2);mujoco.mj_jacDot(m,d,jd1,None,c.pos,b1);mujoco.mj_jacDot(m,d,jd2,None,c.pos,b2)
   bias=R@(jd2-jd1)@d.qvel
   target=-20.*velocity-bias
   names=[m.body(b1).name,m.body(b2).name];pad_contact=any(n.endswith('_tip_sensor_frame') for n in names) and any(n.startswith('shared_joystick') for n in names)
   if not pad_contact:target[0]+=-2000.*float(c.dist)
   foot=m.geom(g1).name=='floor' or m.geom(g2).name=='floor'
   # Table tangential accelerations remain free: the object is not anchored.
   if not foot and not pad_contact:target[1:]=np.nan
   Js.append(J);targets.append(target);mus.append(float(c.friction[0]));identities.append((g1,g2));self.contact_indices.append(contact_index)
  return np.concatenate(Js) if Js else np.empty((0,m.nv)),np.concatenate(targets),np.array(mus),identities
 def solve(self,d):
  m=self.m;J,target,mus,ids=self.contacts(d);nc=len(ids)
  if nc==0:raise RuntimeError('No physical supporting contacts')
  mask=np.flatnonzero(np.isfinite(target));pad_flags=tuple(any(m.body(m.geom_bodyid[g]).name.endswith('_tip_sensor_frame') for g in (a,b)) and any(m.body(m.geom_bodyid[g]).name.startswith('shared_joystick') for g in (a,b)) for a,b in ids);fixed_hand=np.array([i for i in self.hand_indices if i not in self.force_control_indices],dtype=int);hard_hand=np.array(sorted(self.hard_hand_indices),dtype=int);key=(nc,tuple(mask),self.hand_position_mode,tuple(fixed_hand),tuple(hard_hand),pad_flags)
  if key not in self.cache:
   a=cp.Variable(m.nv);tau=cp.Variable(m.nu);f=cp.Variable(3*nc)
   P={name:cp.Parameter(shape) for name,shape in [('M',(m.nv,m.nv)),('h',(m.nv,)),('J',(3*nc,m.nv)),('target',(len(mask),)),('desired',(m.nv,)),('lo',(m.nu,)),('hi',(m.nu,)),('mu',(nc,)),('forcegoal',(3*nc,)),('forceweight',(3*nc,)),('acc_lo',(m.nv,)),('acc_hi',(m.nv,)),('taskJ',(self.task_dim,m.nv)),('tasktarget',(self.task_dim,)),('fixed_hand_tau',(len(self.hand_indices),))]}
   constraints=[P['M']@a+P['h']==self.B@tau+P['J'].T@f,tau>=P['lo'],tau<=P['hi'],a>=P['acc_lo'],a<=P['acc_hi']]

   if self.hand_position_mode and len(hard_hand):constraints.append(tau[hard_hand]==P['fixed_hand_tau'][[int(np.flatnonzero(self.hand_indices==i)[0]) for i in hard_hand]])
   for c in range(nc):
    normal=f[c*3];tang=f[c*3+1:c*3+3];constraints.extend([normal>=0,normal<=1000,cp.norm1(tang)<=P['mu'][c]*normal])
   objective=cp.sum_squares(cp.multiply(np.tile(np.r_[np.full(3,np.sqrt(1000.)),np.full(3,np.sqrt(10.))],len(self.palm_ids)),P['taskJ']@a-P['tasktarget']))+cp.sum_squares(cp.multiply(P['forceweight'],f)-P['forcegoal'])+cp.sum_squares(cp.multiply(np.array([100. if pad_flags[row//3] else 10. for row in mask]),P['J'][mask]@a-P['target']))+cp.sum_squares(cp.multiply(self.weights,a-P['desired']))+1e-4*cp.sum_squares(tau)+1e-6*cp.sum_squares(f)
   if self.hand_position_mode and len(fixed_hand):objective+=1e4*cp.sum_squares(tau[fixed_hand]-P['fixed_hand_tau'][[int(np.flatnonzero(self.hand_indices==i)[0]) for i in fixed_hand]])
   self.cache[key]=(cp.Problem(cp.Minimize(objective),constraints),P,a,tau,f)
  problem,P,a,tau,f=self.cache[key]
  M=np.empty((m.nv,m.nv))
  if hasattr(d,'qM'):mujoco.mj_fullM(m,M,d.qM)
  else:mujoco.mj_fullM(m,d,M)
  h=d.qfrc_bias-d.qfrc_passive
  error=np.zeros(m.nv);mujoco.mj_differentiatePos(m,error,1.,d.qpos,self.qref)
  desired=self.kp*error+self.kd*(self.vref-d.qvel)+self.aref;lo,hi=self.actuator_limits(d)
  desired[self.act_vadr[self.hand_indices]]+=self.kp[self.act_vadr[self.hand_indices]]*self.hand_admittance_offset[self.hand_indices]/self.act_gear[self.hand_indices]
  requested_hand_tau=np.clip(m.actuator_gainprm[self.hand_indices,0]*self.hand_commands[self.hand_indices]+m.actuator_biasprm[self.hand_indices,0]+m.actuator_biasprm[self.hand_indices,1]*d.actuator_length[self.hand_indices]+m.actuator_biasprm[self.hand_indices,2]*d.actuator_velocity[self.hand_indices],lo[self.hand_indices],hi[self.hand_indices])
  if self.hand_position_mode:
   for actuator in fixed_hand:
    if actuator in self.hard_hand_indices:continue
    requested=requested_hand_tau[int(np.flatnonzero(self.hand_indices==actuator)[0])];lo[actuator]=max(lo[actuator],requested-.03);hi[actuator]=min(hi[actuator],requested+.03)
  acc_lo=np.full(m.nv,-200.);acc_hi=np.full(m.nv,200.)
  for joint in range(m.njnt):
   if not m.jnt_limited[joint] or m.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE:continue
   adr=int(m.jnt_dofadr[joint]);q=float(d.qpos[m.jnt_qposadr[joint]]);v=float(d.qvel[adr]);low,high=m.jnt_range[joint];low+=.02;high-=.02
   acc_lo[adr]=max(-200.,-20*v-100*(q-low));acc_hi[adr]=min(200.,-20*v+100*(high-q))
  goals=np.zeros(3*nc);forceweight=np.zeros(3*nc)
  for k,(ga,gb) in enumerate(ids):
   if m.geom(ga).name=='floor' or m.geom(gb).name=='floor':
    measured=np.zeros(6);mujoco.mj_contactForce(m,d,self.contact_indices[k],measured);goals[k*3:k*3+3]=measured[:3];forceweight[k*3:k*3+3]=.3
  pad_contacts={robot:[] for robot in ('g1_a','g1_b')}
  for k,(geom_a,geom_b) in enumerate(ids):
   names=[m.body(m.geom_bodyid[g]).name for g in (geom_a,geom_b)]
   if not any(n.startswith('shared_joystick') for n in names):continue
   for robot in pad_contacts:
    pad_name=next((n for n in names if n.startswith(robot+'/') and n.endswith('_tip_sensor_frame')),None)
    if pad_name:pad_contacts[robot].append((k,'thumb_tip_sensor_frame' in pad_name,pad_name))
  desired_pressure={}
  if self.preload_enabled:
   for robot,entries in pad_contacts.items():
    if robot in self.disabled_pressure_agents:continue
    n_other=len({name for k,is_thumb,name in entries if not is_thumb})
    if n_other and any(is_thumb for k,is_thumb,name in entries):
     for k,is_thumb,name in entries:desired_pressure[name]=2.*n_other if is_thumb else 2.
  dt=.02 if self.pressure_time is None else max(0.,min(.1,float(d.time)-self.pressure_time));self.pressure_time=float(d.time)
  for name in set(self.pressure_cmd)|set(desired_pressure):
   target_pressure=desired_pressure.get(name,0.);alpha=dt/(.30+dt);self.pressure_cmd[name]=self.pressure_cmd.get(name,0.)+alpha*(target_pressure-self.pressure_cmd.get(name,0.))
  for robot,entries in pad_contacts.items():
   if robot in self.disabled_pressure_agents:continue
   for k,is_thumb,name in entries:
    count=sum(n==name for kk,tt,n in entries);goals[k*3]=self.pressure_cmd.get(name,0.)/count;forceweight[k*3]=5.
  for name,force in self.world_force_targets.items():
   old=self.world_force_command.get(name,np.zeros(3));self.world_force_command[name]=old+dt/(1.+dt)*(np.asarray(force)-old)
  for k,(ga,gb) in enumerate(ids):
   if not any(m.body(m.geom_bodyid[g]).name.startswith('shared_joystick') for g in (ga,gb)):continue
   for geom,sign in ((ga,-1.),(gb,1.)):
    name=m.body(m.geom_bodyid[geom]).name
    if name in self.world_force_command:
     count=sum(name in (m.body(m.geom_bodyid[a]).name,m.body(m.geom_bodyid[b]).name) for a,b in ids);goals[k*3:k*3+3]=sign*d.contact[self.contact_indices[k]].frame.reshape(3,3)@self.world_force_command[name]/count;forceweight[k*3:k*3+3]=5.
  ref=self.reference_data;ref.qpos[:]=self.qref;ref.qvel[:]=self.vref;mujoco.mj_kinematics(m,ref);mujoco.mj_comPos(m,ref);mujoco.mj_comVel(m,ref);taskJs=[];taskTargets=[]
  for bid in self.palm_ids:
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);jdp=np.zeros_like(jp);jdr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,bid);mujoco.mj_jacDot(m,d,jdp,jdr,d.xpos[bid],bid)
   rp=np.zeros_like(jp);rr=np.zeros_like(jp);rdp=np.zeros_like(jp);rdr=np.zeros_like(jp);mujoco.mj_jacBody(m,ref,rp,rr,bid);mujoco.mj_jacDot(m,ref,rdp,rdr,ref.xpos[bid],bid)
   Jtask=np.vstack((jp,jr));Jref=np.vstack((rp,rr));Jdot=np.vstack((jdp,jdr));Jrefdot=np.vstack((rdp,rdr));R=d.xmat[bid].reshape(3,3);Rref=ref.xmat[bid].reshape(3,3);orientation=.5*sum((np.cross(R[:,k],Rref[:,k]) for k in range(3)),start=np.zeros(3));error_task=np.r_[ref.xpos[bid]-d.xpos[bid],orientation]
   taskJs.append(Jtask);taskTargets.append(80*error_task+18*(Jref@self.vref-Jtask@d.qvel)+Jref@self.aref+Jrefdot@self.vref-Jdot@d.qvel)
  taskJ=np.concatenate(taskJs);tasktarget=np.concatenate(taskTargets)
  for k,v in {'M':M,'h':h,'J':J,'target':target[mask],'desired':desired,'lo':lo,'hi':hi,'mu':mus,'forcegoal':forceweight*goals,'forceweight':forceweight,'acc_lo':acc_lo,'acc_hi':acc_hi,'taskJ':taskJ,'tasktarget':tasktarget,'fixed_hand_tau':requested_hand_tau}.items():P[k].value=v
  self.last_numeric={'M':M.copy(),'B':self.B.copy(),'J':J.copy(),'h':h.copy(),'mask':mask.copy(),'target':target.copy(),'pad_flags':pad_flags,'weights':self.weights.copy(),'desired':desired.copy(),'taskJ':taskJ.copy(),'tasktarget':tasktarget.copy(),'forceweight':forceweight.copy(),'force_goal_weighted':(forceweight*goals).copy(),'mu':mus.copy(),'acc_lo':acc_lo.copy(),'acc_hi':acc_hi.copy(),'lo':lo.copy(),'hi':hi.copy(),'hard_hand':hard_hand.copy() if self.hand_position_mode else np.array([],dtype=int),'hard_tau':np.array([requested_hand_tau[int(np.flatnonzero(self.hand_indices==i)[0])] for i in hard_hand]) if self.hand_position_mode else np.array([]),'fixed_hand':fixed_hand.copy() if self.hand_position_mode else np.array([],dtype=int),'fixed_tau':P['fixed_hand_tau'].value[[int(np.flatnonzero(self.hand_indices==i)[0]) for i in fixed_hand]].copy() if self.hand_position_mode else np.array([])}
  solve_started=time.perf_counter();used_backend=self.backend;fallback=False;direct_success=False;direct_report=None
  def clarabel_solve():problem.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_feas=1e-9,tol_gap_rel=1e-8,max_iter=120,warm_start=True)
  if self.backend=='DIRECT':
   try:
    from direct_wbc_qp import solve_direct
    av,tv,fv,direct_report=solve_direct(self.last_numeric);a.value=av;tau.value=tv;f.value=fv;direct_success=True
   except (RuntimeError,ValueError):fallback=True;clarabel_solve()
  elif self.backend=='OSQP':
   try:problem.solve(solver='OSQP',eps_abs=1e-7,eps_rel=1e-8,max_iter=20000,warm_start=True,polishing=True)
   except cp.error.SolverError:fallback=True;clarabel_solve()
  else:clarabel_solve()
  def valid_solution():
   if (not direct_success and problem.status!='optimal') or tau.value is None or a.value is None or f.value is None:return False,float('inf')
   residual=float(np.max(np.abs(M@a.value+h-self.B@tau.value-J.T@f.value)));force=f.value.reshape(-1,3);viol=max(0.,float(np.max(lo-tau.value)),float(np.max(tau.value-hi)),float(np.max(-force[:,0])),float(np.max(np.abs(force[:,1:]).sum(axis=1)-mus*force[:,0])),float(np.max(force[:,0]-1000.)),float(np.max(acc_lo-a.value)),float(np.max(a.value-acc_hi)))
   return residual<=1e-4 and viol<=1e-5,residual
  valid,residual=valid_solution()
  if not valid and self.backend in ('OSQP','DIRECT'):fallback=True;direct_success=False;clarabel_solve();valid,residual=valid_solution()
  if not valid:raise RuntimeError('WBC invalid '+str(problem.status)+' residual '+str(residual))
  if fallback:used_backend='CLARABEL_fallback'
  solve_wall=time.perf_counter()-solve_started
  self.hand_tau_correction[:]=0.
  if self.hand_position_mode:
   for actuator in fixed_hand:self.hand_tau_correction[actuator]=tau.value[actuator]-requested_hand_tau[int(np.flatnonzero(self.hand_indices==actuator)[0])]
  self.last_tau=tau.value.copy();self.last_pad_force_targets={}
  for k,(ga,gb) in enumerate(ids):
   if not any(m.body(m.geom_bodyid[g]).name.startswith('shared_joystick') for g in (ga,gb)):continue
   for geom in (ga,gb):
    name=m.body(m.geom_bodyid[geom]).name
    if name.endswith('_tip_sensor_frame'):self.last_pad_force_targets[name]=self.last_pad_force_targets.get(name,0.)+max(0.,float(f.value[k*3]))
  return self.controls(d,tau.value),{'status':('optimal' if direct_success else problem.status),'contacts':nc,'qp_wall_s':solve_wall,'solver_time_s':(direct_report['elapsed_s'] if direct_success else problem.solver_stats.solve_time),'backend':used_backend,'dynamics_residual_N':residual,'max_abs_predicted_acceleration':float(np.max(np.abs(a.value))),'contact_acceleration_task_residual_m_s2':float(np.max(np.abs(J[mask]@a.value-target[mask]))),'max_hand_torque_adjustment_Nm':float(np.max(abs(self.hand_tau_correction))),'torque_limit_violation_Nm':float(max(0.,np.max(lo-tau.value),np.max(tau.value-hi)))},tau.value.copy()

def main():
 m=mujoco.MjModel.from_xml_path(str(OUT/'dual_g1_wuji_joystick.xml'));d=mujoco.MjData(m);d.qpos[:]=np.load(OUT/'dual_g1_wuji_seed.npz')['qpos'];d.qpos[m.joint('shared_joystick_free_base').qposadr[0]+2]-=1e-5;mujoco.mj_forward(m,d)
 ctrl,report,tau=JointWBC(m,d.qpos.copy()).solve(d);report.update(scope='single inverse-dynamics QP at initial stance, not closed-loop validation',qpos_teleport_in_simulation=False,object_external_actuation=False)
 (OUT/'joint_wbc_initial_qp.json').write_text(json.dumps(report,indent=2),encoding='utf-8');np.savez_compressed(OUT/'joint_wbc_initial_qp.npz',qpos=d.qpos,ctrl=ctrl,tau=tau);print(json.dumps(report))
if __name__=='__main__':main()
