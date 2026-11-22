"""Direct matrix form of the same convex WBC QP; no auxiliary residual variables."""
import time
import numpy as np
import scipy.sparse as sp
import clarabel

def solve_direct(data):
 n,u=data['M'].shape[0],data['B'].shape[1];J=data['J'];nc=J.shape[0]//3;nf=3*nc;N=n+u+nf;mask=data['mask'];C=J[mask];cw=data['contact_weight'];C=cw[:,None]*C;ct=cw*data['target'][mask]
 tw=np.asarray(data['task_weight'],dtype=float)
 if tw.shape!=(len(data['tasktarget']),):raise ValueError('Task weight rows do not match task targets')
 T=tw[:,None]*data['taskJ'];tt=tw*data['tasktarget'];w2=data['weights']**2
 H=np.diag(w2)+C.T@C+T.T@T;g=-2*(w2*data['desired']+C.T@ct+T.T@tt)
 fixed=data['fixed_hand'];torque_diag=np.full(u,1e-4);torque_linear=np.zeros(u)
 if len(fixed):torque_diag[fixed]+=1e4;torque_linear[fixed]=-2e4*data['fixed_tau']
 P=sp.triu(sp.block_diag((sp.csc_matrix(2*H),sp.diags(2*torque_diag),sp.diags(2*(data['forceweight']**2+1e-6))),format='csc'),format='csc');q=np.r_[g,torque_linear,-2*data['forceweight']*data['force_goal_weighted']]
 eq=sp.hstack((sp.csc_matrix(data['M']),sp.csc_matrix(-data['B']),sp.csc_matrix(-J.T)),format='csc');beq=-data['h'];fixed=data['fixed_hand']
 hard=data.get('hard_hand',np.array([],dtype=int))
 if len(hard):
  extra=sp.csc_matrix((np.ones(len(hard)),(np.arange(len(hard)),n+hard)),shape=(len(hard),N));eq=sp.vstack((eq,extra),format='csc');beq=np.r_[beq,data['hard_tau']]
 diag=sp.eye(N,format='csc');lower=np.r_[data['acc_lo'],data['lo'],np.full(nf,-np.inf)];upper=np.r_[data['acc_hi'],data['hi'],np.full(nf,np.inf)];lower[n+u::3]=0.;upper[n+u::3]=1000.;low_idx=np.flatnonzero(np.isfinite(lower));up_idx=np.flatnonzero(np.isfinite(upper));ineq=sp.vstack((diag[up_idx],-diag[low_idx]),format='csc');bineq=np.r_[upper[up_idx],-lower[low_idx]]
 rr=[];cc=[];vv=[]
 for contact in range(nc):
  for k,(s1,s2) in enumerate(((1,1),(1,-1),(-1,1),(-1,-1))):
   row=4*contact+k;rr.extend([row]*3);cc.extend([n+u+3*contact,n+u+3*contact+1,n+u+3*contact+2]);vv.extend([-data['mu'][contact],s1,s2])
 friction=sp.csc_matrix((vv,(rr,cc)),shape=(4*nc,N));A=sp.vstack((eq,ineq,friction),format='csc');b=np.r_[beq,bineq,np.zeros(4*nc)];cones=[clarabel.ZeroConeT(len(beq)),clarabel.NonnegativeConeT(len(bineq)+4*nc)]
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=120;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8;settings.tol_feas=1e-9
 start=time.perf_counter();solver=clarabel.DefaultSolver(P,q,A,b,cones,settings);solution=solver.solve();elapsed=time.perf_counter()-start
 if str(solution.status)!='Solved':raise RuntimeError('Direct WBC '+str(solution.status))
 x=np.array(solution.x);return x[:n],x[n:n+u],x[n+u:],{'elapsed_s':elapsed,'status':str(solution.status),'iterations':solution.iterations}
