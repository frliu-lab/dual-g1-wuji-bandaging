"""Linearized contact-point Jacobian for a MuJoCo triangular flex element."""
import mujoco
import numpy as np


def flex_contact_jacobian(model, data, contact, flex_side):
    """Return material-point J/Jdot and triangle weights at the current contact."""
    flex_id=int(contact.flex[flex_side])
    elem=int(contact.elem[flex_side])
    if flex_id<0 or model.flex_dim[flex_id]!=2 or not 0<=elem<model.flex_elemnum[flex_id]:
        raise ValueError('Only 2D triangular flex contacts are supported')
    adr=int(model.flex_elemadr[flex_id])+3*elem
    verts=np.asarray(model.flex_elem[adr:adr+3],dtype=int)
    xyz=np.asarray(data.flexvert_xpos[verts],dtype=float)
    edges=np.column_stack((xyz[1]-xyz[0],xyz[2]-xyz[0]))
    uv=np.linalg.lstsq(edges,contact.pos-xyz[0],rcond=None)[0]
    weights=np.array([1.-uv.sum(),uv[0],uv[1]])
    weights=np.maximum(weights,0.)
    weights/=weights.sum()
    jac=np.zeros((3,model.nv));jacdot=np.zeros_like(jac)
    for vert,weight,point in zip(verts,weights,xyz):
        body=int(model.flex_vertbodyid[vert])
        j=np.zeros_like(jac);jd=np.zeros_like(jac)
        mujoco.mj_jac(model,data,j,None,point,body)
        mujoco.mj_jacDot(model,data,jd,None,point,body)
        jac+=weight*j;jacdot+=weight*jd
    return jac,jacdot,verts,weights
