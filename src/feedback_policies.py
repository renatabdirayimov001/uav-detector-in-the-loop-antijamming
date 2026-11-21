"""Feedback-driven policies: NACK-hop and the proposed Adaptive Feedback Threshold (AFT)."""
import numpy as np
from emulator_and_policies import affected, pick_clean
class NackHop:
    """Detector hard + immediate hop after a NACK on a channel the detector declared clean."""
    name='Det + NACK-hop'
    def reset(s,env): s.a=0; s.bad=-1
    def act(s,env,obs):
        aff=affected(env.G,np.where(obs>=0.5)[0])
        if s.bad>=0: aff=aff|(env.G[:,s.bad]>0)
        if aff[s.a]: s.a=pick_clean(env,aff,-env.G.dot(obs))
        s.bad=-1; return s.a
    def feedback(s,a,ack,obs):
        if not ack: s.bad=a
class AFT:
    """Adaptive feedback threshold on log10 detector score.
    tau starts at log10(0.5). A NACK on a channel declared clean moves tau down towards that channel's score
    (miss evidence); every ACK relaxes tau back towards its start (forgetting). tau never goes below a floor =
    q(1-eps) of training-domain benign calibration scores, which bounds the false-alarm rate."""
    def __init__(s,warm,eps=1e-3,down=0.5,up=0.002,nack_hop=True,margin=0.3):
        lw=np.log10(np.clip(warm,1e-12,1)); s.floor=float(np.quantile(lw,1-eps)); s.t0=np.log10(0.5)
        s.down=down; s.up=up; s.nh=nack_hop; s.m=margin
        s.name='AFT (proposed)'+('' if nack_hop else ' w/o NACK-hop')
    def reset(s,env): s.tau=s.t0; s.a=0; s.bad=-1
    def act(s,env,obs):
        ls=np.log10(np.clip(obs,1e-12,1)); s.ls=ls
        aff=affected(env.G,np.where(ls>=s.tau)[0])
        if s.nh and s.bad>=0: aff=aff|(env.G[:,s.bad]>0)
        if aff[s.a]: s.a=pick_clean(env,aff,-env.G.dot(obs))
        s.bad=-1; s.decl=aff; return s.a
    def feedback(s,a,ack,obs):
        if not ack:
            s.bad=a
            if not s.decl[a]: s.tau=max(s.floor, s.tau - s.down*(s.tau-(s.ls[a]-s.m)))
        else: s.tau=s.tau+s.up*(s.t0-s.tau)
