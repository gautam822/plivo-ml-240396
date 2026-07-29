"""Causal acoustic feature extraction for end-of-turn detection.

Every aggregate used for a pause is computed only from frames whose complete
25 ms window ends at or before ``pause_start``.  The analyzer may precompute
local frame descriptors for efficiency, but ``extract`` slices those arrays to
the causal prefix before any threshold, baseline, trend, or summary is formed.
No feature uses ``pause_end``, current pause duration, the label, file length,
or audio after the pause.
"""

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter

FRAME_MS=25; HOP_MS=10
FEATURE_NAMES=[
 # causal context / hazard
 'pause_index','log_elapsed','pause_rate','prior_pause_count_audio','last_segment_s','mean_segment_s','last_segment_rel',
 # energy / boundary
 'energy_final_rel','energy_last100_rel','energy_last300_rel','energy_drop_100_500','energy_slope_300','energy_slope_700',
 'active_frac_500','active_frac_all','speech_gap_ms','low_energy_tail_ms','final_active_run_ms',
 # pitch / periodicity
 'f0_final_rel_st','f0_delta_700_st','f0_slope_700_st_s','f0_range_700_st','f0_conf_final','f0_conf_rel','voiced_frac_500','voiced_gap_ms',
 # spectrum / voice quality
 'spectral_tilt_final','spectral_tilt_rel','centroid_final_rel','flatness_final_rel','zcr_final_rel','flux_final',
 'band0_rel','band1_rel','band2_rel','band3_rel','band4_rel','band5_rel',
]

def load_wav(path):
    sr,x=wavfile.read(path); x=np.asarray(x)
    if x.ndim>1: x=x.mean(axis=1)
    if x.dtype==np.int16: x=x.astype(np.float32)/32768.
    elif x.dtype==np.int32: x=x.astype(np.float32)/2147483648.
    else: x=x.astype(np.float32)
    return x,sr

def frame_all(x,sr):
    fl=int(sr*FRAME_MS/1000); hop=int(sr*HOP_MS/1000)
    if len(x)<fl:return np.empty((0,fl),np.float32)
    n=1+(len(x)-fl)//hop
    return np.lib.stride_tricks.as_strided(x,shape=(n,fl),strides=(x.strides[0]*hop,x.strides[0])).copy()

def pitch_fft(frames,sr,fmin=60.,fmax=400.,batch=4096):
    nfr,fl=frames.shape; f0=np.zeros(nfr,np.float32); conf=np.zeros(nfr,np.float32)
    lo=int(sr/fmax); hi=min(int(sr/fmin),fl-1); nfft=1
    while nfft<2*fl-1:nfft*=2
    for st in range(0,nfr,batch):
        fr=frames[st:st+batch].astype(np.float64,copy=True); fr-=fr.mean(1,keepdims=True)
        mx=np.max(np.abs(fr),1); F=np.fft.rfft(fr,n=nfft,axis=1); ac=np.fft.irfft(F*F.conj(),n=nfft,axis=1)[:,:fl]
        ac0=ac[:,0]; ac=ac/(ac0[:,None]+1e-30); sl=ac[:,lo:hi]
        lag=lo+np.argmax(sl,1); pk=ac[np.arange(len(fr)),lag]
        valid=(mx>=1e-4)&(ac0>0)
        conf[st:st+len(fr)]=np.where(valid,pk,0).astype(np.float32)
        v=valid&(pk>=.25); vals=np.zeros(len(fr),np.float32); vals[v]=sr/lag[v]
        f0[st:st+len(fr)]=vals
    return f0,conf

def runs(mask):
    # list (start,end_exclusive)
    m=np.r_[False,mask,False].astype(np.int8); d=np.diff(m)
    return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))

def robust_slope(t,y):
    if len(y)<3:return 0.
    # ordinary slope after median smoothing/clipping; t is seconds
    t=t-t.mean(); den=np.dot(t,t)
    return float(np.dot(t,y-y.mean())/den) if den>1e-12 else 0.

class CausalAnalyzer:
    def __init__(self,x,sr):
        self.x=x;self.sr=sr;self.frames=frame_all(x,sr); self.fl=self.frames.shape[1] if len(self.frames) else int(sr*.025); self.hop=int(sr*HOP_MS/1000)
        if len(self.frames)==0:
            self.e=self.f0=self.conf=self.zcr=self.cent=self.flat=self.tilt=self.flux=np.array([],np.float32);self.bands=np.empty((0,6),np.float32);return
        fr=self.frames.astype(np.float64)
        rms=np.sqrt(np.mean(fr*fr,1)+1e-12);self.e=(20*np.log10(rms+1e-12)).astype(np.float32)
        self.zcr=np.mean(np.diff(np.signbit(fr),axis=1),axis=1).astype(np.float32)
        self.f0,self.conf=pitch_fft(self.frames,sr)
        win=fr*np.hanning(self.fl); mag=np.abs(np.fft.rfft(win,axis=1))+1e-10; powr=mag*mag
        freq=np.fft.rfftfreq(self.fl,1/sr); ps=powr.sum(1)+1e-12
        self.cent=((powr*freq).sum(1)/ps).astype(np.float32)
        self.flat=(np.exp(np.mean(np.log(mag),1))/(np.mean(mag,1)+1e-12)).astype(np.float32)
        lo=powr[:,freq<1000].sum(1); hi=powr[:,freq>=1000].sum(1); self.tilt=np.log((lo+1e-8)/(hi+1e-8)).astype(np.float32)
        norm=mag/(mag.sum(1,keepdims=True)+1e-12); self.flux=np.r_[0,np.sqrt(np.sum(np.diff(norm,axis=0)**2,axis=1))].astype(np.float32)
        edges=[0,250,500,1000,2000,4000,8001]; b=[]
        for a,c in zip(edges[:-1],edges[1:]): b.append(np.log(powr[:,(freq>=a)&(freq<c)].sum(1)+1e-10))
        self.bands=np.stack(b,1).astype(np.float32)
    def nframes_at(self,pause_start):
        cut=int(pause_start*self.sr)
        if cut<self.fl:return 0
        return min(len(self.frames),1+(cut-self.fl)//self.hop)
    def extract(self,pause_start,pause_index):
        n=self.nframes_at(pause_start)
        if n<3:return np.zeros(len(FEATURE_NAMES),np.float32)
        e=self.e[:n];f0=self.f0[:n];conf=self.conf[:n];zcr=self.zcr[:n];cent=self.cent[:n];flat=self.flat[:n];tilt=self.tilt[:n];flux=self.flux[:n];bands=self.bands[:n]
        # adaptive causal speech activity threshold
        q10,q50,q90=np.percentile(e,[10,50,90]); thr=max(q10+10.,q90-35.,-58.)
        active=e>thr
        # suppress isolated active spikes, fill tiny holes
        aruns=runs(active); active2=np.zeros_like(active)
        for a,b in aruns:
            if b-a>=3: active2[a:b]=True
        # include confident voiced frames even if quiet
        active=active2|(conf>.42)
        if not active.any(): active=e>(e.max()-25.)
        ai=np.where(active)[0]; lasti=ai[-1]
        speech_e=e[active]; ebase=np.median(speech_e)
        def amean_last(ms,arr=e,mask=active):
            k=max(1,int(ms/HOP_MS)); idx=np.where(mask[max(0,n-k):])[0]+max(0,n-k)
            return float(np.mean(arr[idx])) if len(idx) else float(arr[max(0,n-k):].mean())
        e100=amean_last(100);e300=amean_last(300);e500=amean_last(500)
        energy_final=float(np.mean(e[ai[-min(3,len(ai)):]]))-ebase
        e100rel=e100-ebase;e300rel=e300-ebase;edrop=e100-e500
        def eslope(ms):
            k=max(3,int(ms/HOP_MS)); ids=np.where(active[max(0,n-k):])[0]+max(0,n-k)
            if len(ids)<3: ids=np.arange(max(0,n-k),n)
            return robust_slope(ids*HOP_MS/1000.,e[ids])
        es300=eslope(300);es700=eslope(700)
        k500=max(1,int(500/HOP_MS)); active500=float(active[-k500:].mean()); activeall=float(active.mean())
        speech_gap=(n-1-lasti)*HOP_MS
        # low-energy tail immediately before pause
        low=e < (ebase-18.)
        tail=0
        for v in low[::-1]:
            if v: tail+=1
            else: break
        lowtail=tail*HOP_MS
        rr=runs(active); finalrun=(rr[-1][1]-rr[-1][0])*HOP_MS if rr else 0
        # internal pause / segment timing from causal activity
        silent_runs=[(a,b) for a,b in runs(~active) if b-a>=10 and b<n]
        prior_count=len(silent_runs)
        boundaries=[0]+[b for a,b in silent_runs]
        seg_lens=[]
        for j,s in enumerate(boundaries):
            en=(silent_runs[j][0] if j<len(silent_runs) else n)
            if en>s:seg_lens.append((en-s)*HOP_MS/1000.)
        lastseg=seg_lens[-1] if seg_lens else pause_start; meanseg=np.mean(seg_lens[:-1]) if len(seg_lens)>1 else lastseg
        lastsegrel=lastseg/(meanseg+1e-3)
        # pitch clean-up: active + periodic, speaker relative semitones, preserve actual timing
        voiced=(f0>0)&active&(conf>.28)
        vf=float(voiced[-k500:].mean()); vi=np.where(voiced)[0]
        if len(vi)>=3:
            med=np.median(f0[vi]); st=np.zeros(n,np.float32); st[vi]=12*np.log2(f0[vi]/med)
            good=vi[np.abs(st[vi])<=12]
            if len(good)>=3:
                vals=st[good]; vals=median_filter(vals,size=3,mode='nearest')
                # final 700ms actual frame positions
                ids=good[good>=n-int(700/HOP_MS)]; vals2=st[ids]
                if len(ids)>=3:
                    vals2=median_filter(vals2,size=3,mode='nearest'); fslope=robust_slope(ids*HOP_MS/1000.,vals2)
                    fdelta=float(np.median(vals2[-min(3,len(vals2)):])-np.median(vals2[:min(3,len(vals2))]))
                    frange=float(np.percentile(vals2,90)-np.percentile(vals2,10))
                    ffinal=float(np.median(vals2[-min(3,len(vals2)):]))
                else: fslope=fdelta=frange=ffinal=0.
            else: fslope=fdelta=frange=ffinal=0.
            vgap=(n-1-vi[-1])*HOP_MS
        else: fslope=fdelta=frange=ffinal=0.;vgap=700.
        cfinal=float(np.mean(conf[ai[-min(5,len(ai)):]])); cbase=float(np.median(conf[active])); crel=cfinal-cbase
        # final spectral descriptors relative to active baseline
        fin=ai[-min(5,len(ai)):]
        def rel(arr,logratio=False):
            a=float(np.mean(arr[fin])); b=float(np.median(arr[active]));
            return (np.log((a+1e-6)/(b+1e-6)) if logratio and a>=0 and b>=0 else a-b)
        tiltfin=float(np.mean(tilt[fin])); tiltrel=tiltfin-float(np.median(tilt[active]))
        centrel=np.log((float(np.mean(cent[fin]))+1)/(float(np.median(cent[active]))+1))
        flatrel=np.log((float(np.mean(flat[fin]))+1e-5)/(float(np.median(flat[active]))+1e-5))
        zrel=np.log((float(np.mean(zcr[fin]))+1e-4)/(float(np.median(zcr[active]))+1e-4))
        fluxfin=float(np.mean(flux[fin]))
        brel=(np.mean(bands[fin],0)-np.median(bands[active],0)).tolist()
        return np.asarray([
            float(pause_index),np.log1p(pause_start),float(pause_index)/(pause_start+1.),float(prior_count),lastseg,float(meanseg),lastsegrel,
            energy_final,e100rel,e300rel,edrop,es300,es700,active500,activeall,speech_gap,lowtail,finalrun,
            ffinal,fdelta,fslope,frange,cfinal,crel,vf,vgap,
            tiltfin,tiltrel,centrel,flatrel,zrel,fluxfin,*brel
        ],np.float32)


# Backwards-compatible one-shot API used by the causality test.
def extract_features(x, sr, pause_start, pause_index=0):
    return CausalAnalyzer(x, sr).extract(float(pause_start), int(pause_index))
