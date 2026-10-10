#!/usr/bin/env python3
"""Recalculate manuscript quantities from the companion simulation dataset.

Usage: python audit_final_numbers.py --data SNAPSHOT --output results.json
SNAPSHOT follows tests/paper2026, with optional scpm_faithful/{v0.0,v0.7}
and flex_static directories. Run the existing CV/GCD/EIS analysis scripts in
that snapshot separately; they write outputs beside their inputs. This script
expects completed simulation outputs and leaves the input data unchanged.
"""
import argparse, hashlib, itertools, json, re
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args(); root = a.data; out = {}; sources = {}

def source(rel):
    p = root / rel
    sources[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def arr(rel, **kw):
    x = np.loadtxt(source(rel), **kw)
    assert np.all(np.isfinite(x)), rel
    return x

def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))

bench = {}
for system, name in [('slab','fiveway_values_slab_a100.json'), ('slit','fiveway_values_a100.json')]:
    d = json.loads(source('figures/' + name).read_text()); vals = {}
    for k in ['gmx_conv_long','gmx_cpm_long','lmp_conv_long','lmp_conp_long',
              'gmx_conv_gpu_long','gmx_cpm_gpu_long','gmx_cpm_gpu_64thr_long']:
        x = d[k + '_reps']; m, sd = np.mean(x), np.std(x, ddof=1)
        assert abs(m-d[k]) < .00051 and abs(sd-d[k+'_sd']) < .00051
        vals[k] = dict(mean=float(m), sample_sd=float(sd))
    v = lambda k: vals[k]['mean']
    vals['ratios'] = dict(cpu_cost=v('gmx_conv_long')/v('gmx_cpm_long'),
        gpu_cost=v('gmx_conv_gpu_long')/v('gmx_cpm_gpu_long'),
        vs_lammps=v('gmx_cpm_long')/v('lmp_conp_long'),
        gpu_vs_cpu=v('gmx_cpm_gpu_long')/v('gmx_cpm_long'),
        lammps_cost=v('lmp_conv_long')/v('lmp_conp_long'),
        added_walltime_ratio=(1/v('lmp_conp_long')-1/v('lmp_conv_long')) /
                             (1/v('gmx_cpm_long')-1/v('gmx_conv_long')))
    bench[system] = vals
out['throughput'] = bench

out['arithmetic'] = dict(slab_atoms=2*1144+4576+18400+3680,
    slit_atoms=4536+1288+632+316*5+18720*4,
    projected_residual_rms_V=float(np.sqrt(1e-6/4536)),
    TF_slope=2*1.43996*4*np.pi/(1144*.00878),
    spillover_slope=-2*1.43996*4*np.pi/5.488**2,
    cnt_sigma_C_m2=20*1.602176634e-19/(2*np.pi*.678*4.919*1e-18),
    grid_extra_percent=100*(60*25*360/(56*25*336)-1),
    startup_five_percent_ns=20*311*82.7/86400,
    resistance_Mohm_per_Vps_e=1e-12/1.602176634e-19/1e6,
    flexible_grid_nm=[5.488/48,5.488/48,30.184/132])

q = {k:arr('atomwise/frame0/q_'+k+'.dat') for k in ['matinv','matfree','lammps']}
out['frame0']={k:dict(rmsd=rms(v-q['matinv']),max_abs=float(abs(v-q['matinv']).max()),
    correlation=float(np.corrcoef(v,q['matinv'])[0,1])) for k,v in q.items() if k!='matinv'}
out['frame0']['charge_sd']=float(q['matinv'].std())

forces={}; gaps={}; azimuth={}
for run,start,end in [('q0_force',5,100),('cq10_force',20,120),('cq20_force',20,120),
                     ('cq30_force',200,500),('fq10_force',20,120),('fq20_force',20,120),('fq30_force',200,500)]:
    d=arr('cnt_pair/'+run+'/pullf.xvg',comments=('#','@')); x=d[(d[:,0]>=start)&(d[:,0]<=end),1]
    se=sem([b.mean() for b in np.array_split(x,10)])
    forces[run]=dict(window_ps=[start,end],mean_kJ_mol_nm=float(x.mean()),sem_kJ_mol_nm=se,
                     mean_nN=float(x.mean()/602.214076),sem_nN=se/602.214076)
for run in ['q0_free','q10_free','q20_free','q30_free','fq10_free','fq20_free','fq30_free']:
    d=arr('cnt_pair/'+run+'/pair_distance.dat'); x=d[d[:,0]>=d[-1,0]-200,1]-1.356
    gaps[run]=dict(end_ps=float(d[-1,0]),mean_gap_nm=float(x.mean()),temporal_sd_nm=float(x.std()))
for run in ['q10_free','q20_free','q30_free']:
    with np.load(source('cnt_pair/'+run+'/azimuth.npz')) as z:
        azimuth[run]=[dict(facing_e=float(z['q'+str(k)][abs(z['phi'+str(k)])<30].mean()),
                          outer_e=float(z['q'+str(k)][abs(z['phi'+str(k)])>150].mean())) for k in [1,2]]
out['nanotubes']=dict(forces=forces,gaps=gaps,azimuth_sectors=azimuth,
    fixed_over_CPM={str(k):forces['fq'+str(k)+'_force']['mean_nN']/forces['cq'+str(k)+'_force']['mean_nN'] for k in [10,20,30]})

charging={}
files=list(root.glob('elcons/long*/electrode_charge.dat'))+list(root.glob('elcons/remote_snap/*'))
for f in files:
    if f.suffix not in ['.dat','.log']:continue
    rows=[]
    for l in source(f.relative_to(root)).read_text().splitlines():
        w=l.split()
        try:
            if f.suffix=='.log':
                if len(w)>3 and w[0].isdigit(): rows.append((float(w[0])*1e-6,float(w[3])))
            else:rows.append((float(w[0])*1e-6,float(w[1])))
        except (ValueError,IndexError):pass
    d=np.array(rows);d=d[np.r_[True,np.diff(d[:,0])>0]]
    key='lammps' if f.suffix=='.log' else ('extlag' if 'extlag' in str(f) else 'matfree' if 'matfree' in str(f) else 'matinv')
    charging.setdefault(key,[]).append([float(d[(d[:,0]>=lo)&(d[:,0]<hi),1].mean()) for lo,hi in [(.45,.55),(5,10),(8,10)]])
out['charging']=dict(windows_ns=[[.45,.55],[5,10],[8,10]],methods={k:dict(replica_means=v,mean=np.mean(v,axis=0).tolist(),
    replica_sd=np.std(v,axis=0,ddof=1).tolist() if len(v)>1 else None) for k,v in charging.items()})

profiles={}; grid=np.arange(0,1.5,.02)
for f in root.glob('edl/sym/profiles/*.tsv'):
    d=arr(f.relative_to(root),skiprows=1); k='lammps' if f.name.startswith('lmp') else f.name.split('_r')[0]
    profiles.setdefault(k,[]).append([np.interp(grid,d[:,0]-10.976,d[:,2]),np.interp(grid,(19.208-d[:,0])[::-1],d[:,1][::-1])])
out['EDL']={}
for j,label in enumerate(['anion_positive','cation_negative']):
    means={k:np.mean([v[j] for v in vs],axis=0) for k,vs in profiles.items()}
    within=[np.mean([rms(x[j]-y[j]) for x,y in itertools.combinations(vs,2)]) for vs in profiles.values()]
    out['EDL'][label]=dict(mean_within_method_rms=float(np.mean(within)),max_between_means_rms=max(rms(x-y) for x,y in itertools.combinations(means.values(),2)))

TF={}
for ell in ['0.00','0.10','0.30']:
    g=arr('tf/multilayer/gmx_l'+ell+'/per_atom_charges.dat',comments='#');g=g[g[:,0]==g[:,0].min()];g=g[np.argsort(g[:,1]),2]
    m=arr('tf/multilayer/mw_l'+ell+'/charges.out',comments='#')[:6864]
    TF[ell]=dict(rmsd=rms(g-m),correlation=float(np.corrcoef(g,m)[0,1]),
        layer_percent=[float(100*g[k*1144:(k+1)*1144].sum()/g[:3432].sum()) for k in range(3)])
out['TF']=TF
ls=[];inv=[]
for f in sorted(root.glob('tf/capacitor/mfcg_l*/electrode_charge.dat')):
    ell=float(f.parent.name.split('_l')[1]);d=np.atleast_2d(arr(f.relative_to(root),skiprows=1));ls.append(ell**2);inv.append(4/d[-1,1])
out['TF']['single_layer_slope']=float(np.polyfit(ls,inv,1)[0])

if (root/'flex_static').exists():
    d=arr('flex_static/matinv_extended.txt',comments='#');e=arr('flex_static/matinv_comparison.txt',skiprows=1)
    prod=np.r_[d[:,0]*d[:,2],e[:,1]*e[:,3]]
    out['flexible_static']=dict(Qd_range_percent=float(100*np.ptp(prod)/prod.max()),
        max_relative_charge_difference=float(max(np.max(abs(d[:,2]/d[:,1]-1)),np.max(abs(e[:,3]/e[:,2]-1)))))

if (root/'scpm_faithful').exists():
    sc={}
    for run in ['v0.0','v0.7']:
        d=arr('scpm_faithful/'+run+'/electrode_charge.dat',comments=('#','S'));t=d[:,0]*.002
        q=d[(t>=15000)&(t<20000),1];blocks=[d[(t>=b)&(t<b+1000),1].mean() for b in range(15000,20000,1000)]
        sc[run]=dict(mean_e=float(q.mean()),sem_five_1ns_blocks_e=sem(blocks))
    fac=2*16.02176634/(4.03701*3.99561*.7)
    sc['capacitance_uF_cm2']=fac*(sc['v0.7']['mean_e']-sc['v0.0']['mean_e'])
    sc['sem_uF_cm2']=fac*np.hypot(sc['v0.7']['sem_five_1ns_blocks_e'],sc['v0.0']['sem_five_1ns_blocks_e'])
    out['spillover_last_5ns']=sc

def charge_steps(rel):
    d=arr(rel,comments='#'); out_={}
    for step in np.unique(d[:,0]):
        x=d[d[:,0]==step];out_[int(step)]=x[np.argsort(x[:,1]),2]
    return out_

out['GPU_saved_frames']={}
for run in ['slit','slab3dc','d1cv','d2gcd','mfcg','extlag','e1flex','densecg','single']:
    c=charge_steps('gpuval3/'+run+'/cpu/per_atom_charges.dat');g=charge_steps('gpuval3/'+run+'/gpu/per_atom_charges.dat')
    errors={str(k):rms(c[k]-g[k]) for k in sorted(c.keys() & g.keys())}
    out['GPU_saved_frames'][run]=dict(rmsd_by_step=errors,worst_rmsd=max(errors.values()))

w={k:charge_steps('gpuval3/widths/'+k+'/per_atom_charges.dat')[5] for k in ['file_mf','glob_mf','glob_mi','het_mf','het_mi','uni_mi']}
out['widths']={label:dict(rmsd=rms(w[x]-w[y]),max_abs=float(abs(w[x]-w[y]).max())) for label,x,y in [
    ('file_global_matrix_inversion','uni_mi','glob_mi'),('file_global_matrix_free','file_mf','glob_mf'),('heterogeneous','het_mf','het_mi'),('uniform_control','file_mf','uni_mi'),('width_change','het_mi','uni_mi')]}
x=charge_steps('gpuval3/single/cpu/per_atom_charges.dat')
out['single_electrode']=dict(max_total_charge_e=max(abs(q.sum()) for q in x.values()),
    first_face_charges_e=[float(x[min(x)][:2268].sum()),float(x[min(x)][2268:].sum())])
ref=charge_steps('gpuval3/solvers/mi/per_atom_charges.dat')[1]
out['solver_charge_agreement']={}
for run in ['dcg','mf_1e-4','mf_1e-6','mf_1e-8','mf_1e-10']:
    q=charge_steps('gpuval3/solvers/'+run+'/per_atom_charges.dat')[1]
    out['solver_charge_agreement'][run]=dict(rmsd=rms(q-ref),electrode1_error_e=float((q-ref)[:2268].sum()))

text=source('fairbench/a100/results_s2timing_a100.txt').read_text();out['S2_throughput']={}
for mode in ['mfcg_cpu64','mfcg_hostcg','mfcg_devcg']:
    v=[float(z) for z in re.findall(r'^'+mode+r'\s+rep\d+\s+([\d.]+) ns/day',text,re.M)]
    assert len(v)==3
    out['S2_throughput'][mode]=dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),step_ms=172.8/float(np.mean(v)))

# Preserve the regenerated electrochemical tables as machine-readable evidence.
out['CV_reanalysis_sha256'] = hashlib.sha256(source('timedep/cv5/RESULTS_analysis.md').read_bytes()).hexdigest()
out['GCD_reanalysis'] = {}
for f in sorted(root.glob('timedep/gcd_cutoff/folded_I*.npz')):
    with np.load(source(f.relative_to(root))) as z:
        out['GCD_reanalysis'][f.stem] = {k:float(z[k]) for k in ['I','tau','n','T','swing','Q_at_0V','Q_at_2V','C_tdis_plain','C_tdis_plain_sem','C_dis_slope','eff','gap','R_gap','ReZ_1T']}
out['EIS_reanalysis'] = {}
for f in sorted(root.glob('eis/eis_p*/electrode_charge.dat.gz')):
    run=f.parent.name;T=float(run.split('_p')[1].split('_')[0]);V=.25 if 'a025' in run else .5
    d=arr(f.relative_to(root),skiprows=1);t=d[:,0]*.002;Q=d[:,1];cycles=int(t[-1]//T)
    keep=(t>=T)&(t<cycles*T);t=t[keep];Q=Q[keep];u=(t-t[0])/(t[-1]-t[0]);w=2*np.pi/T
    A=np.column_stack([np.sin(w*t),np.cos(w*t),np.ones(len(t)),u,u*u]);c=np.linalg.lstsq(A,Q,rcond=None)[0];Z=V/(1j*w*complex(c[0],c[1]))
    out['EIS_reanalysis'][run]=dict(f_GHz=1000/T,cycles=cycles-1,Qamp_e=float(np.hypot(*c[:2])),
        Cp=float(c[0]/V),Cpp=float(-c[1]/V),phase_Q_deg=float(np.degrees(np.arctan2(c[1],c[0]))),
        phase_Z_deg=float(np.degrees(np.angle(Z))),ReZ=float(Z.real),negImZ=float(-Z.imag))

out['source_sha256']=sources
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print('Recomputed',len(out)-1,'quantity families from',len(sources),'snapshot files; wrote',a.output)
