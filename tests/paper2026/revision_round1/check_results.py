"""Recompute the published repeat statistics and paired charge-window means."""
from pathlib import Path
import gzip,json
import numpy as np
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'cnt_repeat_results/replica-summary.json').read_text())
for model in ['CPM','uniform']:
    gaps=[]
    for item in s['individual_gaps']:
        if item['model']!=model: continue
        a=np.loadtxt(ROOT/'cnt_repeat_results'/(item['case']+'.dat'))
        m=(a[:,0]>=800)&(a[:,0]<=1000)
        mean=float((a[m,1]-1.356).mean());assert np.isclose(mean,item['mean_gap_nm'],rtol=0,atol=1e-10);gaps.append(mean)
    forces=[x['repulsive_force_nN'] for x in s['individual_contact_forces'] if x['model']==model]
    for label,values in [('gap_nm',gaps),('repulsive_force_nN',forces)]:
        assert len(values)==5
        g=s['groups'][model][label]
        assert np.isclose(np.mean(values),g['mean'],rtol=0,atol=1e-10)
        assert np.isclose(np.std(values,ddof=1)/np.sqrt(5),g['replica_sem'],rtol=0,atol=1e-10)
    print('PASS',model,'five-replica gap and force statistics')
window=json.loads((ROOT/'charging_pair_r1/window_statistics.json').read_text())
for mode in ['predicted','fresh']:
    rows=[]
    with gzip.open(ROOT/'charging_pair_r1'/(mode+'_charge.dat.gz'),'rt') as f:
        for line in f:
            try:
                x=line.split();rows.append((int(x[0]),float(x[1])))
            except (ValueError,IndexError):continue
    a=np.array(rows);t=a[:,0]*.001;q=a[:,1]
    assert len(a)==100000 and t[-1]==10000 and np.all(np.diff(a[:,0])==100)
    for key,lo,hi in [('1.5-2ns',1500,2000),('8-9ns',8000,9000),('9-10ns',9000,10000),('8-10ns',8000,10000)]:
        mask=(t>=lo)&((t<=hi) if hi in (2000,10000) else (t<hi))
        assert np.isclose(q[mask].mean(),window[mode][key],rtol=0,atol=1e-10)
    print('PASS',mode,'complete 10 ns series and window means')
print('All saved-result checks passed.')
