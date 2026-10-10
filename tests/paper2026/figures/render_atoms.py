"""Shared helpers: read .gro, and draw shaded spheres with depth cueing."""
import numpy as np
from matplotlib.patches import Circle
from matplotlib.colors import to_rgb

def read_gro(path):
    L = open(path).read().split('\n'); n = int(L[1])
    res, name, xyz = [], [], []
    for i in range(n):
        s = L[2+i]
        res.append(s[5:10].strip()); name.append(s[10:15].strip())
        xyz.append([float(s[20:28]), float(s[28:36]), float(s[36:44])])
    return np.array(res), np.array(name), np.array(xyz), np.array([float(v) for v in L[2+n].split()[:3]])

STYLE = {
    'C': ('#3F3F3F', 0.070), 'N': ('#3B6EA5', 0.065), 'B': ('#C97B5A', 0.062),
    'F': ('#7FB069', 0.057), 'O': ('#B23A2E', 0.060), 'H': ('#DCDCDC', 0.034),
    'S': ('#D4A017', 0.075), 'NA': ('#8E6FBF', 0.085), 'TI': ('#8C9BA5', 0.098),
    'MW': (None, 0.0),
}
def elem_of(nm):
    nm = nm.upper()
    if nm.startswith(('MW','MV')): return 'MW'
    for k in ('NA','TI'): 
        if nm.startswith(k): return k
    return nm[0]

def _mix(c, t, f):
    c, t = np.array(to_rgb(c)), np.array(to_rgb(t))
    return tuple(c*(1-f) + t*f)

def sphere(ax, cx, cy, rad, color, depth=0.0, zorder=2, lw=0.13):
    """A shaded sphere: depth-cued body + offset specular highlight.
    depth in [0,1]: 0 = front (saturated), 1 = back (washed toward white)."""
    body = _mix(color, '#FFFFFF', 0.42*depth)
    ax.add_patch(Circle((cx, cy), rad, facecolor=body, edgecolor=_mix(body,'#000000',0.45),
                        linewidth=lw, zorder=zorder))
    # inner shading ring gives roundness
    ax.add_patch(Circle((cx - 0.16*rad, cy + 0.16*rad), 0.62*rad,
                        facecolor=_mix(body, '#FFFFFF', 0.30), edgecolor='none',
                        zorder=zorder+0.01, alpha=0.75))
    # small highlight
    ax.add_patch(Circle((cx - 0.30*rad, cy + 0.30*rad), 0.24*rad,
                        facecolor=_mix(body, '#FFFFFF', 0.72), edgecolor='none',
                        zorder=zorder+0.02, alpha=0.9))
