"""Figure 1b: capability gallery, rendered from real coordinates with shaded spheres."""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_atoms import read_gro, STYLE, elem_of, sphere

HERE = os.path.dirname(os.path.abspath(__file__))
GRO  = os.path.normpath(os.path.join(HERE, "..", "..", "..", "example", "start.gro"))
plt.rcParams.update({'font.family':'sans-serif','font.size':6.2,
                     'axes.linewidth':0.5,'savefig.dpi':600})
BLUE, VERM, INK = '#0072B2', '#D55E00', '#333333'
TI, CC, OO = '#8C9BA5', '#3F3F3F', '#B23A2E'

r, nm, x, box = read_gro(GRO)
ELEC = np.isin(r, ['GrO','WaL','WaU'])
YLO, YHI = 2.30, 3.05
ysel = (x[:,1] > YLO) & (x[:,1] < YHI)

def cell(ax, zlo, zhi, elec_color=None):
    m = ysel & (x[:,2] > zlo) & (x[:,2] < zhi)
    idx = np.where(m)[0]; idx = idx[np.argsort(-x[idx,1])]   # far -> near
    for i in idx:
        e = elem_of(nm[i]); col, rad = STYLE.get(e, ('#909090',0.06))
        if col is None: continue
        if elec_color is not None and ELEC[i]: col = elec_color
        d = (x[i,1]-YLO)/(YHI-YLO)                            # 0 near .. 1 far
        sphere(ax, x[i,0], x[i,2], rad, col, depth=1.0-d, zorder=2+d)

def tidy(ax, title):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s in ax.spines.values(): s.set_color('#BBBBBB')
    ax.set_title(title, fontsize=6.3, pad=2.2)

fig = plt.figure(figsize=(3.90, 4.5))
G = gs.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.0],
                hspace=0.22, wspace=0.30,
                left=0.015, right=0.985, top=0.955, bottom=0.055)

# (1) two electrodes ------------------------------------------------------
ax = fig.add_subplot(G[0:2, 0]); ZLO, ZHI = 10.15, 20.05
cell(ax, ZLO, ZHI); tidy(ax, "two electrodes")
ax.set_ylim(ZLO, ZHI); ax.set_xlim(-1.15, box[0])      # margin on the left for the circuit
ZT, ZB, XW_ = 19.25, 11.00, -0.62                      # electrode planes and the wire's x
ax.plot([0.05, XW_, XW_, 0.05], [ZT, ZT, ZB, ZB], '-', color=INK, lw=0.9,
        solid_capstyle='round', zorder=7, clip_on=False)
ax.add_patch(plt.Circle((XW_, 15.12), 0.42, fc='white', ec=INK, lw=0.9, zorder=8))
ax.text(XW_, 15.12, '$V$', color=INK, fontsize=6.6, ha='center', va='center', zorder=9)
ax.text(box[0]*0.50, 15.12, 'fixed $\\Delta U$\nor fixed total charge', color=INK, fontsize=5.8,
        ha='center', va='center', zorder=9,
        bbox=dict(fc='white', ec='none', alpha=0.92, pad=1.0))

# (2) single electrode ----------------------------------------------------
ax = fig.add_subplot(G[0, 1]); cell(ax, ZLO, 14.35)
tidy(ax, "single electrode"); ax.set_xlim(0, box[0]); ax.set_ylim(ZLO, 14.5)
ax.axhline(14.30, color='#999999', ls=(0,(3,2)), lw=0.6)
ax.text(box[0]*0.5, 10.72, 'fixed total charge', color=INK, fontsize=5.8, ha='center', va='center',
        zorder=9, bbox=dict(fc='white', ec='none', alpha=0.94, pad=1.4))

# (3) flexible electrode: multilayer rippled graphene ---------------------
ax = fig.add_subplot(G[1, 1])
acc, XW = 0.142, 1.95
def zrip(xa, ph):
    return (0.105*np.sin(2*np.pi*xa/1.15 + ph)
            + 0.038*np.sin(2*np.pi*xa/0.47 + 1.7*ph + 0.9))
for li, (zoff, ph) in enumerate([(0.00,0.0), (0.345,0.35), (0.690,0.70), (1.035,1.05)]):
    for ri, yy in enumerate([0.42, 0.21, 0.0]):          # rows back in y -> depth
        d = ri/2.0
        k = 0
        while k*acc*3.0 < XW + acc*3:
            for xa in (k*acc*3.0 + 0.5*yy, k*acc*3.0 + acc + 0.5*yy):
                sphere(ax, xa, zoff + zrip(xa, ph) + 0.045*yy, 0.060, CC,
                       depth=d, zorder=2+(1-d)+0.1*li)
            k += 1
tidy(ax, "flexible electrode"); ax.set_xlim(0.02, XW); ax.set_ylim(-0.30, 1.40)

# (4) multicomponent electrode: MXene Ti3C2O2 ----------------------------
ax = fig.add_subplot(G[2, 0])
a, XW = 0.306, 2.75
layers = [(-0.115, OO, 0.062, 'O'), (0.0, TI, 0.098, 'Ti'), (0.118, CC, 0.070, 'C'),
          (0.236, TI, 0.098, 'Ti'), (0.354, CC, 0.070, 'C'), (0.472, TI, 0.098, 'Ti'),
          (0.587, OO, 0.062, 'O')]
for depth, off in ((0.85, 0.5*a), (0.0, 0.0)):
    for zc, col, rad, _ in layers:
        k = 0
        while k*a + off < XW + a:
            sphere(ax, k*a + off, zc + 0.055*depth, rad, col, depth=depth, zorder=2+(1-depth))
            k += 1
ax.set_xlim(-0.10, XW); ax.set_ylim(-0.34, 0.81); tidy(ax, "multicomponent electrode")
for lbl, col in (('Ti', TI), ('C', CC), ('O', OO)):
    ax.scatter([], [], s=11, c=col, label=lbl)
ax.legend(loc='center right', frameon=True, framealpha=0.85, edgecolor='none',
          fontsize=5.5, handletextpad=0.15, labelspacing=0.18, borderpad=0.2)
ax.text(0.02, 0.685, 'MXene Ti$_3$C$_2$O$_2$', color=INK, fontsize=5.8, va='top',
        bbox=dict(boxstyle='round,pad=0.22', fc='#EDF1F4', ec='#C7D1D9', lw=0.4))
ax.text(0.02, -0.28, 'Gaussian width and $l_\\mathrm{TF}$ per element', color=INK, fontsize=5.5)

# (5) time-dependent drive ------------------------------------------------
ax = fig.add_subplot(G[2, 1])
t = np.linspace(0, 2, 1200)
ax.plot(t, 2*np.abs(2*(t-np.floor(t+0.5)))-1, color=BLUE, lw=1.0)
ax.plot(t, 0.60*np.sign(np.sin(2*np.pi*t)), color=VERM, lw=1.0, ls=(0,(3.5,1.5)))
ax.plot(t, 0.80*np.sin(2*np.pi*t), color='#009E73', lw=1.0, ls=(0,(1.2,1.2)))
ax.axhline(0, color='#DDDDDD', lw=0.4, zorder=0)
ax.set_xlim(0,2); ax.set_ylim(-1.5,1.5); ax.set_aspect('auto')
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color('#BBBBBB')
ax.set_title("arbitrary time-dependent drive", fontsize=6.3, pad=2.2)
ax.set_xlabel('time', fontsize=5.9, labelpad=1)
ax.text(0.06, 1.14, 'triangular', color=BLUE, fontsize=5.5)
ax.text(1.02, -1.32, 'square', color=VERM, fontsize=5.5)
ax.text(1.62, 1.14, 'sine', color='#009E73', fontsize=5.5)
ax.text(-0.02, 0.06, '$U(t)$, $Q(t)$', color=INK, fontsize=5.7, rotation=90,
        va='bottom', ha='right', transform=ax.get_yaxis_transform())

out = os.path.join(HERE, "fig_features.pdf")
fig.savefig(out); fig.savefig(out.replace('.pdf','.png'), dpi=230)
print("  wrote", out)
