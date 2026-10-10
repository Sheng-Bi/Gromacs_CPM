"""One plotting style for every figure in the paper.

Import and call `apply()` at the top of a figure script, then take colours from
the role dictionaries below rather than writing hex codes inline.  The point is
that the same thing is the same colour, marker and width wherever it appears.
"""
import matplotlib as mpl

# --- roles -------------------------------------------------------------------
# the four constant-potential treatments, wherever they are compared
SOLVER = {"matinv":  "#E8590C",     # matrix inversion      (orange)
          "matfree": "#087F5B",     # matrix-free CG        (teal)
          "extlag":  "#9C36B5",     # extended Lagrangian   (purple)
          "lammps":  "#1864AB"}     # LAMMPS ELECTRODE      (blue)
# the two codes, where the comparison is code against code
CODE = {"gmx": "#1864AB", "lmp": "#868E96"}
# the two electrode models in the nanotube study
MODEL = {"cpm": "#1864AB", "fixed": "#495057"}
# accents used for annotation and secondary curves
INK, MUTED, GRID = "#212529", "#868E96", "#DEE2E6"
ACCENT = "#C92A2A"

# --- marks -------------------------------------------------------------------
MARK = ("o", "s", "D", "^")          # in order of appearance within a panel
LW = 1.1                             # data line width
MS = 4.2                             # marker size
ELW, CAPS = 0.7, 1.8                 # error bars

def apply(base=8.6):
    """Base size in points. Figures are drawn at the width they are placed at in the
    manuscript, so a point here is a point on the page; keep this close to the 10 pt
    body text (a couple of points smaller at most)."""
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": base,
        "axes.linewidth": 0.7, "axes.labelsize": base + 0.6, "axes.titlesize": base + 0.6,
        "xtick.labelsize": base - 0.4, "ytick.labelsize": base - 0.4,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "xtick.top": True, "ytick.right": True,
        "legend.fontsize": base - 1.0, "legend.frameon": False,
        "lines.linewidth": LW, "lines.markersize": MS,
        "savefig.dpi": 600, "figure.dpi": 110,
    })


def tag(ax, letter, dx=-0.42, dy=0.05, size=10):
    """Panel letter outside the axes, up and to the left of its top-left corner.

    Offsets are in inches, so the letter clears the y label whatever the data
    range is, and it lands in the same place in every figure.
    """
    from matplotlib.transforms import ScaledTranslation
    ax.text(0.0, 1.0, f"({letter})",
            transform=ax.transAxes + ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans),
            fontsize=size, fontweight="bold", ha="left", va="bottom")
