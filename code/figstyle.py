"""Shared figure style for the manuscript (Elsevier column widths, validated palette).

Palette (validated with the dataviz validator against a white surface):
  categorical  #2a78d6 blue, #eb6834 orange, #1baf7a aqua, #eda100 yellow, #e87ba4 magenta
  ordinal      REPORT #184f95, BOUND #3987e5, REFUSE #86b6ef
  sequential   blue ramp 100-700
Ink: primary #0b0b0b, secondary #52514e, muted #898781, grid #e1e0d9, axis #c3c2b7.
Roles are fixed across figures so a colour always means the same thing.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROUND = Path(__file__).resolve().parent.parent
FIG = ROUND / 'figures'
FIG.mkdir(exist_ok=True)

MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM          # Elsevier single / 1.5 / double column

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
STATE = {'REPORT': '#184f95', 'BOUND': '#3987e5', 'REFUSE': '#86b6ef'}
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
       '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ_CMAP = LinearSegmentedColormap.from_list('seqblue', ['#f4f8fd'] + SEQ)
PRIOR = '#c3c2b7'                                  # "nothing gained": the prior box

# fixed roles
RULE = {'naive': ORANGE, 'blind': AQUA, 'axis': YELLOW, 'certified': BLUE, 'loo': BLUE}
RULE_MARK = {'naive': 'o', 'blind': 's', 'axis': 'D', 'certified': '^', 'loo': '^'}
LAYOUT = {'full': BLUE, 'deck': ORANGE}
BUDGET = {'noise': BLUE, 'alias': ORANGE, 'drift': AQUA, 'prior': PRIOR}


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
        'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.8,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.5,
        'legend.frameon': False, 'legend.handlelength': 1.4, 'legend.borderaxespad': 0.3,
        'lines.linewidth': 1.2, 'lines.markersize': 4.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 300, 'figure.dpi': 150, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.titlepad': 4,
        'mathtext.fontset': 'dejavusans',
    })


def panel(ax, letter, title=None, x=-0.02, y=1.02):
    """Panel letter (bold) followed by an optional short title, left-aligned above the axes."""
    txt = f'({letter})' + (f'  {title}' if title else '')
    ax.text(x, y, txt, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5,
            fontweight='bold', color=INK)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02,
                    dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print('saved', name)
