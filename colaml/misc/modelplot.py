import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib.collections import LineCollection, PatchCollection

from ..substModels import BDARD, MarkovModulatedBDARD


def plot_substmodel(model, state_labels=None, ax=None, **heatmapkw):
    if isinstance(model, BDARD):
        ax = _plot_standard_substmodel(
            model, state_labels=state_labels, ax=ax, **heatmapkw
        )
    elif isinstance(model, MarkovModulatedBDARD):
        ax = _plot_Markov_modulated_substmodel(
            model, state_labels=state_labels, ax=ax, **heatmapkw
        )
    elif isinstance(model, tuple):
        ax = _plot_Mirage_substmodels(
            model, state_labels=state_labels, ax=ax, **heatmapkw
        )
    else:
        raise NotImplementedError(f'Unsupported model: {type(model)}')

    return ax


def _plot_standard_substmodel(model, state_labels=None, ax=None, **heatmapkw):
    ax = ax or plt.gca()
    R = model.R
    mask = np.isclose(R, 0) | np.less(R, 0)
    heatmapkw.setdefault('cbar', False)
    heatmapkw.setdefault('annot', True)
    heatmapkw.setdefault('xticklabels', True)
    heatmapkw.setdefault('yticklabels', True)
    heatmapkw.setdefault('cmap', 'copper')
    heatmapkw.setdefault(
        'norm', colors.LogNorm(vmin=R[~mask].min(), vmax=R[~mask].max())
    )
    sns.heatmap(R, mask=mask, ax=ax, **heatmapkw)
    ax.set_xlabel('from', visible=True)
    ax.set_ylabel('to', visible=True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_facecolor('whitesmoke')

    if state_labels:
        ax.set(xticklabels=state_labels, yticklabels=state_labels)

    return ax


def _plot_Mirage_substmodels(models, state_labels=None, ax=None, **heatmapkw):
    from scipy.linalg import block_diag

    ax = ax or plt.gca()
    R = block_diag(*(model.R for model in models))
    mask = np.isclose(R, 0) | np.less(R, 0)
    heatmapkw.setdefault('cbar', False)
    heatmapkw.setdefault('annot', True)
    heatmapkw.setdefault('xticklabels', True)
    heatmapkw.setdefault('yticklabels', True)
    heatmapkw.setdefault('cmap', 'copper')
    heatmapkw.setdefault(
        'norm', colors.LogNorm(vmin=R[~mask].min(), vmax=R[~mask].max())
    )
    sns.heatmap(R, mask=mask, ax=ax, **heatmapkw)
    ax.set_xlabel('from', visible=True)
    ax.set_ylabel('to', visible=True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_facecolor('whitesmoke')

    if state_labels:
        ax.set(xticklabels=state_labels, yticklabels=state_labels)

    return ax


def _plot_Markov_modulated_substmodel(model, state_labels=None, ax=None, **heatmapkw):
    nchrst, ncat = model.ncharstates, model.ncategories
    # plot heatmap
    ax = _plot_standard_substmodel(
        model, state_labels=None, ax=ax, **heatmapkw
    )  # NB: state_labels=None is intentional

    # plot additional elements
    ## 1. frames to highlight transitions within each category
    ## 2. state labels colored by category
    fc, diagfr, xrect, yrect, xsep, ysep = [], [], [], [], [], []
    ax_x = ax.inset_axes([0, 1, 1, 0.06], transform=ax.transAxes, sharex=ax)
    ax_x.set_axis_off()
    ax_y = ax.inset_axes([-0.06, 0, 0.06, 1], transform=ax.transAxes, sharey=ax)
    ax_y.set_axis_off()
    for k in range(ncat):
        fc.append(f'C{k}')
        diagfr.append(plt.Rectangle((k * nchrst, k * nchrst), nchrst, nchrst))
        yrect.append(plt.Rectangle((0, k * nchrst), 1, nchrst))
        xrect.append(plt.Rectangle((k * nchrst, 0), nchrst, 1))
        ysep.extend(
            [[(0, k * nchrst + i), (1, k * nchrst + i)] for i in range(1, nchrst)]
        )
        xsep.extend(
            [[(k * nchrst + i, 0), (k * nchrst + i, 1)] for i in range(1, nchrst)]
        )
        for i, lab in enumerate(state_labels or range(nchrst)):
            ax_y.text(0.5, k * nchrst + i + 0.5, lab, ha='center', va='center')
            ax_x.text(k * nchrst + i + 0.5, 0.5, lab, ha='center', va='center')
    else:
        ax.add_collection(
            PatchCollection(
                diagfr,
                ec='k',
                fc='none',
                lw=plt.rcParams['lines.linewidth'] * 1.5,
                clip_on=False,
            )
        )
        ax_x.add_collection(LineCollection(xsep, ec='w'))
        ax_x.add_collection(PatchCollection(xrect, ec='k', fc=fc, clip_on=False))
        ax_y.add_collection(LineCollection(ysep, ec='w'))
        ax_y.add_collection(PatchCollection(yrect, ec='k', fc=fc, clip_on=False))

    ## 3. partitions between categories
    catsep = []
    for k in range(1, ncat):
        catsep.extend(
            [
                [(nchrst * k, 0), (nchrst * k, nchrst * ncat)],
                [(0, nchrst * k), (nchrst * ncat, nchrst * k)],
            ]
        )
    ax.add_collection(LineCollection(catsep, ls='--', color='k'))
    ax.yaxis.set_tick_params(labelright=False, right=False)
    ax.xaxis.set_tick_params(labelbottom=False, bottom=False)

    return ax
