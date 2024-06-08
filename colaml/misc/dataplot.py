from collections import Counter
from itertools import chain, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import DrawingArea
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import IdentityTransform

from . import phyplot


def draw_extant(phytbl, lmax=None, phyloorient='row', phylotreekw=None, **clustermapkw):
    OGs = pd.DataFrame(phytbl.to_dict(copy=False))
    lmax = lmax or OGs.values.max()
    default_cmap = ListedColormap(
        [*map(plt.get_cmap('bone_r', lmax + 1), range(1, lmax + 1))]
    )

    phylotreekw = phylotreekw or {}
    for config in dict(node_text=None, show_confidence=False).items():
        phylotreekw.setdefault(*config)

    for config in dict(
        figsize=(20, 10),
        xticklabels=False,
        yticklabels=False,
        cmap=default_cmap,
        cbar_pos=(0.02, 0.85, 0.03, 0.1),
        cbar_kws=dict(ticks=range(1, lmax + 1)),
    ).items():
        clustermapkw.setdefault(*config)

    if phyloorient == 'row':
        clustermapkw.setdefault('dendrogram_ratio', (0.2, 0.1))
        clst = sns.clustermap(
            OGs.T,
            mask=OGs.T.eq(0),
            col_cluster=True,
            row_cluster=False,
            vmin=0.5,
            vmax=lmax + 0.5,
            **clustermapkw,
        )
        phyplot.draw(
            phytbl.tree.to_ete3(),
            horizontal=True,
            loffs=0.5,
            lscale=1,
            **phylotreekw,
            ax=clst.ax_row_dendrogram,
        )
        clst.ax_row_dendrogram.sharey(clst.ax_heatmap)
        clst.ax_row_dendrogram.set_ylim(0, phytbl.tree.ntips)

    elif phyloorient == 'col':
        clustermapkw.setdefault('dendrogram_ratio', (0.1, 0.2))
        clst = sns.clustermap(
            OGs,
            mask=OGs.eq(0),
            col_cluster=False,
            row_cluster=True,
            vmin=0.5,
            vmax=lmax + 0.5,
            **clustermapkw,
        )
        phyplot.draw(
            phytbl.tree.to_ete3(),
            horizontal=False,
            loffs=0.5,
            lscale=1,
            **phylotreekw,
            ax=clst.ax_col_dendrogram,
        )
        clst.ax_col_dendrogram.sharex(clst.ax_heatmap)
        clst.ax_col_dendrogram.set_xlim(0, phytbl.tree.ntips)

    else:
        raise ValueError(f'Unknown phyloorient: {phyloorient}')

    return clst


## PROVISIONAL
def _iter_rect_statepatches(wsize, heights, drop_zero, wcenter=0, hoffs=0, cmap=None):
    cmap = plt.get_cmap(cmap)
    start = bool(drop_zero)
    for i, (h, top) in enumerate(
        zip(heights[start:], heights[start:][::-1].cumsum()[::-1]), start=start
    ):
        yield Rectangle(
            (-wsize / 2 + wcenter, top + hoffs),
            wsize,
            -h,
            fc=cmap(i),
            rotation_point=(0, 0),
        )


## PROVISIONAL
def draw_reconstruction(
    recon,
    major='states',
    minor=None,
    drop_zero_state=True,
    horizontal=False,
    style='bar',
    nmajor=None,
    nminor=None,
    cmap_major=None,
    cmap_minor=None,
    size=(0.8, None),
    ax=None,
):
    # resolve configs
    if major is None:
        raise ValueError('\'major\' should not be None.')

    config = dict(
        major=dict(n=nmajor, cmap=cmap_major), minor=dict(n=nminor, cmap=cmap_minor)
    )

    for which, dtype in dict(major=major, minor=minor).items():
        if dtype == 'states' or dtype is None:
            data = recon.to_dict()
            n = config[which]['n'] or max(map(max, data.values())) + 1
            cmap = plt.get_cmap(config[which]['cmap'] or plt.get_cmap('bone_r', n))
            drop_zero = drop_zero_state

        elif dtype == 'categories':
            data = recon.otherstates['categories'].to_dict()
            n = config[which]['n'] or max(map(max, data.values())) + 1
            cmap = plt.get_cmap(config[which]['cmap'] or plt.get_cmap('tab10'))
            drop_zero = False

        else:
            raise ValueError(f'Invalid data type: {dtype}')

        config[which].update(dict(data=data, n=n, cmap=cmap, drop_zero=drop_zero))

    data_maj, nmaj, cmap_maj, drop_zero_maj = map(
        config['major'].get, ('data', 'n', 'cmap', 'drop_zero')
    )
    data_mnr, nmnr, cmap_mnr, drop_zero_mnr = map(
        config['minor'].get, ('data', 'n', 'cmap', 'drop_zero')
    )

    # draw tree
    tree = recon.tree.to_ete3()
    plotter = phyplot.standardTreePlotter(
        node_text=None, show_confidence=False, horizontal=horizontal
    )
    ax = plotter.draw(tree, ax=ax)

    # prep for plotting reconstruction
    ## positions to plot
    lpos = plotter._get_lateral_pos(tree)
    vpos = plotter._get_vertical_pos(tree)
    _, tree_height = tree.get_farthest_leaf()
    for node in tree.iter_leaves():
        vpos[node] = tree_height * 1.01

    ## size caluculation
    mat = ax.transData.get_matrix()
    wscale, hscale = mat[(1, 0), (1, 0)] if horizontal else mat[(0, 1), (0, 1)]
    wsize, hscale = (
        size[0] * wscale,
        (size[1] or tree_height / 25) * hscale / recon.ncols,
    )

    # plot reconstruction
    patches, offsets = [], []
    for node in tree.traverse():
        generators = []
        cnt = np.array(
            [
                *map(
                    Counter(zip(data_maj[node.name], data_mnr[node.name])).__getitem__,
                    product(range(nmaj), range(nmnr)),
                )
            ]
        ).reshape(nmaj, nmnr)
        smaj, smnr = bool(drop_zero_maj), bool(drop_zero_mnr)
        generators.append(
            _iter_rect_statepatches(
                wsize, hscale * cnt[:, smnr:].sum(axis=1), drop_zero_maj, cmap=cmap_maj
            )
        )

        if minor is not None:
            hoffs = cnt[smaj:, smnr:].sum()
            for cntsub in cnt[smaj:, :]:
                hoffs -= cntsub[smnr:].sum()
                generators.append(
                    _iter_rect_statepatches(
                        wsize / 4,
                        hscale * cntsub,
                        drop_zero_mnr,
                        wsize * 3 / 8,
                        hscale * hoffs,
                        cmap=cmap_mnr,
                    )
                )

        offs = (vpos[node], lpos[node]) if horizontal else (lpos[node], vpos[node])
        for rect in chain(*generators):
            if style == 'bar' and horizontal:
                rect.set_angle(-90)
            patches.append(rect)
            offsets.append(offs)

    ax.add_collection(
        PatchCollection(
            patches,
            match_original=True,
            offsets=offsets,
            zorder=2,
            offset_transform=ax.transData,
            transform=IdentityTransform(),
        )
    )
    ax.autoscale_view(scalex=horizontal, scaley=not horizontal)

    # add legends
    handles = []
    handles.append(Patch(visible=False, label=major))
    handles.extend([Patch(fc=cmap_maj(i), label=str(i)) for i in range(smaj, nmaj)])
    if minor is not None:
        handles.append(Patch(visible=False, label=minor))
        handles.extend([Patch(fc=cmap_mnr(j), label=str(j)) for j in range(smnr, nmnr)])
    legend = ax.legend(handles=handles)
    for draw_area in legend.findobj(DrawingArea):
        for handle in draw_area.get_children():
            if handle.get_label() in {major, minor}:
                draw_area.set_visible(False)

    return ax
