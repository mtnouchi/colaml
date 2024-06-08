from itertools import chain

import numpy as np
from numpy import newaxis as newax
from scipy.linalg import toeplitz


def min_changes(phytbl):
    nst = max(chain.from_iterable(phytbl.to_dict(copy=False).values())) + 1
    diff = np.abs(np.arange(nst) - np.arange(nst)[:, newax])

    minchanges = []  # (m, a, n)
    for current, partial_cols in zip(phytbl.tree.nodes, phytbl.inside_encoding):
        if current.is_leaf():
            tmp = np.where(np.eye(nst, dtype=bool), 0, np.inf)
        else:
            tmp = np.sum(
                [
                    (minchanges[ch][:, newax, uniq] + diff[:, :, newax]).min(
                        axis=0
                    )  # an,ab->bn
                    for ch, uniq in zip(current.children_idxs, partial_cols.uniqs)
                ],
                axis=0,
            )
        minchanges.append(tmp)

    else:
        return tmp.min(axis=0)[partial_cols.codes].astype(int)  # .sum()


def mean_mp_changes(phytbl):
    nst = max(chain.from_iterable(phytbl.to_dict(copy=False).values())) + 1
    tree = phytbl.tree
    ncols, nnodes = phytbl.ncols, tree.nnodes

    diff = toeplitz(range(nst))

    nuniqcols = [
        getattr(partial_cols.uniqs, 'shape', (-1, nst))[1]
        for partial_cols in phytbl.inside_encoding
    ]
    min_changes = [
        np.zeros((nst, nuniqcols_m)) for nuniqcols_m in nuniqcols
    ]  # (m, a, n)
    aux_min_changes = [
        np.zeros((nst, nuniqcols_m)) for nuniqcols_m in nuniqcols
    ]  # (m, a, n)
    inside_ways = [
        np.empty((nst, nuniqcols_m), dtype=object) for nuniqcols_m in nuniqcols
    ]  # (m, a, n)
    aux_inside_ways = [
        np.empty((nst, nuniqcols_m), dtype=object) for nuniqcols_m in nuniqcols
    ]  # (m, b, n)
    outside_ways = np.empty((nnodes, nst, ncols), dtype=object)  # (m, b, n)
    aux_outside_ways = np.empty((nnodes, nst, ncols), dtype=object)  # (m, b, n)

    argmin_changes = [
        np.empty((nst, nst, nuniqcols_m), dtype=bool) for nuniqcols_m in nuniqcols
    ]

    tip_ways = np.eye(nst, dtype=object)  # avoid overflow
    tip_cost = np.where(np.eye(nst, dtype=bool), 0, np.inf)
    for current_idx in range(nnodes):
        children_idxs = tree.nodes[current_idx].children_idxs
        if children_idxs:
            children_uniq_codes = phytbl.inside_encoding[current_idx].uniqs
            ch, cuniq = children_idxs[0], children_uniq_codes[0]
            np.copyto(min_changes[current_idx], aux_min_changes[ch][:, cuniq])
            np.copyto(inside_ways[current_idx], aux_inside_ways[ch][:, cuniq])
            for ch, cuniq in zip(children_idxs[1:], children_uniq_codes[1:]):
                min_changes[current_idx] += aux_min_changes[ch][:, cuniq]
                inside_ways[current_idx] *= aux_inside_ways[ch][:, cuniq]
        else:
            min_changes[current_idx] = tip_cost
            inside_ways[current_idx] = tip_ways

        if current_idx == nnodes - 1:
            break

        tmp_changes = (
            min_changes[current_idx][:, newax, :] + diff[:, :, newax]
        )  # (a, b, n)
        aux_min_changes[current_idx][:, :] = tmp_changes.min(axis=0)
        argmin_changes[current_idx][:, :, :] = (
            tmp_changes == aux_min_changes[current_idx]
        )
        aux_inside_ways[current_idx][:, :] = np.where(
            argmin_changes[current_idx], inside_ways[current_idx][:, newax, :], 0
        ).sum(axis=0)

    outside_ways[-1, :, :] = (min_changes[-1] == min_changes[-1].min(axis=0))[
        :, phytbl.inside_encoding[-1].codes
    ]
    total_ways = (
        inside_ways[-1][:, phytbl.inside_encoding[-1].codes] * outside_ways[-1]
    ).sum(axis=0)
    for current_idx in reversed(range(nnodes - 1)):
        current = tree.nodes[current_idx]
        current_pcols = phytbl.inside_encoding[current_idx]
        parent_idx = current.parent_idx
        parent_pcols = phytbl.inside_encoding[parent_idx]
        cur2par_codes = parent_pcols.uniqs[current.sibling_ordinal]

        aux_outside_ways[current_idx][:, :] = (
            outside_ways[parent_idx]
            * (
                inside_ways[parent_idx]
                // aux_inside_ways[current_idx][:, cur2par_codes]
            )[:, parent_pcols.codes]
        )  # bn,(bn~,bn~~)->bn

        outside_ways[current_idx, :, :] = np.where(
            argmin_changes[current_idx][:, :, current_pcols.codes],
            aux_outside_ways[current_idx][newax, :, :],
            0,
        ).sum(axis=1)

        assert np.alltrue(
            total_ways
            == (
                outside_ways[current_idx]
                * inside_ways[current_idx][:, current_pcols.codes]
            ).sum(axis=0)
        )

    mean_mp_changes = []
    for current_idx in range(nnodes - 1):
        current_pcols = phytbl.inside_encoding[current_idx]

        trans_ways = (
            inside_ways[current_idx][:, newax, :] * argmin_changes[current_idx]
        )[:, :, current_pcols.codes] * aux_outside_ways[current_idx][newax, :, :]
        assert np.alltrue(trans_ways.sum(axis=(0, 1)) == total_ways)
        mean_mp_changes.append(
            (trans_ways * diff[:, :, newax]).sum(axis=(0, 1)) / total_ways
        )

    return np.array(mean_mp_changes, dtype=float)
