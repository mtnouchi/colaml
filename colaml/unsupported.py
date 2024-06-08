import warnings
from copy import deepcopy
from functools import cached_property
from inspect import signature
from itertools import zip_longest

import numpy as np
from numpy import newaxis as newax
from numpy.core.multiarray import c_einsum
from scipy.linalg import expm
from scipy.special import xlogy

from .fitting import StopCriteria
from .phyTables import PostorderSerializedTree, ReconPhyTable
from .treeModels import AbstTreeModel, Dirichlet, _to_ndarray_of_shape


class BranchwiseRateStats:
    @staticmethod
    def __hyper_params(model, phytbl):
        return (
            model._ncompoundstates,
            model.ncharstates,
            phytbl.ncols,
            phytbl.tree.nnodes,
        )

    def __init__(self, model, phytbl):
        ncmpst, nchrst, ncols, nnodes = self.__hyper_params(model, phytbl)
        nuniqcols = [
            getattr(partial_cols.uniqs, 'shape', (-1, nchrst))[1]
            for partial_cols in phytbl.inside_encoding
        ]
        self._model = model
        self._phytbl = phytbl
        self.branchP = np.empty((nnodes - 1, ncmpst, ncmpst))  # (m, a, b)
        self.stateP = np.empty((nnodes, ncmpst))  # (m, a)
        self.aux_insideP = [
            np.empty((ncmpst, nuniqcols_m)) for nuniqcols_m in nuniqcols
        ]  # (m, b, n)
        self.insideP = [
            np.empty((ncmpst, nuniqcols_m)) for nuniqcols_m in nuniqcols
        ]  # (m, a, n)
        self.scale = [np.empty(nuniqcols_m) for nuniqcols_m in nuniqcols]  # (m, n)
        self.aux_outsideP = np.empty((nnodes, ncmpst, ncols))  # (m, b, n)
        self.outsideP = np.empty((nnodes, ncmpst, ncols))  # (m, b, n)
        self.tfdns = np.zeros((nnodes - 1, ncmpst, ncmpst))  # (m, a, b)
        self.col_loglik = np.empty(ncols)  # (n,)

    def compute(self):
        model, phytbl = self._model, self._phytbl
        ncmpst = model._ncompoundstates
        tree = phytbl.tree
        nnodes = tree.nnodes

        ## m: node or branch upon it
        ## n: column (cached)
        ## a, b, i, j: state

        # calculate state transtion probs. in each branch
        self.branchP = expm(
            [
                dist * sbstmdl.R
                for dist, sbstmdl in zip(tree.branch_lengths, model.substmodels)
            ]
        )  # mab

        # calculate state probability for each nodes (serves as weights in scaling)
        self.stateP[-1] = model.root_probs
        for current_idx in reversed(range(nnodes - 1)):
            parent_idx = tree.nodes[current_idx].parent_idx
            np.dot(
                self.branchP[current_idx],
                self.stateP[parent_idx],
                out=self.stateP[current_idx],
            )  # ab,b->a

        # calculate inside variables
        for current_idx in range(nnodes):
            children_idxs = tree.nodes[current_idx].children_idxs
            if children_idxs:
                children_uniq_codes = phytbl.inside_encoding[current_idx].uniqs
                ch, cuniq = children_idxs[0], children_uniq_codes[0]
                np.copyto(self.insideP[current_idx], self.aux_insideP[ch][:, cuniq])
                for ch, cuniq in zip(children_idxs[1:], children_uniq_codes[1:]):
                    self.insideP[current_idx] *= self.aux_insideP[ch][:, cuniq]
            else:
                np.copyto(self.insideP[current_idx], model._compound_state_flags)

            np.dot(
                self.stateP[current_idx],
                self.insideP[current_idx],
                out=self.scale[current_idx],
            )  # b,bn->n
            self.insideP[current_idx] /= self.scale[current_idx][newax, :]

            if current_idx != nnodes - 1:
                np.dot(
                    self.branchP[current_idx].T,
                    self.insideP[current_idx],
                    out=self.aux_insideP[current_idx],
                )  # ab,an->bn

        # calculate ouside variables
        self.outsideP[-1] = model.root_probs[:, newax]
        with np.errstate(divide='warn', invalid='warn'):
            for current_idx in reversed(range(nnodes - 1)):
                current = tree.nodes[current_idx]
                parent_idx = current.parent_idx
                parent_pcols = phytbl.inside_encoding[parent_idx]
                cur2par_codes = parent_pcols.uniqs[current.sibling_ordinal]

                with warnings.catch_warnings(record=True) as warn_records:
                    warnings.simplefilter('always')
                    self.aux_outsideP[current_idx] = (
                        self.outsideP[parent_idx]
                        * (
                            self.insideP[parent_idx]
                            / self.aux_insideP[current_idx][:, cur2par_codes]
                        )[:, parent_pcols.codes]
                    )  # bn,(bn~,bn~~)->bn
                    if warn_records:
                        np.nan_to_num(
                            self.aux_outsideP[current_idx], copy=False
                        )  # just first-aid

                np.dot(
                    self.branchP[current_idx],
                    self.aux_outsideP[current_idx],
                    out=self.outsideP[current_idx],
                )  # ab,bn->an

        # obtain likelihoods and posterior probs
        self.post_rootP = c_einsum('an,a->an', self.insideP[-1], model.root_probs)[
            :, phytbl.inside_encoding[-1].codes
        ]

        self.col_loglik.fill(0)
        for sc, partial_cols in zip(self.scale, phytbl.inside_encoding):
            self.col_loglik += np.log(sc)[partial_cols.codes]

        # calculate sufficient statistics
        self.tfdns.fill(0)
        for current_idx in range(nnodes - 1):
            dist = tree.branch_lengths[current_idx]
            partial_cols = phytbl.inside_encoding[current_idx]
            sbstmdl = model.substmodels[current_idx]
            self.tfdns[current_idx, sbstmdl._mask_nonzero] = np.dot(
                # 'ab,abh->h'; h: non-zero (i, j) elememnts
                np.dot(
                    self.insideP[current_idx][:, partial_cols.codes],
                    self.aux_outsideP[current_idx].T,
                ).reshape(-1),
                sbstmdl.sufficient_stats(dist)[:, :, sbstmdl._mask_nonzero].reshape(
                    ncmpst**2, -1
                ),
            )


class BranchwiseTreeModel(AbstTreeModel):
    _params_signature = signature(lambda root_probs, **substmodels_params: ...)

    def __init__(
        self,
        nbranches,
        ncharstates,
        substmodelclass,
        prior_kw=None,
        init_params_method=None,
        init_params_kw=None,
    ):
        # inspect prior_kw and set priors
        assume_priors = prior_kw is not None
        if assume_priors:
            bound_args = self._params_signature.bind(**prior_kw)
            prior_kw = bound_args.kwargs
            (rootP_prior,) = bound_args.args
            rootP_prior = Dirichlet(_to_ndarray_of_shape(rootP_prior, (ncharstates,)))
        else:
            rootP_prior = Dirichlet(np.ones(ncharstates))
        self.__assume_priors = assume_priors
        self.__rootP_prior = rootP_prior

        # define parameters (and init them with NaN)
        self.__substmodels = [
            substmodelclass(
                ncharstates,
                prior_kw=prior_kw,
                init_params_method='skip',
                ss_method='eig',  # Whichever is chosen, it does not seem to have much effect on results.
            )
            for _ in range(nbranches)
        ]
        self.__rootP = np.full((ncharstates,), np.nan)
        self.__ncharstates = ncharstates

        # init parameters
        super().init_params(
            init_params_method=init_params_method, init_params_kw=init_params_kw
        )

    def init_params_manual(self, root_probs, substmodels_params):
        if len(substmodels_params) != self.nbranches:
            raise ValueError(
                f'The length of \'substmodels_params\' must be {self.nbranches}. '
                f'Got: {len(substmodels_params)}'
            )
        self.update(root_probs=root_probs, substmodels_params=None)
        for sbstmdl, kwargs in zip_longest(self.__substmodels, substmodels_params):
            sbstmdl.init_params_manual(**kwargs)

    def init_params_random(self, rng, distr_kw=None):
        if self.__assume_priors:
            if distr_kw is not None:
                warnings.warn(
                    'distr_kw is ignored because this model already assumes priors.'
                )
            rootP_distr = self.__rootP_prior
        else:
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('root_probs')
            bound_args = self._params_signature.bind(**distr_kw)
            distr_kw = bound_args.kwargs
            (rootP_distr,) = bound_args.args
            if rootP_distr is None:
                rootP_distr = Dirichlet(np.ones(self.ncharstates))
            else:
                rootP_distr = Dirichlet(
                    _to_ndarray_of_shape(rootP_distr, (self.ncharstates,))
                )

        for sbstmdl in self.__substmodels:
            sbstmdl.init_params_random(rng, distr_kw)

        self.update(root_probs=rng.dirichlet(rootP_distr))

    def init_params_emEM(
        self,
        phytbl,
        rng,
        ntrial=10,
        max_rounds=100,
        stop_criteria=StopCriteria(1e-4, 1e-4),
        show_init_progress=False,
        distr_kw=None,
        **fit_kw,
    ):
        if not self.__assume_priors:
            tot_brlen = phytbl.tree.branch_lengths.sum()
            mp_rate = phytbl.min_changes.sum() / phytbl.ncols / tot_brlen
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('rates', (3, mp_rate / 2))

        super().init_params_emEM(
            phytbl=phytbl,
            rng=rng,
            ntrial=ntrial,
            show_init_progress=show_init_progress,
            distr_kw=distr_kw,
            max_rounds=max_rounds,
            stop_criteria=stop_criteria,
            **fit_kw,
        )

    @property
    def nbranches(self):
        return len(self.__substmodels)

    @property
    def substmodels(self):
        return self.__substmodels

    @property
    def ncharstates(self):
        return self.__ncharstates

    @property
    def _ncompoundstates(self):
        return self.ncharstates

    @property
    def root_probs(self):
        return self.__rootP

    @property
    def priors(self):
        return dict(
            root_probs=self.__rootP_prior,
            substmodels=[sbstmdl.priors for sbstmdl in self.substmodels],
        )

    @cached_property
    def _compound_state_flags(self):
        csflags = np.eye(self.ncharstates)
        csflags.setflags(write=False)
        return csflags

    def _empty_stats(self, phylodata):
        return BranchwiseRateStats(self, phylodata)

    # With typecheck
    def update(self, root_probs=None, substmodels_params=None):
        if root_probs is None:
            root_probs = self.__rootP
        else:
            root_probs = _to_ndarray_of_shape(root_probs, self.__rootP.shape)

        if substmodels_params is None:
            substmodels_params = []
        else:
            if len(substmodels_params) != self.nbranches:
                raise ValueError(
                    f'The length of \'substmodels_params\' should be {self.nbranches}. Got: {len(substmodels_params)}'
                )

        self._update(root_probs, substmodels_params)

    # Without typecheck
    def _update(self, root_probs, substmodels_params):
        self.__rootP = np.array(root_probs)
        for sbstmdl, kwargs in zip(self.__substmodels, substmodels_params):
            sbstmdl.update(**kwargs or {})

    @property
    def flat_params(self):
        return np.concatenate(
            (self.root_probs, *[sbstmdl.flat_params for sbstmdl in self.substmodels])
        )

    def _decompress_flat_params(self, p):
        nchrst = self.ncharstates
        return dict(
            root_probs=p[:nchrst],
            substmodels_params=[
                sbstmdl._decompress_flat_params(p_sbstmdl)
                # for sbstmdl, p_sbstmdl in zip(self.__substmodels, p[nchrst:].reshape(-1,2,nchrst-1))
                for sbstmdl, p_sbstmdl in zip(
                    self.__substmodels, p[nchrst:].reshape(self.nbranches, -1)
                )
            ],
        )

    def _get_next_params(self, stats):
        rootP = stats.post_rootP.sum(axis=1) + (self.__rootP_prior - 1)
        rootP /= rootP.sum()

        substmodels_params = [
            sbstmdl._get_next_params(tfdns)
            for sbstmdl, tfdns in zip(self.__substmodels, stats.tfdns)
        ]

        return dict(root_probs=rootP, substmodels_params=substmodels_params)

    def _log_prior_prob(self):
        logP_root_prior = xlogy(self.__rootP_prior - 1, self.__rootP).sum()

        return logP_root_prior + sum(
            sbstmdl._log_prior_prob() for sbstmdl in self.__substmodels
        )

    def fit_EM(
        self,
        phylodata,
        stop_atol=1e-8,
        stop_rtol=1e-8,
        max_rounds=100,
        show_progress=False,
        logger=None,
    ):
        return super().fit_EM(
            phylodata=phylodata,
            stop_atol=stop_atol,
            stop_rtol=stop_rtol,
            max_rounds=max_rounds,
            show_progress=show_progress,
            logger=logger,
        )

    def fit_epsR_EM(
        self,
        phylodata,
        stop_atol=1e-8,
        stop_rtol=1e-8,
        max_rounds=1000,
        show_progress=False,
        logger=None,
        restart_params=(1, 0.1),
        restart_check_intv=100,
    ):
        super().fit_epsR_EM(
            phylodata=phylodata,
            stop_atol=stop_atol,
            stop_rtol=stop_rtol,
            max_rounds=max_rounds,
            show_progress=show_progress,
            restart_params=restart_params,
            restart_check_intv=restart_check_intv,
            logger=logger,
        )

    def fit_parabolic_EM(
        self,
        phylodata,
        stop_atol=1e-8,
        stop_rtol=1e-8,
        max_rounds=1000,
        show_progress=False,
        logger=None,
        grid_params=(0.1, 1.5),
        heuristics=False,
    ):
        return super().fit_parabolic_EM(
            phylodata=phylodata,
            stop_atol=stop_atol,
            stop_rtol=stop_rtol,
            max_rounds=max_rounds,
            show_progress=show_progress,
            logger=logger,
            grid_params=grid_params,
            heuristics=heuristics,
        )

    def _reconstruct_joint(self, phytbl):
        tree = phytbl.tree
        ncmpst, ncols, nnodes = self._ncompoundstates, phytbl.ncols, tree.nnodes

        log_branchP = np.log(
            expm(
                [
                    sbstmdl.R * dist
                    for sbstmdl, dist in zip(self.__substmodels, tree.branch_lengths)
                ]
            )
        )  # (m, a, b)

        log_rootP = np.log(self.root_probs)

        tmp_recon = np.zeros((nnodes - 1, ncmpst, ncols), dtype=int)  # (m, j, n)
        loglik = np.zeros((nnodes - 1, ncmpst, ncols))  # (m, j, n)

        recon_charstates = np.zeros((ncols, nnodes), dtype=int)  # (n, m)

        tip_mask = np.ma.log(self._compound_state_flags).filled(-np.inf)
        for current_idx in range(nnodes - 1):
            current = tree.nodes[current_idx]
            if current.is_leaf():
                a = phytbl.inside_encoding[current_idx].codes  # (n, )
                tmp = log_branchP[current_idx, :, :, newax] + tip_mask[:, newax, :]
                tmp_recon[current_idx, :, :] = tmp.argmax(axis=0)[:, a]
                loglik[current_idx, :, :] = tmp.max(axis=0)[:, a]
            else:
                tmp = (
                    log_branchP[current_idx, :, :, newax]
                    + np.sum(
                        [
                            loglik[child_idx, :, :]
                            for child_idx in current.children_idxs
                        ],
                        axis=0,
                    )[:, newax, :]
                )  # (j, i, n)
                tmp_recon[current_idx, :, :] = tmp.argmax(axis=0)
                loglik[current_idx, :, :] = tmp.max(axis=0)

        recon_charstates[:, -1] = (
            log_rootP[:, newax]
            + np.sum(
                [loglik[child_idx, :, :] for child_idx in tree.nodes[-1].children_idxs],
                axis=0,
            )
        ).argmax(axis=0)

        for current_idx in reversed(range(nnodes - 1)):
            parent_idx = tree.nodes[current_idx].parent_idx
            recon_charstates[:, current_idx] = tmp_recon[
                current_idx, recon_charstates[:, parent_idx], np.arange(ncols)
            ]

        return recon_charstates

    def _reconstruct_marginal(self, phytbl):
        stats = self.sufficient_stats(phytbl)
        insideP = np.array(
            [
                insideP_m[:, partial_cols.codes]
                for insideP_m, partial_cols in zip(
                    stats.insideP, phytbl.inside_encoding
                )
            ]
        )
        outsideP = stats.outsideP
        recon_charstates = (insideP * outsideP).argmax(axis=1).T  # (m, n)

        return recon_charstates

    def _get_model_info(self):
        return dict(
            treemodel=type(self),
            tm_args=(self.nbranches, self.ncharstates, type(self.substmodels[0])),
            tm_params=self.flat_params,
        )

    def _wrap_reconstruction(self, phytbl, recon_charstates, method=None):
        tree2 = deepcopy(phytbl.tree)
        tree2.fill_names(ignore_tips=True)
        return ReconPhyTable(
            dict(zip(tree2.names, recon_charstates.T)),
            tree2,
            metadata_kw=dict(method=method, **self._get_model_info()),
        )

    def reconstruct_joint(self, phylodata):
        return self._wrap_reconstruction(
            phylodata, self._reconstruct_joint(phylodata), 'joint'
        )

    def reconstruct_marginal(self, phylodata):
        return self._wrap_reconstruction(
            phylodata, self._reconstruct_marginal(phylodata), 'marginal'
        )

    def simulate(self, tree, N, rng):
        tree2 = PostorderSerializedTree(tree)
        tree2.fill_names(ignore_tips=True)
        charstates = {}
        # substmodels are supposed to be sorted in postorder,
        # while here we need to traverse the tree in preorder.
        sbstmdls = dict(zip(tree2.names, self.__substmodels))
        preorder = iter(tree2.to_ete3().traverse('preorder'))

        # init
        root = next(preorder)
        charstates[root.name] = rng.choice(self.ncharstates, size=N, p=self.root_probs)

        # recursion
        for node in preorder:
            P = sbstmdls[node.name].P(node.dist)
            pa_charstates = charstates[node.up.name]
            my_charstates = np.empty(N, dtype=int)
            for st in set(pa_charstates):
                mask = pa_charstates == st
                my_charstates[mask] = rng.choice(
                    *P[:, st].shape, size=np.count_nonzero(mask), p=P[:, st]
                )
            charstates[node.name] = my_charstates

        return ReconPhyTable(
            charstates,
            tree2,
            metadata_kw=dict(method='simulated', **self._get_model_info()),
        )
