import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
from inspect import signature
from itertools import zip_longest

import numpy as np
from numpy import newaxis as newax
from numpy.core.multiarray import c_einsum
from scipy.linalg import expm
from scipy.special import xlogy
from tqdm.auto import tqdm

from .fitting import EpsRestartEM, ParabolicEM, PlainEM
from .phyTables import PostorderSerializedTree, ReconPhyTable
from .treeStats import MultipleRateTreeStats, SingleRateTreeStats

# import time

# import logging
# timelogger = logging.getLogger(__name__)

# class Timer:
#     def start(self):
#         self.perf_start   = time.perf_counter()
#         self.proc_start   = time.process_time()
#         return self
#
#     def stop(self):
#         self.perf_end     = time.perf_counter()
#         self.proc_end     = time.process_time()
#         self.perf_counter = (self.perf_end - self.perf_start) * 1000
#         self.process_time = (self.proc_end - self.proc_start) * 1000


class Dirichlet(np.ndarray):
    def __new__(cls, input_array, **kwargs):
        return np.asarray(input_array, **kwargs).view(cls)


def _to_ndarray_of_shape(subject, expected_shape):
    subject = np.array(subject)
    if subject.shape == expected_shape:
        return subject

    msg = f'Invalid shape. Expected: {expected_shape}, but got: {subject.shape}.'
    raise ValueError(msg)


_supported_estimator = dict(
    EM=PlainEM,
    epsR_EM=EpsRestartEM,
    parabolic_EM=ParabolicEM,
)


def describe_estimator(run):
    global _supported_estimator
    run.__doc__ = run.__doc__.format(
        table='\n            '.join(
            f' - {repr(meth_key):<15}: {estimable.__name__}(...)'
            for meth_key, estimable in _supported_estimator.items()
        )
    )
    return run


class AbstTreeModel(ABC):
    @describe_estimator
    def fit(self, phytbl, *, method='EM', show_progress=True, logger=None, **fit_kw):
        """
        fits model parameters to input data

        Constructs an estimator instance of the class corresponding to
        the specified `method` string and calls `run()` on it.

        Parameters
        ----------
        phytbl : ExtantPhyTable
            Input data table
        method : str, default='EM'
            Currently supported methods are:
            {table}
        show_progress : bool, default=True
            Set to `False` to suppress progress bar.
        logger : loging.Logger or None, default=None
            If specifeid, fitting process will be recorded.
        fit_kw :
            Extra keyword arguments passed to the specified fitting method.
            See each fitting class for more details.

        Returns
        -------
        fitting_results :
            Redirects the return value from `run()` of the specified estimator.
            Typically, a boolean value indicating convergence.
        """
        try:
            estimator = _supported_estimator[method](**fit_kw)
        except KeyError as err:
            raise ValueError(
                f'Supported methods are {", ".join(map(repr, _supported_estimator.keys()))}'
            ) from err
        return estimator.run(self, phytbl, show_progress=show_progress, logger=logger)

    def sufficient_stats(self, phytbl):
        stats = self._empty_stats(phytbl)
        stats.compute()
        return stats

    @abstractmethod
    def _empty_stats(self, phytbl):
        pass

    @abstractmethod
    def _log_prior_prob(self):
        pass

    @abstractmethod
    def _get_next_params(self, stats):
        pass

    @property
    @abstractmethod
    def flat_params(self):
        pass

    @abstractmethod
    def _decompress_flat_params(self, flat_params):
        pass

    @abstractmethod
    def _update(self, **kwargs):
        pass

    def init_params(self, init_params_method, init_params_kw):
        if init_params_method is None:
            warnings.warn(
                'Parameters are not initialized. Call either \'init_params_manual\', \'init_params_random\' or \'init_params_emEM\' before calculating statistics or fitting parameters.'
            )
        elif init_params_method == 'skip':
            if init_params_kw is not None:
                warnings.warn('When method=\'skip\', init_params_kw is ignored.')
        elif init_params_method == 'manual':
            self.init_params_manual(**init_params_kw or {})
        elif init_params_method == 'random':
            self.init_params_random(**init_params_kw or {})
        elif init_params_method == 'emEM':
            self.init_params_emEM(**init_params_kw or {})
        else:
            raise ValueError(
                f'Unknown value of \'init_params_method\': {init_params_method}'
            )

    @abstractmethod
    def init_params_manual(self, *params):
        pass

    @abstractmethod
    def init_params_random(self, rng, *priors):
        pass

    def init_params_emEM(
        self, phytbl, rng, ntrial, show_init_progress, distr_kw, **fit_kw
    ):
        best_loglik, best_params = -np.inf, self.flat_params

        stats = self._empty_stats(phytbl)
        for _ in tqdm(range(ntrial), disable=not show_init_progress):
            self.init_params_random(rng, distr_kw)
            self.fit(phytbl, **fit_kw)
            stats.compute()
            # timelogger.debug('\t'.join(['###']*6))
            loglik = stats.col_loglik.sum()  # + self._log_prior_prob()
            if best_loglik < loglik:
                best_loglik, best_params = loglik, self.flat_params
        else:
            self._update(**self._decompress_flat_params(best_params))

    def reconstruct(self, phytbl, method='joint', **kwargs):
        if method == 'joint':
            return self.reconstruct_joint(phytbl)
        elif method == 'marginal':
            return self.reconstruct_marginal(phytbl)
        else:
            raise ValueError(f'Unknown reconstruct method: {method}')

    @abstractmethod
    def reconstruct_joint(self, phytbl):
        pass

    @abstractmethod
    def reconstruct_marginal(self, phytbl):
        pass


class PlainTreeModel(AbstTreeModel):
    _params_signature = signature(lambda root_probs, **substmodel_params: ...)

    def __init__(
        self,
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
        self.__substmodel = substmodelclass(
            ncharstates,
            prior_kw=prior_kw,
            init_params_method='skip',
        )
        self.__rootP = np.full((ncharstates,), np.nan)

        # init parameters
        super().init_params(
            init_params_method=init_params_method, init_params_kw=init_params_kw
        )

    def init_params_manual(self, root_probs, **substmodel_params):
        self.substmodel.init_params_manual(**substmodel_params)
        self.update(root_probs=root_probs)

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

        self.substmodel.init_params_random(rng, distr_kw)
        self._update(root_probs=rng.dirichlet(rootP_distr))

    def init_params_emEM(
        self,
        phytbl,
        rng,
        ntrial=10,
        max_rounds=100,
        stop_criteria=(1e-4, 1e-4),
        show_init_progress=False,
        distr_kw=None,
        **fit_kw,
    ):
        if not self.__assume_priors:
            tot_brlen = phytbl.postorder.brlens.sum()
            mp_rate = phytbl.min_changes.sum() / phytbl.ncols / tot_brlen
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('rates', (3.0, mp_rate / 2.0))

        AbstTreeModel.init_params_emEM(
            self,
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
    def substmodel(self):
        return self.__substmodel

    @property
    def ncharstates(self):
        return self.__substmodel.ncharstates

    @property
    def _ncompoundstates(self):
        return self.ncharstates

    @property
    def R(self):
        return self.__substmodel.R

    @property
    def root_probs(self):
        return self.__rootP

    @property
    def priors(self):
        return dict(
            root_probs=self.__rootP_prior,
            **self.substmodel.priors,
        )

    @cached_property
    def _compound_state_flags(self):
        csflags = np.eye(self.ncharstates)
        csflags.setflags(write=False)
        return csflags

    def _empty_stats(self, phytbl):
        return SingleRateTreeStats(self, phytbl)

    # With typecheck
    def update(self, root_probs=None, **kwargs):
        if root_probs is None:
            root_probs = self.__rootP
        else:
            root_probs = _to_ndarray_of_shape(root_probs, self.__rootP.shape)

        self._update(root_probs, **kwargs)

    # Without typecheck
    def _update(self, root_probs, **kwargs):
        self.__rootP = np.array(root_probs)
        self.__substmodel.update(**kwargs)

    @property
    def flat_params(self):
        return np.concatenate([self.root_probs, self.substmodel.flat_params])

    def _decompress_flat_params(self, p):
        return dict(
            root_probs=p[: self.ncharstates],
            **self.substmodel._decompress_flat_params(p[self.ncharstates :]),
        )

    def _get_next_params(self, stats):
        post_rootP = stats.post_rootP.sum(axis=1) + (self.__rootP_prior - 1)
        post_rootP /= post_rootP.sum()

        return dict(
            root_probs=post_rootP, **self.substmodel._get_next_params(stats.tfdns)
        )

    def _log_prior_prob(self):
        log_pri_rootP = xlogy(self.__rootP_prior - 1, self.root_probs).sum()
        return log_pri_rootP + self.substmodel._log_prior_prob()

    def _reconstruct_joint(self, phytbl):
        tree = phytbl.tree
        ncmpst, ncols, nnodes = self._ncompoundstates, phytbl.ncols, tree.nnodes

        log_branchP = np.log(
            expm(c_einsum('m,ij->mij', tree.branch_lengths, self.R))
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
            tm_args=(self.ncharstates, type(self.substmodel)),
            tm_params=self.flat_params,
        )

    def _wrap_reconstruction(self, phytbl, recon_charstates, method=None):
        tree = deepcopy(phytbl.tree)
        tree.fill_names(ignore_tips=True)
        return ReconPhyTable(
            dict(zip(tree.names, recon_charstates.T)),
            tree,
            metadata_kw=dict(method=method, **self._get_model_info()),
        )

    def reconstruct_joint(self, phytbl):
        return self._wrap_reconstruction(
            phytbl, self._reconstruct_joint(phytbl), 'joint'
        )

    def reconstruct_marginal(self, phytbl):
        return self._wrap_reconstruction(
            phytbl, self._reconstruct_marginal(phytbl), 'marginal'
        )

    def _simulate(self, tree, N, rng):
        charstates = {}
        preorder = tree.traverse('preorder')

        # init
        root = next(preorder)
        charstates[root.name] = rng.choice(
            *self.root_probs.shape, size=N, p=self.root_probs
        )

        # recursion
        for node in preorder:
            P = self.substmodel.P(node.dist)
            pa_charstates = charstates[node.up.name]
            my_charstates = np.empty(N, dtype=int)
            for st in set(pa_charstates):
                mask = pa_charstates == st
                my_charstates[mask] = rng.choice(
                    *P[:, st].shape, size=np.count_nonzero(mask), p=P[:, st]
                )

            charstates[node.name] = my_charstates

        return charstates

    def simulate(self, tree, N, rng):
        tree2 = PostorderSerializedTree(tree)
        tree2.fill_names(ignore_tips=True)
        charstates = self._simulate(tree2.to_ete3(), N, rng)
        return ReconPhyTable(
            charstates,
            tree2,
            metadata_kw=dict(method='simulated', **self._get_model_info()),
        )


class MixtureTreeModel(AbstTreeModel):
    _params_signature = signature(
        lambda mixture_probs, root_probs, **substmodels_params: ...
    )

    def __init__(
        self,
        ncharstates,
        nmixtures,
        substmodelclass,
        prior_kw=None,
        init_params_method=None,
        init_params_kw=None,
    ):
        self.__ncharstates = ncharstates

        # inspect prior_kw and set priors
        assume_priors = prior_kw is not None
        if assume_priors:
            bound_args = self._params_signature.bind(**prior_kw)
            prior_kw = bound_args.kwargs
            mixtureP_prior, rootP_prior = bound_args.args
            mixtureP_prior = Dirichlet(
                _to_ndarray_of_shape(mixtureP_prior, (nmixtures,))
            )
            rootP_prior = Dirichlet(_to_ndarray_of_shape(rootP_prior, (ncharstates,)))
        else:
            mixtureP_prior = Dirichlet(np.ones(nmixtures))
            rootP_prior = Dirichlet(np.ones(ncharstates))
        self.__assume_priors = assume_priors
        self.__mixtureP_prior = mixtureP_prior
        self.__rootP_prior = rootP_prior

        # define parameters (and init them with NaN)
        self.__substmodels = tuple(
            substmodelclass(
                ncharstates,
                prior_kw=prior_kw,
                init_params_method='skip',
            )
            for _ in range(nmixtures)
        )
        self.__mixtureP = np.full((nmixtures,), np.nan)
        self.__rootP = np.full((nmixtures, ncharstates), np.nan)

        # init parameters
        super().init_params(
            init_params_method=init_params_method, init_params_kw=init_params_kw
        )

    def init_params_manual(self, mixture_probs, root_probs, substmodels_params):
        if len(substmodels_params) != self.nmixtures:
            raise ValueError(
                f'The length of \'substmodels_params\' must be {self.nmixtures}. '
                f'Got: {len(substmodels_params)}'
            )
        self.update(
            mixture_probs=mixture_probs, root_probs=root_probs, substmodels_params=None
        )
        for sbstmdl, kwargs in zip_longest(self.__substmodels, substmodels_params):
            sbstmdl.init_params_manual(**kwargs)

    def init_params_random(self, rng, distr_kw=None):
        if self.__assume_priors:
            if distr_kw is not None:
                warnings.warn(
                    'distr_kw is ignored because this model already assumes priors.'
                )
            mixtureP_distr = self.__mixtureP_prior
            rootP_distr = self.__rootP_prior
        else:
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('mixture_probs')
            distr_kw.setdefault('root_probs')
            bound_args = self._params_signature.bind(**distr_kw)
            distr_kw = bound_args.kwargs
            mixtureP_distr, rootP_distr = bound_args.args
            if mixtureP_distr is None:
                mixtureP_distr = np.ones(self.nmixtures)
            else:
                mixtureP_distr = _to_ndarray_of_shape(mixtureP_distr, (self.nmixtures,))
            if rootP_distr is None:
                rootP_distr = np.ones(self.ncharstates)
            else:
                rootP_distr = _to_ndarray_of_shape(rootP_distr, (self.ncharstates,))

        for sbstmdl in self.__substmodels:
            sbstmdl.init_params_random(rng, distr_kw)
        self.update(
            root_probs=rng.dirichlet(rootP_distr, size=self.nmixtures),
            mixture_probs=rng.dirichlet(mixtureP_distr),
            substmodels_params=None,  # already initialized
        )

    def init_params_emEM(
        self,
        phytbl,
        rng,
        ntrial=10,
        max_rounds=100,
        stop_criteria=(1e-4, 1e-4),
        show_init_progress=False,
        distr_kw=None,
        **fit_kw,
    ):
        if not self.__assume_priors:
            tot_brlen = phytbl.tree.branch_lengths.sum()
            mp_rate = phytbl.min_changes.sum() / phytbl.ncols / tot_brlen
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('rates', (3.0, mp_rate / 2.0))

        AbstTreeModel.init_params_emEM(
            self,
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
    def substmodels(self):
        return self.__substmodels

    @property
    def ncharstates(self):
        return self.__ncharstates

    @property
    def _ncompoundstates(self):
        return self.__ncharstates

    @property
    def nmixtures(self):
        return len(self.__substmodels)

    @property
    def root_probs(self):
        return self.__rootP

    @property
    def mixture_probs(self):
        return self.__mixtureP

    @property
    def priors(self):
        return dict(
            mixture_probs=self.__mixtureP_prior,
            root_probs=self.__rootP_prior,
            substmodels=[sm.priors for sm in self.substmodels],
        )

    @cached_property
    def _compound_state_flags(self):
        csflags = np.tile(np.eye(self.ncharstates), (self.nmixtures, 1, 1))
        csflags.setflags(write=False)
        return csflags

    def _empty_stats(self, phytbl):
        return MultipleRateTreeStats(self, phytbl)

    # With typecheck
    def update(self, mixture_probs=None, root_probs=None, substmodels_params=None):
        if mixture_probs is None:
            mixture_probs = self.__mixtureP
        else:
            mixture_probs = _to_ndarray_of_shape(mixture_probs, self.__mixtureP.shape)

        if root_probs is None:
            root_probs = self.__rootP
        else:
            root_probs = _to_ndarray_of_shape(root_probs, self.__rootP.shape)

        if substmodels_params is None:
            substmodels_params = []
        elif len(substmodels_params) != self.nmixtures:
            raise ValueError(
                f'The length of \'substmodels_params\' must be {self.nmixtures}. '
                f'Got: {len(substmodels_params)}'
            )

        self._update(mixture_probs, root_probs, substmodels_params)

    # Without typecheck
    def _update(self, mixture_probs, root_probs, substmodels_params):
        self.__mixtureP = np.array(mixture_probs)
        self.__rootP = np.array(root_probs)
        for sbstmdl, kwargs in zip(self.__substmodels, substmodels_params):
            sbstmdl.update(**kwargs or {})

    @property
    def flat_params(self):
        return np.concatenate(
            (
                self.mixture_probs,
                self.root_probs.reshape(-1),
                *[sbstmdl.flat_params for sbstmdl in self.substmodels],
            )
        )

    def _decompress_flat_params(self, p):
        nmixt, nchrst = self.nmixtures, self.ncharstates
        return dict(
            mixture_probs=p[:nmixt],
            root_probs=p[nmixt : nmixt * (nchrst + 1)].reshape((nmixt, nchrst)),
            substmodels_params=[
                sbstmdl._decompress_flat_params(p_sub)
                for sbstmdl, p_sub in zip(
                    self.__substmodels, p[nmixt * (nchrst + 1) :].reshape((nmixt, -1))
                )
            ],
        )

    def _get_next_params(self, stats):
        post_mixtureP = (
            stats.post_mixtureP.sum(axis=1) + self.__mixtureP_prior - 1
        )  # (kn->k) + k
        post_mixtureP /= post_mixtureP.sum()

        post_rootP = (stats.post_mixtureP[:, newax, :] * stats.post_rootP).sum(
            axis=2
        ) + (self.__rootP_prior - 1)[
            newax, :
        ]  # (kn,kan->ka) + a
        post_rootP /= post_rootP.sum(axis=1)[:, newax]

        substmodels_params = [
            sbstmdl._get_next_params(tfdns)
            for sbstmdl, tfdns in zip(self.__substmodels, stats.tfdns)
        ]

        return dict(
            mixture_probs=post_mixtureP,
            root_probs=post_rootP,
            substmodels_params=substmodels_params,
        )

    def _log_prior_prob(self):
        logP_mixture_prior = xlogy(self.__mixtureP_prior - 1, self.__mixtureP).sum()
        logP_root_prior = xlogy(self.__rootP_prior - 1, self.__rootP).sum()

        return (
            logP_mixture_prior
            + logP_root_prior
            + sum(sbstmdl._log_prior_prob() for sbstmdl in self.__substmodels)
        )

    def _reconstruct_joint(self, phytbl):
        tree = phytbl.tree
        ncmpst, ncols, nnodes = self._ncompoundstates, phytbl.ncols, tree.nnodes
        nmixt = self.nmixtures

        log_rootP = np.log(self.root_probs)
        log_branchP = np.zeros((nnodes - 1, nmixt, ncmpst, ncmpst))  # (m, k, j, i)
        for k, sbstmdl in enumerate(self.substmodels):
            log_branchP[:, k] = np.log(
                expm(c_einsum('m,ab->mab', tree.branch_lengths, sbstmdl.R))
            )

        tmp_recon = np.zeros(
            (nnodes - 1, nmixt, ncmpst, ncols), dtype=int
        )  # (m, k, j, n)
        loglik = np.zeros((nnodes - 1, nmixt, ncmpst, ncols))  # (m, k, j, n)
        recon_mixtures = np.zeros(ncols, dtype=int)  # (n,)
        recon_charstates = np.zeros((ncols, nnodes), dtype=int)  # (m, n)

        tip_mask = np.ma.log(self._compound_state_flags).filled(-np.inf)
        for current_idx in range(nnodes - 1):
            current = tree.nodes[current_idx]
            if current.is_leaf():
                a = phytbl.inside_encoding[current_idx].codes  # (n,)
                tmp = (
                    log_branchP[current_idx, :, :, :, newax] + tip_mask[:, :, newax, :]
                )
                tmp_recon[current_idx, :, :, :] = tmp.argmax(axis=1)[:, :, a]
                loglik[current_idx, :, :, :] = tmp.max(axis=1)[:, :, a]
            else:
                tmp = (
                    log_branchP[current_idx, :, :, :, newax]
                    + np.sum(
                        [
                            loglik[child_id, :, :, :]
                            for child_id in current.children_idxs
                        ],
                        axis=0,
                    )[:, :, newax, :]
                )  # kjin
                tmp_recon[current_idx, :, :, :] = tmp.argmax(axis=1)
                loglik[current_idx, :, :, :] = tmp.max(axis=1)

        tmp = log_rootP[:, :, newax] + np.sum(
            [loglik[child_id, :, :, :] for child_id in tree.nodes[-1].children_idxs],
            axis=0,
        )  # (k, j, n)

        recon_mixtures, recon_charstates[:, -1] = np.column_stack(
            np.unravel_index(
                tmp.reshape(-1, tmp.shape[-1]).argmax(axis=0), tmp.shape[:-1]
            )
        ).T  # argmax_kj(k, j, n) -> (n,)

        for current_idx in reversed(range(nnodes - 1)):
            parent_idx = tree.nodes[current_idx].parent_idx
            recon_charstates[:, current_idx] = tmp_recon[
                current_idx,
                recon_mixtures,
                recon_charstates[:, parent_idx],
                np.arange(ncols),
            ]

        return recon_charstates, recon_mixtures

    def _reconstruct_marginal(self, phytbl):
        stats = self.sufficient_stats(phytbl)
        recon_mixtures = stats.post_mixtureP.argmax(axis=0)
        insideP = np.array(
            [
                insideP_m[:, :, partial_cols.codes]
                for insideP_m, partial_cols in zip(
                    stats.insideP, phytbl.inside_encoding
                )
            ]
        )[:, recon_mixtures, :, np.arange(phytbl.ncols)]
        outsideP = stats.outsideP[
            :, recon_mixtures, :, np.arange(phytbl.ncols)
        ]  # (m, k, a, n) -> (n, m, a); the order of axes has been changed by fancy indexing
        recon_charstates = (insideP * outsideP).argmax(axis=2)

        return recon_charstates, recon_mixtures

    def _get_model_info(self):
        return dict(
            treemodel=type(self),
            tm_args=(self.ncharstates, self.nmixtures, type(self.substmodels[0])),
            tm_params=self.flat_params,
        )

    def _wrap_reconstruction(
        self, phytbl, recon_charstates, recon_mixtures, method=None
    ):
        tree = deepcopy(phytbl.tree)
        tree.fill_names(ignore_tips=True)
        return ReconPhyTable(
            dict(zip(tree.names, recon_charstates.T)),
            tree,
            colattrs=(('mixtures', recon_mixtures),),
            metadata_kw=dict(method=method, **self._get_model_info()),
        )

    def reconstruct_joint(self, phytbl):
        return self._wrap_reconstruction(
            phytbl, *self._reconstruct_joint(phytbl), 'joint'
        )

    def reconstruct_marginal(self, phytbl):
        return self._wrap_reconstruction(
            phytbl, *self._reconstruct_marginal(phytbl), 'marginal'
        )

    def simulate(self, tree, N, rng):
        tree2 = PostorderSerializedTree(tree)
        tree2.fill_names(ignore_tips=True)
        charstates = {}
        preorder = iter(tree2.to_ete3().traverse('preorder'))

        # init
        mixture_ids = rng.choice(self.nmixtures, size=N, p=self.mixture_probs)
        root = next(preorder)
        my_charstates = np.empty(N, dtype=int)
        for mix in set(mixture_ids):
            rootP = self.root_probs[mix]
            mask = mixture_ids == mix
            my_charstates[mask] = rng.choice(
                *rootP.shape, size=np.count_nonzero(mask), p=rootP
            )
        charstates[root.name] = my_charstates

        for node in preorder:
            P = expm([model.R * node.dist for model in self.substmodels])
            pa_charstates = charstates[node.up.name]
            my_charstates = np.empty(N, dtype=int)
            for mix, st in set(zip(mixture_ids, pa_charstates)):
                mask = (mixture_ids == mix) & (pa_charstates == st)
                my_charstates[mask] = rng.choice(
                    *P[mix, :, st].shape, size=np.count_nonzero(mask), p=P[mix, :, st]
                )
            charstates[node.name] = my_charstates

        return ReconPhyTable(
            charstates,
            tree2,
            colattrs=(('mixtures', mixture_ids),),
            metadata_kw=dict(method='simulated', **self._get_model_info()),
        )


class MarkovModulatedTreeModel(PlainTreeModel):
    _params_signature = signature(
        lambda cpy_root_probs, cat_root_probs, **substmodel_params: ...
    )

    def __init__(
        self,
        ncharstates,
        ncategories,
        substmodelclass,
        invariant_cat=False,
        prior_kw=None,
        init_params_method=None,
        init_params_kw=None,
    ):
        # inspect prior_kw and set priors
        assume_priors = prior_kw is not None
        if assume_priors:
            bound_args = self._params_signature.bind(**prior_kw)
            prior_kw = bound_args.kwargs
            cpy_rootP_prior, cat_rootP_prior = bound_args.args
            cpy_rootP_prior = Dirichlet(
                _to_ndarray_of_shape(cpy_rootP_prior, (ncharstates,))
            )
            cat_rootP_prior = Dirichlet(
                _to_ndarray_of_shape(cat_rootP_prior, (ncategories,))
            )
        else:
            cpy_rootP_prior = Dirichlet(np.ones(ncharstates))
            cat_rootP_prior = Dirichlet(np.ones(ncategories))
        self.__assume_priors = assume_priors
        self.__cpy_rootP_prior = cpy_rootP_prior
        self.__cat_rootP_prior = cat_rootP_prior

        # define parameters (and init them with NaN)
        self.__substmodel = substmodelclass(
            ncharstates,
            ncategories,
            invariant_cat=invariant_cat,
            prior_kw=prior_kw,
            init_params_method='skip',
        )
        self.__cat_rootP = np.full((ncategories,), np.nan)
        self.__cpy_rootP = np.full((ncategories, ncharstates), np.nan)

        # init parameters
        super().init_params(
            init_params_method=init_params_method, init_params_kw=init_params_kw
        )

    def init_params_manual(self, cpy_root_probs, cat_root_probs, **substmodel_params):
        self.update(
            cpy_root_probs=cpy_root_probs,
            cat_root_probs=cat_root_probs,
        )
        self.__substmodel.init_params_manual(**substmodel_params)

    def init_params_random(self, rng, distr_kw=None):
        if self.__assume_priors:
            if distr_kw is not None:
                warnings.warn(
                    '\'distr_kw\' is ignored because this model already assumes priors.'
                )
            cpy_rootP_distr = self.__cpy_rootP_prior
            cat_rootP_distr = self.__cat_rootP_prior
        else:
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('cpy_root_probs')
            distr_kw.setdefault('cat_root_probs')
            bound_args = self._params_signature.bind(**distr_kw)
            distr_kw = bound_args.kwargs
            cpy_rootP_distr, cat_rootP_distr = bound_args.args
            if cpy_rootP_distr is None:
                cpy_rootP_distr = np.ones(self.ncharstates)
            else:
                cpy_rootP_distr = _to_ndarray_of_shape(
                    cpy_rootP_distr, (self.ncharstates,)
                )
            if cat_rootP_distr is None:
                cat_rootP_distr = np.ones(self.ncategories)
            else:
                cat_rootP_distr = _to_ndarray_of_shape(
                    cat_rootP_distr, (self.ncategories,)
                )

        self.__substmodel.init_params_random(rng, distr_kw)
        self._update(
            cpy_root_probs=rng.dirichlet(cpy_rootP_distr, size=self.ncategories),
            cat_root_probs=rng.dirichlet(cat_rootP_distr),
        )

    def init_params_emEM(
        self,
        phytbl,
        rng,
        ntrial=10,
        max_rounds=100,
        stop_criteria=(1e-4, 1e-4),
        show_init_progress=False,
        distr_kw=None,
        **fit_kw,
    ):
        if not self.__assume_priors:
            tot_brlen = phytbl.tree.branch_lengths.sum()
            mp_rate = phytbl.min_changes.sum() / phytbl.ncols / tot_brlen
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('cpy_change_rates', (3.0, mp_rate / 2.0))
            distr_kw.setdefault(
                'cat_switch_rates', (1.0, mp_rate * 0.1)
            )  # sparse modeling (to be reconsidered)

        AbstTreeModel.init_params_emEM(
            self,
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
    def substmodel(self):
        return self.__substmodel

    @property
    def ncharstates(self):
        return self.__substmodel.ncharstates

    @property
    def ncategories(self):
        return self.__substmodel.ncategories

    @property
    def _ncompoundstates(self):
        return self.ncharstates * self.ncategories

    @property
    def R(self):
        return self.__substmodel.R

    @property
    def root_probs(self):
        return (self.__cat_rootP[:, newax] * self.__cpy_rootP).reshape(-1)

    @property
    def cpy_root_probs(self):
        return self.__cpy_rootP

    @property
    def cat_root_probs(self):
        return self.__cat_rootP

    @property
    def priors(self):
        return dict(
            cpy_root_probs=self.__cpy_rootP_prior,
            cat_root_probs=self.__cat_rootP_prior,
            **self.substmodel.priors,
        )

    @cached_property
    def _compound_state_flags(self):
        csflags = np.tile(np.eye(self.ncharstates), (self.ncategories, 1))
        csflags.setflags(write=False)
        return csflags

    # With typecheck
    def update(self, cpy_root_probs=None, cat_root_probs=None, **kwargs):
        if cpy_root_probs is None:
            cpy_root_probs = self.__cpy_rootP
        else:
            cpy_root_probs = _to_ndarray_of_shape(
                cpy_root_probs, self.__cpy_rootP.shape
            )

        if cat_root_probs is None:
            cat_root_probs = self.__cat_rootP
        else:
            cat_root_probs = _to_ndarray_of_shape(
                cat_root_probs, self.__cat_rootP.shape
            )

        self._update(cpy_root_probs, cat_root_probs, **kwargs)

    # Without typecheck
    def _update(self, cpy_root_probs, cat_root_probs, **kwargs):
        self.__cpy_rootP = np.array(cpy_root_probs)
        self.__cat_rootP = np.array(cat_root_probs)
        self.__substmodel.update(**kwargs)

    @property
    def flat_params(self):
        return np.concatenate(
            (
                self.cat_root_probs,
                self.cpy_root_probs.reshape(-1),
                self.substmodel.flat_params,
            )
        )

    def _decompress_flat_params(self, p):
        nchrst, ncat = self.ncharstates, self.ncategories
        return dict(
            cat_root_probs=p[:ncat],
            cpy_root_probs=p[ncat : ncat * (nchrst + 1)].reshape(ncat, nchrst),
            **self.substmodel._decompress_flat_params(p[ncat * (nchrst + 1) :]),
        )

    # Removed the 'share_switch_rate' argument for simplicity
    # To reduce the number of degrees of freedom, specify an exponential distribution for the prior distribution.
    # If this option is to be reintroduced in the future, it will be specified in the initialization, not the fitting.
    ## FIXME
    def _get_next_params(self, stats):
        nchrst, ncat = self.ncharstates, self.ncategories
        post_rootP = stats.post_rootP.sum(axis=1).reshape((ncat, nchrst))
        tmp_cpy = post_rootP + self.__cpy_rootP_prior - 1
        tmp_cat = post_rootP.sum(axis=1) + self.__cat_rootP_prior - 1
        cpy_rootP = tmp_cpy / tmp_cpy.sum(axis=1)[:, newax]
        cat_rootP = tmp_cat / tmp_cat.sum()

        return dict(
            cat_root_probs=cat_rootP,
            cpy_root_probs=cpy_rootP,
            **self.substmodel._get_next_params(stats.tfdns),
        )

    def _log_prior_prob(self):
        logP_cpy_root_prior = xlogy(
            self.__cpy_rootP_prior - 1, self.cpy_root_probs
        ).sum()
        logP_cat_root_prior = xlogy(
            self.__cat_rootP_prior - 1, self.cat_root_probs
        ).sum()

        return (
            logP_cpy_root_prior
            + logP_cat_root_prior
            + self.__substmodel._log_prior_prob()
        )

    def _reconstruct_joint(self, phytbl):
        recon_compound_states = super()._reconstruct_joint(phytbl)
        recon_categories, recon_charstates = np.divmod(
            recon_compound_states, self.ncharstates
        )
        return recon_charstates, recon_categories

    def _reconstruct_marginal(self, phytbl):
        recon_compound_states = super()._reconstruct_marginal(phytbl)
        recon_categories, recon_charstates = np.divmod(
            recon_compound_states, self.ncharstates
        )
        return recon_charstates, recon_categories

    def _get_model_info(self):
        return dict(
            treemodel=type(self),
            tm_args=(self.ncharstates, self.ncategories, type(self.substmodel)),
            tm_params=self.flat_params,
        )

    def _wrap_reconstruction(self, phytbl, recon_charstates, recon_categories, method):
        tree = deepcopy(phytbl.tree)
        tree.fill_names(ignore_tips=True)
        return ReconPhyTable(
            dict(zip(tree.names, recon_charstates.T)),
            tree,
            otherstates=(('categories', dict(zip(tree.names, recon_categories.T))),),
            metadata_kw=dict(method=method, **self._get_model_info()),
        )

    def reconstruct_joint(self, phytbl):
        return self._wrap_reconstruction(
            phytbl, *self._reconstruct_joint(phytbl), 'joint'
        )

    def reconstruct_marginal(self, phytbl):
        return self._wrap_reconstruction(
            phytbl, *self._reconstruct_marginal(phytbl), 'marginal'
        )

    def simulate(self, tree, N, rng):
        tree2 = PostorderSerializedTree(tree)
        tree2.fill_names(ignore_tips=True)
        compound_states = super()._simulate(tree2.to_ete3(), N, rng)

        charstates = {
            node: states % self.ncharstates for node, states in compound_states.items()
        }
        categories = {
            node: states // self.ncharstates for node, states in compound_states.items()
        }

        return ReconPhyTable(
            charstates,
            tree2,
            otherstates=(('categories', categories),),
            metadata_kw=dict(method='simulated', **self._get_model_info()),
        )
