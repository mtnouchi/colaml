import warnings
from abc import ABC, abstractmethod
from functools import cache
from inspect import signature
from types import MappingProxyType
from typing import NamedTuple

from numba import jit
import numpy as np
from numpy import newaxis as newax
from numpy.core.multiarray import c_einsum
from scipy.linalg import block_diag, expm
from scipy.special import xlogy

class Gamma(NamedTuple):
    shape: float
    scale: float
    
def _to_ndarray_of_shape(subject, expected_shape):
    subject = np.array(subject)
    if subject.shape == expected_shape: 
        return subject
    
    raise ValueError(
        f'Invalid shape. Expected: {expected_shape}, but got: {subject.shape}.'
    )

class abstSuffStatsEngine(ABC):
    def __init__(self, nstates): pass
    @abstractmethod
    def precompute(self, substmodel): pass
    @abstractmethod
    def compute(self, t): pass
    
class eigSuffStatsEngine(abstSuffStatsEngine):
    class precompResult(NamedTuple):
        D     : np.ndarray
        UinvU1: np.ndarray
        UinvU2: np.ndarray
        coef  : np.ndarray
    
    @cache
    def precompute(self, substmodel):
        coef = substmodel.R.copy()
        np.fill_diagonal(coef, 1)
        D, U = np.linalg.eig(substmodel.R)
        nst, = D.shape
        invU = np.linalg.inv(U)
        UinvU1 = c_einsum('au,ui->aiu', U, invU).reshape(-1,nst)
        UinvU2 = c_einsum('jv,vb->vbj', U, invU).reshape(nst,-1)
        return self.precompResult(D=D, UinvU1=UinvU1, UinvU2=UinvU2, coef=coef)
    
    # numba is only used here. 
    # we might be able to get rid of the dependency on numba.
    @staticmethod
    @jit(['f8[:,:](f8[:])', 'c16[:,:](c16[:])'], nopython=True)
    def _exp_difference_quot(tD):
        nst, = tD.shape
        exptD = np.exp(tD)
        integ = np.diag(exptD)
        for u in range(nst):
            for v in range(u):
                den = tD[v] - tD[u]
                if np.abs(den) < 1e-8:
                    # (exp(a) - exp(b)) / (a - b) = exp((a + b) / 2) * sinhc((a - b) / 2)
                    # sinhc(z) = sinh(z) / z = 1 + z^2 / 6 + z^4 / 120 + ...
                    integ[u, v] = integ[v, u] = np.exp((tD[v]+tD[u])/2) * (1 + den**2 / 24)
                else:
                    integ[u, v] = integ[v, u] = (exptD[v] - exptD[u]) / den
        return integ

    def compute(self, substmodel, t):
        cache = self.precompute(substmodel)
        tD, UinvU1, UinvU2, coef = t * cache.D, cache.UinvU1, cache.UinvU2, cache.coef
        nst, = tD.shape
        integ = self._exp_difference_quot(tD)
        tPdP = (UinvU1 @ integ @ UinvU2).reshape(nst,nst,nst,nst).swapaxes(1, 2).real * t
        stats = coef[newax,newax,:,:] * tPdP
        return stats
    
class auxRSuffStatsEngine(abstSuffStatsEngine):
    class precompResult(NamedTuple):
        auxRs      : np.ndarray
        zipped_coef: np.ndarray
    
    @cache
    def precompute(self, substmodel):
        coef = substmodel.R.copy()
        np.fill_diagonal(coef, 1)
        ncmpst, _ = substmodel.R.shape
        Z = np.zeros((ncmpst, ncmpst))
        auxRs = np.array([
            np.block([[substmodel.R, I], [Z, substmodel.R]])
            for I in np.eye(ncmpst*ncmpst).reshape((ncmpst,ncmpst,ncmpst,ncmpst))[substmodel._mask_nonzero]
        ])
        return self.precompResult(auxRs=auxRs, zipped_coef=coef[substmodel._mask_nonzero])
    
    def compute(self, substmodel, t):
        cache = self.precompute(substmodel)
        ncmpst, _ = substmodel.R.shape
        tPdP = expm(t * cache.auxRs)[:,:ncmpst,ncmpst:]
        assert not (tPdP < 0).any(), f'tPdP < 0 in {[*zip(*np.where(tPdP < 0))]}: {tPdP[tPdP < 0]}'
        stats = np.zeros((ncmpst,ncmpst,ncmpst,ncmpst))
        stats[:,:,substmodel.mask_nonzero] = c_einsum('hab,h->abh', tPdP, cache.zipped_coef)
        return stats
    
class spsSuffStatsEngine(abstSuffStatsEngine):
    class precompResult(NamedTuple):
        Es         : np.ndarray
        zipped_coef: np.ndarray
    
    @cache
    def precompute(self, substmodel):
        coef = substmodel.R.copy()
        np.fill_diagonal(coef, 1)
        Es = np.zeros((substmodel._mask_nonzero.sum(), *substmodel.R.shape))
        for n, (i, j) in enumerate(zip(*np.where(substmodel._mask_nonzero))):
            Es[n, i, j] = 1
        return self.precompResult(Es=Es, zipped_coef=coef[substmodel._mask_nonzero])
    
    def compute(self, substmodel, t):
        from scipy.linalg import expm_frechet
        cache = self.precompute(substmodel)
        ncmpst, _ = substmodel.R.shape
        tR = t * substmodel.R
        tPdP = np.array([expm_frechet(tR, tE, compute_expm=False) for tE in t * cache.Es])
        assert not (tPdP < 0).any(), f'tPdP < 0 in {[*zip(*np.where(tPdP < 0))]}: {tPdP[tPdP < 0]}'
        stats = np.zeros((ncmpst,ncmpst,ncmpst,ncmpst))
        stats[:,:,substmodel.mask_nonzero] = c_einsum('hab,h->abh', tPdP, cache.zipped_coef)
        return stats
    
class abstSubstModel(ABC):
    _available_engine = MappingProxyType(dict(
        eig  = eigSuffStatsEngine ,  
        auxR = auxRSuffStatsEngine, 
        sps  = spsSuffStatsEngine
    ))
    def __init__(self, nstates, ss_method):
        self._nstates = nstates
        self._mask_nonzero = np.empty((nstates, nstates), dtype=bool)
        self.__R = np.empty((nstates, nstates))
        self.set_ss_method(ss_method)
        
    @property
    def nstates(self): return self._nstates
    @property
    def R(self): return self.__R
    @R.setter
    def R(self, Rmat):
        #self.P.cache_clear()
        self._ss_engine.precompute.cache_clear()
        self.__R = Rmat
        
    @abstractmethod
    def update(self, **kwargs):
        pass
    
    @property
    def mask_nonzero(self):
        return self._mask_nonzero
    
    #@cache
    def P(self, t):
        return expm(self.R * t)        
        
    def set_ss_method(self, ss_method):
        self._ss_engine = self._available_engine[ss_method](self.nstates)
        self.R = self.R
        
    def sufficient_stats(self, t):
        return self._ss_engine.compute(self, t)
    
    def init_params(self, init_params_method, init_params_kw):
        if init_params_method is None:
            warnings.warn('Parameters are not initialized. Call either \'init_params_manual\', \'init_params_random\' before calculating statistics.')
        elif init_params_method == 'skip':
            if init_params_kw is not None:
                warnings.warn('When init_params_method=\'skip\', init_params_kw is ignored.')
        elif init_params_method == 'manual':
            self.init_params_manual(**init_params_kw or {})
        elif init_params_method == 'random':
            self.init_params_random(**init_params_kw or {})
        else:
            raise ValueError(f'Unknown value of \'init_params_method\': {init_params_method}')
        
class BDARD(abstSubstModel):
    _params_signature = signature(lambda rates: ...)
    
    @property
    def ncharstates(self): return self._ncharstates
    
    def __init__(self, ncharstates, prior_kw=None, init_params_method=None, init_params_kw=None, ss_method='eig'):
        super().__init__(ncharstates, ss_method)
        self._ncharstates = ncharstates
        
        # init masks
        self._mask_ns = np.eye(ncharstates, k=1, dtype=bool) | np.eye(ncharstates, k=-1, dtype=bool)
        self._mask_nonzero = np.eye(ncharstates, dtype=bool) | self._mask_ns
        
        # define parameters (and init them with NaN)
        self.update(rates=np.full((2, ncharstates - 1), np.nan))
        
        # set priors
        assume_priors = prior_kw is not None
        if assume_priors:
            rates_prior, = self._params_signature.bind(**prior_kw).args
            rates_prior = Gamma(*rates_prior)
        else:
            rates_prior = Gamma(shape=1.0, scale=np.inf)
            
        self.__assume_priors = assume_priors
        self.__rates_prior = rates_prior
        
        # init parameters
        self.init_params(init_params_method=init_params_method, init_params_kw=init_params_kw)
        
    def init_params_manual(self, rates):
        #rates = _to_ndarray_of_shape(rates, (2, self.ncharstates-1))
        self.update(rates=rates)
        
    def init_params_random(self, rng, distr_kw=None):
        if self.__assume_priors:
            if distr_kw is not None:
                warnings.warn(
                    '\'distr_kw\' is ignored because this model already assumes priors.'
                )
            rates_distr = self.__rates_prior
            
        else:
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('rates')
            rates_distr, = self._params_signature.bind(**distr_kw).args
            if rates_distr is None:
                rates_distr = Gamma(shape=1.0, scale=1.0)
                warnings.warn(
                    f'Drawing rates from {rates_distr}; '
                    'no prior is assumed in this substitution model. '
                    'Check for consistency with your data and consider explicitly specifying '
                    '\'rates\' in \'distr_kw\'.'
                )
            
        self.update(rates=rng.gamma(*rates_distr, size=(2, self.ncharstates - 1)))
    
    def update(self, rates=None):
        if rates is None: return
    
        rates = _to_ndarray_of_shape(rates, (2, self.ncharstates-1))
        R = np.diag(rates[0], k=1) + np.diag(rates[1], k=-1)
        np.fill_diagonal(R, -R.sum(axis=0))
        self.R = R
        
    @property
    def priors(self):
        return dict(rates=self.__rates_prior)
    
    @property
    def flat_params(self):
        return np.hstack([np.diag(self.R, k=1), np.diag(self.R, k=-1)])
        
    def _decompress_flat_params(self, p):
        return dict(rates=p.reshape(2, self.ncharstates - 1))
    
    def _get_next_params(self, tfdns):
        rates_gamma = self.__rates_prior
        R = (tfdns + rates_gamma.shape - 1) / (np.diagonal(tfdns) + 1 / rates_gamma.scale)[newax,:]
        return dict(rates=R[self._mask_ns].reshape(-1, 2).T)
        
    def _log_prior_prob(self):
        rates_gamma = self.__rates_prior
        logP_rates_prior = xlogy(rates_gamma.shape - 1, self.flat_params).sum() - self.flat_params.sum() / rates_gamma.scale
        return logP_rates_prior
    
#TODO class twoParam(abstSubstModel)
#TODO class CandM(abstSubstModel)
#TODO class BDI(abstSubstModel)

class MarkovModulatedBDARD(abstSubstModel):
    _params_signature = signature(lambda cpy_change_rates, cat_switch_rates: ...)
    
    @property
    def ncharstates(self): return self._ncharstates
    @property
    def ncategories(self): return self._ncategories
        
    @property
    def has_invariant_cat(self):
        return self.__has_invariant_cat
    
    def __init__(self, ncharstates, ncategories, invariant_cat=False, prior_kw=None, init_params_method=None, init_params_kw=None, ss_method='eig'):
        super().__init__(ncharstates * ncategories, ss_method)
        self._ncharstates = ncharstates
        self._ncategories = ncategories
        
        # specify a method for calulating sufficient stats
        self.set_ss_method(ss_method)
        
        # init masks
        self._mask_cpy_ns = np.kron(np.eye(ncategories, dtype=bool), np.eye(ncharstates, k=1, dtype=bool) | np.eye(ncharstates, k=-1, dtype=bool))
        self._mask_switch = np.kron(np.ones((ncategories, ncategories), dtype=bool), np.eye(ncharstates, dtype=bool))
        self._cat_idx = tuple(np.hstack([np.triu_indices(ncategories, 1), np.tril_indices(ncategories, -1)]))
        self._mask_nonzero = self._mask_switch | self._mask_cpy_ns
        
        catidx_row, catidx_col = np.hstack((
            np.triu_indices(ncategories,  1), np.tril_indices(ncategories, -1)
        )) * ncharstates
        icpy, icat = np.arange(1, ncharstates), np.arange(ncategories)
        cpyidx_row = ((icat * ncharstates)[:,None] + np.hstack((icpy-1, icpy))).reshape(-1)
        cpyidx_col = ((icat * ncharstates)[:,None] + np.hstack((icpy, icpy-1))).reshape(-1)
        self._flatidx = (np.hstack((catidx_row, cpyidx_row)), np.hstack((catidx_col, cpyidx_col)))
        
        # set priors
        assume_priors = prior_kw is not None
        if assume_priors:
            cpy_changeR_prior, cat_switchR_prior = self._params_signature.bind(**prior_kw).args
            cpy_changeR_prior = Gamma(*cpy_changeR_prior)
            cat_switchR_prior = Gamma(*cat_switchR_prior)
        else:
            cpy_changeR_prior = Gamma(shape=1.0, scale=np.inf)
            cat_switchR_prior = Gamma(shape=1.0, scale=np.inf)
        
        self.__assume_priors = assume_priors
        self.__cpy_changeR_prior = cpy_changeR_prior
        self.__cat_switchR_prior = cat_switchR_prior
        
        # define parameters (and init them with NaN)
        cpy_change_rates = np.full((ncategories, 2, ncharstates - 1), np.nan)
        cat_switch_rates = np.full((2, ncategories * (ncategories - 1) // 2), np.nan)
        self.update(cpy_change_rates=cpy_change_rates, cat_switch_rates=cat_switch_rates)
        self.__has_invariant_cat = invariant_cat
        
        # init parameters
        self.init_params(init_params_method=init_params_method, init_params_kw=init_params_kw)
        
    def init_params_manual(self, cpy_change_rates, cat_switch_rates):
        nchrst, ncat = self.ncharstates, self.ncategories
        cpy_change_rates = _to_ndarray_of_shape(cpy_change_rates, (ncat, 2, nchrst-1))
        cat_switch_rates = _to_ndarray_of_shape(cat_switch_rates, (2, ncat*(ncat-1)//2))
        if self.__has_invariant_cat:
            cpy_change_rates[0] = 0
        self.update(
            cpy_change_rates=cpy_change_rates,
            cat_switch_rates=cat_switch_rates, 
        )
        
    def init_params_random(self, rng, distr_kw=None):
        nchrst, ncat = self.ncharstates, self.ncategories
        
        if self.__assume_priors:
            if distr_kw is not None:
                warnings.warn(
                    '\'distr_kw\' is ignored because this model already assumes priors.'
                )
            cpy_changeR_distr = self.__cpy_changeR_prior
            cat_switchR_distr = self.__cat_switchR_prior
            
        else:
            distr_kw = (distr_kw or {}).copy()
            distr_kw.setdefault('cpy_change_rates')
            distr_kw.setdefault('cat_switch_rates')
            cpy_changeR_distr, cat_switchR_distr = self._params_signature.bind(**distr_kw).args
            if cpy_changeR_distr is None:
                cpy_changeR_distr = Gamma(shape=2.0, scale=0.5)
                warnings.warn(
                    f'Drawing copy number change rates from {cpy_changeR_distr}; '
                    'no prior is assumed in this substitution model. '
                    'Check for consistency with your data and consider explicitly specifying '
                    '\'cpy_change_rates\' in \'distr_kw\'.'
                )
            if cat_switchR_distr is None:
                cat_switchR_distr = Gamma(shape=1.0, scale=0.1)
                warnings.warn(
                    f'Drawing category switching rates from {cat_switchR_distr}; '
                    'no prior is assumed in this substitution model. '
                    'Check for consistency with your data and consider explicitly specifying '
                    '\'cat_switch_rates\' in \'distr_kw\'.'
                )
            
        self.update(
            cpy_change_rates=(
                np.pad(
                    rng.gamma(*cpy_changeR_distr, size=(ncat - 1, 2, nchrst - 1)), 
                    pad_width=[(1, 0), (0, 0), (0, 0)]
                )
                if self.__has_invariant_cat else
                rng.gamma(*cpy_changeR_distr, size=(ncat, 2, nchrst - 1))
            ),
            cat_switch_rates=rng.gamma(*cat_switchR_distr, size=(2, ncat * (ncat - 1) // 2)),
        )
    
    def update(self, cpy_change_rates=None, cat_switch_rates=None):
        if cpy_change_rates is None and cat_switch_rates is None: 
            return
        
        ncategories, ncharstates = self.ncategories, self.ncharstates
        R_cat = np.zeros((ncategories, ncategories))
        lower_mask = np.tri(ncategories, dtype=bool, k=-1)
        upper_mask = lower_mask.T
        R_cat[upper_mask], R_cat[lower_mask] = cat_switch_rates
        R = block_diag(*[
            np.diag(upper, 1) + np.diag(lower, -1)
            for upper, lower in cpy_change_rates
        ]) + np.kron(R_cat, np.eye(ncharstates, dtype=bool))
        np.fill_diagonal(R, 0)
        np.fill_diagonal(R, -R.sum(axis=0))
        self.R = R
        
    @property
    def priors(self):
        return dict(
            cpy_change_rates=self.__cpy_changeR_prior,
            cat_switch_rates=self.__cat_switchR_prior
        )
    
    @property
    def flat_params(self):
        return self.R[self._flatidx]
    
    def _decompress_flat_params(self, p):
        nchrst, ncat = self.ncharstates, self.ncategories
        return dict(
            cat_switch_rates=p[:ncat*(ncat-1)].reshape(2, ncat*(ncat-1)//2), 
            cpy_change_rates=p[ncat*(ncat-1):].reshape(ncat, 2, nchrst-1)
        )
        
    # Removed the 'share_switch_rate' argument for simplicity
    # To reduce the number of degrees of freedom, specify an exponential distribution for the prior distribution.
    # If this option is to be reintroduced in the future, it will be specified in the initialization, not the fitting.
    def _get_next_params(self, tfdns):
        nchrst, ncat = self.ncharstates, self.ncategories    
        cpy_change_gamma = self.__cpy_changeR_prior
        cat_switch_gamma = self.__cat_switchR_prior
        
        cpy_change_rates = ((tfdns + cpy_change_gamma.shape - 1) / (np.diag(tfdns)[newax,:] + 1 / cpy_change_gamma.scale))[self._mask_cpy_ns].reshape((ncat,nchrst-1,2)).swapaxes(1,2)

        tfdns_cat = tfdns[self._mask_switch].reshape((ncat,nchrst,ncat)).sum(axis=1)
        #if share_switch_rate:
        #    tmp2 = (tfdns_cat.sum() - np.diag(tfdns_cat).sum()  + cat_switch_gamma_shape - 1) / (np.diag(tfdns_cat).sum() + 1 / cat_switch_gamma_scale) / (ncat - 1)
        #    cat_switch_rates = np.full((2, ncat * (ncat - 1) // 2), tmp2)
        #else:
        #    tmp2 = (tfdns_cat + cat_switch_gamma_shape - 1) / (np.diag(tfdns_cat)[newax,:] + 1 / cat_switch_gamma_scale)
        #    cat_switch_rates = tmp2[self._cat_idx].reshape(2, -1) 
        tmp2 = (tfdns_cat + cat_switch_gamma.shape - 1) / (np.diag(tfdns_cat)[newax,:] + 1 / cat_switch_gamma.scale)
        cat_switch_rates = tmp2[self._cat_idx].reshape(2, -1) 
        
        #cat_switch_rates[cat_switch_rates == 0] = 1 / np.nan_to_num(np.inf)
        #cpy_change_rates[cpy_change_rates == 0] = 1 / np.nan_to_num(np.inf)
        if self.has_invariant_cat:
            cpy_change_rates[0] = 0
            
        return dict(cat_switch_rates=cat_switch_rates, cpy_change_rates=cpy_change_rates)
    
    def _log_prior_prob(self):
        cpy_change_gamma = self.__cpy_changeR_prior
        cat_switch_gamma = self.__cat_switchR_prior
        
        nchrst, ncat = self.ncharstates, self.ncategories
        p = self.flat_params
        cpy_change_rates = p[ncat*(ncat-1)+int(self.has_invariant_cat)*(2*nchrst-2):]
        # cat_switch_rates = p[0] if share_switch_rate else p[:ncat*(ncat-1)]
        cat_switch_rates = p[:ncat*(ncat-1)]
        
        logP_cpy_changeR_prior = xlogy(cpy_change_gamma.shape - 1, cpy_change_rates).sum() - cpy_change_rates.sum() / cpy_change_gamma.scale
        logP_cat_switchR_prior = xlogy(cat_switch_gamma.shape - 1, cat_switch_rates).sum() - cat_switch_rates.sum() / cat_switch_gamma.scale
        
        return logP_cpy_changeR_prior + logP_cat_switchR_prior
