import warnings
import weakref
import numpy as np
from numpy import newaxis as newax
from numpy.core.multiarray import c_einsum
from scipy.linalg import expm
from scipy.special import logsumexp

class SingleRateTreeStats:
    __slots__ = (
        '_model', '_phytbl', 
        'branchP', 'stateP', 'aux_insideP', 'insideP', 'scale', 
        'aux_outsideP', 'outsideP', 'tfdns', 'post_rootP', 'col_loglik'
    )
    @staticmethod
    def __hyper_params(model, phytbl):
        return model._ncompoundstates, model.ncharstates, \
               phytbl.ncols, phytbl.tree.nnodes
    
    def __init__(self, model, phytbl):
        ncmpst, nchrst, ncols, nnodes = self.__hyper_params(model, phytbl)
        nuniqcols = [
            getattr(partial_cols.uniqs, 'shape', (-1, nchrst))[1]
            for partial_cols in phytbl.inside_encoding
        ]
        self._model       = weakref.proxy(model)
        self._phytbl      = weakref.proxy(phytbl)
        self.branchP      = np.empty((nnodes-1, ncmpst, ncmpst)) # (m, a, b)
        self.stateP       = np.empty((nnodes, ncmpst))           # (m, a)
        self.aux_insideP  = [np.empty((ncmpst, nuniqcols_m))
                             for nuniqcols_m in nuniqcols]       # (m, b, n)
        self.insideP      = [np.empty((ncmpst, nuniqcols_m))
                             for nuniqcols_m in nuniqcols]       # (m, a, n)
        self.scale        = [np.empty(nuniqcols_m)
                             for nuniqcols_m in nuniqcols]       # (m, n)
        self.aux_outsideP = np.empty((nnodes, ncmpst, ncols))    # (m, b, n)
        self.outsideP     = np.empty((nnodes, ncmpst, ncols))    # (m, b, n)
        self.tfdns        = np.zeros((ncmpst, ncmpst))           # (a, b)
        self.post_rootP   = np.empty((ncmpst, ncols))            # (a, n)
        self.col_loglik   = np.empty(ncols)                      # (n,)
        
    def compute(self):
        model, phytbl = self._model, self._phytbl
        # timer4compute = Timer().start()
        ncmpst = model._ncompoundstates
        sbstmdl = model.substmodel
        tree = phytbl.tree
        nnodes = tree.nnodes
        
        ## m: node or branch upon it
        ## n: column (~: factorized)
        ## a, b, i, j: state
        
        # calculate state transtion probabilities in each branch
        self.branchP = expm(
            c_einsum('m,ij->mij', tree.branch_lengths, sbstmdl.R)
        ) # (m, a, b)
        
        # calculate state probability for each nodes (serves as weights in scaling)
        self.stateP[-1] = model.root_probs
        for current_idx in reversed(range(nnodes - 1)):
            parent_idx = tree.nodes[current_idx].parent_idx
            np.dot(
                self.branchP[current_idx], self.stateP[parent_idx], 
                out=self.stateP[current_idx]
            ) # ab,b->a
        
        # calculate inside variables
        for current_idx in range(nnodes):
            children_idxs = tree.nodes[current_idx].children_idxs
            if children_idxs:
                children_uniq_codes = phytbl.inside_encoding[current_idx].uniqs
                ch, cuniq = children_idxs[0], children_uniq_codes[0]
                np.copyto(self.insideP[current_idx], self.aux_insideP[ch][:,cuniq])
                for ch, cuniq in zip(children_idxs[1:], children_uniq_codes[1:]):
                    self.insideP[current_idx] *= self.aux_insideP[ch][:,cuniq]
            else:
                np.copyto(self.insideP[current_idx], model._compound_state_flags)
                
            np.dot(
                self.stateP[current_idx], self.insideP[current_idx], 
                out=self.scale[current_idx]
            ) # b,bn->n
            self.insideP[current_idx] /= self.scale[current_idx][newax,:]

            if current_idx == nnodes - 1: break

            np.dot(
                self.branchP[current_idx].T, self.insideP[current_idx], 
                out=self.aux_insideP[current_idx]
            ) # ab,an->bn
                
        # calculate ouside variables 
        # timer4outside = Timer().start()
        self.outsideP[-1] = model.root_probs[:,newax]
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
                         * (self.insideP[parent_idx]
                             / self.aux_insideP[current_idx][:,cur2par_codes])[:,parent_pcols.codes]
                    ) # bn,(bn~,bn~~)->bn
                    if warn_records:
                        np.nan_to_num(self.aux_outsideP[current_idx], copy=False) # just first-aid

                np.dot(
                    self.branchP[current_idx], self.aux_outsideP[current_idx], 
                    out=self.outsideP[current_idx]
                ) # ab,bn->an
        # timer4outside.stop()
                
        # obtain likelihoods and posterior probs
        self.post_rootP = c_einsum('an,a->an', self.insideP[-1], model.root_probs)[:,phytbl.inside_encoding[-1].codes]

        self.col_loglik.fill(0)    
        for sc, partial_cols in zip(self.scale, phytbl.inside_encoding):
            self.col_loglik += np.log(sc)[partial_cols.codes]

        # calculate sufficient statistics
        # timer4suffstats = Timer().start()
        self.tfdns[:] = 0 #[sbstmdl._mask_nonzero] = 0
        for current_idx in range(nnodes - 1):
            dist = tree.branch_lengths[current_idx]
            partial_cols = phytbl.inside_encoding[current_idx]
            self.tfdns[sbstmdl._mask_nonzero] += np.dot(
                # '(an,bn->ab),abh->h'; h: non-zero (i, j) elememnts
                np.dot(
                    self.insideP[current_idx][:,partial_cols.codes], 
                    self.aux_outsideP[current_idx].T
                ).reshape(-1),
                sbstmdl.sufficient_stats(dist)[:,:,sbstmdl._mask_nonzero].reshape(ncmpst**2, -1)
            )
        assert (self.tfdns[sbstmdl._mask_nonzero] >= 0).all(), \
           f'tfdns < 0 in {[*zip(*np.where(self.tfdns < 0))]}: {self.tfdns[self.tfdns < 0]}'
        # timer4suffstats.stop()
        # timer4compute.stop()
        # timelogger.debug(
        #     '%f\t%f\t%f\t%f\t%f\t%f',
        #     timer4compute  .perf_counter, timer4compute.  process_time, 
        #     timer4outside  .perf_counter, timer4outside  .process_time, 
        #     timer4suffstats.perf_counter, timer4suffstats.process_time, 
        # )
        
class MultipleRateTreeStats:
    __slots__ = (
        '_model', '_phytbl', 
        'branchP', 'stateP', 'aux_insideP', 'insideP', 'scale', 
        'aux_outsideP', 'outsideP', 'tfdns', 'post_rootP', 'col_loglik', 
        'post_mixtureP'
    )
    @staticmethod
    def __hyper_params(model, phytbl):
        return model._ncompoundstates, model.ncharstates, model.nmixtures, \
               phytbl.ncols, phytbl.tree.nnodes
    
    def __init__(self, model, phytbl):
        ncmpst, nchrst, nmixt, ncols, nnodes = self.__hyper_params(model, phytbl)
        
        nuniqcols = [
            getattr(partial_cols.uniqs, 'shape', (0, nchrst))[1]
            for partial_cols in phytbl.inside_encoding
        ]
        self._model        = weakref.proxy(model)
        self._phytbl       = weakref.proxy(phytbl)
        self.branchP       = np.empty((nnodes-1, nmixt, ncmpst, ncmpst)) # (m, k, a, b)
        self.stateP        = np.empty((nnodes, nmixt, ncmpst))           # (m, k, a)
        self.aux_insideP   = [np.empty((nmixt, ncmpst, nuniqcols_m))
                              for nuniqcols_m in nuniqcols]              # (m, k, b, n)
        self.insideP       = [np.empty((nmixt, ncmpst, nuniqcols_m))
                              for nuniqcols_m in nuniqcols]              # (m, k, a, n)
        self.scale         = [np.empty((nmixt, nuniqcols_m))
                              for nuniqcols_m in nuniqcols]              # (m, k, n)
        self.aux_outsideP  = np.empty((nnodes, nmixt, ncmpst, ncols))    # (m, k, b, n)
        self.outsideP      = np.empty((nnodes, nmixt, ncmpst, ncols))    # (m, k, b, n)
        self.tfdns         = np.zeros((nmixt, ncmpst, ncmpst))           # (k, a, b)
        self.post_rootP    = np.empty((nmixt, ncmpst, ncols))            # (k, a, n)
        self.post_mixtureP = np.empty((nmixt, ncols))                    # (k, n)
        self.col_loglik    = np.empty(ncols)                             # (n,)    
    
    def compute(self):
        model, phytbl = self._model, self._phytbl
        tree = phytbl.tree
        nmixt = model.nmixtures
        nnodes = tree.nnodes
        
        ## m: node or branch upon it
        ## k: mixture
        ## n: column (cached)
        ## a, b, i, j: state
            
        # calculate state transtion probs. in each branch
        for k, sbstmdl in enumerate(model.substmodels):
            self.branchP[:, k] = expm(
                c_einsum('m,ab->mab', tree.branch_lengths, sbstmdl.R)
            )
        
        # calculate state probability for each nodes (serves as weights in scaling)
        self.stateP[-1] = model.root_probs
        for current_idx in reversed(range(nnodes - 1)):
            parent_idx = tree.nodes[current_idx].parent_idx
            c_einsum(
                'kab,kb->ka', 
                self.branchP[current_idx], self.stateP[parent_idx], 
                out=self.stateP[current_idx]
            )
            
        # calculate inside variables
        for current_idx in range(nnodes):
            children_idxs = tree.nodes[current_idx].children_idxs
            if children_idxs:
                children_uniq_codes = phytbl.inside_encoding[current_idx].uniqs
                ch, cuniq = children_idxs[0], children_uniq_codes[0]
                np.copyto(self.insideP[current_idx], self.aux_insideP[ch][:,:,cuniq])
                for ch, cuniq in zip(children_idxs[1:], children_uniq_codes[1:]):
                    self.insideP[current_idx] *= self.aux_insideP[ch][:,:,cuniq] 
            else:
                np.copyto(self.insideP[current_idx], model._compound_state_flags)
            
            c_einsum(
                'kbn,kb->kn', 
                self.insideP[current_idx], self.stateP[current_idx],
                out=self.scale[current_idx]
            ) # kbn,kb->kn
            
            self.insideP[current_idx] /= self.scale[current_idx][:,newax,:]
            
            if current_idx == nnodes - 1: break
            np.matmul(
                self.branchP[current_idx].swapaxes(1, 2), self.insideP[current_idx], 
                out=self.aux_insideP[current_idx]
            ) # kan,kab->kbn
                    
        # calculate ouside variables
        self.outsideP[-1] = model.root_probs[:,:,newax]
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
                         * (self.insideP[parent_idx] 
                             / self.aux_insideP[current_idx][:,:,cur2par_codes])[:,:,parent_pcols.codes]
                    ) # kbn,(kbn~,kbn~~)->kbn
                    if warn_records:
                        np.nan_to_num(self.aux_outsideP[current_idx], copy=False) # just first-aid
                    
                np.matmul(
                    self.branchP[current_idx], self.aux_outsideP[current_idx], 
                    out=self.outsideP[current_idx]
                ) # kab,kbn->kan
                
        self.post_rootP = c_einsum(
            'kan,ka->kan',
            self.insideP[-1], model.root_probs
        )[:,:,phytbl.inside_encoding[-1].codes] # kan,ka->kan

        with np.errstate(divide='ignore'):
            log_mixtureP = np.log(model.mixture_probs)

        self.post_mixtureP.fill(0)
        for sc, partial_cols in zip(self.scale, phytbl.inside_encoding):
            self.post_mixtureP += np.log(sc)[:,partial_cols.codes]
        self.post_mixtureP += log_mixtureP[:,newax]

        self.col_loglik = logsumexp(self.post_mixtureP, axis=0) # (n,)
        self.post_mixtureP -= self.col_loglik[newax,:]
        np.exp(self.post_mixtureP, out=self.post_mixtureP) # (k, n)

        # calculate sufficient statistics
        self.tfdns.fill(0) # (k, i, j)
        mask = np.array([sm._mask_nonzero for sm in model.substmodels])
        rept = np.repeat(np.arange(nmixt), np.count_nonzero(mask, axis=(1,2)))
        for current_idx in range(nnodes - 1):
            dist = tree.branch_lengths[current_idx]
            partial_cols = phytbl.inside_encoding[current_idx]
            self.tfdns[mask] += c_einsum(
                'abh,hab->h', # h: non-zero (k,i,j) element
                np.dstack([
                    sm.sufficient_stats(dist)[:,:,sm._mask_nonzero]
                    for sm in model.substmodels
                ]), 
                np.matmul( # kan,kbn,kn->kab
                    self.insideP[current_idx][:,:,partial_cols.codes], 
                    (self.aux_outsideP[current_idx] * self.post_mixtureP[:,newax,:]).swapaxes(1, 2)
                )[rept,:,:],
            )
                
        assert (self.tfdns >= 0).all(), \
               f'tdfns < 0 in {[*zip(*np.where(self.tfdns < 0))]}: {self.tfdns[self.tfdns < 0]}'
