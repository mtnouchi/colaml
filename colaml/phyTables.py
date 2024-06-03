from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from itertools import compress
from operator import attrgetter
from typing import NamedTuple, Any
from types import MappingProxyType # something like "frozendict"

import ete3
import numpy as np
from numpy.typing import NDArray

class AssumePostorder(tuple):
    """Tuple *expected* to be sorted in post-order.
    
    Note that users are responsible for sorting the input iterable.
    """
    pass


class PartialColumn(NamedTuple):
    uniqs: np.ndarray | None
    codes: np.ndarray


class PostorderSerializedTree:
    __slots__ = ('__names', '__branch_lengths', '__node_relations')
    
    class NodeDump(NamedTuple):
        parent_idx     : int
        children_idxs  : tuple[int, ...]
        sibling_ordinal: int

        def is_leaf(self):
            return not bool(self.children_idxs)

    def __init__(self, tree: ete3.Tree):
        nodes = AssumePostorder(tree.traverse('postorder'))
        dumper = attrgetter('name', 'dist', 'up', 'children')
        names, branch_lengths, parents, children_ls = zip(*map(dumper, nodes))

        batch_indexer = lambda nds: tuple(map(nodes.index, nds))
        parent_idxs   = AssumePostorder(batch_indexer(parents[:-1]))
        children_idxs = AssumePostorder(map(batch_indexer, children_ls))
        sibling_ordinal = AssumePostorder(
            children_idxs[par].index(cur) for cur, par in enumerate(parent_idxs)
        )
        
        self.__names = AssumePostorder(names)
        self.__branch_lengths = AssumePostorder(branch_lengths[:-1])
        self.__node_relations = AssumePostorder(map(
            self.NodeDump, 
            parent_idxs + (-1,), children_idxs, sibling_ordinal + (0,)
        ))

    def fill_names(self, *, ignore_tips=True, fmt='_Node{}'):
        self.__names = AssumePostorder(
            name if name or (ignore_tips and node.is_leaf()) else fmt.format(i)
            for i, (name, node) in enumerate(zip(self.names, self.nodes))
        )
        
    @property
    def names(self): 
        return self.__names
        
    @property
    def branch_lengths(self):
        arr = np.array(self.__branch_lengths)
        arr.setflags(write=False)
        return arr
    
    @branch_lengths.setter
    def branch_lengths(self, val):
        val = np.asarray(val)
        if val.shape != self.branch_lengths.shape:
            msg = f'Invalid shape. Expected: {self.branch_lengths.shape}, but got: {val.shape}.'
            raise ValueError(msg)
        self.__branch_lengths = AssumePostorder(val)
    
    @property
    def node_relations(self):
        return self.__node_relations
    nodes = node_relations # alias

    @property
    def nnodes(self):
        return len(self.__node_relations)
    
    @property
    def ntips(self):
        return sum(node.is_leaf() for node in self.nodes)

    def to_ete3(self):
        tree = ete3.Tree.from_parent_child_table([
            (current_idx, child_idx, self.__branch_lengths[child_idx]) 
            for current_idx, current in enumerate(self.__node_relations)
            for child_idx in current.children_idxs
        ])
        tree.dist = 0.0
        for node in tree.traverse():
            node.name = self.__names[node.name]
        return tree

def _detect_ncols(data):
    try:
        ncols, = set(map(len, data.values()))
    except ValueError as err:
        msg = 'All rows must have the same number of columns.'
        raise ValueError(msg) from err
    return ncols

class ExtantPhyTable:
    __slots__ = ('__index_in_postorder', '__tipstates', '__ncols', '__weakref__')

    def __init__(self, tipstates, index_in_postorder):
        # validate arguments
        ncols = _detect_ncols(tipstates)
        
        tips = tuple(compress(
            index_in_postorder.names, 
            (nd.is_leaf() for nd in index_in_postorder.nodes)
        ))
        if len(tips) != len(set(tips)):
            msg = 'All tips must be uniquely named.'
            raise ValueError(msg)
        if set(tips) != set(tipstates.keys()):
            msg = 'Missing or extra node(s) in \'tiptates\'.'
            raise ValueError(msg)
        
        # set values
        self.__index_in_postorder = index_in_postorder
        self.__tipstates = tipstates
        self.__ncols = ncols
        
    def to_dict(self, *, copy=True):
        if not copy:
            return self.__tipstates
        return deepcopy(self.__tipstates)
    
    @property
    def ncols(self):
        return self.__ncols
    
    @property
    def index_in_postorder(self):
        return self.__index_in_postorder
    tree = index_in_postorder

    @property
    @cache
    def inside_encoding(self):
        inside = []
        for name, node in zip(self.tree.names, self.tree.nodes):
            if node.is_leaf():
                uniqs, codes = None, np.array(self.__tipstates[name])
                codes.setflags(write=False)
            else:
                uniqs, codes = np.unique(
                    np.array([
                        inside[child_id].codes for child_id in node.children_idxs
                    ]), 
                    return_inverse=True, axis=1
                )
                uniqs.setflags(write=False)
                codes.setflags(write=False)
            inside.append(PartialColumn(uniqs, codes))
        
        return AssumePostorder(inside)
        
    @property
    def min_changes(self):
        from .misc.parsimony import min_changes
        return min_changes(self)
        
class ReconPhyTable:
    @dataclass
    class ColumnAttr:
        label  : str
        __data : NDArray
        def to_list(self, *, copy=True):
            if not copy:
                return list(self.__data)
            return list(deepcopy(self.__data))

    @dataclass
    class NodeAttr:
        label  : str
        __data : dict[str, Any]
        def to_dict(self, *, copy=True):
            if not copy:
                return dict(self.__data)
            return dict(deepcopy(self.__data))
    
    StateAttr = NodeAttr
    
    @dataclass(frozen=True)
    class ReconContext():
        method    : str
        treemodel : type
        tm_args   : tuple[Any, ...]
        tm_params : NDArray
        notes     : None | str = None
        
    def __init__(self, charstates, index_in_postorder, 
                 otherstates=(), colattrs=(), nodeattrs=(), metadata_kw=None):
        ncols = _detect_ncols(charstates)
        if len(index_in_postorder.names) != len(set(index_in_postorder.names)):
            msg = 'All nodes must be uniquely named.'
            raise ValueError(msg)
        if set(index_in_postorder.names) != set(charstates.keys()):
            msg = 'Missing or extra node(s) in \'charstates\'.'
            raise ValueError(msg)
        
        otherstates_dict = {}
        for label, data in otherstates:
            if label in otherstates_dict:
                msg = f'Duplicated label \'{label}\' in \'otherstates\'.'
                raise ValueError(msg)
            if set(index_in_postorder.names) != set(data.keys()):
                msg = f'Missing or extra node(s) in \'otherstates\' ({label}).'
                raise ValueError(msg)
            if _detect_ncols(data) != ncols:
                msg = f'Invalid column counts ({label}).'
                raise ValueError(msg)
            otherstates_dict[label] = self.StateAttr(label, data)
            
        colattrs_dict = {}
        for label, data in colattrs:
            if label in colattrs_dict:
                msg = f'Duplicated label \'{label}\' in colattrs.'
                raise ValueError(msg)
            if len(data) != ncols:
                msg = f'Invalid column counts ({label}).'
                raise ValueError(msg)
            colattrs_dict[label] = self.ColumnAttr(label, data)
            
        nodeattrs_dict = {}
        for label, data in nodeattrs:
            if set(index_in_postorder.names) != set(data.keys()):
                msg = f'Inconsitent node(s) found in \'tree\' and \'nodeattrs\'.'
                raise ValueError(msg)
            if label in nodeattrs_dict:
                msg = f'Duplicated label \'{label}\' in nodeattrs.'
                raise ValueError(msg)
            nodeattrs_dict[label] = self.NodeAttr(label, data)
        
        metadata = None
        if metadata_kw is not None:
            metadata = self.ReconContext(**metadata_kw)
            
        self.__index_in_postorder = index_in_postorder
        self.__ncols = ncols
        self.__charstates = charstates
        self.__otherstates = MappingProxyType(otherstates_dict)
        self.__colattrs  = MappingProxyType(colattrs_dict )
        self.__nodeattrs = MappingProxyType(nodeattrs_dict)
        self.__metadata = metadata
        
    @property
    def ncols(self):
        return self.__ncols
    
    @property
    def metadata(self):
        return self.__metadata
    
    @property
    def index_in_postorder(self):
        return self.__index_in_postorder
    tree = index_in_postorder

    def to_dict(self, *, copy=True):
        if not copy:
            return self.__charstates
        return deepcopy(self.__charstates)
        
    @property
    def otherstates(self):
        return self.__otherstates
    
    @property
    def colattrs(self):
        return self.__colattrs
        
    @property
    def nodeattrs(self):
        return self.__nodeattrs 

    def drop_past(self):
        tipstates = {
            name: self.__charstates[name]
            for name, node in zip(self.tree.names, self.tree.nodes)
            if node.is_leaf()
        }
        return ExtantPhyTable(tipstates, self.tree)
