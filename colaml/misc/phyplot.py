import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ete3 import Tree
from abc import ABCMeta, abstractmethod

class abstTreePlotter(metaclass=ABCMeta):
    def __init__(
        self, branch_length=True, lc='k', lw=None, align=False, align_linekw=None,
        node_text='name', node_textkw=None, node_spacing='auto',   # global settings for nodes 
        branch_text=None, branch_textkw=None, show_confidence=True # global settings for branches
    ):
        self.branch_length = branch_length
        self.globallc = lc
        self.globallw = lw or plt.rcParams['lines.linewidth']
        if node_text is None:
            self.node_text_fn = lambda _: None
        elif isinstance(node_text, str):
            self.node_text_fn = lambda node: getattr(node, node_text, None)
        elif callable(node_text):
            self.node_text_fn = node_text
        else:
            raise ValueError('node_text should be None, str, or callable')
        self.node_textkw = node_textkw or {}
        self.node_spacing = node_spacing
        
        if show_confidence and branch_text is not None:
            warnings.warn('If branch_text is not None, show_confidence is ignored.')
            
        if branch_text is None:
            self.branch_text_fn = (lambda node: None if node.is_leaf() else node.support) if show_confidence else (lambda _: None)
        elif isinstance(branch_text, str):
            self.branch_text_fn = lambda node: getattr(node, branch_text, None)
        elif callable(branch_text):
            self.branch_text_fn = branch_text
        else:
            raise ValueError('branch_text should be None, str, or callable')
        self.branch_textkw = branch_textkw or {}
        
        self.align = align
        self.align_linekw = align_linekw or {}
        self.align_linekw.setdefault('color', 'gray')
        self.align_linekw.setdefault('lw', plt.rcParams['lines.linewidth'] * 0.5)
        self.align_linekw.setdefault('ls', '--')
        
    def _get_lateral_pos(self, tree):
        lpos = {}
        lim = -1
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                lpos[node] = lim = lim + 1
            else:
                lpos[node] = (lpos[node.children[0]] + lpos[node.children[-1]]) / 2
        return lpos
    
    def _get_vertical_pos(self, tree):
        return {
            node: tree.get_distance(node, topology_only=not self.branch_length) + tree.dist
            for node in tree.traverse('postorder')
        }
    
    @abstractmethod
    def _make_segments(self, lateral_pos, vertical_pos):
        pass
    
    @abstractmethod
    def _iter_node_text_args(self, tree, lateral_pos, vertical_pos, spacing):
        pass
    
    @abstractmethod
    def _iter_branch_text_args(self, tree, lateral_pos, vertical_pos):
        pass
    
    @abstractmethod
    def _make_align_segments(self, lateral_pos, vertical_pos):
        pass
    
    @abstractmethod
    def _adjust_ax(self, ax):
        return ax
    
    def _resolve_tree_config(self, tree, config, default, clade='auto', node=None):
        if clade == 'auto' and node == 'auto':
            raise ValueError('clade** and node** cannot be \'auto\' at the same time.')
        
        if clade == 'auto':
            get_clade_config_val = lambda n: getattr(n, config, None)
        elif isinstance(clade, dict):
            get_clade_config_val = clade.get
        elif callable(clade):
            get_clade_config_val = clade
        else:
            get_clade_config_val = lambda _: clade
        
        if node == 'auto':
            get_node_config_val = lambda n: getattr(n, config, None)
        elif isinstance(node, dict):
            get_node_config_val = node.get
        elif callable(node):
            get_node_config_val = node
        else:
            get_node_config_val = lambda _: node
        
        config_vals = {}
        for n in tree.traverse('preorder'):
            config_vals[n] = get_clade_config_val(n) or config_vals.get(n.up, default)
        for n in tree.traverse('preorder'):
            config_vals[n] = get_node_config_val(n) or config_vals[n]
    
        return config_vals
    
    def draw(self, tree, cladelc='auto', nodelc=None, cladelw='auto', nodelw=None, local_node_textkw=None, local_branch_textkw=None, ax=None): 
        if ax is None: ax = plt.gca()
        
        local_node_textkw = local_node_textkw or {}
        if isinstance(local_node_textkw, dict):
            local_node_textkw_fn = local_node_textkw.get
        elif callable(local_node_textkw):
            local_node_textkw_fn = local_node_textkw
        else:
            raise ValueError('local_node_textkw should be dict or callable')
            
        local_branch_textkw = local_branch_textkw or {}
        if isinstance(local_branch_textkw, dict):
            local_branch_textkw_fn = local_branch_textkw.get
        elif callable(local_branch_textkw):
            local_branch_textkw_fn = local_branch_textkw
        else:
            raise ValueError('local_branch_textkw should be dict or callable')
            
        lateral_pos, vertical_pos = self._get_lateral_pos(tree), self._get_vertical_pos(tree)
        segments = self._make_segments(lateral_pos, vertical_pos)
        lc = self._resolve_tree_config(tree, 'color', self.globallc, clade=cladelc, node=nodelc)
        lw = self._resolve_tree_config(tree, 'width', self.globallw, clade=cladelw, node=nodelw)
        
        lc = [*map(lc.__getitem__, segments.keys())]
        lw = [*map(lw.__getitem__, segments.keys())]
        ax.add_collection(LineCollection(segments.values(), color=lc, lw=lw))
        
        spacing = max(vertical_pos.values()) / 100 if self.node_spacing == 'auto' else self.node_spacing
        for node, args, kwargs in self._iter_node_text_args(tree, lateral_pos, vertical_pos, spacing):
            tmp_kw = self.node_textkw.copy()
            tmp_kw.update(kwargs or {})
            tmp_kw.update(local_node_textkw_fn(node) or {})
            ax.text(*args, **tmp_kw)
        
        for node, args, kwargs in self._iter_branch_text_args(tree, lateral_pos, vertical_pos):
            tmp_kw = self.branch_textkw.copy()
            tmp_kw.update(kwargs or {})
            tmp_kw.update(local_branch_textkw_fn(node) or {})
            ax.text(*args, **tmp_kw)
        
        if self.align:
            align_segments = self._make_align_segments(lateral_pos, vertical_pos, spacing)
            ax.add_collection(LineCollection(align_segments.values(), **self.align_linekw))
        
        self._adjust_ax(ax)
        return ax
    
class standardTreePlotter(abstTreePlotter):
    def __init__(self, horizontal=False, loffs=0, lscale=1, voffs=0, **kwargs):
        super().__init__(**kwargs)
        self.horizontal = horizontal
        self.node_textkw.setdefault('ha', 'left'  )
        self.node_textkw.setdefault('va', 'center')
        self.node_textkw.setdefault('rotation', 0 if horizontal else -90)
        self.node_textkw.setdefault('rotation_mode', 'anchor')
        self.loffs, self.lscale = loffs, lscale
        self.voffs = voffs
        
    def _vadjust(self, v):
        return v + self.voffs
    
    def _ladjust(self, l):
        return l * self.lscale + self.loffs
    
    def _make_segments(self, lateral_pos, vertical_pos):
        step = -1 if self.horizontal else 1
        return {
            node: 
            [(self._ladjust(lateral_pos[node   ]), self._vadjust(vertical_pos[node   ]))[::step],
             (self._ladjust(lateral_pos[node   ]), self._vadjust(vertical_pos[node.up]))[::step],
             (self._ladjust(lateral_pos[node.up]), self._vadjust(vertical_pos[node.up]))[::step]]
            if not node.is_root() else
            [(self._ladjust(lateral_pos[node]), self._vadjust(vertical_pos[node]))[::step],
             (self._ladjust(lateral_pos[node]), self._vadjust(0))[::step]]
            for node in lateral_pos.keys() & vertical_pos.keys()
        }
    
    def _iter_node_text_args(self, tree, lateral_pos, vertical_pos, spacing):
        tree_height = max(vertical_pos.values())
        for node in tree.traverse('preorder'):
            text = self.node_text_fn(node)
            if text is None: continue
            l = self._ladjust(lateral_pos[node]) 
            if self.align and node.is_leaf():
                v = self._vadjust(tree_height + spacing)
            else:
                v = self._vadjust(vertical_pos[node] + spacing)
            
            if self.horizontal:
                yield node, (v, l, text), None
            else:
                yield node, (l, v, text), None
            
    def _iter_branch_text_args(self, tree, lateral_pos, vertical_pos):
        for node in tree.traverse('preorder'):
            if node.is_root(): continue
            text = self.branch_text_fn(node)
            if text is None: continue
            v = self._vadjust((vertical_pos[node] + vertical_pos[node.up]) / 2)
            l = self._ladjust(lateral_pos[node])
            if self.horizontal:
                yield node, (v, l, text), None
            else:
                yield node, (l, v, text), None
            
    def _make_align_segments(self, lateral_pos, vertical_pos, spacing):
        tree_height = max(vertical_pos.values())
        step = -1 if self.horizontal else 1
        return {
            node: [
                (self._ladjust(lateral_pos[node]), self._vadjust(vertical_pos[node]))[::step],
                (self._ladjust(lateral_pos[node]), self._vadjust(tree_height + spacing))[::step],
            ]
            for node in lateral_pos.keys() & vertical_pos.keys()
            if node.is_leaf()
        }
    
    def _adjust_ax(self, ax):
        if self.horizontal:
            ax.margins(x=0.05, y=0)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - abs(self.lscale/2), ymax + abs(self.lscale/2))
        else:
            ax.margins(x=0, y=0.05)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - abs(self.lscale/2), xmax + abs(self.lscale/2))
        ax.invert_yaxis()
        return ax

class slantedTreePlotter(standardTreePlotter):
    def _make_segments(self, lateral_pos, vertical_pos):
        step = -1 if self.horizontal else 1
        return {
            node: [
                (self._ladjust(lateral_pos[node]), self._vadjust(vertical_pos[node]))[::step],
                (self._ladjust(lateral_pos.get(node.up, lateral_pos[node])), self._vadjust(vertical_pos.get(node.up, 0)))[::step]
            ]
            for node in lateral_pos.keys() & vertical_pos.keys()
            #if not node.is_root()
        }
    
    def _iter_branch_text_args(self, tree, lateral_pos, vertical_pos):
        for node in tree.traverse('preorder'):
            if node.is_root(): continue
            text = self.branch_text_fn(node)
            if text is None: continue
            v = self._vadjust((vertical_pos[node] + vertical_pos[node.up]) / 2)
            l = self._ladjust((lateral_pos[node] + lateral_pos[node.up]) / 2)
            if self.horizontal:
                yield node, (v, l, text), None
            else:
                yield node, (l, v, text), None

class circularTreePlotter(abstTreePlotter):
    def __init__(self, aoffs=0, ascale=1, roffs=0, **kwargs):
        super().__init__(**kwargs)
        self.node_textkw.setdefault('ha', 'left'  )
        self.node_textkw.setdefault('va', 'center')
        self.node_textkw.setdefault('rotation_mode', 'anchor')        
        self.branch_textkw.setdefault('ha', 'left'  )
        self.branch_textkw.setdefault('va', 'center')
        self.branch_textkw.setdefault('rotation_mode', 'anchor')        
        if ascale > 1 or ascale < -1:
            raise ValueError('ascale should be within [-1, 1].')
        self.aoffs, self.ascale = np.deg2rad(aoffs), ascale
        self.roffs = roffs
        
    def _radjust(self, r):
        return r + self.roffs
    
    def _aadjust(self, a):
        return a * self.ascale + self.aoffs
    
    # returns radian
    def _get_lateral_pos(self, tree):
        lpos = super()._get_lateral_pos(tree)
        tree_width = max(lpos.values()) + 1
        return {n: np.deg2rad(p * 360 / tree_width) for n, p in lpos.items()}
    
    def _make_segments(self, lateral_pos, vertical_pos):
        segments = {}
        
        for node in lateral_pos.keys() & vertical_pos.keys():
            r0, rad0 = self._radjust(vertical_pos[node]), self._aadjust(lateral_pos[node])
            r1, rad1 = self._radjust(vertical_pos.get(node.up, 0)), self._aadjust(lateral_pos.get(node.up, lateral_pos[node]))
                
            n = int(2 ** np.ceil(abs(rad0 - rad1) / np.pi * 2)) * 10 + 1
            steps = np.linspace(rad1, rad0, n, endpoint=True)
            
            segments[node] = np.concatenate([
                np.array([np.cos(steps), np.sin(steps)]).T * r1,
                np.array([[np.cos(rad0), np.sin(rad0)]]) * r0,
            ])
            
        return segments
    
    def _iter_node_text_args(self, tree, lateral_pos, vertical_pos, spacing):
        rmax = self._radjust(max(vertical_pos.values()))
        
        for node in tree.traverse('preorder'):
            if self.align and node.is_leaf():
                r = rmax
            else:
                r = self._radjust(vertical_pos[node] + spacing)
            rad = self._aadjust(lateral_pos[node])
            text = self.node_text_fn(node)
            if text is None: continue
            yield node, (r * np.cos(rad), r * np.sin(rad), text), dict(rotation=np.rad2deg(rad))
        
    def _iter_branch_text_args(self, tree, lateral_pos, vertical_pos):
        for node in tree.traverse('preorder'):
            if node.is_root(): continue
            text = self.branch_text_fn(node)
            if text is None: continue
            r = self._radjust((vertical_pos[node] + vertical_pos[node.up]) / 2)
            rad = self._aadjust(lateral_pos[node])
            yield node, (r * np.cos(rad), r * np.sin(rad), text), dict(rotation=np.rad2deg(rad) + 90)
    
    def _make_align_segments(self, lateral_pos, vertical_pos, spacing):
        r1 = self._radjust(max(vertical_pos.values()))
        align_segments = {}
        for node in lateral_pos.keys() & vertical_pos.keys():
            if not node.is_leaf(): continue
            r0, rad0 = self._radjust(vertical_pos[node]), self._aadjust(lateral_pos[node])
            align_segments[node] = [
                (r0 * np.cos(rad0), r0 * np.sin(rad0)),
                (r1 * np.cos(rad0), r1 * np.sin(rad0)),
            ]
        return align_segments
    
    def _adjust_ax(self, ax):
        ax.autoscale()
        ax.set_aspect('equal')
        return ax

_SupportedModes = {
    'standard': standardTreePlotter,
    'slanted' : slantedTreePlotter ,
    'circular': circularTreePlotter,
}

def draw(
    tree, mode='standard', ax=None, 
    cladelc='auto', nodelc=None, cladelw='auto', nodelw=None, 
    local_node_textkw=None, local_branch_textkw=None, 
    **general_kw
):
    plotter = _SupportedModes[mode](**general_kw)
    
    return plotter.draw(
        tree, ax=ax, cladelc=cladelc, nodelc=nodelc, cladelw=cladelw, nodelw=nodelw, 
        local_node_textkw=local_node_textkw, local_branch_textkw=local_branch_textkw
    )
