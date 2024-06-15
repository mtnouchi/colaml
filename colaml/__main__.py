import argparse
import gzip
import json
import warnings
from logging import DEBUG, Formatter, StreamHandler, getLogger
from pathlib import Path

import numpy as np

# TODO fix slow import (substModels)
import colaml
import colaml.unsupported

logger = getLogger()


def phytbl_from_json(path: Path, lmax: int):
    # from itertools import compress
    # from operator import methodcaller

    # import pandas as pd
    from ete3 import Tree

    open_func = open
    if path.name.endswith('.gz'):
        open_func = gzip.open
    with open_func(path, 'rt') as input_file:
        input_data = json.load(input_file)

    tree = colaml.PostorderSerializedTree(Tree(input_data['tree'], format=3))

    OGs = dict(
        zip(
            input_data['OGs']['index'],
            np.nan_to_num(input_data['OGs']['data'], nan=0).clip(0, lmax).astype(int),
        )
    )

    # OGs = (
    #     pd.DataFrame(**input_data['OGs'])
    #     .T.reindex(columns=compress(tree.names, map(methodcaller('is_leaf'), tree.nodes)))
    #     .fillna(0)
    #     .astype(int)
    #     .clip(upper=lmax)
    #     .to_dict(orient='list')
    # )

    return colaml.ExtantPhyTable(OGs, tree), input_data['OGs'].get('columns')


def gamma_par(string: str):
    from colaml.substModels import Gamma

    return Gamma(*map(float, string.split(',')))


def dirichlet_par(string):
    from colaml.treeModels import Dirichlet

    return Dirichlet([*map(float, string.split(','))])


def get_parser():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@', description='Command line interface of CoLaML'
    )
    parser.add_argument(
        '--version', action='version', version=f'CoLaML v{colaml.__version__}'
    )
    subparser = parser.add_subparsers(
        title='subcommand', dest='subcommand', required=True
    )

    # fit
    fit_parser = subparser.add_parser('fit', help='Fit model parameters to data')

    ## fit-commons
    fit_parser_commons = argparse.ArgumentParser(add_help=False)
    fit_parser_commons.add_argument(
        '-q',
        '--no-progress',
        dest='pbar',
        action='store_false',
        help='suppress progress bar',
    )
    fit_parser_commons.add_argument(
        '-i',
        '--input',
        required=True,
        type=Path,
        help='path to input json file (can be gzipped)',
    )
    fit_parser_commons.add_argument(
        '-o', '--output', required=True, type=Path, help='path+prefix for output files'
    )
    fit_parser_commons.add_argument(
        '--max-iter',
        dest='max_rounds',
        default=5000,
        type=int,
        help='maximum iterations in EM',
    )
    fit_parser_commons.add_argument(
        '--seed', required=True, type=int, help='random seed'
    )
    fit_parser_commons.add_argument(
        '--lmax', required=True, type=int, help='max gene copy number'
    )

    ## fit models
    fit_subparser = fit_parser.add_subparsers(
        title='model', dest='model', required=True
    )

    mmm_parser = fit_subparser.add_parser(
        'mmm', parents=[fit_parser_commons], help='Markov-modulated model'
    )
    mmm_parser.add_argument(
        '--ncat', required=True, type=int, help='#(rate categories)'
    )

    mmm_adv_group = mmm_parser.add_argument_group(
        title='advanced options for MAP estimation'
    )
    mmm_adv_group.add_argument(
        '--map', action='store_true', help='enable MAP estimation '
    )
    mmm_adv_group.add_argument(
        '--gainloss-gamma',
        dest='prior_cpy_change_rates',
        type=gamma_par,
        metavar='SHAPE,SCALE',
        help='gamma prior of gain/loss rate',
    )
    mmm_adv_group.add_argument(
        '--switch-gamma',
        dest='prior_cat_switch_rates',
        type=gamma_par,
        metavar='SHAPE,SCALE',
        help='gamma prior of switch rate',
    )
    mmm_adv_group.add_argument(
        '--copy-root-dirichlet',
        dest='prior_cpy_root_probs',
        type=dirichlet_par,
        metavar='ALPHA[,ALPHA[,...]]',
        help='Dirichlet prior of copy root probs',
    )
    mmm_adv_group.add_argument(
        '--cat-root-dirichlet',
        dest='prior_cat_root_probs',
        type=dirichlet_par,
        metavar='ALPHA[,ALPHA[,...]]',
        help='Dirichlet prior of category root probs',
    )

    mirage_parser = fit_subparser.add_parser(
        'mirage', parents=[fit_parser_commons], help='Mirage: mixture model'
    )
    mirage_parser.add_argument(
        '--nmixtures', required=True, type=int, help='#(rate mixtures)'
    )

    mirage_adv_group = mirage_parser.add_argument_group(
        title='advanced options for MAP estimation'
    )
    mirage_adv_group.add_argument(
        '--map', action='store_true', help='enable MAP estimation '
    )
    mirage_adv_group.add_argument(
        '--rate-gamma',
        dest='prior_rates',
        type=gamma_par,
        metavar='SHAPE,SCALE',
        help='gamma prior of gain/loss rate',
    )
    mirage_adv_group.add_argument(
        '--root-dirichlet',
        dest='prior_root_probs',
        type=dirichlet_par,
        metavar='ALPHA[,ALPHA[,...]]',
        help='Dirichlet prior of root probs',
    )
    mirage_adv_group.add_argument(
        '--mixture-dirichlet',
        dest='prior_mixture_probs',
        type=dirichlet_par,
        metavar='ALPHA[,ALPHA[,...]]',
        help='Dirichlet prior of mixture probs',
    )

    branch_parser = fit_subparser.add_parser(
        'branch', parents=[fit_parser_commons], help='Branch model'
    )

    branch_adv_group = branch_parser.add_argument_group(
        title='advanced options for MAP estimation'
    )
    branch_adv_group.add_argument(
        '--map', action='store_true', help='enable MAP estimation '
    )
    branch_adv_group.add_argument(
        '--rate-gamma',
        dest='prior_rates',
        type=gamma_par,
        metavar='SHAPE,SCALE',
        help='gamma prior of gain/loss rate',
    )
    branch_adv_group.add_argument(
        '--root-dirichlet',
        dest='prior_root_probs',
        type=dirichlet_par,
        metavar='ALPHA[,ALPHA[,...]]',
        help='Dirichlet prior of root probs',
    )

    # recon
    recon_parser = subparser.add_parser('recon', help='Reconstruct ancestral states')
    recon_parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=Path,
        help='path to input json file (can be gzipped)',
    )
    recon_parser.add_argument(
        '-m',
        '--model',
        required=True,
        type=Path,
        help='path to fitting json file (can be gzipped)',
    )
    recon_parser.add_argument(
        '-o', '--output', required=True, type=Path, help='path to output file'
    )
    recon_parser.add_argument(
        '--method',
        required=True,
        choices=['joint', 'marginal'],
        help='reconstruction method',
    )

    # TODO file converter
    # conv_parser = subparser.add_parser(
    #    'conv', help='Convert file formats')

    return parser


def configure_emEM_init(args):
    from colaml.fitting import StopCriteria

    init_kw = dict(
        ntrial=25,
        max_rounds=100,
        stop_criteria=StopCriteria(atol=0, rtol=0.001),
        show_init_progress=False,
        show_progress=False,
    )

    init_kw['show_init_progress'] = args.pbar

    return init_kw


def configure_parabolicEM(args):
    from colaml.fitting import BezierGeomGrid, StopCriteria

    fit_kw = dict(
        method='parabolic_EM',
        max_rounds=5000,
        stop_criteria=StopCriteria(atol=1e-6, rtol=1e-6),
        grid=BezierGeomGrid(first=0.1, ratio=1.5),
        show_progress=False,
    )

    fit_kw['max_rounds'] = args.max_rounds
    fit_kw['show_progress'] = args.pbar

    return fit_kw


class CoLaMLEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def fit_mmm(args):
    phytbl, _ = phytbl_from_json(args.input, args.lmax)
    approx_rate = phytbl.min_changes.mean() / phytbl.tree.branch_lengths.sum()

    # additional parsing
    prior_cpy_change_rates = args.prior_cpy_change_rates
    prior_cat_switch_rates = args.prior_cat_switch_rates
    prior_cpy_root_probs = args.prior_cpy_root_probs
    prior_cat_root_probs = args.prior_cat_root_probs

    if args.map:
        if prior_cpy_change_rates is None:
            prior_cpy_change_rates = (2, approx_rate / 2)
        if prior_cat_switch_rates is None:
            prior_cat_switch_rates = (3, approx_rate / 3 * 0.1)
        if prior_cpy_root_probs is None:
            prior_cpy_root_probs = np.full(args.lmax + 1, 1.1)
        if prior_cat_root_probs is None:
            prior_cat_root_probs = np.full(args.ncat, 1.1)
    else:
        if (
            prior_cpy_change_rates is not None
            or prior_cat_switch_rates is not None
            or prior_cpy_root_probs is not None
            or prior_cat_root_probs is not None
        ):
            warnings.warn('prior aguments are ignored when --map is not specified')

    prior_kw = None
    if args.map:
        prior_kw = dict(
            cpy_change_rates=prior_cpy_change_rates,
            cat_switch_rates=prior_cat_switch_rates,
            cpy_root_probs=prior_cpy_root_probs,
            cat_root_probs=prior_cat_root_probs,
        )

    init_model_kw = dict(
        ncharstates=args.lmax + 1,
        ncategories=args.ncat,
        invariant_cat=False,
        prior_kw=prior_kw,
        init_params_method='skip',
    )

    init_params_kw = configure_emEM_init(args)
    if not args.map:
        init_params_kw['distr_kw'] = dict(
            cpy_root_probs=np.ones(args.lmax + 1),
            cat_root_probs=np.ones(args.ncat),
            cpy_change_rates=(2, approx_rate / 2),
            cat_switch_rates=(3, approx_rate / 3 * 0.1),
        )
    rng = np.random.default_rng(seed=args.seed)
    rng_state = rng.bit_generator.state

    fit_kw = configure_parabolicEM(args)
    fit_kw['heuristics'] = True

    for ss_method in ('eig', 'auxR'):
        try:
            mmm = colaml.MarkovModulatedTreeModel(
                substmodelclass=colaml.MarkovModulatedBDARD, **init_model_kw
            )
            mmm.substmodel.set_ss_method(ss_method)
            mmm.init_params_emEM(phytbl, rng, **init_params_kw)
            init = mmm._decompress_flat_params(mmm.flat_params)
            converged = mmm.fit(phytbl, **fit_kw)
            break

        except Exception:
            logger.warning('Caught internal error:', exc_info=True)
            logger.warning('Finding next ss_method...')
            rng.bit_generator.state = rng_state  # reset
            continue
    else:
        msg = 'Fitting failed. Try with another seed.'
        logger.error(msg)
        raise RuntimeError(msg)

    run = dict(
        args=vars(args),
        init_model=init_model_kw,
        init_params_method='emEM',
        init_params_kw=init_params_kw,
        fit_kw=fit_kw,
        internal=dict(
            ss_method=ss_method,
        ),
        result=dict(
            init=init,
            params=mmm._decompress_flat_params(mmm.flat_params),
            converged=converged,
        ),
    )
    with gzip.open(args.output, 'wt') as output_file:
        json.dump(run, output_file, indent=2, cls=CoLaMLEncoder)


def fit_mirage(args):
    phytbl, _ = phytbl_from_json(args.input, args.lmax)
    approx_rate = phytbl.min_changes.mean() / phytbl.tree.branch_lengths.sum()

    # additional parsing
    prior_rates = args.prior_rates
    prior_root_probs = args.prior_root_probs
    prior_mixture_probs = args.prior_mixture_probs

    if args.map:
        if prior_rates is None:
            prior_rates = (2, approx_rate / 2)
        if prior_root_probs is None:
            prior_root_probs = np.full(args.lmax + 1, 1.1)
        if prior_mixture_probs is None:
            prior_mixture_probs = np.full(args.nmixtures, 1.1)
    else:
        if (
            prior_rates is not None
            or prior_root_probs is not None
            or prior_mixture_probs is not None
        ):
            warnings.warn('prior aguments are ignored when --map is not specified')

    prior_kw = None
    if args.map:
        prior_kw = dict(
            rates=prior_rates,
            root_probs=prior_root_probs,
            mixture_probs=prior_mixture_probs,
        )

    init_model_kw = dict(
        ncharstates=args.lmax + 1,
        nmixtures=args.nmixtures,
        prior_kw=prior_kw,
        init_params_method='skip',
    )

    init_params_kw = configure_emEM_init(args)
    if not args.map:
        init_params_kw['distr_kw'] = dict(
            root_probs=np.ones(args.lmax + 1),
            mixture_probs=np.ones(args.nmixtures),
            rates=(2, approx_rate / 2),
        )
    rng = np.random.default_rng(seed=args.seed)
    rng_state = rng.bit_generator.state

    fit_kw = configure_parabolicEM(args)

    for ss_method in ('eig', 'auxR'):
        try:
            mirage = colaml.MixtureTreeModel(
                substmodelclass=colaml.BDARD, **init_model_kw
            )
            for sm in mirage.substmodels:
                sm.set_ss_method(ss_method)
            mirage.init_params_emEM(phytbl, rng, **init_params_kw)
            init = mirage._decompress_flat_params(mirage.flat_params)
            converged = mirage.fit(phytbl, **fit_kw)
            break

        except Exception:
            logger.warning('Caught internal error:', exc_info=True)
            logger.warning('Finding next ss_method...')
            rng.bit_generator.state = rng_state  # reset
            continue
    else:
        msg = 'Fitting failed. Try with another seed.'
        logger.error(msg)
        raise RuntimeError(msg)

    run = dict(
        args=vars(args),
        init_model=init_model_kw,
        init_params_method='emEM',
        init_params_kw=init_params_kw,
        fit_kw=fit_kw,
        internal=dict(ss_method=ss_method),
        result=dict(
            init=init,
            params=mirage._decompress_flat_params(mirage.flat_params),
            converged=converged,
        ),
    )
    with gzip.open(args.output, 'wt') as output_file:
        json.dump(run, output_file, indent=2, cls=CoLaMLEncoder)


def fit_branch(args):
    phytbl, _ = phytbl_from_json(args.input, args.lmax)
    approx_rate = phytbl.min_changes.mean() / phytbl.tree.branch_lengths.sum()

    # additional parsing
    prior_rates = args.prior_rates
    prior_root_probs = args.prior_root_probs

    if args.map:
        if prior_rates is None:
            prior_rates = (2, approx_rate / 2)
        if prior_root_probs is None:
            prior_root_probs = np.full(args.lmax + 1, 1.1)
    else:
        if prior_rates is not None or prior_root_probs is not None:
            warnings.warn('prior aguments are ignored when --map is not specified')

    prior_kw = None
    if args.map:
        prior_kw = dict(
            rates=prior_rates,
            root_probs=prior_root_probs,
        )

    init_model_kw = dict(
        nbranches=phytbl.tree.nnodes - 1,
        ncharstates=args.lmax + 1,
        prior_kw=prior_kw,
        init_params_method='skip',
    )

    init_params_kw = configure_emEM_init(args)
    if not args.map:
        init_params_kw['distr_kw'] = dict(
            root_probs=np.ones(args.lmax + 1),
            rates=(2, approx_rate / 2),
        )
    rng = np.random.default_rng(seed=args.seed)
    rng_state = rng.bit_generator.state

    fit_kw = configure_parabolicEM(args)

    for ss_method in ('eig', 'auxR'):
        try:
            branch = colaml.unsupported.BranchwiseTreeModel(
                substmodelclass=colaml.BDARD, **init_model_kw
            )
            for sm in branch.substmodels:
                sm.set_ss_method(ss_method)
            branch.init_params_emEM(phytbl, rng, **init_params_kw)
            init = branch._decompress_flat_params(branch.flat_params)
            converged = branch.fit(phytbl, **fit_kw)
            break

        except Exception:
            logger.warning('Caught internal error:', exc_info=True)
            logger.warning('Finding next ss_method...')
            rng.bit_generator.state = rng_state  # reset
            continue
    else:
        msg = 'Fitting failed. Try with another seed.'
        logger.error(msg)
        raise RuntimeError(msg)

    run = dict(
        args=vars(args),
        init_model=init_model_kw,
        init_params_method='emEM',
        init_params_kw=init_params_kw,
        fit_kw=fit_kw,
        internal=dict(ss_method=ss_method),
        result=dict(
            init=init,
            params=branch._decompress_flat_params(branch.flat_params),
            converged=converged,
        ),
    )
    with gzip.open(args.output, 'wt') as output_file:
        json.dump(run, output_file, indent=2, cls=CoLaMLEncoder)


_fit_callers = dict(
    mmm=fit_mmm,
    mirage=fit_mirage,
    branch=fit_branch,
)


def fit(args):
    args.output.parent.mkdir(exist_ok=True)
    assert args.model in _fit_callers
    _fit_callers[args.model](args)


def model_from_json(path):
    open_func = open
    if path.name.endswith('.gz'):
        open_func = gzip.open
    with open_func(path, 'rt') as runinfo_file:
        run = json.load(runinfo_file)

    model_name = run['args']['model']
    if model_name == 'mmm':
        model = colaml.MarkovModulatedTreeModel(
            substmodelclass=colaml.MarkovModulatedBDARD, **run['init_model']
        )
        model.substmodel.set_ss_method(run['internal']['ss_method'])
    elif model_name == 'mirage':
        model = colaml.MixtureTreeModel(
            substmodelclass=colaml.BDARD, **run['init_model']
        )
        for sm in model.substmodels:
            sm.set_ss_method(run['internal']['ss_method'])
    elif model_name == 'branch':
        model = colaml.unsupported.BranchwiseTreeModel(
            substmodelclass=colaml.BDARD, **run['init_model']
        )
        for sm in model.substmodels:
            sm.set_ss_method(run['internal']['ss_method'])
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model.update(**run['result']['params'])
    return model


def recon(args):
    model = model_from_json(args.model)
    phytbl, columns = phytbl_from_json(args.input, model.ncharstates - 1)
    phytbl.tree.fill_names(ignore_tips=True)

    recon = model.reconstruct(phytbl, method=args.method)

    result = dict(
        args=vars(args),
        tree=phytbl.tree.to_ete3().write(format=3, format_root_node=True),
        recon=dict(
            columns=columns,
            index=[*recon.to_dict(copy=False).keys()],
            data=[*recon.to_dict(copy=False).values()],
        ),
        otherstates=[
            dict(
                label=other.label,
                states=dict(
                    columns=columns,
                    index=[*other.to_dict(copy=False).keys()],
                    data=[*other.to_dict(copy=False).values()],
                ),
            )
            for other in recon.otherstates.values()
        ],
        colattrs=[
            dict(name=colattr.label, index=columns, data=colattr.to_list(copy=False))
            for colattr in recon.colattrs.values()
        ],
        nodeattrs=[
            dict(
                name=nodeattr.label,
                index=[*nodeattr.to_dict(copy=False).keys()],
                data=[*nodeattr.to_dict(copy=False).values()],
            )
            for nodeattr in recon.nodeattrs.values()
        ],
    )

    with gzip.open(args.output, 'wt') as output_file:
        json.dump(result, output_file, indent=2, cls=CoLaMLEncoder)


_subcommand_callers = dict(
    fit=fit,
    recon=recon,
)


class LoggingContext:
    # from https://docs.python.org/ja/3/howto/logging-cookbook.html
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


def main():
    from threadpoolctl import threadpool_limits

    handler = StreamHandler()
    handler.setFormatter(Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))

    with (
        LoggingContext(logger, DEBUG, handler, close=False),
        threadpool_limits(limits=1, user_api='blas'),
    ):
        parser = get_parser()
        args = parser.parse_args()
        assert args.subcommand in _subcommand_callers
        _subcommand_callers[args.subcommand](args)


if __name__ == '__main__':
    main()
