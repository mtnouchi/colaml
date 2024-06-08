import weakref
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import NamedTuple

import numpy as np
from tqdm.auto import tqdm

from .misc.logging import EMrecord, get_loggingformat


def dist(a, b):
    return np.linalg.norm(b - a)


def _pinv(vec):
    return vec / (vec @ vec)


def dropwhileN(predicate, iterable, *, max_drop):
    iterable = iter(iterable)
    for _, x in zip(range(max_drop), iterable):
        if predicate(x):
            continue
        yield x
        break
    yield from iterable


class EMProgress:
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.pbar = tqdm(*args, **kwargs)
        self._context = False

    def update(self):
        if not self._context:
            return
        self.count += 1
        self.pbar.update()

    def __enter__(self):
        self._context = True
        self.pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.pbar.__exit__(exc_type, exc_value, trace)
        self._context = False


class StatsHandler:
    def __init__(self, model, phytbl, progress, logger):
        self.stats = model._empty_stats(phytbl)
        self.progress = weakref.proxy(progress)
        self.logger = weakref.proxy(logger) if logger else None

    def __getattr__(self, name):
        return getattr(self.stats, name)

    def compute(self, *, status='stdEM', return_logP=True):
        self.stats.compute()
        self.progress.update()

        # skip logP calculation if not needed
        if not return_logP and self.logger is None:
            return

        logP = (self.stats.col_loglik.sum(), self.stats._model._log_prior_prob())
        if self.logger:
            self.logger.info(
                get_loggingformat(),
                EMrecord(*logP, self.stats._model.flat_params, status),
            )

        if return_logP:
            return logP


class StopCriteria(NamedTuple):
    atol: float
    rtol: float

    def is_met(self, p1, p2):
        return np.allclose(p1, p2, atol=self.atol, rtol=self.rtol)


class Estimable(ABC):
    @abstractmethod
    def run(self, model, data, *, show_progress=False, logger=None):
        return False


class PlainEM(Estimable):
    def __init__(self, stop_criteria=StopCriteria(1e-8, 1e-8), max_rounds=1000):
        self.stop_criteria = StopCriteria(*stop_criteria)
        self.max_rounds = max_rounds

    def run(self, model, phytbl, show_progress=True, logger=None):
        with EMProgress(total=self.max_rounds, disable=not show_progress) as prog:
            stats = StatsHandler(model, phytbl, prog, logger)
            params = model.flat_params
            while prog.count < self.max_rounds:
                stats.compute(return_logP=False)
                model._update(**model._get_next_params(stats))
                new_params = model.flat_params
                if self.stop_criteria.is_met(params, new_params):
                    return True
                params = new_params

        return False


class EpsRestartConf(NamedTuple):
    initial_thresh: float
    update_ratio: float
    check_interval: int


class EpsRestartEM(Estimable):
    def __init__(
        self,
        stop_criteria=StopCriteria(atol=1e-8, rtol=1e-8),
        max_rounds=1000,
        restart_conf=EpsRestartConf(1, 0.1, 100),
    ):
        self.stop_criteria = StopCriteria(*stop_criteria)
        self.max_rounds = max_rounds
        self.restart_conf = EpsRestartConf(*restart_conf)

    def run(self, model, phytbl, show_progress=True, logger=None):
        with EMProgress(total=self.max_rounds, disable=not show_progress) as prog:
            stats = StatsHandler(model, phytbl, prog, logger)
            tmp_stats = StatsHandler(model, phytbl, prog, logger)

            # init run
            p0 = model.flat_params
            stats.compute(return_logP=False)

            model._update(**model._get_next_params(stats))

            p1 = model.flat_params
            stats.compute(return_logP=False)

            p_acc_old = p1

            # main run
            restart_thresh = self.restart_conf.initial_thresh
            update_ratio = self.restart_conf.update_ratio
            last_restart = -1
            while prog.count < self.max_rounds:
                # M step
                model._update(**model._get_next_params(stats))
                p2 = model.flat_params
                if self.stop_criteria.is_met(p1, p2):
                    return True

                # E step
                loglik, log_priorP = stats.compute()

                # generate accelerated series
                p_acc_new = p1 + _pinv(_pinv(p2 - p1) - _pinv(p1 - p0))

                acc_delta = dist(p_acc_old, p_acc_new)

                # update params for the next iteration
                p0 = p1
                p1 = p2
                p_acc_old = p_acc_new

                # test whether to restart
                ### all parameters are valid?
                if not (p_acc_new >= 0).all():
                    continue

                ### step size has shrunk enough since the last restart?
                step_shrinks = acc_delta < restart_thresh
                ### time for a periodic check, counting from the last restart?
                on_period = (
                    prog.count - last_restart
                ) % self.restart_conf.check_interval == 0
                if not (step_shrinks or on_period):
                    continue

                ### likelihood is improved by acceleration?
                model._update(**model._decompress_flat_params(p_acc_new))
                try:
                    tmp_stats.compute(status='acc-search', return_logP=False)
                except:
                    continue

                model._update(**model._get_next_params(tmp_stats))
                tmp_p = model.flat_params
                try:
                    tmp_loglik, tmp_log_priorP = tmp_stats.compute(status='acc-search')
                except:
                    continue

                if step_shrinks:
                    restart_thresh *= update_ratio

                if tmp_loglik + tmp_log_priorP < loglik + log_priorP:
                    continue

                # restart
                p0, p1 = p_acc_new, tmp_p
                stats, tmp_stats = tmp_stats, stats  # swap "refs"
                loglik, log_priorP = tmp_loglik, tmp_log_priorP
                last_restart = prog.count

        return False


class BezierGeomGrid(namedtuple('_BezierGeomGrid', ['first', 'ratio'])):
    MAX_SEARCH = 2500

    def __new__(cls, first, ratio):
        if not first > 0:
            raise ValueError(f'\'first\' must be positive: {first}')
        if not ratio > 1:
            raise ValueError(f'\'ratio\' must be greater than 1: {ratio}')
        return super().__new__(cls, first, ratio)

    def search(self, p0, p1, p2):
        dt = self.first
        for _ in range(BezierGeomGrid.MAX_SEARCH):  # for safety
            yield p2 + dt * (2 * (p2 - p1) + dt * (p2 - 2 * p1 + p0))
            dt *= self.ratio


class ParabolicEM(Estimable):
    def __init__(
        self,
        stop_criteria=StopCriteria(1e-8, 1e-8),
        max_rounds=1000,
        grid=BezierGeomGrid(0.1, 1.5),
        heuristics=False,
    ):
        self.stop_criteria = StopCriteria(*stop_criteria)
        self.max_rounds = max_rounds
        self.grid = BezierGeomGrid(*grid)
        self.heuristics = heuristics

    def run(self, model, phytbl, show_progress=True, logger=None):
        with EMProgress(total=self.max_rounds, disable=not show_progress) as prog:
            stats = StatsHandler(model, phytbl, prog, logger)
            tmp_stats = StatsHandler(model, phytbl, prog, logger)

            # init run
            p0 = model.flat_params
            stats.compute(return_logP=False)

            model._update(**model._get_next_params(stats))
            p1 = model.flat_params
            stats.compute(return_logP=False)

            model._update(**model._get_next_params(stats))
            p2 = model.flat_params

            # main run
            em_delta = dist(p0, p1)
            while prog.count < self.max_rounds:
                loglik, log_priorP = stats.compute()

                # p_best = p2
                best_log_jointP = loglik + log_priorP

                sequence = self.grid.search(p0, p1, p2)
                if self.heuristics:
                    sequence = dropwhileN(
                        lambda p: not dist(p2, p) >= em_delta,
                        sequence,
                        max_drop=10,  # to avoid infinite loop
                    )
                for _ in range(1000):
                    p_new = next(sequence)
                    if (p_new < 0).any():
                        break

                    model._update(**model._decompress_flat_params(p_new))
                    try:
                        tmp_loglik, tmp_log_priorP = tmp_stats.compute(
                            status='acc-search'
                        )
                    except:
                        break

                    tmp_log_jointP = tmp_loglik + tmp_log_priorP
                    if tmp_log_jointP <= best_log_jointP:
                        break

                    best_log_jointP = tmp_log_jointP
                    stats, tmp_stats = tmp_stats, stats  # swap "refs"

                p0 = p1
                p1 = p2

                model._update(**model._get_next_params(stats))
                p_tmp = model.flat_params
                stats.compute(return_logP=False)

                model._update(**model._get_next_params(stats))
                p2 = model.flat_params
                if self.stop_criteria.is_met(p_tmp, p2):
                    return True

                em_delta = dist(p_tmp, p2)

        return False
