import json
from contextlib import contextmanager

import numpy as np

LOGFMT = '%s'


def set_loggingformat(logfmt):
    global LOGFMT
    LOGFMT = logfmt


def get_loggingformat():
    return LOGFMT


@contextmanager
def loggingformat(logfmt):
    try:
        prev_logfmt = get_loggingformat()
        set_loggingformat(logfmt)
        yield
    finally:
        set_loggingformat(prev_logfmt)


class _NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class EMrecord(dict):
    def __init__(self, loglik, logprior, params, status):
        super().__init__(loglik=loglik, logprior=logprior, params=params, status=status)

    def __repr__(self):
        return json.dumps(self, cls=_NDArrayEncoder)
