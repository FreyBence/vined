"""Content identities for the decoded neural inputs actually consumed."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from utils.provenance import fingerprint


def _content(value):
    if isinstance(value, pd.Series):
        return dict(name=value.name, index=_content(value.index.to_numpy()),
                    data=_content(value.to_numpy()))
    if isinstance(value, pd.DataFrame):
        return dict(columns=list(value.columns), index=_content(value.index.to_numpy()),
                    data={str(key): _content(value[key].to_numpy()) for key in value})
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            data = _content(value.tolist())
        else:
            data = hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")
                                  if value.size else b"").hexdigest()
        return dict(dtype=str(value.dtype), shape=list(value.shape), data=data)
    if isinstance(value, dict):
        return {str(key): _content(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_content(item) for item in value]
    if isinstance(value, np.generic):
        return _content(value.item())
    if value is pd.NA:
        return {"missing": "pandas.NA"}
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported source metadata type: {type(value).__name__}")


def content_hash(value):
    return fingerprint(_content(value))


def json_value(value):
    """Normalize metadata containers; unavailable scalar metadata becomes null."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported generation metadata type: {type(value).__name__}")
