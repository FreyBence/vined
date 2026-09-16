"""Numeric visual feature archive and physical-time alignment contracts."""
import numpy as np

SCHEMA_VERSION = 2
FEATURE_WIDTH = 768


def validate_ids(values):
    ids = np.asarray(values)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Trial IDs must be unique, nonnegative integers")
    return ids.astype(np.int64)


def load_archive(path, eid):
    """Legacy archives are rejected: their trial IDs and clock are ambiguous."""
    with np.load(path, allow_pickle=False) as data:
        required = {"schema_version", "eid", "trial_ids", "offsets", "times", "features", "valid", "clock"}
        if not required.issubset(data.files):
            raise ValueError("Legacy/incomplete visual archive; regenerate features from timestamped replays")
        if data["schema_version"].item() != SCHEMA_VERSION or data["clock"].item() != "session_seconds":
            raise ValueError("Unsupported visual schema or clock")
        if data["eid"].item() != eid:
            raise ValueError("Visual archive EID does not match requested session")
        ids = validate_ids(data["trial_ids"])
        offsets, times = data["offsets"], data["times"]
        features, valid = data["features"], data["valid"]
        if (offsets.dtype.kind not in "iu" or offsets.shape != (len(ids)+1,)
                or offsets[0] != 0 or offsets[-1] != len(times)
                or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError("Invalid visual trial offsets")
        if (times.ndim != 1 or times.dtype.kind != "f" or not np.isfinite(times).all()
                or features.shape != (len(times), FEATURE_WIDTH)
                or features.dtype.kind != "f" or not np.isfinite(features).all()
                or valid.shape != times.shape or valid.dtype.kind != "b"):
            raise ValueError("Invalid visual timestamps, features, width, or validity mask")
        time_list, value_list, masks = [], [], []
        for start, stop in zip(offsets[:-1], offsets[1:]):
            t, v, m = times[start:stop], features[start:stop], valid[start:stop]
            if np.any(np.diff(t) <= 0) or np.any(np.linalg.norm(v[m], axis=-1) == 0):
                raise ValueError("Visual timestamps must increase and valid features must be nonzero")
            time_list.append(t.copy())
            value_list.append(v.copy())
            masks.append(m.copy())
    return dict(trial_ids=ids, times=time_list, values=value_list, valid=masks, skip=False)


def resample_features(times, values, valid, queries):
    """Linear interpolation at bin centers; no extrapolation or invalid-gap bridging.

    Interpolated CLIP vectors are renormalized. Unavailable bins contain zero
    placeholders and a false validity mask, never a synthetic blank embedding.
    """
    result = np.zeros((len(queries), values.shape[1]), dtype=np.float32)
    right = np.searchsorted(times, queries, side="left")
    right = np.clip(right, 0, len(times)-1)
    exact = np.isclose(times[right], queries, rtol=0, atol=1e-9)
    left = np.where(exact, right, np.maximum(right-1, 0))
    available = ((queries >= times[0]) & (queries <= times[-1])
                 & valid[left] & valid[right])
    denominator = times[right] - times[left]
    weight = np.divide(queries-times[left], denominator,
                       out=np.zeros_like(queries), where=denominator > 0)
    interpolated = values[left] * (1-weight[:, None]) + values[right] * weight[:, None]
    norms = np.linalg.norm(interpolated, axis=-1)
    available &= norms > 1e-12
    result[available] = interpolated[available] / norms[available, None]
    return result, available
