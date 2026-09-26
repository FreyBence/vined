"""Insertion-specific electrophysiology access and source clock conversion."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .access import AccessPolicy, DatasetSource, SessionAccessError


@dataclass(frozen=True)
class EphysSource:
    eid: str
    pid: str | None
    pname: str
    band: str
    collection: str
    revision: str
    streaming: bool
    datasets: tuple[DatasetSource, ...]


@dataclass(frozen=True)
class LoadedEphys:
    recording: object
    source: EphysSource
    _forward: object
    _reverse: object

    def samples_to_times(self, samples):
        """Convert this stream's fractional sample indices to session seconds."""
        return self._forward(samples)

    def times_to_samples(self, times):
        """Convert session seconds to fractional indices in this stream."""
        return self._reverse(times)


def load_ephys(access, eid, *, pid=None, pname=None, band="lf", stream=False, revision=None):
    eid = access.canonical_eid(eid)
    if band not in ("ap", "lf") or not isinstance(stream, bool):
        raise ValueError("band must be 'ap' or 'lf', and stream must be boolean")
    if pid is not None:
        try:
            pid = str(UUID(pid))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("pid must be a UUID string") from exc
    if pname is not None and (not isinstance(pname, str) or not pname
                              or any(c in pname for c in "/\\*?[]")):
        raise ValueError("pname must be an exact probe name")
    if pid is None and pname is None:
        raise ValueError("Provide pid or pname to select one insertion")
    if revision is not None and (not isinstance(revision, str)
                                  or any(c in revision for c in "/\\*?[]")):
        raise ValueError("revision must be an exact revision string")
    local = access.policy == AccessPolicy.LOCAL_ONLY
    missing = "local_source_unavailable" if local else "source_unavailable"
    collection = None
    try:
        if stream and local:
            raise SessionAccessError("local_source_unavailable",
                                     "Remote streaming is disabled; request stream=False for local files")
        metadata = access.metadata(eid, include_probes=True)
        if metadata.probes is None:
            if pid is not None:
                raise SessionAccessError(missing, "Insertion identity is unresolved in the local catalog")
        else:
            probes = [p for p in metadata.probes
                      if (pid is None or str(p.get("id")) == pid)
                      and (pname is None or p.get("name") == pname)]
            if not probes:
                raise SessionAccessError(missing, "Requested insertion is absent from session metadata")
            if len(probes) != 1:
                raise SessionAccessError("source_conflict", "Insertion selection is ambiguous")
            pid, pname = str(UUID(str(probes[0]["id"]))), probes[0]["name"]
        collection = f"raw_ephys_data/{pname}"
        catalog = access.discover(eid, collection=collection, revision=revision)
        binaries = [s for s in catalog.datasets if s.name.endswith(f".{band}.cbin")]
        if not binaries:
            reason = "revision_unavailable" if revision is not None and not local else missing
            raise SessionAccessError(reason, f"No {band} compressed recording is available")
        if len(binaries) != 1:
            raise SessionAccessError("source_conflict", "Multiple recordings/revisions; specify an exact revision")
        binary = binaries[0]
        revision = binary.revision
        stem = binary.name[:-5]
        # Brainbox's stream loader does not accept revision/collection arguments
        # and takes the first URL. Reject ambiguity before it can choose a source.
        if stream:
            candidates = access.discover(eid, collection=f"*{pname}").datasets
            if any(s.name.endswith(f".{band}.bin") for s in candidates):
                raise SessionAccessError("source_conflict", "Streaming catalog also contains an uncompressed recording")
            for suffix in (f".{band}.cbin", f".{band}.meta", f".{band}.ch"):
                matches = [s for s in candidates if s.name.endswith(suffix)]
                if len(matches) != 1 or (matches[0].collection, matches[0].revision) != (collection, revision):
                    raise SessionAccessError("source_conflict",
                                             "Streaming source is ambiguous; use stream=False with an exact revision")
        names = [stem + ".meta", stem + ".ch", "_spikeglx_*.timestamps.npy"]
        if band != "ap":
            names.append("_spikeglx_*.ap.meta")
        if not stream:
            names.append(binary.name)
        loaded = access.load_datasets(eid, names, collection=collection,
                                      revision=revision, download_only=True)
        paths = {item.source.name: item.data for item in loaded}
        import numpy as np
        import spikeglx
        from scipy.interpolate import interp1d

        timestamps = np.load(next(path for name, path in paths.items() if name.endswith(".timestamps.npy")))
        if (timestamps.ndim != 2 or timestamps.shape[1] != 2 or len(timestamps) < 2
                or not np.isfinite(timestamps).all() or not (np.diff(timestamps, axis=0) > 0).all()):
            raise SessionAccessError("source_inconsistent", "Invalid probe-sample/session-time mapping")
        ap_meta = next(path for name, path in paths.items() if name.endswith(".ap.meta"))
        ap_rate = spikeglx._get_fs_from_meta(spikeglx.read_meta_data(ap_meta))
        if not np.isfinite(ap_rate) or ap_rate <= 0:
            raise SessionAccessError("source_inconsistent", "Invalid AP sampling frequency")
        if stream:
            from spikeinterface.extractors.iblextractors import IblRecordingExtractor

            recording = IblRecordingExtractor(pid=pid, stream_type=band,
                                               one=access._client(), stream=True)
            reader = recording._file_streamer
            if (str(recording.ssl.eid) != eid or recording.ssl.pname != pname
                    or Path(reader.file_meta_data).resolve() != Path(paths[stem + ".meta"]).resolve()
                    or reader.url_cbin != access._client().record2url(
                        access._client()._cache["datasets"].loc[(UUID(eid), UUID(binary.dataset_id))])):
                reader.close()
                raise SessionAccessError("source_conflict", "Stream loader selected a different source")
        else:
            from spikeinterface.extractors.cbin_ibl import CompressedBinaryIblExtractor

            recording = CompressedBinaryIblExtractor(cbin_file_path=paths[binary.name])
        rate = recording.get_sampling_frequency()
        if not np.isfinite(rate) or rate <= 0:
            raise SessionAccessError("source_inconsistent", "Invalid source sampling frequency")
        # IBL timestamps use AP sample indices. Scale using recorded rates, never
        # assume a fixed AP/LF ratio of twelve for all source recordings.
        samples = timestamps[:, 0] * rate / ap_rate
        forward = interp1d(samples, timestamps[:, 1], fill_value="extrapolate")
        reverse = interp1d(timestamps[:, 1], samples, fill_value="extrapolate")
        sources = tuple(item.source for item in loaded)
        if stream:
            sources += (binary,)
        source = EphysSource(eid, pid, pname, band, collection, revision, stream, sources)
        return LoadedEphys(recording, source, forward, reverse)
    except SessionAccessError as exc:
        raise SessionAccessError(exc.reason, f"Ephys EID={eid}, PID={pid}, probe={pname}, "
                                 f"band={band}, collection={collection!r}, revision={revision!r}: {exc}",
                                 eid=eid) from exc
    except Exception as exc:
        raise SessionAccessError("access_failed", f"Could not open ephys EID={eid}, PID={pid}, "
                                 f"probe={pname}, band={band}, collection={collection!r}, "
                                 f"revision={revision!r}", eid=eid) from exc
