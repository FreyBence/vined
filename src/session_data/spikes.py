"""Probe-aware source access using Brainbox selection and ALF decoding."""

from dataclasses import dataclass
from uuid import UUID

from .access import AccessPolicy, DatasetSource, SessionAccessError


@dataclass(frozen=True)
class SpikeSortingSource:
    eid: str
    pid: str | None
    pname: str
    collection: str
    revision: str
    datasets: tuple[DatasetSource, ...]


@dataclass(frozen=True)
class LoadedSpikeSorting:
    spikes: object
    clusters: object
    channels: object
    source: SpikeSortingSource


def load_spike_sorting(access, eid, *, pid=None, pname=None, collection=None, revision=None):
    eid = access.canonical_eid(eid)
    if pid is not None:
        try:
            pid = str(UUID(pid))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("pid must be a UUID string") from exc
    if pname is not None and (not isinstance(pname, str) or not pname
                              or any(c in pname for c in "/\\*?[]")):
        raise ValueError("pname must be an exact probe name")
    if pid is None and pname is None:
        raise ValueError("Provide pid or pname to select one probe")
    for value in (collection, revision):
        if value is not None and (not isinstance(value, str) or any(c in value for c in "*?[]")):
            raise ValueError("Loading requires exact collection/revision strings")
    missing = ("local_source_unavailable" if access.policy == AccessPolicy.LOCAL_ONLY
               else "source_unavailable")
    try:
        metadata = access.metadata(eid, include_probes=True)
        if metadata.probes is None:
            # A collection can establish a probe label, but never invent a PID.
            if pid is not None:
                raise SessionAccessError("local_source_unavailable",
                                         "Insertion identity is unresolved in the local catalog")
        else:
            matches = [probe for probe in metadata.probes
                       if (pid is None or str(probe.get("id")) == pid)
                       and (pname is None or probe.get("name") == pname)]
            if not matches:
                raise SessionAccessError(missing, "Requested insertion is absent from session metadata")
            if len(matches) != 1:
                raise SessionAccessError("source_conflict", "Insertion selection is ambiguous")
            probe = matches[0]
            pid, pname = str(UUID(str(probe["id"]))), probe["name"]
        probe_collection = f"alf/{pname}"
        if collection is not None and not (collection == probe_collection
                                           or collection.startswith(probe_collection + "/")):
            raise SessionAccessError("source_conflict", "Collection does not belong to the requested probe")

        from brainbox.io.one import SpikeSortingLoader
        from one.alf.path import filename_parts

        # Only object decoding and sorter selection are used. Disable the unused
        # atlas constructor: AllenAtlas can download files even with a local ONE.
        # Resolve PID ourselves so Brainbox cannot override the requested EID.
        loader = SpikeSortingLoader(one=access._client(), eid=eid, pname=pname, atlas=False)
        loader.collections = [c for c in loader.collections
                              if c == probe_collection or c.startswith(probe_collection + "/")]
        if collection is None:
            collection = loader._get_spike_sorting_collection(spike_sorter="iblsorter")
        if collection is None:
            raise SessionAccessError(missing, "No spike-sorting collection is available")
        discovery = access.discover(eid, collection=collection, revision=revision)
        attributes = loader._get_attributes(None)
        names = {obj: set() for obj in ("spikes", "clusters", "channels")}
        for source in discovery.datasets:
            namespace, obj, attribute, *_ = filename_parts(source.name)
            # Match Brainbox's default uncurated namespace selection. Explicit
            # curation and additional attributes are not part of this request.
            if namespace is None and obj in names and (
                    obj == "channels" or attribute in attributes[obj]):
                names[obj].add(source.name)
        if any(not group for group in names.values()):
            reason = "revision_unavailable" if revision is not None and missing == "source_unavailable" else missing
            raise SessionAccessError(reason, "Required spike, cluster, or channel sources are unavailable")
        loaded = access.load_datasets(eid, sorted(set.union(*names.values())),
                                      collection=collection, revision=revision, download_only=True)
        objects = {}
        for obj, group in names.items():
            paths = [item.data for item in loaded if item.source.name in group]
            objects[obj] = loader._load_object(paths, wildcards=loader.one.wildcards)
        # Missing core attributes are not an empty recording. Zero-length arrays
        # carrying these attributes remain valid and are returned unchanged.
        if not {"times", "clusters"}.issubset(objects["spikes"]) or "channels" not in objects["clusters"]:
            raise SessionAccessError(missing, "Required spike times/cluster identities or cluster channels are missing")
        sources = tuple(item.source for item in loaded)
        source = SpikeSortingSource(eid, pid, pname, collection, sources[0].revision, sources)
        return LoadedSpikeSorting(objects["spikes"], objects["clusters"], objects["channels"], source)
    except SessionAccessError as exc:
        raise SessionAccessError(
            exc.reason, f"Spike sources for EID={eid}, PID={pid}, probe={pname}, "
            f"collection={collection!r}, revision={revision!r}: {exc}", eid=eid
        ) from exc
    except Exception as exc:
        raise SessionAccessError(
            "access_failed", f"Could not load spike sources for EID={eid}, PID={pid}, "
            f"probe={pname}, collection={collection!r}, revision={revision!r}", eid=eid
        ) from exc
