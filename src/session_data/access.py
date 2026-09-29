"""Session identity and acquisition context shared by source-family adapters."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import UUID
import hashlib
import warnings
from contextlib import ExitStack
from tempfile import TemporaryDirectory

from utils.paths import REPO_ROOT, dataset_dir
from utils.sessions import select_sessions


class AccessPolicy(str, Enum):
    LOCAL_ONLY = "local-only"
    REMOTE_ALLOWED = "remote-allowed"


class SessionAccessError(RuntimeError):
    """Source-boundary failure with a stable reason and optional session identity."""

    def __init__(self, reason, message, *, eid=None):
        super().__init__(message)
        self.reason = reason
        self.eid = eid


@dataclass(frozen=True)
class SessionSource:
    """Resolved identity and expected cache path, not a dataset availability claim."""

    eid: str
    path: Path


@dataclass(frozen=True)
class DatasetSource:
    eid: str
    dataset_id: str
    name: str
    collection: str
    revision: str
    path: Path
    availability: str


@dataclass(frozen=True)
class DatasetDiscovery:
    eid: str
    datasets: tuple[DatasetSource, ...]
    status: str
    origin: str


@dataclass(frozen=True)
class LoadedDataset:
    data: object
    source: DatasetSource


@dataclass(frozen=True)
class LoadedTrials:
    data: object
    sources: tuple[DatasetSource, ...]


@dataclass(frozen=True)
class SessionMetadata:
    eid: str
    reported: dict
    derived: dict
    probes: tuple[dict, ...] | None
    origin: str


class SessionAccess:
    """Own one source-cache/client context with an explicit acquisition policy.

    Clients are lazy and private: source-family adapters must use this context's
    policy and client rather than accepting consumer-configured ONE instances.
    """

    def __init__(self, *, policy, cache_dir=None,
                 base_url="https://openalyx.internationalbrainlab.org", force_reload=False):
        self._policy = AccessPolicy(policy)
        if type(force_reload) is not bool:
            raise TypeError("force_reload must be a boolean")
        if force_reload and self._policy != AccessPolicy.REMOTE_ALLOWED:
            raise ValueError("force_reload requires remote-allowed access")
        self._force_reload = force_reload
        self._reloaded = set()
        cache_path = Path(cache_dir).expanduser() if cache_dir is not None else dataset_dir()
        self._cache_dir = (cache_path if cache_path.is_absolute()
                           else REPO_ROOT / cache_path).resolve()
        self._base_url = base_url
        self._one = None

    @property
    def force_reload(self):
        return self._force_reload

    @property
    def policy(self):
        return self._policy

    @property
    def cache_dir(self):
        return self._cache_dir

    @property
    def base_url(self):
        return self._base_url

    @staticmethod
    def canonical_eid(eid):
        """Reuse generic manifest identity rules without assigning experiment roles."""
        if not isinstance(eid, str):
            raise SessionAccessError("invalid_eid", "EID must be a UUID string")
        try:
            return select_sessions(eid=eid)[0]
        except ValueError as exc:
            raise SessionAccessError("invalid_eid", f"Invalid session EID: {eid}") from exc

    def _client(self):
        if self._one is None:
            # Do not use the cached ONE factory: it can select OneAlyx from local
            # cache configuration and can share mutable clients between contexts.
            from one.api import One, OneAlyx

            try:
                if self.policy == AccessPolicy.LOCAL_ONLY:
                    self._one = One(cache_dir=self.cache_dir, mode="local")
                else:
                    self._one = OneAlyx(cache_dir=self.cache_dir,
                                        base_url=self.base_url, mode="local", silent=True)
                    # Load existing local tables before permitting remote lookup.
                    self._one.mode = "remote"
                    if self.force_reload:
                        # No cached HTTP response or offline fallback for this context.
                        self._one.alyx.cache_mode = None
            except Exception as exc:
                raise SessionAccessError(
                    "access_failed", "Could not initialize session source access"
                ) from exc
        return self._one

    def resolve_session(self, eid, *, require_local=False):
        """Resolve an EID using cached identity first, then permitted remote lookup.

        ``require_local`` requires the session directory to exist, not any specific
        dataset. Resolution never downloads datasets or fabricates session paths.
        """
        eid = self.canonical_eid(eid)
        try:
            client = self._client()
            if self.policy == AccessPolicy.LOCAL_ONLY:
                path = client.eid2path(eid)
            else:
                path = None if self.force_reload else client.eid2path(eid, query_type="local")
                if path is None:
                    path = client.eid2path(eid, query_type="remote")
        except SessionAccessError as exc:
            raise SessionAccessError(exc.reason, str(exc), eid=eid) from exc
        except Exception as exc:
            raise SessionAccessError(
                "access_failed", f"Could not resolve session {eid}", eid=eid
            ) from exc
        if path is None:
            scope = "local cache" if self.policy == AccessPolicy.LOCAL_ONLY else "configured source"
            raise SessionAccessError(
                "session_unresolved", f"Session {eid} is unresolved in the {scope}", eid=eid
            )
        path = Path(path)
        if require_local:
            try:
                available = path.is_dir()
            except OSError as exc:
                raise SessionAccessError(
                    "access_failed", f"Could not inspect session path {path}", eid=eid
                ) from exc
            if not available:
                raise SessionAccessError(
                    "local_source_unavailable", f"Session directory is not local: {path}", eid=eid
                )
        return SessionSource(eid=eid, path=path)

    def metadata(self, eid, *, include_probes=False):
        """Return factual session fields and optionally insertion records.

        Missing fields are unavailable, never inferred from other sessions.
        Uncached offline insertion metadata is unresolved (None), not an empty list.
        """
        session = self.resolve_session(eid)
        client = self._client()
        remote = self.policy == AccessPolicy.REMOTE_ALLOWED
        try:
            if remote:
                reported = dict(client.get_details(session.eid, full=True, query_type="remote"))
            else:
                reported = client.get_details(UUID(session.eid)).to_dict()
            probes = None
            if include_probes and remote:
                records = client.alyx.rest("insertions", "list", session=session.eid)
                probes = tuple(records or ())
            elif include_probes:
                from one.api import OneAlyx

                try:
                    # The installed offline One lacks this public conversion;
                    # reuse its local-only implementation without a web client.
                    pids, labels = OneAlyx.eid2pid(client, UUID(session.eid), query_type="local")
                    if pids is not None:
                        probes = tuple(dict(id=str(pid), name=label, session=session.eid)
                                       for pid, label in zip(pids, labels))
                except NotImplementedError:
                    pass
            if remote:
                self._save_catalog()
        except Exception as exc:
            raise SessionAccessError(
                "access_failed", f"Could not obtain metadata for {session.eid}", eid=session.eid
            ) from exc
        return SessionMetadata(session.eid, reported, {"session_path": session.path},
                               probes, "remote" if remote else "cache")

    def _save_catalog(self):
        from one.api import One

        client = self._client()
        # Brainbox's PID lookups create an in-memory insertion table. Installed
        # ONE cannot merge it into an on-disk cache without insertions.pqt. Keep
        # these transient records in memory while saving the normal ONE catalog.
        insertions = None
        insertion_meta = client._cache['_meta']['raw'].get('insertions')
        if not (self.cache_dir / 'insertions.pqt').is_file():
            insertions = client._cache.pop('insertions', None)
        try:
            client.save_cache()
            # Installed ONE serializes origin sets in place; reload through the
            # offline implementation to restore its expected in-memory types.
            One.load_cache(client)
        finally:
            if insertions is not None:
                client._cache['insertions'] = insertions
                if insertion_meta is not None:
                    client._cache['_meta']['raw']['insertions'] = insertion_meta

    def _catalog(self, eid):
        session = self.resolve_session(eid)
        origin = "remote" if self.policy == AccessPolicy.REMOTE_ALLOWED else "cache"
        try:
            table = self._client().list_datasets(
                session.eid, details=True, keep_eid_index=True,
                query_type="remote" if origin == "remote" else "local")
            if origin == "remote":
                self._save_catalog()
        except Exception as exc:
            raise SessionAccessError(
                "access_failed", f"Could not discover datasets for {session.eid}", eid=session.eid
            ) from exc
        return session, table, origin

    def _describe(self, session, record, origin):
        from one.alf.path import rel_path_parts

        record_eid, dataset_id = record.name
        if str(record_eid) != session.eid:
            raise SessionAccessError("source_conflict", "Dataset belongs to another session",
                                     eid=session.eid)
        collection, revision, *_ = rel_path_parts(record.rel_path)
        # ONE handles its configured UUID filename convention and cache layout.
        path = Path(self._client().record2path(record))
        state = "local" if path.is_file() else (
            "remote-known" if origin == "remote" else "not-local")
        return DatasetSource(session.eid, str(dataset_id), Path(record.rel_path).name,
                             collection or "", revision or "", path, state)

    def discover(self, eid, *, filename=None, collection=None, revision=None):
        """List matching datasets without downloading; filters support ONE wildcards."""
        from one.util import filter_datasets

        session, table, origin = self._catalog(eid)
        try:
            matches = filter_datasets(table, filename=filename, collection=collection,
                                      revision=revision, revision_last_before=False,
                                      assert_unique=False, wildcards=True)
            sources = tuple(self._describe(session, row, origin) for _, row in matches.iterrows())
        except SessionAccessError:
            raise
        except Exception as exc:
            raise SessionAccessError("access_failed", "Could not inspect dataset catalog",
                                     eid=session.eid) from exc
        status = "available" if sources else ("unavailable" if origin == "remote" else "unresolved")
        return DatasetDiscovery(session.eid, sources, status, origin)

    def load_dataset(self, eid, dataset, *, collection=None, revision=None, download_only=False):
        """Load one named dataset with its effective identity; explicit revisions are exact."""
        return self.load_datasets(eid, [dataset], collection=collection, revision=revision,
                                  download_only=download_only)[0]

    def probes(self, eid):
        """Return reported insertions, or catalog-derived labels with unknown PIDs."""
        metadata = self.metadata(eid, include_probes=True)
        if metadata.probes is not None:
            return metadata.probes
        discovery = self.discover(eid, filename="spikes.times.npy", collection="alf/*")
        labels = sorted({s.collection.split('/')[1] for s in discovery.datasets
                         if s.collection.startswith('alf/')})
        if not labels:
            raise SessionAccessError("local_source_unavailable", "Probe identities are unresolved",
                                     eid=metadata.eid)
        return tuple(dict(id=None, name=name, session=metadata.eid, origin="derived") for name in labels)

    def load_trials(self, eid, *, collection="alf", revision=None):
        """Load the trial table required by active consumers, retaining original rows."""
        table = self.load_dataset(eid, "_ibl_trials.table.pqt", collection=collection,
                                   revision=revision)
        return LoadedTrials(table.data, (table.source,))

    def load_spike_sorting(self, eid, *, pid=None, pname=None, collection=None, revision=None):
        """Load one probe's unfiltered ALF spike, cluster, and channel sources."""
        from .spikes import load_spike_sorting

        return load_spike_sorting(self, eid, pid=pid, pname=pname,
                                 collection=collection, revision=revision)

    def load_ephys(self, eid, *, pid=None, pname=None, band="lf", stream=False,
                   revision=None):
        """Open one insertion's recording and its source-clock conversion."""
        from .ephys import load_ephys

        return load_ephys(self, eid, pid=pid, pname=pname, band=band,
                          stream=stream, revision=revision)

    def load_datasets(self, eid, datasets, *, collection=None, revision=None, download_only=False,
                      latest_common_revision=False):
        """Load a coherent group from one catalog snapshot, preserving request order.

        Each name must select one dataset. Joint selections must share a collection
        and revision; request distinct source groups separately if they differ.
        Opt in to latest_common_revision to resolve mixed automatic revisions to
        the newest shared revision, without overriding explicit requests.
        """
        from one.alf.exceptions import ALFError, ALFObjectNotFound
        from one.alf.io import load_file_content
        from one.util import filter_datasets

        if isinstance(datasets, str):
            raise ValueError("datasets must be a sequence of dataset names")
        names = tuple(datasets)
        if not names or any(not isinstance(name, str) or not name or "/" in name or "\\" in name
                            for name in names):
            raise ValueError("Provide nonempty dataset filenames; use collection/revision arguments")
        if any(value is not None and any(char in value for char in "*?[]")
               for value in (collection, revision)):
            raise ValueError("Loading requires exact collection/revision values")
        session, table, origin = self._catalog(eid)
        selected = []
        candidates = []
        for name in names:
            try:
                # Exact filtering first prevents ONE's last-before revision fallback.
                matches = filter_datasets(table, filename=name, collection=collection,
                                          revision=revision, revision_last_before=False,
                                          assert_unique=False, wildcards=True)
                if matches.empty:
                    reason = "source_unavailable" if origin == "remote" else "local_source_unavailable"
                    if revision is not None and origin == "remote":
                        reason = "revision_unavailable"
                    raise SessionAccessError(reason, f"No source matches {name}", eid=session.eid)
                candidates.append(matches)
                matches = filter_datasets(matches, revision=revision,
                                          revision_last_before=revision is None, assert_unique=True)
                row = matches.iloc[0]
                selected.append((row, self._describe(session, row, origin)))
            except ALFError as exc:
                raise SessionAccessError("source_conflict", f"Ambiguous source for {name}",
                                         eid=session.eid) from exc
            except SessionAccessError:
                raise
            except Exception as exc:
                raise SessionAccessError("access_failed", f"Could not resolve source {name}",
                                         eid=session.eid) from exc
        if (latest_common_revision and revision is None
                and len({source.collection for _, source in selected}) == 1
                and len({source.revision for _, source in selected}) > 1):
            effective_collection = selected[0][1].collection
            groups = [
                [(row, self._describe(session, row, origin)) for _, row in matches.iterrows()]
                for matches in candidates
            ]
            shared = set.intersection(*[
                {source.revision for _, source in group
                 if source.collection == effective_collection}
                for group in groups
            ])
            if not shared:
                available = {
                    name: sorted({source.revision for _, source in group
                                  if source.collection == effective_collection})
                    for name, group in zip(names, groups)
                }
                raise SessionAccessError(
                    "source_conflict",
                    f"Joint sources have no shared revision for {session.eid}: {available}",
                    eid=session.eid,
                )
            # ONE orders revision labels lexicographically; unrevisioned ('') is oldest.
            common_revision = max(shared)
            resolved = []
            for name, group in zip(names, groups):
                matches = [(row, source) for row, source in group
                           if source.collection == effective_collection
                           and source.revision == common_revision]
                if len(matches) != 1:
                    raise SessionAccessError("source_conflict", f"Ambiguous source for {name}",
                                             eid=session.eid)
                resolved.append(matches[0])
            selected = resolved
            warnings.warn(
                f"{session.eid}: joint sources {names} have different default revisions; "
                f"using newest shared revision {common_revision!r} in {effective_collection!r}",
                stacklevel=2,
            )
        if len({(source.collection, source.revision) for _, source in selected}) > 1:
            raise SessionAccessError("source_conflict", "Joint sources have different collections/revisions",
                                     eid=session.eid)
        loaded = []
        for row, source in selected:
            staging = ExitStack()
            try:
                refresh = self.force_reload and source.dataset_id not in self._reloaded
                if refresh:
                    temporary = staging.enter_context(TemporaryDirectory(prefix="reload-", dir=self.cache_dir))
                    # Installed ONE has no public force option on load_dataset_from_id.
                    # Its download primitive always transfers, even for cached datasets.
                    path = self._client()._download_dataset(row, cache_dir=temporary, update_cache=False)
                    if path is None:
                        raise SessionAccessError("source_unavailable", f"Remote reload unavailable: {source.name}", eid=session.eid)
                else:
                    path = self._client().load_dataset_from_id(source.dataset_id, download_only=True)
                path = Path(path)
                # ONE can return an inconsistent local file in offline mode. Do not
                # decode it as valid source data merely because the path exists.
                if row.get("file_size") is not None and path.stat().st_size != row.file_size:
                    raise SessionAccessError("source_inconsistent", f"File size mismatch: {path}",
                                             eid=session.eid)
                expected_hash = row.get("hash")
                if expected_hash:
                    digest = hashlib.md5()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected_hash:
                        raise SessionAccessError("source_inconsistent", f"File hash mismatch: {path}",
                                                 eid=session.eid)
                data = path if download_only else load_file_content(path)
                if refresh:
                    source.path.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(source.path)
                    path = source.path
                    if download_only:
                        data = path
                    self._reloaded.add(source.dataset_id)
                identity = DatasetSource(source.eid, source.dataset_id, source.name,
                                         source.collection, source.revision, path, "local")
                loaded.append(LoadedDataset(data, identity))
            except SessionAccessError:
                raise
            except ALFObjectNotFound as exc:
                reason = ("local_source_unavailable" if self.policy == AccessPolicy.LOCAL_ONLY
                          else "source_unavailable")
                raise SessionAccessError(reason, f"Dataset unavailable: {source.name}",
                                         eid=session.eid) from exc
            except Exception as exc:
                raise SessionAccessError("access_failed", f"Could not load {source.name}",
                                         eid=session.eid) from exc
            finally:
                staging.close()
        return tuple(loaded)
