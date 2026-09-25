"""
Find and rank IBL Brain Wide Map sessions by visual-region relevance.

Output:
    output/eid-relevanc.txt

Relevance definition:
    relevant good mapped units / all good mapped units * 100

Only sessions with relevance > 0 are written to the output.
The optional --min-relevance argument can raise this threshold.

Example:
    python find_visual_eids.py
    python find_visual_eids.py --min-relevance 50

Dependencies:
    one-api
    ibllib
    iblatlas
    numpy

Notes:
    - The IBL project is intentionally fixed to "brainwide".
    - The script uses cluster quality label == 1 as the "good unit" criterion.
    - Atlas descendants are included, therefore cortical layers such as
      VISp1, VISp2/3, VISp4, etc. are counted under VISp.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable  # noqa: UP035

import numpy as np
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from one.api import ONE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"

# Fixed as requested: only Brain Wide Map data are considered.
IBL_PROJECT = "brainwide"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.paths import dataset_dir, output_dir

OUTPUT_FILE = output_dir() / "eid-relevanc.txt"

# Default selection for this discovery script, not a neural-data requirement.
# docs/neural-data/spec.md imposes no fixed region list or weights. Each good
# mapped unit contributes equally; --regions overrides the anatomical selection.
#
# Allen/IBL acronym for dorsal lateral geniculate complex is LGd.
RELEVANT_REGIONS = (
    "LGd",
    # Allen CCF visual cortical areas.
    #
    # A build_relevant_region_ids() az atlaszhierarchia alapján
    # automatikusan hozzáveszi az alrégiókat/rétegeket is.
    # Például VISa -> VISa1, VISa2/3, VISa4, VISa5, VISa6a, VISa6b.
    #
    # Nem elég csak a közös "VIS" szülőt használni, mert például
    # VISa és VISrl az Allen hierarchiában a PTLp ág alatt található.
    "VISp",
    "VISl",
    "VISli",
    "VISpl",
    "VISpor",
    "VISpm",
    "VISam",
    "VISa",
    "VISrl",
    "VISal",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionRelevance:
    eid: str
    relevance_percent: float
    relevant_units: int
    mapped_good_units: int
    total_good_units: int
    mapping_coverage_percent: float
    region_counts: dict[str, int]

    @property
    def regions_text(self) -> str:
        return ", ".join(
            f"{region}({count})"
            for region, count in sorted(self.region_counts.items())
        )


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------

def build_relevant_region_ids(
    atlas: AllenAtlas,
    regions: Iterable[str] = RELEVANT_REGIONS,
) -> dict[str, set[int]]:
    """
    Build Allen atlas ID sets for each relevant region.

    Descendants are included, e.g. VISp includes VISp1, VISp2/3, VISp4,
    VISp5, VISp6a and VISp6b.

    Absolute IDs are stored because IBL atlas IDs may be lateralized:
    negative IDs represent the left hemisphere.
    """
    result: dict[str, set[int]] = {}

    for acronym in regions:
        if acronym not in atlas.regions.acronym:
            raise ValueError(f"Unknown Allen atlas acronym: {acronym}")
        atlas_ids = np.atleast_1d(
            atlas.regions.acronym2id(acronym)
        )

        if atlas_ids.size == 0:
            raise ValueError(
                f"Unknown Allen atlas acronym: {acronym}"
            )

        descendant_ids: set[int] = set()

        # acronym2id may expose lateralized IDs, therefore normalize them.
        for atlas_id in np.unique(np.abs(atlas_ids.astype(int))):
            descendants = atlas.regions.descendants(ids=int(atlas_id))
            descendant_ids.update(
                abs(int(region_id))
                for region_id in descendants["id"]
            )

        result[acronym] = descendant_ids

    return result


def build_atlas_id_to_region(
    region_ids: dict[str, set[int]],
) -> dict[int, str]:
    """
    Create a reverse lookup from Allen atlas ID to project-level region.

    Raises if two configured project regions overlap in the Allen hierarchy.
    This protects the relevance count from double counting.
    """
    reverse: dict[int, str] = {}

    for region, ids in region_ids.items():
        for atlas_id in ids:
            previous = reverse.get(atlas_id)

            if previous is not None and previous != region:
                raise ValueError(
                    "Relevant region definitions overlap: "
                    f"atlas ID {atlas_id} belongs to both "
                    f"{previous} and {region}."
                )

            reverse[atlas_id] = region

    return reverse


# ---------------------------------------------------------------------------
# ONE discovery
# ---------------------------------------------------------------------------

def find_candidate_eids(
    one: ONE,
    regions: Iterable[str] = RELEVANT_REGIONS,
) -> set[str]:
    """
    Find Brain Wide Map sessions touching at least one relevant region.

    Searching is performed at probe-insertion level first, which avoids
    analyzing every electrophysiology session in the database.
    """
    candidate_eids: set[str] = set()

    for region in regions:
        print(f"Searching Brain Wide Map insertions in {region} ...")

        pids = one.search_insertions(
            atlas_acronym=region,
            datasets="clusters.metrics.pqt",
            project=IBL_PROJECT,
            query_type="remote",
        )

        print(f"  found {len(pids)} insertion(s)")

        for pid in pids:
            eid, _ = one.pid2eid(pid, query_type="remote")

            if eid is not None:
                candidate_eids.add(str(eid))

    return candidate_eids


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------

def load_probe_clusters(
    one: ONE,
    atlas: AllenAtlas,
    pid: str,
):
    """
    Load only cluster/channel information required for relevance analysis.

    Spike time arrays are intentionally not loaded because this script only
    needs cluster quality and anatomical location.
    """
    loader = SpikeSortingLoader(
        pid=pid,
        one=one,
        atlas=atlas,
    )

    clusters = loader.load_spike_sorting_object(
        "clusters",
        missing="raise",
    )
    # Read the supplied IBL quality label only. merge_clusters recomputes
    # missing metrics even with compute_metrics=False, requiring spike arrays.
    if "label" not in clusters:
        metrics = clusters.get("metrics")
        if metrics is not None and "label" in metrics:
            labels = np.asarray(metrics["label"])
            if labels.shape != np.asarray(clusters["channels"]).shape:
                raise ValueError("Cluster metrics and cluster channels differ in shape")
            clusters["label"] = labels

    if "label" not in clusters:
        return clusters

    # Raw ALF anatomy uses brainLocationIds_ccf_2017; merged Brainbox data
    # uses atlas_id. Both contain Allen CCF IDs, not positions or acronyms.
    if "atlas_id" not in clusters:
        if "brainLocationIds_ccf_2017" in clusters:
            clusters["atlas_id"] = clusters["brainLocationIds_ccf_2017"]
        else:
            channels = loader.load_spike_sorting_object(
                "channels", missing="raise",
            )
            anatomy = channels.get("atlas_id")
            if anatomy is None:
                anatomy = channels.get("brainLocationIds_ccf_2017")
            if anatomy is not None:
                anatomy = np.asarray(anatomy)
                indices = np.asarray(clusters["channels"])
                if (
                    anatomy.ndim != 1
                    or indices.ndim != 1
                    or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0)
                    or np.any(indices >= anatomy.size)
                ):
                    raise ValueError("Invalid cluster-to-channel anatomy mapping")
                clusters["atlas_id"] = anatomy[indices]

    return clusters


def analyze_session(
    one: ONE,
    atlas: AllenAtlas,
    eid: str,
    atlas_id_to_region: dict[int, str],
) -> SessionRelevance | None:
    """
    Calculate session-level visual relevance across every probe in the EID.

    Relevance:
        number of relevant good mapped clusters
        ---------------------------------------
        number of all good mapped clusters
    """
    pids, probe_names = one.eid2pid(
        eid,
        query_type="remote",
    )

    total_good_units = 0
    mapped_good_units = 0
    relevant_units = 0
    region_counts: dict[str, int] = defaultdict(int)

    for pid, probe_name in zip(pids, probe_names):
        try:
            clusters = load_probe_clusters(
                one=one,
                atlas=atlas,
                pid=str(pid),
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  WARNING: skipped {eid}/{probe_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if "label" not in clusters:
            # Missing required QC never passes the good-unit filter. Do not
            # substitute a different metric with different source semantics.
            print(
                f"  WARNING: {eid}/{probe_name} has no cluster "
                "'label'; probe skipped."
            )
            continue

        if "atlas_id" not in clusters:
            print(
                f"  WARNING: {eid}/{probe_name} has no merged "
                "'atlas_id'; probe skipped."
            )
            continue

        labels = np.asarray(clusters["label"])
        atlas_ids_raw = np.asarray(clusters["atlas_id"])

        if labels.ndim != 1 or atlas_ids_raw.shape != labels.shape:
            print(
                f"  WARNING: cluster label/atlas lengths differ for "
                f"{eid}/{probe_name}; probe skipped."
            )
            continue

        good_mask = labels == 1
        total_good_units += int(np.count_nonzero(good_mask))

        # atlas_id is expected to be numeric. Non-finite and zero values are
        # considered unmapped and intentionally excluded from the denominator.
        try:
            atlas_ids_float = atlas_ids_raw.astype(float)
        except (TypeError, ValueError):
            print(
                f"  WARNING: non-numeric atlas IDs for "
                f"{eid}/{probe_name}; probe skipped."
            )
            continue

        mapped_mask = (
            good_mask
            & np.isfinite(atlas_ids_float)
            & (atlas_ids_float != 0)
        )

        mapped_atlas_ids = np.abs(
            atlas_ids_float[mapped_mask].astype(int)
        )

        mapped_good_units += int(mapped_atlas_ids.size)

        for atlas_id in mapped_atlas_ids:
            region = atlas_id_to_region.get(int(atlas_id))

            if region is None:
                continue

            relevant_units += 1
            region_counts[region] += 1

    if mapped_good_units == 0 or relevant_units == 0:
        return None

    relevance_percent = (
        relevant_units / mapped_good_units * 100.0
    )

    mapping_coverage_percent = (
        mapped_good_units / total_good_units * 100.0
        if total_good_units > 0
        else 0.0
    )

    return SessionRelevance(
        eid=eid,
        relevance_percent=relevance_percent,
        relevant_units=relevant_units,
        mapped_good_units=mapped_good_units,
        total_good_units=total_good_units,
        mapping_coverage_percent=mapping_coverage_percent,
        region_counts=dict(region_counts),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(
    results: Iterable[SessionRelevance],
    output_file: Path,
    min_relevance: float,
    regions: Iterable[str] = RELEVANT_REGIONS,
) -> None:
    """
    Write a human-readable EID relevance table.

    Example:
        EID                                    RELEVANCE   UNITS       REGIONS
        754b...                                 82.14%      184/224     LGd(20), VISp(164)
    """
    rows = list(results)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        file.write("IBL Brain Wide Map - visual-region relevance\n")
        file.write(
            f"Minimum relevance: {min_relevance:.2f}%\n"
        )
        file.write(
            "Relevance = relevant good mapped units / "
            "all good mapped units\n"
        )
        file.write(
            "Regions = relevant project regions with good-unit counts\n"
        )
        file.write(f"Selected Allen CCF regions (including descendants): {', '.join(regions)}\n")
        file.write("Quality: supplied IBL label == 1; equal weight per unit\n")
        file.write("Probes missing required quality/anatomy are skipped; scores use available probes\n")
        file.write("\n")

        header = (
            f"{'EID':36}  "
            f"{'RELEVANCE':>10}  "
            f"{'UNITS':>13}  "
            f"{'MAP COVERAGE':>12}  "
            f"REGIONS\n"
        )
        file.write(header)
        file.write("-" * 110 + "\n")

        for result in rows:
            units = (
                f"{result.relevant_units}/"
                f"{result.mapped_good_units}"
            )

            file.write(
                f"{result.eid:36}  "
                f"{result.relevance_percent:9.2f}%  "
                f"{units:>13}  "
                f"{result.mapping_coverage_percent:11.2f}%  "
                f"{result.regions_text}\n"
            )

        file.write("\n")
        file.write(f"Sessions: {len(rows)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find IBL Brain Wide Map EIDs containing good units from "
            "visual-processing-related brain regions."
        )
    )

    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.0,
        help=(
            "Minimum session relevance percentage in [0, 100]. "
            "Example: --min-relevance 50"
        ),
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        default=RELEVANT_REGIONS,
        metavar="ACRONYM",
        help="Allen region acronyms, including descendants (default: %(default)s).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.min_relevance <= 100.0:
        raise ValueError(
            "--min-relevance must be between 0 and 100."
        )

    print("Initializing ONE and Allen atlas ...")
    one = ONE(base_url=ONE_BASE_URL, cache_dir=str(dataset_dir()))
    atlas = AllenAtlas()

    regions = tuple(dict.fromkeys(args.regions))
    region_ids = build_relevant_region_ids(atlas, regions)
    atlas_id_to_region = build_atlas_id_to_region(region_ids)

    candidate_eids = find_candidate_eids(one, regions)

    print()
    print(f"Candidate EIDs: {len(candidate_eids)}")
    print("Analyzing candidate sessions ...")

    results: list[SessionRelevance] = []

    for index, eid in enumerate(
        sorted(candidate_eids),
        start=1,
    ):
        print(
            f"[{index}/{len(candidate_eids)}] {eid}"
        )

        try:
            result = analyze_session(
                one=one,
                atlas=atlas,
                eid=eid,
                atlas_id_to_region=atlas_id_to_region,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  WARNING: session failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if result is None:
            continue

        # A 0% session is never emitted, even if min relevance is 0.
        if result.relevance_percent <= 0.0:
            continue

        if (
            result.relevance_percent
            < args.min_relevance
        ):
            continue

        results.append(result)

    # Highest relevance first. For equal relevance prefer sessions with more
    # relevant units, then use EID for deterministic output.
    results.sort(
        key=lambda item: (
            -item.relevance_percent,
            -item.relevant_units,
            item.eid,
        )
    )

    write_results(
        results=results,
        output_file=OUTPUT_FILE,
        min_relevance=args.min_relevance,
        regions=regions,
    )

    print()
    print(
        f"Done: {len(results)} session(s) written to "
        f"{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()
