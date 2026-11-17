"""Train one independent filtered run using the Experiment 2 production lifecycle."""
from common import module, bind_experiment


def bind_shared_experiment(out, record):
    # Both delay pairs share one experiment but have distinct native supports.
    shared = dict(record)
    support = record["support"]
    shared["support"] = {key: support[key] for key in
        ("parent_support", "selection_identity", "selection", "unfiltered_sources")}
    bind_experiment(out, shared)

if __name__ == "__main__":
    lifecycle = module("subregion-aggregation", "train.py", "information_training_lifecycle")
    lifecycle.bind_experiment = bind_shared_experiment
    lifecycle.main()
