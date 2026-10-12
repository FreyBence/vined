"""Scenario attempt naming and selection of training-owned checkpoints."""

from pathlib import Path


def scenario_directory(root, scenario_id):
    if type(scenario_id) is not int or scenario_id < 1:
        raise ValueError("Scenario ID must be a positive integer")
    return Path(root).resolve() / f"scenario-{scenario_id}"


def attempt_name(value):
    text = str(value)
    if not text.isascii() or not text.isdecimal() or int(text) < 1:
        raise ValueError("Attempt must be a positive number, such as 001")
    return f"{int(text):03d}"


def new_attempt_directory(root, scenario_id):
    """Reserve the next number without reusing failed or concurrent attempts."""
    parent = scenario_directory(root, scenario_id)
    parent.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name) for path in parent.iterdir()
               if path.name.isascii() and path.name.isdecimal()]
    number = max(numbers, default=0) + 1
    while True:
        destination = parent / attempt_name(number)
        try:
            destination.mkdir(exist_ok=False)
            return destination
        except FileExistsError:
            number += 1


def selected_attempt(root, scenario_id, attempt=None):
    """Choose the latest attempt, then its preselected best checkpoint."""
    parent = scenario_directory(root, scenario_id)
    if attempt is None:
        if not parent.is_dir():
            raise ValueError(f"No training attempts for scenario {scenario_id} in {parent}")
        candidates = [path for path in parent.iterdir()
                      if path.is_dir() and path.name.isascii() and path.name.isdecimal() and int(path.name) > 0]
        if not candidates:
            raise ValueError(f"No training attempts for scenario {scenario_id} in {parent}")
        directory = max(candidates, key=lambda path: int(path.name))
    else:
        directory = parent / attempt_name(attempt)
    checkpoint = directory / "model_best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Training attempt {directory} has no selected model_best.pt; finish training or choose --attempt")
    return directory.name, checkpoint
