# E1b confound-controlled visual motion probe

Run from the checkout root in the existing environment:

```powershell
.venv\Scripts\python.exe script/experiment/visual-motion-controlled/run.py
```

The default destination is `output/experiment/visual-information/e1b`.
Use `--output PATH` for another fresh destination; existing directories are
rejected. `--e1 PATH` selects the completed E1 directory and `--threads N`
sets the existing BLAS thread limit (default four). No installation is needed.

This diagnostic reuses frozen E1 qualified labels and retained normalized
features. It reads only selected train/validation numeric payloads. There is no
rendering, encoder inference, neural training, production change or test-data
evaluation. E1's prior whole-generation integrity verification is inherited;
the E1b reader does not repeat its test-payload access.

Source and aligned stages, median-limited and full-trial intervals, and
unrestricted and matched comparisons remain separate. Sparse independent-trial
strata are inconclusive. See [interface.md](interface.md) for contracts and
the generated `report.md` and `verification.md` for measured conclusions.

The completed follow-up run is under
[`e1b-completed`](../../../output/experiment/visual-information/e1b-completed/).
Its [compact report](../../../output/experiment/visual-information/e1b-completed/summary.md)
and [manual verification](../../../output/experiment/visual-information/e1b-completed/verification.md)
accompany the detailed generated score tables. The actual invocation used
`--output output/experiment/visual-information/e1b-completed`.
