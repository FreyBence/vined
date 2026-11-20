# E1 visual information probes

From the checkout root, using the existing project environment:

```powershell
.venv\Scripts\python.exe script/experiment/visual-information/run.py
```

The default output is `output/experiment/visual-information/e1`. It must not
already exist. Use `--output output/experiment/visual-information/another-run`
for an independent rerun. `--threads` controls existing BLAS thread pools
(default four). No dependency installation or network access is needed.

This experiment implements the requested retained-feature E1 comparison.
It does not invoke the old motion audit, encoder, renderer or neural model.
Its train/validation views retain original trial identities and the selected
dataset's persisted membership. See [interface.md](interface.md) for frozen
inputs, qualification, outputs and verification limits.

The report and scientific outputs are generated under the chosen output
directory, which is ignored by Git. The implementation and usage documentation
are tracked separately from these research artifacts.
