# Training scenarios

Each numbered file contains one scenario object: `1.json` through `10.json`.
The filename matches the object's integer `id`. Select scenario N by reading
`scenarios/N.json`; its settings are at the JSON root.

`shared.json` contains the schema version, fixed sources, available modalities,
supported prediction tasks, masking capabilities, comparison groups, and
experimental axes that apply across scenarios. Read it alongside the selected
scenario; it is shared metadata, not a defaults file to merge into the scenario.

Scenarios 1–5 describe encoding and scenarios 6–10 describe decoding, each with
1, 3, 6, 9, or 12 context bins at 60 Hz.

These files describe experiment configurations. The training commands do not
currently load them automatically. Unspecified training values remain `null`.
