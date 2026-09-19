# Repository instructions

These instructions apply to the entire repository. Read [instructions.md](instructions.md) for project context and development guidance.

## No test writing

- Do not create, add, or modify test code in this repository, including unit, integration, regression, end-to-end, snapshot, or temporary test scripts.
- Validate changes through code inspection, existing checks, or manual verification without writing tests.
- Do not delete existing tests unless the user explicitly requests their removal.

## No CI/CD automation

- This repository does not use continuous integration, continuous delivery, or continuous deployment (CI/CD).
- Do not add CI/CD workflows, pipeline configuration, automated dependency-update bots (including Dependabot), or scripts and documentation intended to support those systems.
- Use code inspection, existing local checks, and manual verification. Review dependency updates manually and keep them consistent with the repository's constraints files.
