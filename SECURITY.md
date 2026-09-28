# Security Policy

This policy covers `ml-pipes-supervision` and follows the shared
[ml-pipes Security Policy](https://github.com/trained-by-humans/ml-pipes/blob/main/SECURITY.md).

## Supported releases

Report issues affecting the latest published release or the current `main`
branch. Release candidates are supported while they are the current candidate
for the package.

## Reporting a vulnerability

Report suspected vulnerabilities through GitHub Private Vulnerability
Reporting for this repository. Do not open a public issue for a sensitive
report.

Include the package version or commit, a minimal reproduction, the affected
Python and dependency versions, and any required environment details.

## In scope

- vulnerabilities in the `ml_pipes.supervision` package code;
- packaging, GitHub Actions, or release-process mistakes that could compromise
  published distributions; and
- unsafe behavior introduced by this package's Supervision, tracker, or
  optional Inference integration boundaries.

## Out of scope

- vulnerabilities in Supervision, Roboflow Inference, trackers, or other
  dependencies that this package does not introduce or modify; report those to
  their upstream maintainers;
- applications built on top of this package; and
- infrastructure, credentials, or deployment configuration owned by downstream
  users.
