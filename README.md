# CMOR-438

This repository demonstrates how to organize GitHub automation assets alongside a lightweight Python project.

## Repository Layout

- `.github/workflows/ci.yml` runs `pytest` on every push and pull request using Python 3.11.
- `.github/ISSUE_TEMPLATE/bug_report.md` guides contributors when opening bug reports.
- `.github/PULL_REQUEST_TEMPLATE.md` provides a checklist for pull requests.
- `.github/CODEOWNERS` assigns default reviewers (replace `@your-username` with a real handle).
- `requirements.txt` lists Python dependencies required by the workflow.
- `tests/test_placeholder.py` contains a starter test to keep the CI green.

## Getting Started

1. Install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Run the test suite locally: `pytest`
3. Update `.github/CODEOWNERS` and extend the templates as your project grows.

With this structure in place, GitHub will automatically validate contributions and provide consistent collaboration guidance.
