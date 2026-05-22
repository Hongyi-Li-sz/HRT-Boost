# Release checklist

Confirm these items before making the repository public or creating a GitHub
release.

- [ ] Confirm the final repository name and URLs in `pyproject.toml`.
- [ ] Confirm the final package version in `pyproject.toml` and `hrt_boost/__init__.py`.
- [ ] Add the paper venue, DOI, arXiv, OpenReview, or project-page URL when available.
- [ ] Confirm the corresponding-author notation for Jun Xu in the README or paper link.
- [ ] Add dataset source links, licenses, and citations in `docs/DATASETS.md`.
- [ ] Run `python scripts/run_quick_demo.py`.
- [ ] Run `python scripts/run_quick_demo.py --include-hrt`.
- [ ] Run `python -m pytest`.
- [ ] Run the intended benchmark command and add final benchmark results to the README if desired.
- [ ] Create a GitHub release tag such as `v0.1.0` after the repository is ready.
