# Release checklist

- [ ] `python run_all.py` passes.
- [ ] `python scripts/verify_repository.py` passes.
- [ ] K25 status is not described as a full pass unless its benchmark changes.
- [ ] `U=-1 != KEY=-1` still has an explicit regression test.
- [ ] New research files are indexed.
- [ ] New third-party material is recorded in `THIRD_PARTY_NOTICES.md`.
- [ ] `CHANGELOG.md` and `CITATION.cff` version/date are updated.
- [ ] No secrets, tokens, private paths or `__pycache__` files are committed.
- [ ] README limitations and non-claims still match the code.
