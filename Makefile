# PEEL-Local's local gate. `make check` is what CI runs and what to run
# before opening a pull request -- see CONTRIBUTING.md.
#
# The gate is deliberately cheap: it excludes the `slow` test tier (GPU,
# model downloads, a running Ollama), so it stays fast enough to run on every
# change. The expensive tier is `make check-slow`.

PYTHON ?= python3

.PHONY: check check-slow lint docs test test-slow freeze-check help

help:
	@echo "make check       -- lint + docs + the test tiers that need no GPU/Ollama (the gate)"
	@echo "make check-slow  -- the same, plus the slow tier"
	@echo "make lint        -- ruff, errors only"
	@echo "make docs        -- verify docs/ symbol references and FUNCTIONS.md coverage"
	@echo "make test        -- pytest, excluding the slow tier"
	@echo "make freeze-check-- verify data/ still matches the v1.0.0-paper hashes"

lint:
	$(PYTHON) -m ruff check --select=F,E9 .

docs:
	$(PYTHON) scripts/check_docs.py

test:
	$(PYTHON) -m pytest

test-slow:
	$(PYTHON) -m pytest -m slow

# The behaviour freeze, on its own, for when that is the only question.
freeze-check:
	shasum -a 256 -c tests/golden/data.sha256

check: lint docs test
	@# Records that the gate ran, for the PreToolUse guard that refuses
	@# `gh pr create` over an ungated branch. Harmless when absent.
	@[ -x "$(HOME)/.claude/hooks/git-guard.sh" ] \
		&& "$(HOME)/.claude/hooks/git-guard.sh" --stamp || true

check-slow: lint docs test test-slow
	@[ -x "$(HOME)/.claude/hooks/git-guard.sh" ] \
		&& "$(HOME)/.claude/hooks/git-guard.sh" --stamp || true
