# Plenty of Fish - housekeeping
# `make` or `make help` lists targets. .venv/ and .git/ are never touched.

# Prune .venv and .git so we never clean dependencies or repo internals.
PRUNE := -path ./.venv -prune -o -path ./.git -prune -o

.DEFAULT_GOAL := help

.PHONY: help clean clean-pyc clean-cache clean-os clean-results clean-all untrack

help: ## Show this help
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

clean: clean-pyc clean-cache clean-os ## Remove all build/cache/OS junk (keeps results)

clean-pyc: ## Remove __pycache__ dirs and .pyc/.pyo files
	@find . $(PRUNE) -type d -name '__pycache__' -exec rm -rf {} +
	@find . $(PRUNE) -type f \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f {} +
	@echo "Removed Python bytecode caches."

clean-cache: ## Remove tool caches (.pytest_cache, .mypy_cache, .ruff_cache, *.egg-info)
	@find . $(PRUNE) -type d \( -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '*.egg-info' \) -exec rm -rf {} +
	@rm -rf .coverage htmlcov build dist
	@echo "Removed tool caches."

clean-os: ## Remove OS cruft (.DS_Store, Thumbs.db)
	@find . $(PRUNE) -type f \( -name '.DS_Store' -o -name 'Thumbs.db' \) -exec rm -f {} +
	@echo "Removed OS junk files."

clean-results: ## Remove generated simulation outputs in results/
	@find results -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	@echo "Cleared results/ (generated outputs)."

clean-all: clean clean-results ## Everything above, including generated results

untrack: ## Stop tracking committed caches in git (run once, then commit)
	@git rm -r --cached --quiet $$(git ls-files | grep -E '__pycache__|\.pyc$$|\.DS_Store') 2>/dev/null || true
	@echo "Untracked cached files. Review with 'git status', then commit."
