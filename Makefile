# Copyright 2026 Fraunhofer AISEC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
.PHONY: install install-test install-docs install-dev \
        docs docs-serve docs-build docs-build-strict docs-deploy \
        test test-clean format license-header clean

# --- Installation ---
# Install the package only (for end users)
install:
	pip install -e .

# Install with test dependencies (used by CI and developers)
install-test:
	pip install -e ".[test]"

# Install with documentation dependencies (used by CI and developers)
install-docs:
	pip install -e ".[docs]"

# Full development setup (local development only)
install-dev:
	pip install -e ".[test,docs]"
	pre-commit install

# --- Documentation ---
docs-serve: install-docs
	mkdocs serve -a 127.0.0.1:8000

docs-build: install-docs
	mkdocs build

docs-build-strict: install-docs
	mkdocs build --strict

docs-deploy: install-docs
	mkdocs gh-deploy --force

# Default docs target: start the local preview server
docs: docs-serve

# --- Tests ---
# Install test dependencies, then run the tests
test: install-test
	pytest

# Clear stale caches, reinstall, then run the tests
test-clean: clean install-test
	pytest

# --- Code Quality ---
# Format the entire code base
format: install-dev
	black --target-version=py312 .

# --- License ---
# Apply license headers to all relevant files
license-header: install-dev
	@for t in .py .md .yml .yaml .toml; do \
		./tools/license_header.py . -e $$t; \
	done
	./tools/license_header.py . -n Makefile

# --- Cleanup ---
clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache