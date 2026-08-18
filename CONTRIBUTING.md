<!--
Copyright 2026 Fraunhofer AISEC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# CONTRIBUTING

Thank you for your interest in improving **QACE-SDK**!  
We welcome issues, bug fixes, new features, and documentation updates.  
Before contributing, please read this short guide and, **most importantly, sign the Fraunhofer AISEC Contributor License Agreement (CLA)**.

---

## Table of Contents
1. Contributor License Agreement (CLA) ★  
2. Code of Conduct  
3. Getting Started  
4. Development Workflow  
5. Style & Tooling  
6. Testing  
7. Documentation  
8. Commit & PR Checklist  
9. License

---

## 1. Contributor License Agreement (CLA) ★

All external contributors **must have a signed Fraunhofer AISEC CLA on file _before_ we can accept any Pull/Merge Request.**  
A blank copy will be provided to you when you open your first PR.

### How to sign

1. Request the CLA via mail at qace-sdk@aisec.fraunhofer.de
2. Download the CLA PDF we send you.  
3. Read it carefully, sign it (electronically or by hand), and email the signed copy back to us.  
4. We will confirm receipt; you can then proceed with your contribution.

### CLA – key points (non-binding summary)

* You grant AISEC and all downstream users a **perpetual, worldwide, royalty-free** copyright license to use, reproduce, modify, and distribute your contribution.
* You grant a matching **patent license** for any patents you hold that would otherwise be infringed by your contribution.
* You confirm you have the legal right to contribute the code (e.g., your employer has approved it or waived their rights).
* You affirm that, to the best of your knowledge, the contribution does **not infringe** third-party IP.
* Contributions are provided **as-is** without warranties; you are not required to provide support.
* The Project may decline to accept a contribution or re-license the Project in the future.
* German law applies; courts in Munich have jurisdiction.

The full, legally binding text is provided separately and must be signed by every contributor (or their employer) prior to merge.

---

## 2. Code of Conduct
This project follows the [Contributor Covenant v3.0](https://www.contributor-covenant.org/version/3/0/code_of_conduct/).  
Be respectful, inclusive, and constructive in all interactions.

---

## 3. Getting Started

```bash
git clone https://github.com/<your-user>/qace-sdk.git
cd qace-sdk
python -m venv .venv
source .venv/bin/activate
make install          # installs runtime + dev deps & pre-commit hooks
```

---

## 4. Development Workflow

1. `git checkout -b feat/<short-description>`
2. Implement your change. Keep commits small & focused.
3. Run pre-commit hooks + tests locally (`make test`).
4. Open a PR. The CI will run the same checks.
5. A maintainer will review, request changes if needed, and merge.

---

## 5. Style & Tooling

| Tool                 | Purpose                         | Run locally with             |
|----------------------|---------------------------------|------------------------------|
| black                | Code formatting                 | `black .`                    |
| beartype             | Runtime type enforcement        | automatic                    |
| pre-commit           | Runs all linters on commit      | `pre-commit run --all-files` |
| license_header.py    | Apache-2.0 headers              | `make license_header`        |
| generate_licenses.py | Generating thrid party licenses | `./generate_licences.py`     |

Guidelines:

* Follow **PEP 8**, enforced by `black`.
* Add **type hints** for all public APIs.
* Maintain **MSB₀ qubit ordering** via helpers in `qace.vbf._circuit_conventions`.
* Write Google-style docstrings.

---

## 6. Testing

```bash
make test   # installs test extras & runs pytest
```

* Cover new code with unit tests.  
* Tests must pass on Python 3.12+.  
* Use `IterativeExecutor` for deterministic single-shot tests, which incorporate quantum amplitude amplification.

---

## 7. Documentation

Live preview:

```bash
make docs-serve   # http://127.0.0.1:8000
```

Static build:

```bash
make docs-build   # output in site/
```

Update docstrings + `docs/` Markdown when you change public APIs.

---

## 8. Commit & PR Checklist

- [ ] CLA signed and confirmed by AISEC.  
- [ ] `black`, and `pre-commit` pass.  
- [ ] Unit tests added/updated and `make test` passes.  
- [ ] Public APIs are typed and documented.  
- [ ] Apache-2.0 headers added (`make license_header`).  
- [ ] No secrets / large binaries committed.  
- [ ] PR description explains _what_ and _why_.

---

## 9. License

All contributions are released under the existing [Apache License 2.0](LICENSE).
