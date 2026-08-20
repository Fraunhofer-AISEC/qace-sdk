#!/usr/bin/env python3
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
"""Generate third-party license report and aggregated NOTICE file."""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import site
import subprocess
import sys
import sysconfig
from collections import Counter

# --- Configuration --------------------------------------------------------
FORBIDDEN = ["GPL", "AGPL", "LGPL", "UNKNOWN"]
OUTPUT_DIR = "third_party_licenses"
NOTICE_AGG = "NOTICE"  # aggregated project-level NOTICE file
REPORT_FILE = "REPORT.md"  # report next to this script

# Own project metadata (written to the top of NOTICE)
PROJECT_NAME = "qace"
PROJECT_COPYRIGHT = "Copyright 2026 Fraunhofer AISEC"
PROJECT_OUTBOUND_LICENSE = "Apache-2.0"

# Own packages that must NOT appear in the third-party report
SELF_PACKAGES = {"qace"}

# Manually verified false positives: (name, version) -> reason / real license
ALLOWLIST = {}
# --------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def install_pip_licenses() -> None:
    log(">> Installing pip-licenses (if needed)...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "pip-licenses",
        ],
        check=True,
    )


def collect_licenses() -> list[dict]:
    log(">> 1/4: Gathering package license metadata...")
    log(f"   Excluding self packages: {', '.join(sorted(SELF_PACKAGES)) or '(none)'}")
    cmd = [
        "pip-licenses",
        "--format=json",
        "--with-authors",
        "--with-urls",
        "--with-license-file",
        "--no-license-path",
    ]
    if SELF_PACKAGES:
        cmd += ["--ignore-packages", *sorted(SELF_PACKAGES)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def analyze(pkgs: list[dict]) -> tuple[list, list, list, list]:
    """Return (rows, violations, allowlisted, unknowns).

    rows      = (name, ver, effective, is_forbidden_unhandled)
    unknowns  = (name, ver, effective, note)  -> packages whose metadata
                license was UNKNOWN (regardless of later detection).
    """
    log(f">> 2/4: Analyzing licenses (forbidden: {';'.join(FORBIDDEN)})...")
    rows, violations, allowlisted, unknowns = [], [], [], []
    for p in sorted(pkgs, key=lambda x: x["Name"].lower()):
        name, version = p["Name"], p["Version"]
        if name.lower() in SELF_PACKAGES:
            continue
        lic = p.get("License") or "UNKNOWN"
        effective = lic
        lic_up = lic.upper()
        was_unknown = lic_up == "UNKNOWN"
        note = None

        if was_unknown:
            text = (p.get("LicenseText") or "").upper()
            if "APACHE LICENSE" in text and "VERSION 2.0" in text:
                effective, lic_up = "Apache-2.0 (detected)", "APACHE-2.0"
                note = "Metadata UNKNOWN - detected as Apache-2.0 (verify manually)"
            elif "GNU GENERAL PUBLIC LICENSE" in text:
                effective, lic_up = "GPL (detected)", "GPL (DETECTED)"
                note = "Metadata UNKNOWN - detected as GPL (forbidden, verify)"
            else:
                note = "No license detectable - verify manually"

        allow_key = next(
            (k for k in ALLOWLIST if k[0].lower() == name.lower() and k[1] == version),
            None,
        )
        is_forbidden = any(bad in lic_up for bad in FORBIDDEN)

        if is_forbidden and allow_key:
            effective = f"{lic} (allowlisted)"
            allowlisted.append((name, version, lic, ALLOWLIST[allow_key]))
        elif is_forbidden:
            violations.append((name, version, lic))

        if was_unknown:
            unknowns.append((name, version, effective, note))

        rows.append((name, version, effective, is_forbidden and not allow_key))
    return rows, violations, allowlisted, unknowns


def write_license_files(
    pkgs: list[dict], rows, violations, allowlisted, unknowns
) -> list:
    log(f">> 3/4: Writing per-package license files into '{OUTPUT_DIR}/'...")
    if os.path.isdir(OUTPUT_DIR):
        for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
            os.remove(f)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    by_name = {p["Name"].lower(): p for p in pkgs}
    file_rows, missing = [], []
    for name, version, effective, _ in rows:
        p = by_name[name.lower()]
        lic = p.get("License") or "UNKNOWN"
        author = p.get("Author") or ""
        url = p.get("URL") or ""
        text = p.get("LicenseText") or ""

        fname = f"{safe(name)}-{safe(version)}.LICENSE.txt"
        header = (
            f"Package: {name}\nVersion: {version}\nLicense: {lic}\n"
            f"Author:  {author}\nURL:     {url}\n" + "-" * 72 + "\n\n"
        )
        if not text or text.strip().upper() in ("UNKNOWN", ""):
            text = "[No license text found in package metadata - verify manually!]\n"
            missing.append(f"{name} {version}")
        with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as lf:
            lf.write(header + text + "\n")
        file_rows.append((name, version, effective, fname))

    write_index(file_rows)
    write_report(file_rows, missing, violations, allowlisted, unknowns)
    return missing


def write_index(file_rows) -> None:
    """Pure package index inside OUTPUT_DIR."""
    today = datetime.date.today().isoformat()
    with open(os.path.join(OUTPUT_DIR, "INDEX.md"), "w", encoding="utf-8") as idx:
        idx.write("# Third-Party Licenses - Index\n\n")
        idx.write(f"Project license (outbound): {PROJECT_OUTBOUND_LICENSE}\n\n")
        idx.write(f"Generated: {today}\n")
        idx.write(f"Total packages: {len(file_rows)}\n\n")
        idx.write("| Package | Version | License | File |\n")
        idx.write("|---------|---------|---------|------|\n")
        for name, version, effective, fname in file_rows:
            idx.write(f"| {name} | {version} | {effective} | [{fname}](./{fname}) |\n")


def write_report(file_rows, missing, violations, allowlisted, unknowns) -> None:
    """Report with status / actions / breakdown next to the script."""
    today = datetime.date.today().isoformat()
    action_needed = bool(violations or missing or unknowns)
    status = "⚠️ ACTION REQUIRED" if action_needed else "✅ CLEAN"

    with open(REPORT_FILE, "w", encoding="utf-8") as rep:
        rep.write("# Third-Party License Report\n\n")

        # --- Summary block ---
        rep.write(f"**Status:** {status}  \n")
        rep.write(f"**Project license (outbound):** {PROJECT_OUTBOUND_LICENSE}  \n")
        rep.write(f"**Generated:** {today}  \n")
        rep.write(f"**Total packages:** {len(file_rows)}  \n")
        rep.write(
            f"**Details:** [`{OUTPUT_DIR}/INDEX.md`](./{OUTPUT_DIR}/INDEX.md)\n\n"
        )

        # --- Action blocks (prominent) ---
        if violations:
            rep.write("## ⚠️ Forbidden licenses — resolve before release\n\n")
            rep.write("| Package | Version | License |\n")
            rep.write("|---------|---------|---------|\n")
            for name, version, lic in violations:
                rep.write(f"| {name} | {version} | **{lic}** |\n")
            rep.write(
                "\n> Fix the dependency or add a verified entry to `ALLOWLIST`.\n\n"
            )

        if missing:
            rep.write("## ⚠️ Missing license text — verify manually\n\n")
            for m in missing:
                rep.write(f"- {m}\n")
            rep.write("\n")

        # --- UNKNOWN licenses (metadata) ---
        if unknowns:
            rep.write("## ℹ️ UNKNOWN licenses (metadata) — note\n\n")
            rep.write(
                "These packages reported `UNKNOWN` in their metadata. "
                "Where a detection was possible, the effective license "
                "is shown — please still verify manually.\n\n"
            )
            rep.write("| Package | Version | Effective | Note |\n")
            rep.write("|---------|---------|-----------|------|\n")
            for name, version, effective, note in unknowns:
                rep.write(f"| {name} | {version} | {effective} | {note} |\n")
            rep.write("\n")

        # --- License breakdown ---
        breakdown = Counter(r[2].split(" (")[0] for r in file_rows)
        rep.write("## License breakdown\n\n")
        rep.write("| License | Count |\n|---------|-------|\n")
        for lic, n in sorted(breakdown.items(), key=lambda x: (-x[1], x[0])):
            rep.write(f"| {lic} | {n} |\n")
        rep.write("\n")

        if allowlisted:
            rep.write("## Allowlisted (verified) exceptions\n\n")
            for name, version, lic, reason in allowlisted:
                rep.write(f"- **{name} {version}** ({lic}) — {reason}\n")
            rep.write("\n")


def collect_notices() -> None:
    log(">> 4/4: Collecting NOTICE files...")

    paths = set(site.getsitepackages() if hasattr(site, "getsitepackages") else [])
    try:
        paths.add(site.getusersitepackages())
    except Exception:
        pass
    paths.add(sysconfig.get_paths()["purelib"])

    found, seen = [], set()
    for sp in paths:
        if not sp or not os.path.isdir(sp):
            continue
        patterns = [
            os.path.join(sp, "*.dist-info", "NOTICE*"),
            os.path.join(sp, "*.dist-info", "licenses", "NOTICE*"),
        ]
        for pat in patterns:
            for f in glob.glob(pat):
                real = os.path.realpath(f)
                if real in seen:
                    continue
                seen.add(real)
                distinfo = f.split(os.sep)
                pkgver = next(
                    (
                        d[: -len(".dist-info")]
                        for d in distinfo
                        if d.endswith(".dist-info")
                    ),
                    "unknown",
                )
                found.append((pkgver, f))

    for pkgver, f in sorted(found):
        dest = os.path.join(OUTPUT_DIR, f"{pkgver}.NOTICE.txt")
        with (
            open(f, "r", encoding="utf-8", errors="replace") as src,
            open(dest, "w", encoding="utf-8") as out,
        ):
            out.write(f"NOTICE for {pkgver}\n" + "-" * 72 + "\n\n")
            out.write(src.read())

    with open(NOTICE_AGG, "w", encoding="utf-8") as agg:
        agg.write("NOTICE\n")
        agg.write("=" * 72 + "\n\n")
        agg.write(f"{PROJECT_NAME}\n")
        agg.write(f"{PROJECT_COPYRIGHT}\n\n")
        agg.write("This product includes third-party software.\n")
        agg.write("The following NOTICE contents are reproduced as required by\n")
        agg.write("Section 4(d) of the Apache License 2.0.\n\n")
        agg.write(f"Generated: {datetime.date.today().isoformat()}\n\n")
        if not found:
            agg.write("(No NOTICE files were found among installed packages.)\n")
        for pkgver, f in sorted(found):
            agg.write("=" * 72 + "\n")
            agg.write(f"NOTICE for {pkgver}\n")
            agg.write("=" * 72 + "\n\n")
            with open(f, "r", encoding="utf-8", errors="replace") as src:
                agg.write(src.read().rstrip() + "\n\n")

    log(f"   Found {len(found)} NOTICE file(s).")
    log(f"   Aggregated project NOTICE written to: {NOTICE_AGG}")


def main() -> None:
    install_pip_licenses()
    pkgs = collect_licenses()
    rows, violations, allowlisted, unknowns = analyze(pkgs)
    missing = write_license_files(pkgs, rows, violations, allowlisted, unknowns)
    collect_notices()

    log(">> Done.")
    log(f"   Report: {REPORT_FILE}")
    log(f"   Index:  {OUTPUT_DIR}/INDEX.md")
    log(f"   Aggregated NOTICE: {NOTICE_AGG}")

    # Prominent console summary (script does NOT exit on findings)
    if violations:
        log("\n" + "!" * 60)
        log(f"!! {len(violations)} FORBIDDEN LICENSE(S) FOUND — see report top:")
        for name, version, lic in violations:
            log(f"!!   - {name} {version} -> {lic}")
        log("!" * 60)
    if missing:
        log(f"\n!! {len(missing)} package(s) missing license text (see report).")
    if unknowns:
        log(
            f"\n!! {len(unknowns)} package(s) with UNKNOWN metadata license "
            f"(see report section 'UNKNOWN licenses')."
        )
        for name, version, effective, note in unknowns:
            log(f"!!   - {name} {version} -> {effective} ({note})")
    if not violations and not missing and not unknowns:
        log("\n✅ All clean — no forbidden licenses, no missing texts, no UNKNOWNs.")


if __name__ == "__main__":
    main()
