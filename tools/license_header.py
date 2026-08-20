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
"""Insert or replace a license header in source files.

Supports both line-style comments (e.g. ``# ``, ``// ``) and block-style
comments (e.g. ``<!-- ... -->``, ``/* ... */``). Files can be selected by
extension (``-e .py``) or by filename/glob (``-n Makefile``). Existing
headers at the top of a file are replaced; a shebang line (``#!...``) is
preserved.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


# ─── Comment styles ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class CommentStyle:
    """How to render a comment header for a given file type.

    Line style: each line is prefixed with ``line_prefix``.
        Example: ``CommentStyle(line_prefix="# ")``
            # Copyright ...
            # Licensed ...

    Block style: header is wrapped between ``block_open`` and ``block_close``.
        Example: ``CommentStyle(block_open="<!--", block_close="-->")``
            <!--
            Copyright ...
            Licensed ...
            -->

    Combining both yields banner-style comments:
        ``CommentStyle(block_open="/*", line_prefix=" * ", block_close=" */")``
            /*
             * Copyright ...
             * Licensed ...
             */
    """

    line_prefix: str = ""
    block_open: str = ""
    block_close: str = ""

    @property
    def is_block(self) -> bool:
        return bool(self.block_open or self.block_close)

    def __post_init__(self) -> None:
        if bool(self.block_open) != bool(self.block_close):
            raise ValueError("Block comments need both block_open and block_close.")
        if not self.is_block and not self.line_prefix:
            raise ValueError("Either line_prefix or block markers must be set.")


DEFAULT_COMMENT_STYLES: dict[str, CommentStyle] = {
    # ─ by extension: line style ─────────
    ".py": CommentStyle(line_prefix="# "),
    ".sh": CommentStyle(line_prefix="# "),
    ".fish": CommentStyle(line_prefix="# "),
    ".rb": CommentStyle(line_prefix="# "),
    ".yml": CommentStyle(line_prefix="# "),
    ".yaml": CommentStyle(line_prefix="# "),
    ".toml": CommentStyle(line_prefix="# "),
    ".cfg": CommentStyle(line_prefix="# "),
    ".ini": CommentStyle(line_prefix="# "),
    ".c": CommentStyle(line_prefix="// "),
    ".h": CommentStyle(line_prefix="// "),
    ".cpp": CommentStyle(line_prefix="// "),
    ".hpp": CommentStyle(line_prefix="// "),
    ".java": CommentStyle(line_prefix="// "),
    ".js": CommentStyle(line_prefix="// "),
    ".ts": CommentStyle(line_prefix="// "),
    ".go": CommentStyle(line_prefix="// "),
    ".rs": CommentStyle(line_prefix="// "),
    ".sql": CommentStyle(line_prefix="-- "),
    ".lua": CommentStyle(line_prefix="-- "),
    ".hs": CommentStyle(line_prefix="-- "),
    # ─ by extension: block style ────────
    ".md": CommentStyle(block_open="<!--", block_close="-->"),
    ".html": CommentStyle(block_open="<!--", block_close="-->"),
    ".htm": CommentStyle(block_open="<!--", block_close="-->"),
    ".xml": CommentStyle(block_open="<!--", block_close="-->"),
    ".svg": CommentStyle(block_open="<!--", block_close="-->"),
    ".css": CommentStyle(block_open="/*", block_close="*/"),
    ".scss": CommentStyle(block_open="/*", block_close="*/"),
    # ─ by filename (no extension) ───────
    "Makefile": CommentStyle(line_prefix="# "),
    "makefile": CommentStyle(line_prefix="# "),
    "GNUmakefile": CommentStyle(line_prefix="# "),
    "Dockerfile": CommentStyle(line_prefix="# "),
    "Containerfile": CommentStyle(line_prefix="# "),
    "Jenkinsfile": CommentStyle(line_prefix="// "),  # Groovy
    "CMakeLists.txt": CommentStyle(line_prefix="# "),
    ".gitignore": CommentStyle(line_prefix="# "),
    ".gitattributes": CommentStyle(line_prefix="# "),
    ".dockerignore": CommentStyle(line_prefix="# "),
    ".editorconfig": CommentStyle(line_prefix="# "),
    ".env": CommentStyle(line_prefix="# "),
}

DEFAULT_EXCLUDES: set[str] = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".tox",
    "build",
    "dist",
}


# ─── Core logic ───────────────────────────────────────────────────────────
def build_header(license_path: Path, style: CommentStyle, encoding: str) -> str:
    """Return the license text wrapped in ``style`` + one blank line."""
    text = license_path.read_text(encoding=encoding)
    lines = text.splitlines()

    if style.is_block:
        out = [style.block_open]
        out.extend(f"{style.line_prefix}{ln}".rstrip() for ln in lines)
        out.append(style.block_close)
    else:
        out = [f"{style.line_prefix}{ln}".rstrip() for ln in lines]

    return "\n".join(out) + "\n"


def strip_header(source: str, style: CommentStyle) -> tuple[str, str]:
    """Split ``source`` into (shebang, body_without_leading_header)."""
    lines = source.splitlines(keepends=True)
    shebang = ""

    if lines and lines[0].startswith("#!"):
        shebang, lines = lines[0], lines[1:]

    if style.is_block:
        # Skip leading blank lines
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1

        # If it starts with block_open, consume up to and including block_close
        if i < len(lines) and lines[i].lstrip().startswith(style.block_open):
            j = i
            found = False
            while j < len(lines):
                if style.block_close in lines[j]:
                    j += 1
                    found = True
                    break
                j += 1

            if not found:
                # Malformed: no closing marker -> leave file untouched
                return shebang, "".join(lines)

            # Consume trailing blank lines after the block
            while j < len(lines) and not lines[j].strip():
                j += 1
            return shebang, "".join(lines[j:])

        return shebang, "".join(lines)

    # Line style
    marker = style.line_prefix.strip()
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith(marker):
            i += 1
        else:
            break
    return shebang, "".join(lines[i:])


def update_file(
    path: Path,
    header: str,
    style: CommentStyle,
    encoding: str,
    dry_run: bool,
) -> bool:
    """Rewrite ``path`` so it starts with ``header``. Return True if changed."""
    original = path.read_text(encoding=encoding)
    shebang, body = strip_header(original, style)
    new_content = shebang + header + body

    if new_content == original:
        return False

    if dry_run:
        print(f"would update: {path}")
    else:
        path.write_text(new_content, encoding=encoding)
        print(f"updated:      {path}")
    return True


# ─── File discovery ───────────────────────────────────────────────────────
def is_excluded(path: Path, root: Path, excludes: set[str]) -> bool:
    """Return True if any parent directory of ``path`` is in ``excludes``."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part in excludes for part in rel.parts)


def collect_files(
    root: Path,
    pattern: str,
    excludes: set[str],
    recursive: bool,
) -> list[Path]:
    """Return files under ``root`` matching ``pattern`` (a glob)."""
    finder = root.rglob if recursive else root.glob
    return sorted(
        p for p in finder(pattern) if p.is_file() and not is_excluded(p, root, excludes)
    )


# ─── CLI ──────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert or replace a license header in source files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                    # .py files in cwd\n"
            "  %(prog)s ./my-repo -e .c                    # C files, auto '// '\n"
            "  %(prog)s . -e .md                           # Markdown, auto '<!-- -->'\n"
            "  %(prog)s . -n Makefile                      # Makefiles, auto '# '\n"
            "  %(prog)s . -n 'Dockerfile*'                 # glob works too\n"
            "  %(prog)s . -e .rs -c '// '                  # explicit line prefix\n"
            "  %(prog)s . -e .tpl --block-open '{#' --block-close '#}'\n"
            "  %(prog)s . -n custom.conf -c '; '           # custom file + prefix\n"
            "  %(prog)s . --dry-run                        # preview only\n"
            "  %(prog)s . --exclude vendor                 # skip additional folder\n"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="root directory to process (default: cwd)",
    )

    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-e",
        "--ext",
        default=None,
        help="file extension to target (e.g. .py, .md)",
    )
    target.add_argument(
        "-n",
        "--name",
        default=None,
        metavar="PATTERN",
        help="filename or glob to target (e.g. Makefile, Dockerfile*)",
    )

    parser.add_argument(
        "-c",
        "--comment",
        default=None,
        help="line comment prefix (e.g. '# ', '// ')",
    )
    parser.add_argument(
        "--block-open",
        default=None,
        help="opening marker for block-style comments (e.g. '<!--')",
    )
    parser.add_argument(
        "--block-close",
        default=None,
        help="closing marker for block-style comments (e.g. '-->')",
    )
    parser.add_argument(
        "-l",
        "--license",
        type=Path,
        default=None,
        help="path to license file (default: <path>/LICENSE)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="file encoding (default: utf-8)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="only process files directly under <path>",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="additional directory name to skip (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report changes without writing files",
    )
    return parser.parse_args(argv)


def resolve_style(args: argparse.Namespace, style_key: str) -> CommentStyle | None:
    """Build a CommentStyle from CLI args, falling back to the defaults."""
    has_block = bool(args.block_open) or bool(args.block_close)

    if has_block:
        if not (args.block_open and args.block_close):
            print(
                "error: --block-open and --block-close must both be provided.",
                file=sys.stderr,
            )
            return None
        return CommentStyle(
            line_prefix=args.comment or "",
            block_open=args.block_open,
            block_close=args.block_close,
        )

    if args.comment:
        return CommentStyle(line_prefix=args.comment)

    return DEFAULT_COMMENT_STYLES.get(style_key)


def describe_style(style: CommentStyle) -> str:
    if style.is_block:
        inner = f", inner {style.line_prefix!r}" if style.line_prefix else ""
        return f"block {style.block_open!r} … {style.block_close!r}{inner}"
    return f"line {style.line_prefix!r}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root: Path = args.path.resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    # Determine what to match and which style to use
    if args.name:
        pattern = args.name
        style_key = args.name
        label = args.name
    else:
        ext = args.ext or ".py"
        if not ext.startswith("."):
            ext = f".{ext}"
        pattern = f"*{ext}"
        style_key = ext
        label = pattern

    style = resolve_style(args, style_key)
    if style is None:
        print(
            f"error: no default comment style known for '{style_key}'. "
            f"Provide --comment or --block-open/--block-close.",
            file=sys.stderr,
        )
        return 1

    license_file = (args.license or (root / "LICENSE")).resolve()
    if not license_file.is_file():
        print(f"error: license file not found: {license_file}", file=sys.stderr)
        return 1

    excludes = DEFAULT_EXCLUDES | set(args.exclude)
    header = build_header(license_file, style, args.encoding)
    files = collect_files(root, pattern, excludes, not args.no_recursive)

    scope = "recursive" if not args.no_recursive else "top level only"
    print(f"root:     {root}")
    print(f"license:  {license_file}")
    print(f"pattern:  {label}  ({scope})")
    print(f"style:    {describe_style(style)}")
    print(f"files:    {len(files)}")
    print("─── header preview ───")
    print(header, end="")
    print("──────────────────────")

    changed = sum(
        update_file(f, header, style, args.encoding, args.dry_run) for f in files
    )

    # license_header.py should run in an automated test workflow returning 1 means
    # the user has forgotten to add a license header.
    exit_code = changed

    action = "would change" if args.dry_run else "changed"
    print(f"Done. {action} {changed} of {len(files)} file(s).")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
