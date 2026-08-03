#!/usr/bin/env python3
"""Print the PE import table of one or more binaries (CI diagnostic helper).

Usage:
    python scripts/dump_pe_imports.py <binary> [<binary> ...]

Each binary's import-table DLLs are printed as a comma-separated list. This is
used by the Windows CUDA CI job to see exactly which DLLs the loader must
resolve when the CLI starts.
"""

import sys

try:
    import pefile
except ImportError:
    print("pefile is not installed; run: python -m pip install pefile")
    sys.exit(0)


def main(argv):
    if not argv:
        print(__doc__)
        return 0
    for path in argv:
        try:
            pe = pefile.PE(path, fast_load=True)
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
            )
            deps = [e.dll.decode() for e in pe.DIRECTORY_ENTRY_IMPORT]
            print(f"{path} -> {', '.join(deps)}")
        except Exception as exc:
            print(f"{path} -> ERROR: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
