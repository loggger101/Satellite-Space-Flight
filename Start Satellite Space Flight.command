#!/bin/bash
# Double-click in Finder to start Satellite Space Flight (macOS; on Linux run
# "python3 start.py"). The first start installs what it needs; see start.py.
cd "$(dirname "$0")" || exit 1

# Prefer a Python that already has numpy and pygame-ce, then any Python 3.10+
# (start.py then sets one up).
ready='import sys, numpy, pygame; sys.exit(sys.version_info < (3, 10) or not getattr(pygame, "IS_CE", 0))'
recent='import sys; sys.exit(sys.version_info < (3, 10))'
for check in "$ready" "$recent"; do
    for py in python3 python; do
        if command -v "$py" >/dev/null 2>&1 && "$py" -c "$check" >/dev/null 2>&1; then
            "$py" start.py "$@"
            status=$?
            if [ "$status" -ne 0 ]; then
                read -r -p "Press Enter to close this window..."
            fi
            exit "$status"
        fi
    done
done

echo
echo "  Satellite Space Flight needs Python 3.10 or newer, and none was found."
echo
echo "   1. Install Python from https://www.python.org/downloads/"
echo "   2. Double-click this file again."
echo
read -r -p "Press Enter to close this window..."
exit 1
