"""Python's own reading of a study folder, to check the MATLAB tables against.

    python study_sums.py FOLDER

For every CSV the study writes, one line per column:

    file,column,rows,filled,sum

``filled`` counts non-empty cells and ``sum`` adds the numeric ones (empty when
the column is text), so a reader that drops rows, shifts a column or turns a
blank into zero shows up as a difference.
"""

import csv
import math
import sys
from pathlib import Path


def main(folder):
    for path in sorted(Path(folder).glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        columns = list(rows[0]) if rows else []
        for c in columns:
            cells = [r[c] for r in rows if r[c] != ""]
            try:
                total = math.fsum(float(v) for v in cells)
                text = repr(total)
            except ValueError:
                text = ""
            print(f"{path.name},{c},{len(rows)},{len(cells)},{text}")


if __name__ == "__main__":
    main(sys.argv[1])
