"""Write the frozen carrier velocity field 0/U with the matched ramp profile.

The carrier velocity is uniform / linear-ramp / uniform in x, identical to the
lptlib verification field: u = 12 m/s for x <= 1e-4, linear down to 4 m/s by
x = 2e-4, then 4 m/s downstream. v = w = 0.

It reads the cell-centre x-coordinates from 0/Cx (produced by
``postProcess -func writeCellCentres``) so the internalField order matches the
mesh exactly, then writes 0/U as a nonuniform field. This avoids depending on
setExprFields or swak4Foam and works on any OpenFOAM build.

Usage (from the case directory, after blockMesh and writeCellCentres):
    python set_U.py
"""

import re
from pathlib import Path

U1, U2, X1, X2 = 12.0, 4.0, 1e-4, 2e-4

CASE = Path(__file__).resolve().parent


def ramp(x):
    if x <= X1:
        return U1
    if x >= X2:
        return U2
    return U1 + (U2 - U1) * (x - X1) / (X2 - X1)


def read_scalar_internalfield(path):
    """Parse internalField nonuniform List<scalar> N ( v0 v1 ... ) from an OF file."""
    text = Path(path).read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\(", text)
    if not m:
        # Uniform field fallback (single value): rare for Cx but handle it.
        mu = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", text)
        if mu:
            return [float(mu.group(1))]
        raise RuntimeError(f"Could not parse internalField in {path}")
    n = int(m.group(1))
    start = m.end()
    depth = 1
    i = start
    while depth and i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    body = text[start:i - 1]
    vals = [float(t) for t in body.split()]
    if len(vals) != n:
        raise RuntimeError(f"Expected {n} values, parsed {len(vals)} in {path}")
    return vals


def main():
    # writeCellCentres writes into the latest time dir; try 0 then constant.
    for cand in [CASE / "0" / "Cx", CASE / "0.000000" / "Cx", CASE / "constant" / "Cx"]:
        if cand.exists():
            cx_path = cand
            break
    else:
        raise SystemExit("0/Cx not found. Run: postProcess -func writeCellCentres")

    xs = read_scalar_internalfield(cx_path)
    us = [ramp(x) for x in xs]

    out = CASE / "0" / "U"
    header = (
        "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n\n"
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   nonuniform List<vector>\n"
        f"{len(us)}\n(\n"
    )
    body = "\n".join(f"({u} 0 0)" for u in us)
    boundary = (
        "\n)\n;\n\n"
        "boundaryField\n{\n"
        "    inlet     { type fixedValue; value uniform (12 0 0); }\n"
        "    outlet    { type zeroGradient; }\n"
        "    walls     { type zeroGradient; }\n"
        "    frontBack { type empty; }\n"
        "}\n"
    )
    out.write_text(header + body + boundary)
    print(f"Wrote {out} with {len(us)} cells; "
          f"u range {min(us):.2f}..{max(us):.2f} m/s")


if __name__ == "__main__":
    main()
