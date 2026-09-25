"""
plot_route_molecules.py
────────────────────────
Renders each molecule in a route CSV (role,smiles columns, e.g.
figures/routes_batch_01/example_18_row11475.csv) as its own individual
figure -- one PNG (always) per row, drawn on a fixed --size x --size pixel
canvas shared by every molecule in the file, optionally also PDF and/or SVG.

By default RDKit auto-fits each 2D depiction to the fixed canvas, so every
image comes out at exactly the same pixel dimensions regardless of the
molecule's own size -- but a tiny building block gets blown up and a large
PROTAC gets shrunk, so ring sizes and bond lengths differ molecule to
molecule. Passing --bond-length switches to a fixed chemical scale instead
(RDKit's fixedBondLength draw option): every molecule is drawn with the same
pixel bond length -- same ring size, same bond length everywhere -- and
centered, untrimmed, on the same --size x --size canvas, so both the output
dimensions *and* the chemical scale are identical across molecules. The
tradeoff is that --size must be chosen generously enough to fit the largest
molecule at that scale, since anything that doesn't fit is silently clipped
by RDKit; the script checks each render for content touching the canvas edge
and warns when that happens, as a sign --size is too small for --bond-length.

Usage
-----
    python scripts/dataset/plot_route_molecules.py \\
        figures/routes_batch_01/example_18_row11475.csv \\
        --output-dir figures/routes_batch_01/example_18_row11475_mols \\
        --size 400 --pdf --svg

    # Same chemical scale (bond length/ring size) across every molecule,
    # instead of same-canvas auto-fit -- needs a larger --size to fit the
    # biggest molecule without clipping:
    python scripts/dataset/plot_route_molecules.py \\
        figures/routes_batch_01/example_18_row11475.csv \\
        --size 900 --bond-length 30

Arguments:
    input_csv      Input CSV path with role,smiles columns.
    --output-dir   Output directory for the per-molecule images (default:
                   "<input_csv stem>_mols" next to the input CSV).
    --role-col     Role column name (default: "role").
    --smiles-col   SMILES column name (default: "smiles").
    --size         Square canvas size in pixels, shared by every molecule
                   (default: 400).
    --bond-length  Fixed bond length in pixels. When set, every molecule is
                   drawn at this same chemical scale instead of being
                   auto-fit to the canvas (default: unset, i.e. auto-fit).
    --pdf          Also save each molecule as PDF (wraps the same PNG raster
                   RDKit has no native vector-PDF drawer).
    --svg          Also save each molecule as SVG (RDKit's native vector SVG
                   drawer, independent of the PNG raster).
    --dpi          DPI recorded on the PDF page (default: 300); this only
                   sets the PDF's physical page size, not the pixel
                   dimensions.
"""

import argparse
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from retrotac.chem_utils import smiles_to_mol

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=str, help="Input CSV path with role,smiles columns.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the per-molecule images (default: '<input_csv stem>_mols').",
    )
    p.add_argument("--role-col", type=str, default="role", help="Role column name (default: 'role').")
    p.add_argument("--smiles-col", type=str, default="smiles", help="SMILES column name (default: 'smiles').")
    p.add_argument(
        "--size", type=int, default=400, help="Square canvas size in pixels, shared by every molecule (default: 400)."
    )
    p.add_argument(
        "--bond-length",
        type=int,
        default=None,
        help="Fixed bond length in pixels; when set, every molecule is drawn at this same chemical scale "
        "instead of being auto-fit to the canvas (default: unset, i.e. auto-fit).",
    )
    p.add_argument("--pdf", action="store_true", help="Also save each molecule as PDF.")
    p.add_argument("--svg", action="store_true", help="Also save each molecule as SVG.")
    p.add_argument("--dpi", type=int, default=300, help="DPI recorded on the PDF page (default: 300).")
    return p.parse_args()


def render_png_bytes(mol, size: int, bond_length: Optional[int] = None) -> bytes:
    """Draw *mol* as PNG bytes on a fixed size x size canvas.

    Args:
        mol: A valid (non-None) RDKit molecule.
        size: Canvas width and height, in pixels.
        bond_length: If given, fix the drawn bond length to this many pixels
            (same chemical scale for every molecule) instead of auto-fitting
            the depiction to the canvas.

    Returns:
        Raw PNG bytes.
    """
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    if bond_length is not None:
        drawer.drawOptions().fixedBondLength = bond_length
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_svg_text(mol, size: int, bond_length: Optional[int] = None) -> str:
    """Draw *mol* as SVG markup on a fixed size x size canvas.

    Args:
        mol: A valid (non-None) RDKit molecule.
        size: Canvas width and height, in pixels.
        bond_length: If given, fix the drawn bond length to this many pixels
            (same chemical scale for every molecule) instead of auto-fitting
            the depiction to the canvas.

    Returns:
        SVG markup text.
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    if bond_length is not None:
        drawer.drawOptions().fixedBondLength = bond_length
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def touches_canvas_edge(png_bytes: bytes, margin: int = 1) -> bool:
    """Check whether drawn (non-white) content reaches the canvas border.

    Used as a clipping heuristic for --bond-length renders: content flush
    against the edge means the molecule didn't fully fit in --size at that
    fixed scale and got cut off by RDKit.

    Args:
        png_bytes: Raw PNG bytes from :func:`render_png_bytes`.
        margin: Border thickness, in pixels, to inspect.

    Returns:
        True if any border pixel is not (near-)white.
    """
    arr = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    border = np.concatenate(
        [arr[:margin].reshape(-1, 3), arr[-margin:].reshape(-1, 3), arr[:, :margin].reshape(-1, 3), arr[:, -margin:].reshape(-1, 3)]
    )
    return bool(np.any(border < 250))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent / f"{input_path.stem}_mols"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    for col in (args.role_col, args.smiles_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.input_csv}. Available: {df.columns.tolist()}")

    print(f"Loaded {len(df)} rows from {args.input_csv}")
    n_written = 0
    for idx, row in df.iterrows():
        smi = row[args.smiles_col]
        role = row[args.role_col]
        mol = smiles_to_mol(smi)
        if mol is None:
            print(f"Skipping row {idx} ({role}): unparseable SMILES '{smi}'")
            continue

        stem = f"{idx:02d}_{role}"
        png_bytes = render_png_bytes(mol, args.size, args.bond_length)
        if args.bond_length is not None and touches_canvas_edge(png_bytes):
            print(
                f"Warning: row {idx} ({role}) touches the canvas edge at --size {args.size} with "
                f"--bond-length {args.bond_length}; it may be clipped -- consider increasing --size."
            )
        (out_dir / f"{stem}.png").write_bytes(png_bytes)

        if args.svg:
            (out_dir / f"{stem}.svg").write_text(render_svg_text(mol, args.size, args.bond_length))

        if args.pdf:
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            img.save(out_dir / f"{stem}.pdf", "PDF", resolution=args.dpi)

        n_written += 1

    print(f"Wrote {n_written} molecule image(s) ({args.size}x{args.size}px) → {out_dir}")


if __name__ == "__main__":
    main()
