"""Tests for sample_route_tree.py.

Run inside a container that has rdkit/pandas/matplotlib (no .venv exists on
disk yet in this checkout):

    apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \\
        python -m unittest scripts/dataset/test_sample_route_tree.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_route_tree import (  # noqa: E402
    build_children_map,
    compute_layout,
    filter_by_shape,
    format_target_label,
    load_resolved_candidates,
    main,
    parse_route_cell,
    pick_examples,
    plot_route_tree,
    render_molecule_image,
    trim_whitespace,
    write_route_csv,
)
from route_scores.route_tree_score import parse_route  # noqa: E402

# T -> [A, BB]; A -> [B]; B -> [C, D] -- same convergent-then-linear-then-fork
# shape as the reference figure (one immediate leaf, one branch that keeps
# disconnecting for two more steps before splitting into two more leaves).
_SHAPED_ROUTE = {
    1: [["T => A.BB", ""]],
    2: [["A => B", ""]],
    3: [["B => C.D", ""]],
}

# Same shape as _SHAPED_ROUTE but with real, RDKit-parseable SMILES, for the
# rendering smoke test (chemically nonsensical as reactions -- only the
# string structure and parseability matter here).
_DRAWABLE_ROUTE = {
    1: [["CCOC(=O)c1ccccc1 => CCO.OC(=O)c1ccccc1N", ""]],
    2: [["CCO => CCBr", ""]],
    3: [["CCBr => CC.Br", ""]],
}


class ParseRouteCellTests(unittest.TestCase):
    def test_parses_a_python_literal_string(self) -> None:
        self.assertEqual(parse_route_cell("{1: [['T => A.B', '']]}"), {1: [["T => A.B", ""]]})

    def test_passes_through_an_already_parsed_dict(self) -> None:
        route = {1: [["T => A.B", ""]]}
        self.assertEqual(parse_route_cell(route), route)

    def test_returns_empty_dict_for_a_blank_or_nan_cell(self) -> None:
        self.assertEqual(parse_route_cell(float("nan")), {})
        self.assertEqual(parse_route_cell(""), {})

    def test_returns_empty_dict_for_unparseable_text(self) -> None:
        self.assertEqual(parse_route_cell("not a dict at all {"), {})


class LoadResolvedCandidatesTests(unittest.TestCase):
    def test_skips_unresolved_and_empty_routes(self) -> None:
        df = pd.DataFrame(
            {
                "resolved": [True, False, True, "True"],
                "route": [
                    "{1: [['T1 => A.B', '']]}",
                    "{1: [['T2 => A.B', '']]}",
                    "{}",
                    "{1: [['T4 => A.B', '']]}",
                ],
            }
        )

        candidates = load_resolved_candidates(df, route_col="route", resolved_col="resolved")

        targets = sorted(tree.target for _idx, tree in candidates)
        self.assertEqual(targets, ["T1", "T4"])

    def test_skips_a_blank_route_cell_instead_of_raising(self) -> None:
        # pandas reads a blank CSV cell as NaN, a float -- not the string
        # "{}" the "purchasable as-is" rows actually use.
        df = pd.DataFrame({"resolved": [True], "route": [float("nan")]})

        candidates = load_resolved_candidates(df)

        self.assertEqual(candidates, [])


class FilterByShapeTests(unittest.TestCase):
    def test_keeps_only_trees_within_step_and_leaf_bounds(self) -> None:
        df = pd.DataFrame(
            {
                "resolved": [True, True, True],
                "route": [
                    "{1: [['T1 => A', '']]}",  # 1 step, 1 leaf -> too small
                    "{1: [['T2 => A.B', '']], 2: [['A => C.D', '']]}",  # 2 steps, 3 leaves
                    "{1: [['T3 => A', '']], 2: [['A => B', '']], 3: [['B => C', '']], "
                    "4: [['C => D', '']], 5: [['D => E', '']]}",  # 5 steps, 1 leaf -> too many steps
                ],
            }
        )
        candidates = load_resolved_candidates(df)

        kept = filter_by_shape(candidates, min_steps=2, max_steps=4, min_leaves=2, max_leaves=4)

        targets = [tree.target for _idx, tree in kept]
        self.assertEqual(targets, ["T2"])


class PickExamplesTests(unittest.TestCase):
    def test_same_seed_is_reproducible(self) -> None:
        candidates = [(i, i) for i in range(10)]

        first = pick_examples(candidates, seed=0, n_examples=3)
        second = pick_examples(candidates, seed=0, n_examples=3)

        self.assertEqual(first, second)

    def test_n_examples_larger_than_pool_returns_whole_pool(self) -> None:
        candidates = [(i, i) for i in range(3)]

        picked = pick_examples(candidates, seed=0, n_examples=100)

        self.assertEqual(len(picked), 3)
        self.assertCountEqual(picked, candidates)


class BuildChildrenMapTests(unittest.TestCase):
    def test_maps_each_product_to_its_reactants(self) -> None:
        tree = parse_route(_SHAPED_ROUTE)

        children_of = build_children_map(tree)

        self.assertEqual(
            children_of, {"T": ["A", "BB"], "A": ["B"], "B": ["C", "D"]}
        )


class ComputeLayoutTests(unittest.TestCase):
    def test_x_is_bfs_depth_from_target(self) -> None:
        tree = parse_route(_SHAPED_ROUTE)
        children_of = build_children_map(tree)

        positions = compute_layout(tree.target, children_of)

        xs = {node: xy[0] for node, xy in positions.items()}
        self.assertEqual(xs, {"T": 0, "A": 1, "BB": 1, "B": 2, "C": 3, "D": 3})

    def test_internal_node_y_is_mean_of_children_y(self) -> None:
        tree = parse_route(_SHAPED_ROUTE)
        children_of = build_children_map(tree)

        positions = compute_layout(tree.target, children_of)
        y = {node: xy[1] for node, xy in positions.items()}

        # A chains to a single child (B): a 1-child internal node keeps that
        # child's y exactly, so the connector is a straight horizontal line.
        self.assertEqual(y["A"], y["B"])
        self.assertAlmostEqual(y["B"], (y["C"] + y["D"]) / 2)
        self.assertAlmostEqual(y["T"], (y["A"] + y["BB"]) / 2)
        # The three leaves must each land on a distinct row.
        self.assertEqual(len({y["BB"], y["C"], y["D"]}), 3)


class WriteRouteCsvTests(unittest.TestCase):
    def test_writes_target_then_building_blocks_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "example.csv"

            write_route_csv(out_path, target="TSMI", leaves=["L1", "L2"])

            written = pd.read_csv(out_path)
            self.assertEqual(written["role"].tolist(), ["target", "building_block", "building_block"])
            self.assertEqual(written["smiles"].tolist(), ["TSMI", "L1", "L2"])

    def test_creates_missing_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "nested" / "example.csv"

            write_route_csv(out_path, target="TSMI", leaves=[])

            self.assertTrue(out_path.exists())


class TrimWhitespaceTests(unittest.TestCase):
    def test_crops_to_content_plus_padding(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (200, 150), "white")
        img.putpixel((100, 70), (0, 0, 0))  # single content pixel, easy to reason about

        trimmed = trim_whitespace(img, pad=5)

        # A single-pixel content region trims to exactly 2*pad on each side.
        self.assertEqual(trimmed.size, (10, 10))

    def test_clamps_padding_to_image_bounds(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (20, 20), "white")
        img.putpixel((1, 1), (0, 0, 0))  # near the top-left corner

        trimmed = trim_whitespace(img, pad=50)  # padding far exceeds the image itself

        self.assertEqual(trimmed.size, (20, 20))  # clamped to the original image, not negative/huge


class RenderMoleculeImageTests(unittest.TestCase):
    def test_larger_molecule_renders_wider_at_a_fixed_scale(self) -> None:
        from rdkit import Chem

        small = Chem.MolFromSmiles("CCO")
        large = Chem.MolFromSmiles("C" * 30)  # a long carbon chain

        small_img = render_molecule_image(small, bond_px=28)
        large_img = render_molecule_image(large, bond_px=28)

        # Forcing both onto the same fixed-size box (the previous approach)
        # would have made these come out the same width; at a fixed
        # pixels-per-bond scale the 30-carbon chain must be substantially wider.
        self.assertGreater(large_img.width, small_img.width * 3)

    def test_does_not_clip_within_the_default_canvas(self) -> None:
        from rdkit import Chem

        from sample_route_tree import DEFAULT_RENDER_CANVAS

        # A real, sizeable route target (~70 heavy atoms) -- must render
        # fully within the default canvas, not get cut off at its edge.
        mol = Chem.MolFromSmiles(
            "NC(=O)c1c(-c2ccc(Oc3ccccc3)cc2)nn2c1NCC[C@H]2C1CCN(C(=O)c2cn"
            "(CCOCCOCCOCCOCCNc3ccc4c(c3)C(=O)N(C3CCC(=O)NC3=O)C4=O)nn2)CC1"
        )

        img = render_molecule_image(mol, bond_px=28)

        self.assertLess(img.width, DEFAULT_RENDER_CANVAS[0])
        self.assertLess(img.height, DEFAULT_RENDER_CANVAS[1])


class FormatTargetLabelTests(unittest.TestCase):
    def test_rounds_score_to_one_decimal(self) -> None:
        self.assertEqual(format_target_label(0.846), "TARGET - SCORE: 0.8")

    def test_keeps_a_whole_score_at_one_decimal(self) -> None:
        self.assertEqual(format_target_label(1.0), "TARGET - SCORE: 1.0")


class PlotRouteTreeTests(unittest.TestCase):
    def test_renders_a_nonempty_png(self) -> None:
        tree = parse_route(_DRAWABLE_ROUTE)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tree.png"

            returned = plot_route_tree(tree, out_path, score=1.0)

            self.assertEqual(returned, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)

    def test_tolerates_an_unparseable_smiles(self) -> None:
        route = {1: [["not a smiles => CCO.CCN", ""]]}
        tree = parse_route(route)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tree.png"

            plot_route_tree(tree, out_path, score=1.0)  # must not raise

            self.assertTrue(out_path.exists())

    def test_also_writes_a_pdf_sibling(self) -> None:
        tree = parse_route(_DRAWABLE_ROUTE)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tree.png"

            plot_route_tree(tree, out_path, score=1.0)

            pdf_path = out_path.with_suffix(".pdf")
            self.assertTrue(pdf_path.exists())
            self.assertGreater(pdf_path.stat().st_size, 500)


def _write_fixture_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "resolved": [True, True, False, True],
            "route": [
                _SHAPED_ROUTE,  # 3 steps, 3 leaves -> matches the default 2-4/2-4 window
                {1: [["BIG => A", ""]], 2: [["A => B", ""]], 3: [["B => C", ""]],
                 4: [["C => D", ""]], 5: [["D => E", ""]]},  # 5 steps -> outside window
                _SHAPED_ROUTE,  # resolved=False -> excluded regardless of shape
                {1: [["SMALL => A.B", ""]]},  # 1 step -> outside window
            ],
        }
    )
    df.to_csv(path, index=False)


class CliTests(unittest.TestCase):
    def test_writes_csv_for_the_one_matching_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "routes.csv"
            output_dir = Path(tmp) / "out"
            _write_fixture_csv(input_csv)

            main([str(input_csv), "--output-dir", str(output_dir), "--seed", "0", "--no-plot"])

            csvs = sorted(output_dir.glob("*.csv"))
            self.assertEqual(len(csvs), 1)
            written = pd.read_csv(csvs[0])
            self.assertEqual(written.loc[written["role"] == "target", "smiles"].tolist(), ["T"])
            self.assertEqual(
                sorted(written.loc[written["role"] == "building_block", "smiles"]), ["BB", "C", "D"]
            )
            self.assertEqual(list(output_dir.glob("*.png")), [])

    def test_writes_png_when_plot_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "routes.csv"
            output_dir = Path(tmp) / "out"
            pd.DataFrame({"resolved": [True], "route": [_DRAWABLE_ROUTE]}).to_csv(input_csv, index=False)

            main([str(input_csv), "--output-dir", str(output_dir), "--seed", "0", "--plot"])

            self.assertEqual(len(list(output_dir.glob("*.png"))), 1)

    def test_exits_with_message_when_no_route_matches_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "routes.csv"
            output_dir = Path(tmp) / "out"
            pd.DataFrame({"resolved": [True], "route": [{1: [["SMALL => A.B", ""]]}]}).to_csv(
                input_csv, index=False
            )

            with self.assertRaises(SystemExit):
                main(
                    [
                        str(input_csv), "--output-dir", str(output_dir),
                        "--min-steps", "2", "--max-steps", "4", "--no-plot",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
