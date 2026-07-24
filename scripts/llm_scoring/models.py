"""DSPy signatures, pydantic output schemas, and scoring/classification modules.

Kept separate from `llm_scoring.py` (LM configuration, batch runner, CLI) so the
schema and module definitions can be imported on their own -- e.g. by
`classify_protac.py`, which only needs `ProtacClassifier` -- without pulling in
argparse / threading / CSV I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import dspy
from pydantic import BaseModel, Field

# Add src/ to the path so protac_synth.route_parsing is importable.
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from scripts.llm_scoring.route_parsing import ParsedRoute, describe_molecule, render_route_for_llm  # noqa: E402

# Default reasoning engine. See RouteScorer / the CoT note at the bottom: DSPy's
# ChainOfThought is zero-shot (it needs no labelled examples); we default to bare
# Predict because the schema already elicits per-step and per-route rationales.
USE_COT_DEFAULT = False


# --------------------------------------------------------------------------- #
# 1. Output schema (Pydantic v2) -- route/molecule scoring
# --------------------------------------------------------------------------- #
ReactionCategory = Literal[
    "Reaction feasible, all good",
    "Reaction feasible, unexpected disconnection",
    "Unlikely disconnection",
    "Selectivity (regio-, stereo-, chemo-) issues",
    "Functional group compatibility problems",
    "Protecting group strategy is wrong/non-optimal",
    "Unnecessary step",
    "Non-optimal reagent",
]

RouteCategory = Literal[
    "Route feasible as it is",
    "Route feasible with few modifications",
    "Route feasible with significant modifications",
    "Route unfeasible",
    "Route was not solved to building blocks",
]

MoleculeCategory = Literal[
    "Readily synthesizable (standard building blocks & couplings)",
    "Synthesizable with routine medicinal-chemistry effort",
    "Challenging but feasible (multi-step / sensitive chemistry)",
    "Very challenging (long sequence / difficult selectivity or stereocontrol)",
    "Likely not synthesizable as drawn",
]


class StepEvaluation(BaseModel):
    step_label: str = Field(description="The step tag exactly as given, e.g. 'S1'.")
    rationale: str = Field(description="Concise chemical reasoning (mechanism, selectivity, precedent).")
    suggested_fix: str = Field(default="", description="Concrete alternative if not 'all good'.")
    categories: list[ReactionCategory] = Field(
        min_length=1, max_length=2,
        description="1-2 categories, most appropriate first.")
    feasibility_score: float = Field(
        ge=0, le=100,
        description="0 = chemically impossible; 40 = plausible but low-yield / "
                    "selectivity-fraught; 70 = literature-precedented with care; "
                    "90-100 = robust, routine transformation.")
    confidence: float = Field(ge=0, le=100, description="Confidence in THIS step assessment.")


class RouteEvaluation(BaseModel):
    rationale: str = Field(description="Concise overall route feasibility reasoning.")
    convergence_comment: str = Field(description="Convergent vs linear; longest linear sequence.")
    risk_factors: list[str] = Field(description="Ranked list of the main synthetic risks.")
    category: RouteCategory
    route_score: float = Field(
        ge=0, le=100,
        description="PRIMARY overall quality/feasibility score. Rubric: "
                    "90-100 robust, short, convergent, cheap accessible BBs; "
                    "70-89 sound, minor modifications; "
                    "50-69 core idea works but needs significant changes; "
                    "25-49 major problems, likely fails as written; "
                    "0-24 unfeasible OR not solved to purchasable building blocks.")
    confidence: float = Field(ge=0, le=100)


class MoleculeEvaluation(BaseModel):
    """Route-free assessment used when `resolved == False` (no usable route)."""
    rationale: str = Field(description="Concise de novo synthesizability reasoning.")
    likely_disconnections: list[str] = Field(
        default_factory=list,
        description="Brief strategic disconnections you would attempt.")
    risk_factors: list[str] = Field(description="Ranked list of the main synthetic risks.")
    category: MoleculeCategory
    synthesizability_score: float = Field(
        ge=0, le=100,
        description="Overall ease of synthesis, HIGHER = EASIER (same orientation "
                    "as route_score). 90-100 trivial modular assembly from stock "
                    "building blocks; 70-89 routine medicinal-chemistry synthesis; "
                    "50-69 feasible but multi-step / some sensitive chemistry; "
                    "25-49 difficult (long sequence, hard selectivity/stereocontrol, "
                    "awkward core); 0-24 very hard or impractical as drawn.")
    confidence: float = Field(ge=0, le=100)


# --------------------------------------------------------------------------- #
# 2. Prompts, encoded as signature instructions -- route/molecule scoring
# --------------------------------------------------------------------------- #
class RetrosynthesisRouteAndStepsScore(dspy.Signature):
    """You are a senior medicinal / synthetic-organic chemist. Critically evaluate a
proposed RETROSYNTHETIC route and score it, with the precision of an expert reviewer.

INPUT FORMAT
- `route` gives a COMPOUND LEGEND (every structure listed once with a tag: [T] target,
  [I*] intermediate, [B*] building block assumed purchasable, [U*] UNRESOLVED leaf that
  is NOT a building block) followed by RETROSYNTHETIC STEPS.
- A step "S_n [depth d]: [A] <= [B] + [C]" is a single reaction written retrosynthetically:
  A is DISCONNECTED into B and C, i.e. the FORWARD reaction makes A FROM B and C.
- Presence of any [U*] compound means the route did not reach purchasable material on that
  branch.

DOMAIN NOTE (these targets are often PROTAC / bifunctional degraders)
- Expect modular assembly of a warhead + linker + E3-ligase ligand. Common couplings:
  amide formation, CuAAC "click" (azide + alkyne -> triazole), SNAr, Buchwald-Hartwig,
  reductive amination, Suzuki, sulfonyl-amide formation, and Boc protection/deprotection.
- Watch chemoselectivity: these molecules carry many amides, sulfonamides, basic amines,
  aryl halides, phenols, and a glutarimide (base- and epimerization-sensitive). Flag acyl
  chlorides / activated esters reacting with the WRONG amine or a free OH; azide handling
  and safety; and orthogonality of any protecting groups across the WHOLE route.

PER-STEP: assess feasibility & likelihood of success, appropriateness of the disconnection,
regio/stereo/chemoselectivity, functional-group compatibility, protecting-group logic (in
the context of the full route), step necessity/efficiency, reagent choice, and probable side
reactions. Then pick the MOST APPROPRIATE 1-2 reaction categories and a continuous
`feasibility_score` using the anchors in the schema.

WHOLE-ROUTE: assess overall feasibility, step economy, convergence, use of accessible
starting materials, and scalability. Pick one route category and a continuous `route_score`
using the rubric in the schema. Score DOWN hard for any [U*] leaf (assign "Route was not
solved to building blocks" and a low score).

RULES
- Return exactly one StepEvaluation per step, keyed by the given `step_label`.
- Prioritize chemical accuracy over comprehensiveness. Where genuinely uncertain, say so and
  lower `confidence` rather than inventing precedent.
- Be terse and technical in rationales; no hedging boilerplate."""

    target_smiles: str = dspy.InputField(desc="Target molecule SMILES (also [T] in the legend).")
    route: str = dspy.InputField(desc="Compound legend + retrosynthetic steps.")
    search_metadata: str = dspy.InputField(desc="Route flags: resolved, depth, #steps, #BBs.")

    step_evaluations: list[StepEvaluation] = dspy.OutputField(
        desc="One entry per step Sx, keyed by step_label.")
    route_evaluation: RouteEvaluation = dspy.OutputField(
        desc="Whole-route assessment incl. the primary route_score.")


class RetrosynthesisRouteScore(dspy.Signature):
    """You are a senior medicinal / synthetic-organic chemist. Critically evaluate a
proposed RETROSYNTHETIC route and score it, with the precision of an expert reviewer.

INPUT FORMAT
- `route` gives a COMPOUND LEGEND (every structure listed once with a tag: [T] target,
  [I*] intermediate, [B*] building block assumed purchasable, [U*] UNRESOLVED leaf that
  is NOT a building block) followed by RETROSYNTHETIC STEPS.
- A step "S_n [depth d]: [A] <= [B] + [C]" is a single reaction written retrosynthetically:
  A is DISCONNECTED into B and C, i.e. the FORWARD reaction makes A FROM B and C.
- Presence of any [U*] compound means the route did not reach purchasable material on that
  branch.

DOMAIN NOTE (these targets are often PROTAC / bifunctional degraders)
- Expect modular assembly of a warhead + linker + E3-ligase ligand. Common couplings:
  amide formation, CuAAC "click" (azide + alkyne -> triazole), SNAr, Buchwald-Hartwig,
  reductive amination, Suzuki, sulfonyl-amide formation, and Boc protection/deprotection.
- Watch chemoselectivity: these molecules carry many amides, sulfonamides, basic amines,
  aryl halides, phenols, and a glutarimide (base- and epimerization-sensitive). Flag acyl
  chlorides / activated esters reacting with the WRONG amine or a free OH; azide handling
  and safety; and orthogonality of any protecting groups across the WHOLE route.

WHOLE-ROUTE: read every step, then assess overall feasibility, step economy, convergence,
use of accessible starting materials, and scalability -- weighing per-step feasibility,
regio/stereo/chemoselectivity, functional-group compatibility, protecting-group logic, and
probable side reactions along the way even though no per-step scores are returned. Pick one
route category and a continuous `route_score` using the rubric in the schema. Score DOWN
hard for any [U*] leaf (assign "Route was not solved to building blocks" and a low score).

RULES
- Prioritize chemical accuracy over comprehensiveness. Where genuinely uncertain, say so and
  lower `confidence` rather than inventing precedent.
- Be terse and technical in the rationale; no hedging boilerplate."""

    target_smiles: str = dspy.InputField(desc="Target molecule SMILES (also [T] in the legend).")
    route: str = dspy.InputField(desc="Compound legend + retrosynthetic steps.")
    search_metadata: str = dspy.InputField(desc="Route flags: resolved, depth, #steps, #BBs.")

    route_evaluation: RouteEvaluation = dspy.OutputField(
        desc="Whole-route assessment incl. the primary route_score.")


class MoleculeSynthesizabilityScore(dspy.Signature):
    """You are a senior medicinal / synthetic-organic chemist. Estimate how readily a single
target molecule can be MADE when NO retrosynthetic route is available (a route search failed
to resolve one to purchasable building blocks). Judge de novo synthesizability from your own
chemical knowledge -- do NOT assume a route; reason about how you WOULD make it.

WHAT TO ASSESS
- Strategic disconnections you would attempt, and whether they terminate in accessible,
  likely-purchasable building blocks.
- Structural complexity: ring systems and fused/bridged/macrocyclic motifs, stereocentres
  (and how they would be set/controlled), dense or unusual heterocycles, functional-group
  density and mutual compatibility, and any obviously fragile motifs.
- Protecting-group burden and chemoselectivity challenges across a plausible synthesis.
- Rough sense of the longest linear sequence / number of steps and whether convergent
  assembly is possible.

DOMAIN NOTE (these targets are often PROTAC / bifunctional degraders)
- Expect a warhead + linker + E3-ligase ligand assembled modularly (amide coupling, CuAAC
  click, SNAr, Buchwald-Hartwig, reductive amination, Suzuki; Boc protect/deprotect).
- A glutarimide (base-/epimerization-sensitive), many amides/sulfonamides, basic amines,
  aryl halides and phenols are common and constrain selectivity and protecting-group choice.
- Modular PROTAC assembly is usually tractable; score DOWN mainly for genuinely hard cores,
  difficult macrocyclization, uncontrolled stereochemistry, or exotic/unstable motifs.

`molecular_context` provides a few RDKit-derived descriptors as weak grounding; use them but
rely on your own reading of the structure.

SCORING -- use the SAME orientation as route scoring: HIGHER = EASIER to synthesize.
- 90-100 trivial modular assembly from stock building blocks, few steps.
- 70-89 routine medicinal-chemistry synthesis, standard couplings, minor challenges.
- 50-69 feasible but multi-step / some sensitive or selective chemistry.
- 25-49 difficult: long sequence, hard selectivity/stereocontrol, or awkward core.
- 0-24 very hard or likely impractical to make as drawn.

Be terse and technical. Where genuinely uncertain, say so and lower `confidence` rather than
inventing precedent."""

    target_smiles: str = dspy.InputField(desc="Target molecule SMILES.")
    molecular_context: str = dspy.InputField(desc="A few RDKit descriptors (weak grounding).")
    note: str = dspy.InputField(desc="Why no route is provided.")

    molecule_evaluation: MoleculeEvaluation = dspy.OutputField(
        desc="De novo synthesizability assessment incl. the synthesizability_score.")


# --------------------------------------------------------------------------- #
# 3. Modules -- route/molecule scoring
# --------------------------------------------------------------------------- #
class RouteScorer(dspy.Module):
    """Single-pass route grader.

    `score_steps` picks the signature: RetrosynthesisRouteAndStepsScore (route + every
    step) if True, else the cheaper route-only RetrosynthesisRouteScore
    (default -- route_score is the primary signal, so it is prioritized over
    step-level detail). `pred.step_evaluations` is normalised to `[]` in the
    route-only case so downstream aggregation doesn't need to special-case it.
    `use_cot` toggles dspy.ChainOfThought vs dspy.Predict. CoT is zero-shot: it
    only prepends a free-text reasoning field, so it needs NO labelled examples.
    We default it OFF because the schema already forces per-step / per-route
    rationales; flip it on to A/B whether reason-then-score improves calibration."""

    def __init__(self, use_cot: bool = USE_COT_DEFAULT, score_steps: bool = False):
        super().__init__()
        engine = dspy.ChainOfThought if use_cot else dspy.Predict
        self.score_steps = score_steps
        signature = RetrosynthesisRouteAndStepsScore if score_steps else RetrosynthesisRouteScore
        self.score = engine(signature)

    def forward(self, parsed: ParsedRoute) -> dspy.Prediction:
        rendered = render_route_for_llm(parsed)
        pred = self.score(
            target_smiles=parsed.target,
            route=rendered.route_text,
            search_metadata=rendered.meta_text,
        )
        pred.mode = "route"
        pred.parsed = parsed
        # Compound legend + steps text (route_text), NOT the whole prompt -- persisted
        # downstream so tags like [I2]/[U1] referenced in the rationale stay decodable.
        pred.route_description = rendered.route_text
        if not self.score_steps:
            pred.step_evaluations = []
        return pred


class MoleculeScorer(dspy.Module):
    """Route-free synthesizability scorer for `resolved == False` targets."""

    def __init__(self, use_cot: bool = USE_COT_DEFAULT):
        super().__init__()
        engine = dspy.ChainOfThought if use_cot else dspy.Predict
        self.score = engine(MoleculeSynthesizabilityScore)

    def forward(self, parsed: ParsedRoute) -> dspy.Prediction:
        pred = self.score(
            target_smiles=parsed.target,
            molecular_context=describe_molecule(parsed.target),
            note="No route resolved to purchasable building blocks for this target; "
                 "score its de novo synthesizability without a route.",
        )
        pred.mode = "molecule_only"
        pred.parsed = parsed
        pred.route_description = ""  # no route was resolved -- nothing to render
        return pred


class SynthesisScorer(dspy.Module):
    """Route each target to route-grading or route-free synthesizability scoring.

    Molecule mode is used when `resolved is False` (by contract) OR the parsed
    route has no reactions. Both paths return a 0-100 score, higher = better."""

    def __init__(self, use_cot: bool = USE_COT_DEFAULT, score_steps: bool = False):
        super().__init__()
        self.route = RouteScorer(use_cot=use_cot, score_steps=score_steps)
        self.molecule = MoleculeScorer(use_cot=use_cot)

    def forward(self, parsed: ParsedRoute) -> dspy.Prediction:
        molecule_only = (parsed.resolved is False) or (not parsed.reactions)
        return self.molecule(parsed) if molecule_only else self.route(parsed)


class EnsembleScorer(dspy.Module):
    """Sample K times and aggregate. Denoises the label and yields an agreement
    estimate (score_std, category_agreement). Works for both scoring modes."""

    def __init__(self, k: int = 3, use_cot: bool = USE_COT_DEFAULT, score_steps: bool = False):
        super().__init__()
        self.k = k
        self.base = SynthesisScorer(use_cot=use_cot, score_steps=score_steps)

    def forward(self, parsed: ParsedRoute) -> dspy.Prediction:
        from statistics import mean, median, pstdev

        samples = []
        for _ in range(self.k):
            try:
                samples.append(self.base(parsed))
            except Exception as e:  # keep partial ensembles alive
                samples.append(e)
        good = [s for s in samples if not isinstance(s, Exception)]
        if not good:
            raise samples[-1]

        mode = good[0].mode
        if mode == "route":
            scores = [s.route_evaluation.route_score for s in good]
            cats = [s.route_evaluation.category for s in good]
            confs = [s.route_evaluation.confidence for s in good]
            per_step: dict[str, list[float]] = {}
            for s in good:
                for se in s.step_evaluations:
                    per_step.setdefault(se.step_label, []).append(se.feasibility_score)
            step_score_mean = {k: mean(v) for k, v in per_step.items()}
            rep = min(good, key=lambda s: abs(s.route_evaluation.route_score - median(scores)))
            risk_factors = rep.route_evaluation.risk_factors
            rationale = rep.route_evaluation.rationale
        else:
            scores = [s.molecule_evaluation.synthesizability_score for s in good]
            cats = [s.molecule_evaluation.category for s in good]
            confs = [s.molecule_evaluation.confidence for s in good]
            step_score_mean = None
            rep = min(good, key=lambda s: abs(
                s.molecule_evaluation.synthesizability_score - median(scores)))
            risk_factors = rep.molecule_evaluation.risk_factors
            rationale = rep.molecule_evaluation.rationale

        top_cat = max(set(cats), key=cats.count)
        return dspy.Prediction(
            mode=mode,
            score_mean=mean(scores),
            score_median=median(scores),
            score_std=pstdev(scores) if len(scores) > 1 else 0.0,
            category=top_cat,
            category_agreement=cats.count(top_cat) / len(cats),
            confidence_mean=mean(confs),
            step_score_mean=step_score_mean,
            risk_factors=risk_factors,
            rationale=rationale,
            n_samples=len(good),
            samples=good,
            parsed=parsed,
            # Identical across samples (rendered once per call from the same `parsed`,
            # not model-dependent), so any sample's copy is representative.
            route_description=good[0].route_description,
        )


# --------------------------------------------------------------------------- #
# 4. Output schema + signature + module -- PROTAC/non-PROTAC classification
# --------------------------------------------------------------------------- #
ProtacLabel = Literal["PROTAC", "non-PROTAC"]


class ProtacLabelOnly(BaseModel):
    label: ProtacLabel = Field(
        description="'PROTAC' if the molecule is a bifunctional degrader (warhead + "
                    "linker + E3-ligase ligand, connected so as to bring a target "
                    "protein and an E3 ligase into proximity), else 'non-PROTAC'.")
    confidence: float = Field(ge=0, le=100, description="Confidence in the label, 0-100.")


class ProtacLabelWithRationale(BaseModel):
    rationale: str = Field(description="Concise (1-2 sentence) structural justification for the label.")
    label: ProtacLabel = Field(
        description="'PROTAC' if the molecule is a bifunctional degrader (warhead + "
                    "linker + E3-ligase ligand, connected so as to bring a target "
                    "protein and an E3 ligase into proximity), else 'non-PROTAC'.")
    confidence: float = Field(ge=0, le=100, description="Confidence in the label, 0-100.")


class ClassifyProtac(dspy.Signature):
    """You are a medicinal chemist. Decide whether the given molecule is a PROTAC
(a bifunctional degrader) or not.

WHAT MAKES A MOLECULE A PROTAC
- Two ligand-like moieties (a "warhead" binding a protein of interest, and an
  E3-ligase-recruiting ligand such as a thalidomide/lenalidomide-derived
  glutarimide, VHL-binding hydroxyproline scaffold, or a cereblon/MDM2/IAP binder)
  joined by a covalent linker (e.g. PEG chain, alkyl chain, triazole from CuAAC,
  piperazine/piperidine spacer).
- Molecular weight is usually large (typically >700-800 Da) and there is a clear
  three-part modular architecture: warhead - linker - E3 ligand.
- A molecule that is merely large, or merely contains a glutarimide/heterocycle
  in isolation, is NOT automatically a PROTAC -- the linked bifunctional
  architecture is the deciding feature, not size or a single substructure alone.
- Molecular glues, single-ligand inhibitors, natural products, and ordinary
  drug-like small molecules are non-PROTAC even if structurally complex.

`molecular_context` gives a few RDKit-derived descriptors (size, rings,
stereocentres) as weak grounding; rely primarily on your own reading of the
SMILES structure.

Be terse and technical. Where genuinely uncertain, say so and lower `confidence`
rather than guessing."""

    target_smiles: str = dspy.InputField(desc="Molecule SMILES to classify.")
    molecular_context: str = dspy.InputField(desc="A few RDKit descriptors (weak grounding).")

    classification: ProtacLabelOnly = dspy.OutputField(
        desc="PROTAC vs non-PROTAC label with confidence.")


class ClassifyProtacWithRationale(dspy.Signature):
    __doc__ = ClassifyProtac.__doc__

    target_smiles: str = dspy.InputField(desc="Molecule SMILES to classify.")
    molecular_context: str = dspy.InputField(desc="A few RDKit descriptors (weak grounding).")

    classification: ProtacLabelWithRationale = dspy.OutputField(
        desc="PROTAC vs non-PROTAC label with confidence and a concise rationale.")


class ProtacClassifier(dspy.Module):
    """Standalone PROTAC / non-PROTAC classifier for a bare molecule (no route).

    Deliberately a separate module from RouteScorer/MoleculeScorer/SynthesisScorer:
    this is a label+confidence classification task, not route/molecule feasibility
    scoring, so it gets its own dedicated dspy.Predict rather than sharing the
    scoring modules' route-specific plumbing. `include_rationale` picks between
    ClassifyProtac and ClassifyProtacWithRationale -- mirroring RouteScorer's
    `score_steps` switch -- since the rationale costs extra tokens and is not
    needed for bulk classification runs (default off)."""

    def __init__(self, include_rationale: bool = False):
        super().__init__()
        self.include_rationale = include_rationale
        signature = ClassifyProtacWithRationale if include_rationale else ClassifyProtac
        self.classify = dspy.Predict(signature)

    def forward(self, smiles: str) -> dspy.Prediction:
        pred = self.classify(
            target_smiles=smiles,
            molecular_context=describe_molecule(smiles),
        )
        pred.smiles = smiles
        return pred
