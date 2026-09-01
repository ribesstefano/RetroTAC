# PROTAC Synthesizability Score Prediction

> Transcribed from the uploaded handover PDF. Body text, equations, and tables are transcribed faithfully. The document contains eight embedded charts/figures; since a flat markdown file can't embed the original raster images, each is replaced below with a detailed inline description built from its caption, axis labels, legend, and the surrounding discussion (marked as `> **Figure N description:**`). Figure 8 is a numeric correlation matrix, so it is reproduced as an exact data table rather than a prose description.

---

## 1. Overview

### 1.1 How to read this document

| Section | Contents |
|---|---|
| Background | Literature and rationale for the approach |
| Methodology | Data, labels, features, validation protocol |
| Results | Model performance and interpretation |
| Status | What is finished, what is open |
| Appendix | Code and data locations |

---

## 2. Background

### 2.1 PROTACs and why synthesizability is hard

Whole-molecule retrosynthetic planning can capture the full complexity of a PROTAC but is expensive and often yields routes that are hard to interpret. The faster alternative is a learned synthesizability score, yet almost every score in use was developed on drug-like small molecules: their training distributions sit well below the mass of a typical degrader, and their fragment vocabularies were never fitted to linker chemistry. Applying them to PROTACs is an extrapolation that is seldom acknowledged. Only FSscore and DeepPSA have addressed these molecules directly, the latter motivated by the gap between the proliferation of PROTAC generative models and the tools to judge whether their output can be made (Neeser et al. 2024; Zhang et al. 2025). How the remaining scores behave here is largely unmeasured.

### 2.2 Existing synthesizability scores

Six published scores are integrated in this project, falling into three families by what they measure.

| Score | Approach | Training data | Output | Orientation |
|---|---|---|---|---|
| SA score | Fragment frequency and complexity penalty, rule-based | 1M PubChem molecules | 1 to 10 | Higher is harder |
| SCScore | Neural net, pairwise reaction ordering | 12M Reaxys reactions | 1 to 5 | Higher is harder |
| RAscore | Classifier on retrosynthesis solvability | 200k ChEMBL, 100k GDB-derived | Probability | Higher is easier |
| SYBA | Bernoulli naive Bayes, fragment log-odds | ZINC15 vs Nonpher | Log-odds | Higher is easier |
| GASA | Graph attention classifier | 800k ChEMBL, GDBChEMBL, ZINC15 | Class and ES probability | Higher is easier |
| FSscore | GNN ranker, pairwise preferences | Reaction pairs, then expert feedback | Learned scale | Higher is easier |

**Structural complexity heuristics.** SA score (Ertl and Schuffenhauer 2009) combines a fragment contribution term, rewarding substructures frequent in PubChem, with a penalty for rings, stereocenters, macrocycles and size. It needs no training and was validated against medicinal chemists' ease ratings. SYBA (Voršilák et al. 2020) learns the same fragment-level view, summing per-fragment log-odds from a classifier trained to separate ZINC15 molecules from Nonpher-generated structures. Both measure how complex a structure looks, which is distinct from whether a route exists.

**Reaction-corpus models.** SCScore (Coley et al. 2018) trains a neural network on ECFP fingerprints under one constraint: on average a published reaction's product is more complex than its reactants, so complexity accumulates along a synthesis. FSscore (Neeser et al. 2024) also learns from reactant-product pairs, but as a RankNet-style pairwise ranker over molecular graphs, with a second stage fine-tuning on expert preferences within a target chemical space, one of which the authors demonstrate is PROTAC-like. It is therefore a relative score, meaningful within the domain it was tuned for.

**Route-aware and graph-based models.** RAscore (Thakkar et al. 2021) is the only one whose labels come from an actual search. The authors ran AiZynthFinder over 200k ChEMBL and 100k generated GDBChEMBL and GDBMedChem compounds (3 min limit, 7 steps, 200 iterations, policy cutoff 0.995), then trained classifiers on ECFP6 fingerprints with and without descriptors to predict whether a route was found. Their baselines, SA score, SCScore and SYBA, did not separate solved from unsolved compounds cleanly. The best model per dataset was named separately, RAscore and GDBscore, reflecting different provenance. GASA (Yu et al. 2022) is a graph attention network classifying molecules as easy or hard, trained on ~800k compounds sampled near the decision boundary so it discriminates between similar pairs; use the probability rather than the class.

### 2.3 Shared limitations

- **No experimental ground truth.** None is validated against laboratory outcomes; each optimises a proxy, so "synthesizable" means something different in each case.
- **Complexity is not accessibility.** A complex-looking molecule may be one coupling of two purchasable fragments; only RAscore reasons about routes.
- **Route-blind.** They take a SMILES in isolation and know nothing about what is purchasable.
- **Training-distribution dependence.** All are fitted to drug-like ChEMBL, ZINC or Reaxys space.
- **Artificial hard labels.** SYBA's hard set is generated and GASA's split is database membership, so both may detect artificial rather than real difficulty.
- **Incomparable scales.** Differing ranges, orientations and arbitrary cutoffs require harmonisation before combination.
- **Mutual disagreement.** Correlations are modest with no consensus on which is correct, so any single score is a weak signal.

Method-specific issues compound these. SA score's size, macrocycle and stereocenter penalties make it sensitive to molecular weight above all. SCScore's ordering assumption fails for simplifying reactions, its 1-5 range compresses differences, and it depends on proprietary Reaxys data. SYBA is blind to global topology and stereochemistry. RAscore inherits AiZynthFinder's templates, stock set and search limits wholesale. GASA's binary output loses granularity. FSscore needs expert fine-tuning per space, on small and subjective data.

### 2.4 Why this matters for PROTACs

PROTACs are large, bifunctional and built around long flexible linkers, placing them outside the distribution all six were fitted to. SA score's size penalty rates almost any PROTAC as hard regardless of whether it is a trivial modular coupling of purchasable warhead, linker and E3 ligand. The reaction-trained and route-based scores depend on templates and stock sets that under-represent linker chemistry. This plausibly explains the poor agreement observed between the scores on the PROTAC set, both with one another and with route-derived measures; the correlation structure appears in Results.

These remain useful, cheap triage heuristics, but they are proxies fitted to general chemistry, they disagree, and they are least trustworthy exactly where this work operates. Their relative ranking within a PROTAC set is more defensible than any absolute claim, which is the argument for grounding labels in an explicit route search.

### 2.5 Scoring retrosynthetic routes

The scores above judge a molecule. A second body of work judges a route, which carries different information: step count, intermediates, terminal building blocks, and a sequence of transformations. Methods differ in which they treat as signal.

**RetroScore** (Gao et al. 2026) measures structural change via graph edit distance between successive steps. The end distance, between starting building blocks and target, indicates how much assembly the route performs; the step distance localises it, identifying the key step where most complexity is introduced. Step count is penalised non-linearly:

$$\text{RLScore} = 1 - \log_{10}(\text{route\_len})$$

so the gap between two and four steps matters more than between eight and ten.

**Retro-BLEU** (Li et al. 2024) borrows from machine translation evaluation, asking whether the transformation sequence resembles published chemistry:

$$\text{Score}_{\text{Retro-BLEU}}(r) = \exp\!\left(\frac{L}{\max(L, \text{len}(r))}\right) + \exp\big(f_n(r)\big)$$

*(the source PDF's rendering of this equation loses some operator grouping in extraction; consult Li et al. 2024 directly before implementing this term)*

where $\text{len}(r)$ is the number of reaction steps, $f_n(r)$ is the n-gram overlap of the route's template sequence with known routes, and $L$ is a hyperparameter setting the length beyond which routes are penalised. The authors use bigrams and set $L$ to 3, the average length of patent routes in their corpus being 2.79.

The length reference is worth noting for this project: a threshold calibrated to routes averaging under three steps is far below what PROTAC assembly requires, so the term would penalise essentially every route here unless $L$ were reset.

Its caveat: Retro-BLEU measures plausibility, not whether the starting building blocks can be obtained. The two are separable, suggesting a route is best characterised along three axes: structural change, building block accessibility, and reaction sequence plausibility. Its n-gram statistics also come from USPTO patent routes, where PROTAC chemistry is thinly represented.

**RPScore** (Kreutter and Reymond 2023) scores a linear sequence of $N$ steps as a product of three factors:

$$\text{RPScore} = SP^N \cdot \prod_i CS_i \cdot \prod \text{Simplicity(intermediate)}$$

The step penalty $SP$ ($0 < SP \le 1$, default 0.8) shrinks geometrically with length, promoting short routes and demoting unproductive protection and deprotection cycles. The confidence scores $CS_i$ come from a forward validation transformer, so each step carries a measure of whether a forward model believes the reaction works. Simplicity runs 0 to 1, is derived from SCScore, and is set to 1 for any molecule in the commercial building block set; reagents are excluded, since their calculated simplicity says little about availability.

**ChemiRise** (Lin et al. 2021) defines route quality recursively as a cost to minimise, so it can guide the search rather than only rank output:

$$\text{Cost}(m) = \text{GetComplexity}(m) \quad \text{if } m \text{ is a purchasable leaf}$$

$$\text{Cost}(m) = \min_r \frac{\sum_{\text{reactant} \in r} \text{Cost}(\text{reactant})}{\text{GetFeasibility}(r)} \quad \text{otherwise}$$

Feasibility is reactivity, from template frequency, times functional group compatibility from a naive Bayes model. It also contributes an expert rubric grading routes A (ready to run) to D (no route to stock), one of the few attempts to anchor a computational score to expert judgement.

**What transfers.** All four penalise length and supplement it with either step quality or intermediate accessibility. Length transfers directly, since the longest linear sequence is available here. The other two do not: RPScore's $CS_i$ and ChemiRise's Feasibility both require a trained reaction model, which is not part of the route output used here; and both weight building blocks by complexity, which matters less when building blocks come from a stock set and are synthesizable by construction.

#### 2.5.1 The score used here

The score is a composite of route-level properties. In its original form it combined three terms as a weighted mean:

$$S = \frac{w_l \cdot \text{length} + w_c \cdot \text{coup} + w_b \cdot \text{bal}}{w_l + w_c + w_b}$$

**Length** decays to zero at a saturation point. Originally logarithmic:

$$\text{length} = \max\left(0,\ 1 - \frac{\log_{10}(\text{LLS})}{\log_{10}(\text{Sat})}\right)$$

later changed to linear decay:

$$\text{length} = \max\left(0,\ 1 - \frac{\text{LLS}}{\text{Sat}}\right)$$

LLS is the number of reactions along the longest path from target to building blocks: for a linear route the step count, for a convergent route the longest branch. Taking the longest branch rather than total steps is what makes the term sensitive to convergence, a criterion the metrics above largely miss.

A logarithm compresses differences at the long end, as RLScore and RPScore's geometric penalty both do; linear decay treats each additional reaction as an equal cost. On a dataset where most routes are short the two behave similarly, and the linear form is easier to interpret.

> **Figure 1 description — "Distribution of the length term under logarithmic and linear decay":** an overlaid histogram with x-axis "LLS score" (0.0 to 1.0) and y-axis "Count" (0 to a little over 5000). Two series are compared: "Old (log)" and "New (linear)", i.e. the length term computed under the two decay formulas above, across the full PROTAC route dataset. The point of the figure (per the surrounding text) is that the logarithmic formulation compresses differences among longer routes toward the higher end of the [0,1] range, while the linear formulation spreads scores more evenly and proportionally to step count. Because most routes in the dataset are short relative to the saturation constant, the two distributions occupy broadly similar territory, which is the empirical justification given later for preferring the linear form (simpler, equally informative, easier to interpret).

**Coupling** measured the fraction of steps joining two or more separate fragments into one product:

$$\text{coup} = \frac{n_{\text{coupling}}}{n_{\text{steps}}}$$

distinguishing modular assembly from decoration of a single scaffold, which for PROTACs maps onto the warhead, linker and E3 ligand architecture.

> **Figure 2 description — "Distribution of the composite score under logarithmic and linear length terms":** an overlaid histogram, x-axis "Score" (0.0 to 1.0), y-axis "Count" (0 to roughly 2500). Series: "Old (log) — unweighted" and "New (linear) — unweighted", i.e. the full composite score (at this stage still combining length and balance) recomputed under each length-term formulation. Both distributions cluster in the middle-to-upper part of the range, with the linear-term version shifted slightly toward higher scores, mirroring the shift already seen in Figure 1 for the length term alone — showing that the length-term choice propagates into the composite but doesn't change its overall shape drastically.

**Fragment balance** asks how evenly matched the pieces joined at each coupling step are, comparing heavy atom counts pairwise:

$$\text{bal} = \text{mean}_{i \in \text{coupling steps}}\left(\text{mean}_{j<k} \frac{\min(HAC_j, HAC_k)}{\max(HAC_j, HAC_k)}\right)$$

The ratio is near 1 when comparably sized fragments are joined and near 0 when a small group is appended to a large intermediate. This separates genuine module coupling from decoration and carries the PROTAC-specific intuition: a degrader built from three substantial fragments differs from one built by progressive functionalisation even at equal route length.

> **Figure 3 description — "Distributions of the three score terms across the dataset":** three side-by-side histogram panels sharing the same layout (x-axis 0.0 to 1.0, y-axis "Count"). Panel 1, "LLS score", shows the length term's distribution (comparable to Figure 1's "New (linear)" series, up to ~5000 count). Panel 2, "Coupling score", is heavily right-skewed and piles up near 1.0 (up to ~8000 count at the top bin), showing that almost every step in almost every found route couples two separate fragments. Panel 3, "Balance score", is roughly bell-shaped and centered around 0.5–0.6 (up to ~1750 count), spreading fairly evenly across the range. The takeaway stated in the text: length and balance spread across their range while coupling concentrates near its upper bound — this is the empirical finding that later motivates dropping the coupling term as uninformative (it isn't wrong, it's just nearly constant, so it consumes weight without adding discriminative power).

Length and balance spread across their range while coupling concentrates near its upper bound, which is what later motivates its removal.

Because balance is a ratio of heavy atom counts it is scale-invariant, so unlike SA score the composite does not penalise a molecule for being large. That is the mechanism by which size bias is avoided, and it differs from dividing through by molecular size.

#### 2.5.2 Simplification

Both the coupling term and the weights were dropped, leaving an unweighted mean of length and balance:

$$S = \frac{\text{length} + \text{bal}}{2}$$

which is the general form with $w_c = 0$ and $w_l = w_b$. Substituting the linear length term gives the score as currently computed:

$$S = \frac{1}{2}\left[\max\left(0,\ 1 - \frac{\text{LLS}}{\text{Sat}}\right) + \text{mean}_{i \in \text{coupling steps}}\left(\text{mean}_{j<k} \frac{\min(HAC_j, HAC_k)}{\max(HAC_j, HAC_k)}\right)\right]$$

Both terms lie in $[0,1]$, so the composite does too, with higher values indicating an easier synthesis.

**Coupling was uninformative.** Removing it required only setting its weight to zero. Roughly 80% of molecules clustered near a coup value of 1, meaning almost every step in almost every route joins separate fragments, so the term contributed little to the ranking while consuming weight.

The clustering is a finding rather than a nuisance: routes found for PROTACs are overwhelmingly modular, which the architecture predicts. What varies is not whether fragments are joined but how evenly matched they are, which balance measures.

> **Figure 4 description — "Composite score distribution with and without the coupling term":** an overlaid histogram, x-axis "Unweighted Score" (0.0 to 1.0), y-axis "Count" (0 to roughly 2500+). Series: "With coupling" vs "Without coupling". The two distributions are visually near-identical, both broadly bell-shaped and centered around 0.6–0.8, with a spike near 0 (the boundary-case "unsolved" molecules that score 0 by convention). This near-overlap is the direct empirical evidence that dropping the coupling term barely changes the score distribution, supporting its removal.

This also means balance is defined for nearly every route; the minority without any coupling step falls back to 0.5, as described under Labels.

**The weights made little difference.** Score distributions were largely unchanged without them. The unweighted mean is more defensible in any case: with no validation target to fit against, any weighting would be arbitrary, and equal weighting makes that explicit rather than implying a tuned trade-off.

> **Figure 5 description — "Composite score distribution under weighted and unweighted aggregation":** an overlaid histogram, x-axis "Score" (0.0 to 1.0), y-axis "Count" (0 to roughly 1400+). Series: "Unweighted (equal-average)" vs "Weighted (current: 0.50/0.30/0.20)" — note this legend label is a concrete numeric detail from the source: it implies an earlier three-term weighting scheme used weights of 0.50/0.30/0.20 (presumably across length/coupling/balance in some order) before the coupling term and weighting were dropped entirely. Both distributions nearly overlap, broadly bell-shaped and centered around 0.6–0.7, reinforcing the text's conclusion that the specific weighting scheme makes little difference and that the simpler unweighted mean is preferable.

#### 2.5.3 Relation to the literature

The composite sits closest to RPScore structurally, aggregating several route-level properties, and to RLScore in its logarithmic treatment of length. It differs in what fills the remaining terms: RPScore uses per-step confidence and intermediate simplicity, both needing a trained reaction model, while this score uses fragment geometry derived entirely from heavy atom counts, computable from route output alone. That makes it cheaper and lets it capture whether a route assembles a molecule from comparably sized modules, which for PROTACs is closer to the property of interest.

The weakness is that nothing in it judges whether individual reactions are plausible. On Retro-BLEU's three axes, it covers structural change well, treats building block accessibility as given by the stock set, and leaves reaction sequence plausibility unaddressed.

#### 2.5.4 What makes a good retrosynthetic route

The criteria below are long established in synthetic practice and set the ceiling on what any of these metrics can capture.

- **Few steps.** Yield compounds multiplicatively, so each step is costly.
- **Convergent rather than linear.** Assembling fragments in parallel preserves more material than carrying one intermediate through. Equal-length routes can differ substantially here.
- **Simplifying disconnections.** Good disconnections break the target where complexity reduces most, typically at rings, branch points, or carbon-heteroatom bonds formed by reliable couplings.
- **Reliable, well-precedented reactions.** Broad substrate scope and predictable selectivity beat steps that work only on close analogues.
- **Available starting materials.** A route ending in commercial compounds is executable now.
- **Minimal protecting group manipulation.** These add steps without building the target, which is why RPScore's geometric penalty demotes such cycles.

The metrics above capture step count well, building block availability partially, and step reliability only through learned proxies. Convergence, disconnection quality and protecting group strategy are largely outside what any of them measure, which is why expert rubrics like ChemiRise's A to D grading remain necessary.

Useful entry points are the Synthia introduction to retrosynthesis (*A Brief Guide to Retrosynthesis in Organic Chemistry*, n.d.) and the route prioritisation discussion in Kreutter and Reymond (Kreutter and Reymond 2023).

### 2.6 Where this work sits

This work combines two strands that have so far stayed separate. The route scoring literature judges routes but was developed on drug-like targets, while the PROTAC-specific scores judge molecules without reference to how they would be assembled. The composite defined above is a route score built around a property specific to degraders, namely whether the route joins comparably sized modules, and it is deliberately cheap: it needs only heavy atom counts and step structure, with no reaction model of the kind RPScore and ChemiRise depend on.

The surrogate then serves a different purpose from DeepPSA, the closest comparison. DeepPSA scores PROTACs directly, motivated by filtering the output of generative models. Here the score is defined over a route, and the model exists to approximate it for molecules whose routes have not been computed, since route search is the expensive step. The two are complementary rather than competing: one asks whether a proposed molecule looks makeable, the other how costly the synthesis a planner would find is likely to be.

---

## 3. Methodology

### 3.1 Data

Routes are supplied by an external team's Shallow Tree algorithm; see Labels. Each row is one PROTAC with its route and the scores derived from it.

| Column group | Contents |
|---|---|
| `smiles` | The target molecule |
| `route` | Reaction steps keyed by depth, each a product-to-reactants transformation with template annotation |
| `buildingBlocks` | Terminal compounds the route resolves to |
| `resolved`, `resolvedDepth`, `searchDuration`, `error` | Search outcome and cost |
| `struct_*` | Route descriptors: `n_steps`, `n_BB`, `max_depth`, `lls`, `coupling_fraction`, `avg_branching`, `fragment_balance` |
| `synthesizability` | The composite route score, the regression target |
| `score`, `score_note` | Search-level score and status label |

The `struct_*` columns are the composite's inputs, so the score can be recomputed under new parameters without rerunning the search.

#### 3.1.1 Boundary cases

Two categories have no route to score and are assigned values by convention: purchasable targets, already in the stock set, score **1**; unsolved targets, where the search hit its limits, score **0**. In both the route is empty.

Two consequences. First, the label is a mixture: point masses at 0 and 1 sit on a continuous interior, so part of the explained variance in R² comes from separating the extremes rather than ranking genuine routes. Second, the extremes are not symmetric: a 1 is a positive fact about the molecule, while a 0 is a statement about the search, which may mean unfinished rather than inaccessible.

#### 3.1.2 Quality control and filtering

Filtering targets bad fragment splits, not near-duplicate structures. Each PROTAC is decomposed into E3 ligand, linker and POI ligand, with descriptors per fragment and for the whole molecule. A molecule can look reasonable overall while its split is wrong, and a wrong split corrupts every fragment-level feature downstream.

**Rule-based flags.** Per-feature statistical flags run over PROTAC- and linker-level descriptors. High-kurtosis features, mostly constant with rare extremes where a z-score is unstable, are flagged by percentile outside 0.5 to 99.5; the rest use a robust median and MAD z-score at threshold 3.5. Each molecule gets an outlier fraction per group. Domain hard rules catch what no distributional test would see:

| Rule | Rationale |
|---|---|
| Missing E3, linker or POI fragment | Split failed; nothing to compute a z-score on |
| PROTAC MW outside 400 to 2200 Da | Outside the plausible degrader range |
| PROTAC with no rings | Not a PROTAC |
| Linker with zero rotatable bonds and >8 heavy atoms | Rigid ring system mis-assigned as linker |
| Linker with <2 heavy atoms | Degenerate split |
| PROTAC with zero rotatable bonds and >8 heavy atoms | Macrocyclic or mis-parsed |

These combine into `rule_based_score`, weighting linker outlier fraction at 1.5 times PROTAC and hard rules at 2.0, since the linker is the strongest indicator of a bad split. A molecule is flagged at the 95th percentile, or if any hard rule fires.

**Multivariate flags.** Rule-based flags are univariate and hand-specified, so they miss molecules ordinary on every feature individually but jointly implausible. Four IsolationForest pipelines cover PROTAC-only, linker-only, PROTAC plus linker, and all fragments, each running median imputation, near-zero-variance removal, standardisation, PCA for the two larger feature sets, then IsolationForest (300 estimators, subsample 0.8, bootstrapped). Contamination is anchored to the rate at which the rule-based signal already flags on the matching subset, clipped to 1–10%, so the two methods are calibrated against one another.

**Combination.** All five scores are rank-normalised and averaged; molecules below the 95th percentile are retained. Training and evaluation run on the retained set only, with flagged molecules excluded entirely rather than down-weighted.

Reported metrics therefore describe behaviour on well-formed PROTACs with clean splits, and say nothing about the ~5% removed. If the surrogate is applied to unfiltered input, the same quality control should run first.

### 3.2 Labels

**Route generation.** Routes are not generated in this project. They come from a Shallow Tree retrosynthesis algorithm developed and run by an external team, received as scored route trees. This is a hard dependency. The code here can rescore existing routes under new parameters but cannot produce routes for new molecules. Extending the dataset, changing the stock set, or altering search limits all require the external team. It also bounds what the caveats above can be checked against: search-limit and template-coverage concerns apply to Shallow Tree as to any planner, but its configuration is not visible here.

**Scoring.** `route_tree_score.py` walks each route tree and computes the composite score, with all parameters in `route_scoring.yaml`. Scoring is therefore reproducible from the config alone and fully decoupled from route generation, which is what makes rescoring possible despite the dependency.

Because the score is a weighted mean, the weights double as an on-off switch. Setting a term's weight to zero removes it from numerator and denominator alike, so remaining terms stay correctly averaged and on the same scale; this is how coupling was dropped, with no code change. Equal weights recover the unweighted mean now in use. The same works in reverse: a new term can be added with a non-zero weight and evaluated by rescoring the same trees twice. Several `struct_*` columns are computed but unscored, including `avg_branching` and `n_BB`, so some candidate terms need no new route traversal.

Balance is defined only over coupling steps, so routes containing none fall back to a fixed value of 0.5. This is a neutral midpoint rather than a judgement: such a route is neither rewarded nor penalised on that term, and its score is determined by length alone. The case is rare here, given how strongly coupling clusters near 1, but the fallback should be kept in mind when interpreting scores at the low end.

Two cautions follow. The terms are not independent, so scoring balance while suppressing coupling relies on nearly all routes having coupling steps, which holds here but is not guaranteed on another dataset; where it does not, a large share of molecules would sit at the 0.5 fallback and balance would stop discriminating. And any weight change alters the target, so every model must be retrained before results are comparable: cheap for XGBoost, not for the GNN.

Purchasable and unsolved targets are assigned 1 and 0 rather than computed; see Boundary cases.

> **Worked example (useful as a reimplementation test case):** LLS 9 with fragment balance 0.377 gives a composite of 0.288, implying a length term of exactly 0.200 under an unweighted mean.

### 3.3 Features

**XGBoost and MLP.** Morgan fingerprints, radius 3, folded to 512 bits, concatenated with 2D RDKit descriptors.

512 bits is narrow for molecules of this size: PROTACs generate many distinct radius-3 environments, so bit collisions are more frequent than for drug-like compounds and distinct substructures can share a bit. This plausibly contributes to the gap between these models and the GNN.

**GNN.** Molecular graphs computed directly from SMILES, with no fingerprint or descriptor input. This avoids the collision problem and preserves connectivity that folding discards, consistent with the GNN's advantage in Results.

### 3.4 Models

XGBoost, a multilayer perceptron and a graph neural network, each trained as a regressor.

### 3.5 Validation protocol

Nested cross-validation, 5 outer by 5 inner folds, with scaffold-based splitting following Ash et al. (Ash et al. 2025). The outer loop estimates generalisation and the inner selects hyperparameters, so no model is evaluated on data that influenced its own configuration.

Splits use Bemis-Murcko scaffolds rather than random assignment. PROTAC datasets are built largely from congeneric series, since published degraders typically explore linker variations around a fixed warhead and E3 ligand pair. A random split scatters series members across train and test, scoring the model on near-duplicates of what it has seen. This matters more here than for typical property prediction, because the label depends on a route and routes to closely related molecules are themselves closely related. Scaffold splitting forces the held-out set to contain unseen chemistry, which is the setting the surrogate would face in use.

Hyperparameter tuning used Optuna with a TPE sampler and a median pruner, 20 trials per fold and seed, targeting R². The pruner terminates trials falling below the running median at the same step, which matters at this budget: with 20 trials the search cannot afford to run poor configurations to completion.

**Wall time per fold.** XGBoost and the MLP take roughly 30 minutes; the GNN around a day, some 50 times more, dominating the cost of any full run. A full nested run is 25 fold-model fits, so a complete GNN sweep is weeks of sequential compute against roughly half a day for XGBoost. Expect to iterate on the cheaper models and treat GNN runs as infrequent confirmations.

---

## 4. Results and interpretation

### 4.1 Model performance

| Model | R² (mean ± SD) |
|---|---|
| XGBoost | 0.470 ± 0.026 |
| MLP | 0.541 ± 0.033 |
| GNN | 0.631 ± 0.051 |

The ordering is clear and the gaps large relative to spread: roughly 0.07 between XGBoost and the MLP, 0.09 between the MLP and the GNN. Operating on the molecular graph appears to help, plausible given that the label derives from route structure and graph topology is closer to what a disconnection acts on.

> **Figure 6 description — "Outer-fold R² across the three models, 25 folds each":** three box plots (one per model: XGB, MLP, GNN), each annotated with its mean and std exactly matching the table above (XGB mean=0.470 std=0.026; MLP mean=0.541 std=0.033; GNN mean=0.631 std=0.051), with individual fold results overlaid as scatter points (25 points per box, from the 5×5 nested CV). The XGB box is tight and low, roughly 0.42–0.52. The MLP box is wider and higher, roughly 0.48–0.58. The GNN box sits highest, roughly 0.61–0.69 for most folds, but one visible outlier point falls to about 0.39, which is what inflates the GNN's standard deviation relative to its interquartile range (the text explicitly notes this: "one GNN fold falls to about 0.39 while the rest cluster between 0.61 and 0.69, so its interquartile range is in fact narrower than the MLP's and the standard deviation overstates its instability").

Variance runs the other way, with XGBoost most stable and the GNN least. But one GNN fold falls to about 0.39 while the rest cluster between 0.61 and 0.69, so its interquartile range is in fact narrower than the MLP's and the standard deviation overstates its instability.

> **Figure 7 description — "Tukey HSD multiple comparison of mean cross-validated R², FWER 0.05":** a horizontal forest/interval plot. Y-axis lists the three models (XGB at bottom, MLP in the middle, GNN at top). X-axis is "Mean CV R²", spanning roughly 0.475 to 0.650+. Each model is plotted as a point estimate with a horizontal confidence-interval bar. Per the legend: purple marks the "Best method" (GNN, positioned highest with its interval roughly 0.61–0.65), red marks "Significantly worse" (both XGB, interval roughly 0.465–0.480, and MLP, interval roughly 0.525–0.555, are shown in red since both fall significantly below GNN), and a third legend entry, "Similar" (teal), is defined but not used on any model, since no pair was found statistically indistinguishable. Dashed vertical reference lines mark the GNN's confidence bounds so the other two intervals can be visually compared against them. This is the graphical form of the Tukey HSD result described in the text.

The test separates all three: the GNN is significantly better than both and the MLP better than XGBoost, with no pair declared similar, so GNN > MLP > XGBoost holds at FWER 0.05. The choice of test follows the same protocol as the validation design, which recommends repeated measures ANOVA with post hoc Tukey HSD for pairwise model comparison (Ash et al. 2025).

The best model accounts for roughly 63% of variance in the route-derived score on unseen scaffolds, against 54% and 47%. That is useful for triage but not a substitute for scoring the route, and should be read against label noise: routes to closely related molecules are found independently, so some variance reflects the labels rather than the models.

The GNN's advantage is real but costs roughly fifty times the compute. At 0.09 R² it is worth paying for in most settings, though for high-throughput triage the MLP remains a reasonable compromise.

### 4.2 Comparison against published scores

> **Figure 8 — "Spearman correlations between the route-derived scores and six published synthesizability scores":** this is a 9×9 correlation heatmap. Since the source gives the exact numeric values, they are reproduced below as a table rather than a prose description (color scale in the original ran from −1.00, dark blue, to +1.00, dark red, with the diagonal at 1.00).

| | synthesizability | aizynthfinder_score | hac_weighted_score | sa_score | sc_score | ra_score | syba_score | gasa_es_prob | fs_score |
|---|---|---|---|---|---|---|---|---|---|
| **synthesizability** | 1.00 | 0.41 | 0.42 | -0.37 | -0.12 | 0.25 | 0.17 | 0.34 | 0.33 |
| **aizynthfinder_score** | 0.41 | 1.00 | 0.97 | -0.36 | -0.14 | 0.22 | 0.20 | 0.28 | 0.33 |
| **hac_weighted_score** | 0.42 | 0.97 | 1.00 | -0.37 | -0.15 | 0.21 | 0.20 | 0.29 | 0.35 |
| **sa_score** | -0.37 | -0.36 | -0.37 | 1.00 | 0.38 | -0.38 | 0.00 | -0.31 | -0.76 |
| **sc_score** | -0.12 | -0.14 | -0.15 | 0.38 | 1.00 | -0.01 | 0.22 | 0.00 | -0.51 |
| **ra_score** | 0.25 | 0.22 | 0.21 | -0.38 | -0.01 | 1.00 | 0.21 | 0.38 | 0.24 |
| **syba_score** | 0.17 | 0.20 | 0.20 | 0.00 | 0.22 | 0.21 | 1.00 | 0.62 | -0.01 |
| **gasa_es_prob** | 0.34 | 0.28 | 0.29 | -0.31 | 0.00 | 0.38 | 0.62 | 1.00 | 0.28 |
| **fs_score** | 0.33 | 0.33 | 0.35 | -0.76 | -0.51 | 0.24 | -0.01 | 0.28 | 1.00 |

No published score reproduces the route-derived score. Its strongest relationship is with SA score at -0.37, an agreement rather than a disagreement given the opposite orientation, followed by GASA at 0.34, FSscore at 0.33, RAscore at 0.25, SYBA at 0.17, and SCScore essentially unrelated at -0.12. None is strong enough to serve as a substitute, which is the result the approach rests on.

That SCScore is weakest fits the Background argument: it measures complexity accumulated along a published reaction corpus, the property least aligned with how many steps a modular PROTAC assembly takes.

The published scores disagree with one another. SYBA and SA score are uncorrelated at 0.00, as are SCScore and GASA, and RAscore and SCScore at -0.01. The strongest pair is SA score and FSscore at -0.76, again an agreement once orientation is accounted for. This supports the earlier claim that any single published score is a weak signal here.

The two route-derived variants (`aizynthfinder_score` and `hac_weighted_score`) are near-identical, correlating at 0.97 with their correlations against every published score agreeing to within about 0.02. The weighting changes very little in rank terms.

`synthesizability` is the composite route score used as the regression target, correlating with the two route-length scores at 0.41 and 0.42. Only moderate agreement is expected, since it is built from route geometry rather than length alone.

---

## 5. Current status and open threads

### 5.1 Known issues

**Routes cannot be regenerated in-house.** The Shallow Tree algorithm belongs to an external team. Existing routes can be rescored, but no new molecule can be added without them, which constrains most obvious extensions.

### 5.2 Suggested next steps

**Multiple routes per molecule.** The dataset holds one route per target, so the label inherits whatever a single stochastic search happened to find. Requesting several routes per molecule from the external team, under different seeds, would let the label be defined over that distribution rather than one draw: the best route found, an average across seeds, or the spread itself, which measures how reliably a molecule can be made and flags the least trustworthy labels.

The immediate gain is on unsolved cases. A target that fails under one seed but solves under another is unlucky rather than inaccessible, yet currently receives a 0 indistinguishable from genuine failure. Repeated searches separate the two and would also quantify the label noise the results section only gestures at, setting a realistic ceiling on achievable R².

Scoring needs little change, since the composite is already computed per route; the work is aggregation plus a schema allowing several routes per target. The constraint is the external dependency, so this needs scoping early.

---

## 6. Appendix: code and data

### 6.1 Repository

Source: `github.com/andreaz-u/PROTAC-Synthesizability`

Working copy on Berzelius:
`/proj/berzelius-2026-62/users/x_jzhuz/PROTAC-Synthesizability`

> **Note for whoever picks this up:** the work described here lives on the **`scoring`** branch, which has an open pull request and has **not** been merged. Check out that branch rather than `main`, and check whether the pull request has since been merged or has drifted behind.

| File | Purpose |
|---|---|
| `route_tree_score.py` | Scores route trees into the composite label |
| `route_scoring.yaml` | Scoring parameters (terms, weights, saturation) |
| `analyze_splits.ipynb` | Fragment-split quality control and filtering |

---

## References

- *A Brief Guide to Retrosynthesis in Organic Chemistry.* n.d. Synthia Online. https://www.synthiaonline.com/resources/articles/brief-guide-retrosynthesis-organic-chemistry.
- Ash, Jeremy R., Cas Wognum, Raquel Rodríguez-Pérez, et al. 2025. "Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery." *Journal of Chemical Information and Modeling* 65 (18): 9398–411. https://doi.org/10.1021/acs.jcim.5c01609.
- Coley, Connor W., Luke Rogers, William H. Green, and Klavs F. Jensen. 2018. "SCScore: Synthetic Complexity Learned from a Reaction Corpus." *Journal of Chemical Information and Modeling* 58 (2): 252–61. https://doi.org/10.1021/acs.jcim.7b00622.
- Ertl, Peter, and Ansgar Schuffenhauer. 2009. "Estimation of Synthetic Accessibility Score of Drug-Like Molecules Based on Molecular Complexity and Fragment Contributions." *Journal of Cheminformatics* 1 (1): 8. https://doi.org/10.1186/1758-2946-1-8.
- Gao, Sinuo, Xiaofei Zhou, Lu Liang, and Jianping Lin. 2026. "RetroScore: Graph Edit Distance-Guided Retrosynthesis for Accessibility Scoring with Route Metrics." *Journal of Cheminformatics* 18 (1): 10. https://doi.org/10.1186/s13321-025-01138-6.
- Kreutter, David, and Jean-Louis Reymond. 2023. "Multistep Retrosynthesis Combining a Disconnection Aware Triple Transformer Loop with a Route Penalty Score Guided Tree Search." *Chemical Science* 14 (36): 9959–69. https://doi.org/10.1039/D3SC01604H.
- Li, Junren, Lei Fang, and Jian-Guang Lou. 2024. "Retro-BLEU: Quantifying Chemical Plausibility of Retrosynthesis Routes Through Reaction Template Sequence Analysis." *Digital Discovery* 3 (3): 482–90. https://doi.org/10.1039/D3DD00219E.
- Lin, Yuquan, Minghong Gao, Suocheng Tan, et al. 2021. *ChemiRise: A Data-Driven Retrosynthesis Engine.* https://doi.org/10.48550/arXiv.2108.04682.
- Neeser, Rebecca M., Bruno Correia, and Philippe Schwaller. 2024. "FSscore: A Personalized Machine Learning-Based Synthetic Feasibility Score." *Chemistry–Methods*, ahead of print. https://doi.org/10.1002/cmtd.202400024.
- Thakkar, Amol, Veronika Chadimová, Esben Jannik Bjerrum, Ola Engkvist, and Jean-Louis Reymond. 2021. "Retrosynthetic Accessibility Score (RAscore): Rapid Machine Learned Synthesizability Classification from AI Driven Retrosynthetic Planning." *Chemical Science* 12 (9): 3339–49. https://doi.org/10.1039/D0SC05401A.
- Voršilák, Milan, Michal Kolář, Ivan Čmelo, and Daniel Svozil. 2020. "SYBA: Bayesian Estimation of Synthetic Accessibility of Organic Compounds." *Journal of Cheminformatics* 12 (1): 35. https://doi.org/10.1186/s13321-020-00439-2.
- Yu, Jiahui, Jike Wang, Hong Zhao, et al. 2022. "Organic Compound Synthetic Accessibility Prediction Based on the Graph Attention Mechanism." *Journal of Chemical Information and Modeling* 62 (12): 2973–86. https://doi.org/10.1021/acs.jcim.2c00038.
- Zhang, Ran, Shihang Wang, Lin Wang, Siyuan Tian, Yilin Tang, and Fang Bai. 2025. "DeepPSA: A Geometric Deep Learning Model for PROTAC Synthetic Accessibility Prediction." *Journal of Chemical Information and Modeling* 65 (13): 6861–73. https://doi.org/10.1021/acs.jcim.5c00366.