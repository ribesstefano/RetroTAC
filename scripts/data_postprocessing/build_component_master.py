import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

# ─────────────────────────────────────────────────────────────────────────────
# ── Build component master table
# input : component_capped.csv, get_score(AiZyntFinder output results) dir,
#         component_vendor.csv
# output: component_master.csv with columns:
    # "component_id",  # component_with_dummy_smiles id
    # "component", # warhead, linker, e3_ligase_ligand
    # "component_smiles_with_dummy",
    # "cap_type",
    # "cap_smiles",
    # "cap_smiles_canon",
    # "aizynth_score",
    # "aizynth_search_time",
    # "aizynth_steps",
    # "aizynth_solved_tag", # if all precursors in stock, aizynth_solved_tag == 1, else 0
    # "hac_score",
    # "pubchem_cid",
    # "pubchem_vendor_tag",# if vendor_statu ==1 means pubchem has a vendor,else 0
    # "final_solved_tag", # either aizynth_solved_tag ==1 or pubchem_vendor_tag==1, else 0
#         
# ─────────────────────────────────────────────────────────────────────────────


capped_csv = "/mimer/NOBACKUP/groups/naiss2023-6-290/tingtingmo/data/01_components_capping/component_capped.csv"

results_dir = Path("/mimer/NOBACKUP/groups/naiss2023-6-290/tingtingmo/get_score_fullstock0515")

cid_vendor_csv = "/mimer/NOBACKUP/groups/naiss2023-6-290/tingtingmo/data/03_pubchem_vendor/component_check_cid_vendor.csv"

output_csv = "/mimer/NOBACKUP/groups/naiss2023-6-290/tingtingmo/data/04_get_score/component_master.csv"

# Capping map
single_cap_map = { # warhead, e3 ligase ligand
    "H": "H",
    "CH3": "CH3",
    "OH": "OH",
    "NH2": "NH2",
    "COOH": "COOH",
    "=O": "eqO",
}


linker_cap_map = {
    "CH3|H": "CH3-H",
    "CH3|CH3": "CH3-CH3",
    "CH3|OH": "CH3-OH",
    "CH3|NH2": "CH3-NH2",
    "CH3|COOH": "CH3-COOH",
    "CH3|=O": "CH3-eqO",

    "OH|H": "OH-H",
    "OH|CH3": "OH-CH3",
    "OH|OH": "OH-OH",
    "OH|NH2": "OH-NH2",
    "OH|COOH": "OH-COOH",
    "OH|=O": "OH-eqO",

    "NH2|H": "NH2-H",
    "NH2|CH3": "NH2-CH3",
    "NH2|OH": "NH2-OH",
    "NH2|NH2": "NH2-NH2",
    "NH2|COOH": "NH2-COOH",
    "NH2|=O": "NH2-eqO",

    "COOH|H": "COOH-H",
    "COOH|CH3": "COOH-CH3",
    "COOH|OH": "COOH-OH",
    "COOH|NH2": "COOH-NH2",
    "COOH|COOH": "COOH-COOH",
    "COOH|=O": "COOH-eqO",

    "=O|H": "eqO-H",
    "=O|CH3": "eqO-CH3",
    "=O|OH": "eqO-OH",
    "=O|NH2": "eqO-NH2",
    "=O|COOH": "eqO-COOH",
    "=O|=O": "eqO-eqO",
}

component_cap_maps = {
    "warhead": single_cap_map,
    "e3": single_cap_map,
    "linker": linker_cap_map,
}


def clean_str(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def compute_hac_weighted_score(precursor_rows, route_length=None, alpha=0.95, max_route_length=20):
    hac_total = 0
    hac_solved = 0

    for _, row in precursor_rows.iterrows():
        mol = Chem.MolFromSmiles(str(row["precursor"]))
        if mol is None:
            continue

        hac = Descriptors.HeavyAtomCount(mol)
        hac_total += hac

        if bool(row["in_stock"]):
            hac_solved += hac

    if hac_total == 0:
        return 0.0

    availability = hac_solved / hac_total

    if route_length is None:
        return availability

    penalty = 1 - (route_length / max_route_length)
    return alpha * availability + (1 - alpha) * penalty


# get main table
base_df = pd.read_csv(capped_csv)

base_df = base_df[
    base_df["cap_smiles_canon"].notna()
].copy()

base_df = base_df[
    [
        "component_id",
        "component",
        "component_smiles_with_dummy",
        "cap_type",
        "cap_smiles",
        "cap_smiles_canon",
    ]
].copy()


# get summary
summary_all = []

for component, cap_map in component_cap_maps.items():
    for cap_type, file_cap in cap_map.items():

        summary_file = results_dir / f"{component}_cap_type_{file_cap}_results_summary.csv"

        if not summary_file.exists():
            print(f"Missing: {summary_file.name}")
            continue

        df = pd.read_csv(summary_file)
        df = clean_str(df, ["molecule"])

        df = df[[
            "component_id",
            "molecule",
            "aizynthfinder_score",
            "search_time",
            "steps",
            "precursors",
            "in_stock",
        ]].copy()

        df = df.rename(columns={
            "aizynthfinder_score": "aizynth_score",
            "search_time": "aizynth_search_time",
            "steps": "aizynth_steps",
        })

        df["aizynth_solved_tag"] = (
            pd.to_numeric(df["precursors"], errors="coerce").fillna(0) ==
            pd.to_numeric(df["in_stock"], errors="coerce").fillna(-1)
        ).astype(int)

        df["cap_type"] = cap_type

        summary_all.append(df)

summary_df = pd.concat(summary_all, ignore_index=True)


# compute hac_score
hac_rows = []

for component, cap_map in component_cap_maps.items():
    for cap_type, file_cap in cap_map.items():

        prec_file = results_dir / f"{component}_cap_type_{file_cap}_results_precursors.csv"
        steps_file = results_dir / f"{component}_cap_type_{file_cap}_results_steps.csv"

        if not prec_file.exists():
            print(f"Missing: {prec_file.name}")
            continue

        prec_df = pd.read_csv(prec_file)
        prec_df = clean_str(prec_df, ["molecule", "precursor"])

        step_counts = {}
        if steps_file.exists():
            steps_df = pd.read_csv(steps_file)
            step_counts = steps_df.groupby("molecule").size().to_dict()

        # BUG FIX 1: group by both component_id and molecule so component_id
        # is available inside the loop and correctly captured per row
        for (component_id, mol), group in prec_df.groupby(["component_id", "molecule"]):
            route_len = step_counts.get(mol, 0)
            hac = compute_hac_weighted_score(group, route_length=route_len)

            hac_rows.append({
                "component_id": component_id,  # now correctly defined
                "cap_type": cap_type,
                "molecule": mol,
                "hac_score": hac,
            })

# BUG FIX 2: moved outside both for-loops so all cap_types are included
hac_df = pd.DataFrame(hac_rows).drop_duplicates()


# merge summary and hac into base_df on component_id + cap_type
# (component_id uniquely identifies a component_smiles_with_dummy,
#  and cap_type distinguishes which capped variant was scored)
merged = base_df.merge(
    summary_df.drop(columns=["molecule"]),  # BUG FIX 3: drop molecule (=cap_smiles_canon)
    on=["component_id", "cap_type"],        # since base_df already has cap_smiles_canon
    how="left"
)

# merge hac score
merged = merged.merge(
    hac_df.drop(columns=["molecule"]),  # same: drop redundant molecule column
    on=["component_id", "cap_type"],
    how="left"
)

# merge CID + vendor status
cid_vendor_df = pd.read_csv(cid_vendor_csv)
cid_vendor_df = clean_str(cid_vendor_df, ["cap_smiles_canon"])

# support either CID or cid column
if "CID" in cid_vendor_df.columns:
    cid_col = "CID"
elif "cid" in cid_vendor_df.columns:
    cid_col = "cid"
else:
    raise ValueError("No CID column found in component_check_cid_vendor.csv")

if "vendor_status" not in cid_vendor_df.columns:
    raise ValueError("No vendor_status column found in component_check_cid_vendor.csv")

cid_vendor_df = cid_vendor_df[["component_id", cid_col, "vendor_status"]].copy()
cid_vendor_df = cid_vendor_df.drop_duplicates(subset=["component_id"])
cid_vendor_df = cid_vendor_df.rename(columns={cid_col: "pubchem_cid"})

merged = merged.merge(cid_vendor_df, on="component_id", how="left")

# fill missing CID
merged["pubchem_cid"] = pd.to_numeric(merged["pubchem_cid"], errors="coerce").fillna(0).astype(int)

# convert vendor_status -> 0/1
merged["pubchem_vendor_tag"] = merged["vendor_status"].notna().astype(int)
merged = merged.drop(columns=["vendor_status"])

print("Rows with CID:", (merged["pubchem_cid"] != 0).sum())
print("Rows with vendor:", merged["pubchem_vendor_tag"].sum())

merged["aizynth_solved_tag"] = pd.to_numeric(
    merged["aizynth_solved_tag"], errors="coerce"
).fillna(0).astype(int)

merged["final_solved_tag"] = (
    (merged["aizynth_solved_tag"] == 1) | (merged["pubchem_vendor_tag"] == 1)
).astype(int)

print("Rows with aizynth_solved_tag=1:", merged["aizynth_solved_tag"].sum())
print("Rows with pubchem_vendor_tag=1:", merged["pubchem_vendor_tag"].sum())
print("Rows with final_solved_tag=1:", merged["final_solved_tag"].sum())


# clean
for col in ["aizynth_score", "aizynth_search_time", "aizynth_steps", "hac_score"]:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")


# get final table
final_cols = [
    "component_id",
    "component",
    "component_smiles_with_dummy",
    "cap_type",
    "cap_smiles",
    "cap_smiles_canon",
    "aizynth_score",
    "aizynth_search_time",
    "aizynth_steps",
    "aizynth_solved_tag",
    "hac_score",
    "pubchem_cid",
    "pubchem_vendor_tag",
    "final_solved_tag",
]

# remove component with the same cap_smiles
merged = merged[final_cols].drop_duplicates(subset=["component_id", "cap_smiles_canon"])

merged.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print("Rows:", len(merged))