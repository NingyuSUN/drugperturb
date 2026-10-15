"""Create source/design/structure manifests without inspecting expression outcomes."""
from pathlib import Path
import datetime, hashlib, json
import pandas as pd
from drugperturb.design import select_design, match_controls
from drugperturb.chemistry import standardize_smiles

root = Path(__file__).resolve().parents[1]
raw = root / "data/raw"
metadata = pd.read_csv(raw/"GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz",sep="\t",dtype={"dose_pattern":str})
drugs, controls = select_design(metadata)
mapping = match_controls(drugs, controls)
mapping.to_csv(root/"manifests/control_mapping.tsv",sep="\t",index=False)
plates = set(drugs.plate_oligo)
selected_controls = controls[controls.plate_oligo.isin(plates)]
wells = pd.concat([drugs,selected_controls],ignore_index=True).sort_values("Combo")
wells.to_csv(root/"manifests/selected_wells.tsv",sep="\t",index=False)
rows=[]
for compound, group in drugs.groupby("treatment",sort=True):
    if group.SMILES.nunique()!=1:
        raise ValueError(f"Conflicting SMILES for {compound}")
    source=group.iloc[0]
    row={"compound_id":compound,"name":source["name"],"original_smiles":source.SMILES,"status":"accepted","exclusion_reason":""}
    try:
        row.update(standardize_smiles(source.SMILES))
        row["n_removed_fragments"]=len(row["removed_fragments"])
        row["removed_fragments"]=json.dumps(row["removed_fragments"])
    except ValueError as error:
        row.update(status="excluded",exclusion_reason=str(error))
    rows.append(row)
chemistry=pd.DataFrame(rows)
# Select the least modified formulation, then catalog ID, WITHOUT responses.
accepted=chemistry[chemistry.status.eq("accepted")].sort_values(["n_removed_fragments","compound_id"])
for _,group in accepted.groupby("canonical_smiles",sort=True):
    if len(group)>1:
        representative=group.iloc[0].compound_id
        for index in group.index[1:]:
            chemistry.loc[index,"status"]="excluded"
            chemistry.loc[index,"exclusion_reason"]="duplicate_parent_represented_by:"+representative
chemistry.to_csv(root/"manifests/chemistry_manifest.tsv",sep="\t",index=False)
manifest=[]
for file in sorted(raw.glob("*.gz")):
    digest=hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    manifest.append({"filename":file.name,"url":"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/"+file.name,"bytes":file.stat().st_size,"sha256":digest.hexdigest(),"verified_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"public_source":True})
pd.DataFrame(manifest).to_csv(root/"manifests/source_manifest.tsv",sep="\t",index=False)
receipt={"metadata_wells":len(metadata),"drug_wells":len(drugs),"treatment_labels":drugs.treatment.nunique(),"relevant_plates":len(plates),"strict_control_wells":mapping.control_Combo.nunique(),"same_plate_vehicle_wells":len(selected_controls),"selected_wells":len(wells),"valid_structure_labels_before_parent_dedup":len(accepted),"unique_parent_representatives":int(chemistry.status.eq("accepted").sum()),"exclusions":chemistry.loc[chemistry.status.eq("excluded"),["compound_id","name","exclusion_reason"]].to_dict("records"),"expression_outcomes_inspected":False,"final_cell_QC_done":False,"models_trained":False}
(root/"receipts/design_qualification.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
