#!/usr/bin/env python3
"""Freeze public gene-set membership and metadata-only mapping; never read responses."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.parse
import urllib.request
import zipfile

AUTHOR_COMMIT = "079639c50811dd43a206a779ab2f0199a147c98f"
SOURCES = {
    "author_cell_cycle": {
        "url": f"https://raw.githubusercontent.com/cole-trapnell-lab/sci-plex/{AUTHOR_COMMIT}/bin/cc.genes.RDS",
        "filename": "cc.genes.RDS", "sha256": "c7e83853b825d59e77055bcae140815a5b7161f47be1f3fed2e5407efe688640",
        "version": AUTHOR_COMMIT,
    },
    "reactome": {
        "url": "https://download.reactome.org/97/ReactomePathways.gmt.zip",
        "filename": "ReactomePathways.v97.gmt.zip", "sha256": "8c1dbc8578431da5d2d5118262718c60b553a9be3398e93658daa069e4a9afd4",
        "version": "97",
    },
}
PATHWAYS = {
    "R-HSA-3214815": ("HDACs deacetylate histones", 60),
    "R-HSA-3214847": ("HATs acetylate histones", 108),
    "R-HSA-2262752": ("Cellular responses to stress", 791),
}

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def fetch_cached(url, path, expected_sha=None, offline=False):
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists():
        data = path.read_bytes()
        if not receipt_path.exists():
            raise ValueError(f"Cache has no provenance receipt: {path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt["url"] != url or receipt["sha256"] != sha256(data):
            raise ValueError(f"Cache provenance mismatch: {path}")
    else:
        if offline:
            raise FileNotFoundError(f"Offline cache missing: {path}")
        request = urllib.request.Request(url, headers={"Accept": "application/json"} if "/fetch/" in url else {})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read(10_000_001)
                    if len(data) > 10_000_000:
                        raise ValueError("Response exceeds bounded small-source limit")
                    receipt = {"url": url, "retrieved_at_utc": utcnow(), "bytes": len(data),
                               "sha256": sha256(data), "http_status": response.status,
                               "content_type": response.headers.get("Content-Type"),
                               "last_modified": response.headers.get("Last-Modified")}
                if "/fetch/" in url:
                    parsed = json.loads(data)
                    if "response" not in parsed or "docs" not in parsed["response"]:
                        raise ValueError("HGNC response is not the expected JSON schema")
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Unable to fetch {url}: {exc}") from exc
                time.sleep(attempt + 1)
        if expected_sha and sha256(data) != expected_sha:
            raise ValueError(f"Pinned source hash mismatch: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".part")
        temp.write_bytes(data)
        temp.replace(path)
        atomic_json(receipt_path, receipt)
    if expected_sha and sha256(data) != expected_sha:
        raise ValueError(f"Pinned source hash mismatch: {path}")
    return data, receipt

def decode_pinned_cc_rds(data):
    """Read the two string vectors in this hash-pinned R XDR-v2 named-list file.

    A deliberately bounded decoder, not a general R object deserializer. Validate
    serialization header, list/vector types, lengths, ASCII CHAR types and names.
    No gene symbols are supplied by the implementation.
    """
    if sha256(data) != SOURCES["author_cell_cycle"]["sha256"]:
        raise ValueError("Only the inspected hash-pinned author RDS is supported")
    raw = gzip.decompress(data)
    if raw[:2] != b"X\n" or struct.unpack_from(">i", raw, 2)[0] != 2:
        raise ValueError("Unexpected R serialization header")
    pos = 14
    def integer():
        nonlocal pos
        value = struct.unpack_from(">i", raw, pos)[0]
        pos += 4
        return value
    def chars():
        nonlocal pos
        if integer() != 0x40009:
            raise ValueError("Unexpected R CHAR type/encoding")
        length = integer()
        if not 0 < length < 100:
            raise ValueError("Unexpected string length")
        value = raw[pos:pos + length].decode("ascii")
        pos += length
        return value
    def vector():
        if integer() != 16:
            raise ValueError("Expected R STRSXP vector")
        n = integer()
        if not 0 < n < 100:
            raise ValueError("Unexpected string vector length")
        return [chars() for _ in range(n)]
    if integer() != 0x213 or integer() != 2:
        raise ValueError("Expected attributed list with two marker vectors")
    first, second = vector(), vector()
    if integer() != 0x402 or integer() != 1 or chars() != "names":
        raise ValueError("Expected named-list attribute")
    names = vector()
    if names != ["s.genes", "g2m.genes"] or integer() != 254 or pos != len(raw):
        raise ValueError("Unexpected named-list trailer")
    if (len(first), len(second)) != (43, 54):
        raise ValueError("Author marker denominator changed")
    return dict(zip(names, [first, second]))

def load_genes(path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"), delimiter="\t"))
    if not rows or not {"gene_id", "gene_symbol"}.issubset(rows[0]):
        raise ValueError("Expected gene_id/gene_symbol gene metadata")
    if len({r["gene_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate original gene IDs")
    by_symbol, by_stable = collections.defaultdict(list), collections.defaultdict(list)
    for row in rows:
        by_symbol[row["gene_symbol"]].append(row["gene_id"])
        by_stable[row["gene_id"].split(".", 1)[0]].append(row["gene_id"])
    return rows, by_symbol, by_stable

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--gene-metadata", type=Path, default=root / "data/processed/gene_metadata.tsv")
    parser.add_argument("--cache", type=Path, default=root / "data/raw/biology")
    parser.add_argument("--output", type=Path, default=root / "configs/biology_v1.json")
    parser.add_argument("--offline", action="store_true", help="Use only hash-verified archived inputs")
    args = parser.parse_args()
    args.gene_metadata = args.gene_metadata.resolve()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    genes, by_symbol, by_stable = load_genes(args.gene_metadata)
    source_records, source_data = {}, {}
    for key, spec in SOURCES.items():
        data, receipt = fetch_cached(spec["url"], args.cache / spec["filename"], spec["sha256"], args.offline)
        source_data[key] = data
        source_records[key] = {**receipt, "version": spec["version"], "archive_path": str((args.cache / spec["filename"]).relative_to(root))}
    cc = decode_pinned_cc_rds(source_data["author_cell_cycle"])
    sets = [
        {"set_id": "author_s_phase", "name": "Author S-phase markers", "source": "author_cell_cycle", "use": "single_cell_description", "source_members": cc["s.genes"]},
        {"set_id": "author_g2m_phase", "name": "Author G2/M markers", "source": "author_cell_cycle", "use": "single_cell_description", "source_members": cc["g2m.genes"]},
    ]
    with zipfile.ZipFile(io.BytesIO(source_data["reactome"])) as archive:
        if archive.namelist() != ["ReactomePathways.gmt"]:
            raise ValueError("Unexpected Reactome archive members")
        gmt = archive.read("ReactomePathways.gmt").decode()
    found = {}
    for line in gmt.splitlines():
        name, set_id, *members = line.split("\t")
        if set_id in PATHWAYS:
            if set_id in found or (name, len(members)) != PATHWAYS[set_id] or len(set(members)) != len(members):
                raise ValueError(f"Unexpected or duplicate pathway definition: {set_id}")
            found[set_id] = {"set_id": set_id, "name": name, "source": "reactome", "use": "pathway_response_and_description" if set_id == "R-HSA-2262752" else "pathway_response", "source_members": members}
    if set(found) != set(PATHWAYS):
        raise ValueError("A predefined pathway is missing")
    sets.extend(found[k] for k in PATHWAYS)
    missing = sorted({s for item in sets for s in item["source_members"] if not by_symbol[s]})
    def query_hgnc(symbol):
        encoded = urllib.parse.quote(symbol, safe="")
        url = "https://rest.genenames.org/fetch/symbol/" + encoded
        path = args.cache / "hgnc" / (encoded + ".json")
        try:
            raw, receipt = fetch_cached(url, path, offline=args.offline)
            response = json.loads(raw)["response"]
            provenance = {**receipt, "archive_path": str(path.relative_to(root)), "lookup_field": "symbol"}
            if not response["docs"]:
                previous_url = "https://rest.genenames.org/fetch/prev_symbol/" + encoded
                previous_path = args.cache / "hgnc" / ("prev_symbol__" + encoded + ".json")
                previous_raw, previous_receipt = fetch_cached(previous_url, previous_path, offline=args.offline)
                response = json.loads(previous_raw)["response"]
                provenance = {**previous_receipt, "archive_path": str(previous_path.relative_to(root)),
                              "lookup_field": "prev_symbol", "prior_approved_symbol_query": provenance}
            return symbol, (response, provenance, None)
        except Exception as exc:
            return symbol, (None, {"url": url}, str(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hgnc = dict(pool.map(query_hgnc, missing))
    errors = {s: result[2] for s, result in hgnc.items() if result[2]}
    for item in sets:
        mappings = []
        for symbol in item["source_members"]:
            exact = by_symbol[symbol]
            record = {"source_symbol": symbol, "gene_id": None, "stable_gene_id": None}
            if exact:
                record.update(method="exact_source_symbol", candidates=exact,
                              status="matched" if len(exact) == 1 else "ambiguous")
                if len(exact) == 1:
                    record.update(gene_id=exact[0], stable_gene_id=exact[0].split(".", 1)[0])
            else:
                response, provenance, error = hgnc[symbol]
                record.update(method="hgnc_official_symbol_to_ensembl", hgnc_source=provenance)
                if error:
                    record.update(status="retrieval_failed", error=error)
                else:
                    docs = response["docs"]
                    record["hgnc_num_found"] = response.get("numFound", len(docs))
                    record["hgnc_records"] = [{k: d.get(k) for k in ("hgnc_id", "symbol", "ensembl_gene_id", "prev_symbol", "alias_symbol")} for d in docs]
                    exact_hgnc_match = len(docs) == 1 and (
                        docs[0].get("symbol") == symbol if provenance["lookup_field"] == "symbol"
                        else symbol in docs[0].get("prev_symbol", []))
                    if not exact_hgnc_match:
                        record["status"] = "unmatched" if not docs else "ambiguous"
                    else:
                        stable = docs[0].get("ensembl_gene_id")
                        candidates = by_stable.get(stable, []) if isinstance(stable, str) else []
                        record.update(stable_gene_id=stable, candidates=candidates,
                                      status="matched" if len(candidates) == 1 else "ambiguous" if len(candidates) > 1 else "unmatched")
                        if len(candidates) == 1:
                            record["gene_id"] = candidates[0]
            mappings.append(record)
        mapped_ids = sorted({r["gene_id"] for r in mappings if r["status"] == "matched"})
        mapped_members = sum(r["status"] == "matched" for r in mappings)
        item.update(source_member_count=len(item["source_members"]), mapping_records=mappings,
                    mapping_status_counts=dict(collections.Counter(r["status"] for r in mappings)),
                    mapped_source_member_count=mapped_members, mapped_unique_gene_count=len(mapped_ids),
                    mapped_unique_gene_ids=mapped_ids, mapped_source_member_fraction=mapped_members / len(mappings),
                    unique_gene_fraction_of_original_members=len(mapped_ids) / len(mappings),
                    selected_gene_count=None, selected_gene_fraction=None, quantitatively_evaluable=None)
    payload = {
        "schema_version": 1, "status": "FROZEN_METADATA_ONLY" if not errors else "INCOMPLETE_SOURCE_RETRIEVAL",
        "source_definition_policy": "Pinned author commit and Reactome 97; no response-dependent selection",
        "sources": source_records,
        "gene_metadata": {"path": str(args.gene_metadata.relative_to(root)), "sha256": sha256(args.gene_metadata.read_bytes()), "rows": len(genes)},
        "hgnc_policy": "Exact source symbol first; then exact HGNC approved symbol, or official exact previous symbol only when approved lookup is empty -> unique stable Ensembl ID -> unique original ID; no guessed aliases",
        "mapping_retrieval_errors": errors,
        "single_cell_score": {"formula": "log1p(10000 * sum(counts for unique mapped set genes) / sum(counts for all human genes))", "zero_library_policy": "NA; count and report", "well_summary": ["n_cells", "median", "q25", "q75", "matched_vehicle_difference"], "interpretation": "Fixed marker expression score, not a cell-phase classifier or pathway-activity measurement"},
        "pathway_readout": {"minimum_unique_selected_genes": 10, "minimum_fraction_original_source_members": 0.25,
                            "denominator": "Full original source member count, including unmapped entries; deduplicate mapped Ensembl IDs in numerator",
                            "coverage_failure": "NA, no response-driven gene rescue", "metrics": ["mean_delta", "gene_level_RMSE", "direction_agreement"],
                            "truth": "Measured delta in the same selected gene subset, never PCA-reconstructed truth",
                            "interpretation": "Transcriptional response among covered genes; not enzyme activity or validated pathway activation"},
        "sets": sets,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": payload["status"], "gene_rows": len(genes), "hgnc_queries": len(missing), "retrieval_errors": errors,
                      "sets": [{"id": x["set_id"], "members": x["source_member_count"], "mapped_unique": x["mapped_unique_gene_count"], "statuses": x["mapping_status_counts"]} for x in sets]}, indent=2))
    return 2 if errors else 0

if __name__ == "__main__":
    raise SystemExit(main())
