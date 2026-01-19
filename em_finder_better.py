#!/usr/bin/env python3
"""
em_finder_better.py

Better Em/pH/T extraction for a UniProt list:
1) Pull UniProt JSON:
   - recommended protein name, gene, organism, PDB IDs
   - try to extract UniProt "Redox potential" annotation (FASTEST when present)
2) If no UniProt redox potential, search Europe PMC by NAME/GENE/ORG/PDB IDs
3) Extract Em candidates only when strong midpoint/redox context exists
4) Extract pH and temperature from local context and whole text fallback

Run:
  python3 em_finder_better.py --input uniprot_ids.txt --out em_results.csv --workers 16 --sleep 0.2 --stop-after 2

Notes:
- Non-open-access papers often provide only abstracts: pH/temp may remain blank.
- This aims for higher recall than searching the accession itself.
"""

import argparse
import concurrent.futures
import csv
import html
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests


# ----------------------------
# Config / endpoints
# ----------------------------

UA = {"User-Agent": "em-finder-better/2.0"}

UNIPROT_JSON = "https://rest.uniprot.org/uniprotkb/{}.json"

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_ARTICLE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EPMC_LANDING = "https://europepmc.org/article/{}/{}"


# ----------------------------
# Patterns: Em + context scoring
# ----------------------------

MIDPOINT_WORDS = re.compile(
    r"(?is)\b("
    r"midpoint|mid-point|reduction potential|redox potential|"
    r"potentiometr|redox titrat|"
    r"E0'|E°'|E0\s*'|Em\b|E1/2|E\s*=\s*"
    r")\b"
)

FLAVIN_WORDS = re.compile(r"(?is)\b(FAD|FMN|flavin|flavoprotein|flavodoxin|isoalloxazine)\b")
APPLIED_WORDS = re.compile(r"(?is)\b(appl(?:y|ied)|held at|poised at|polariz(?:ed|ation)|bias)\b")

PH_ANY_PATTERN = re.compile(r"(?is)\bpH\s*([0-9]{1,2}(?:\.[0-9]+)?)\b")
TEMP_C_PATTERN = re.compile(r"(?is)\b([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:°\s*C|degrees?\s*C|deg(?:ree)?s?\s*C)\b")
TEMP_K_PATTERN = re.compile(r"(?is)\b([0-9]{2,3}(?:\.[0-9]+)?)\s*K\b")
ROOM_TEMP_PATTERN = re.compile(r"(?is)\broom temperature\b")

NUM_UNIT_PATTERN = re.compile(
    r"(?is)"
    r"(?P<sign>[+\-−–]?)\s*"
    r"(?P<num>\d+(?:\.\d+)?)"
    r"(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?"
    r"\s*(?P<unit>mV|millivolts?|V|volts?)\b"
)

_TAG_RE = re.compile(r"(?s)<[^>]*>")


# ----------------------------
# Output schema
# ----------------------------

@dataclass
class EmRow:
    uniprot_id: str
    source: str              # "uniprot" or "europepmc"
    pmid: str
    year: str
    title: str
    normalized_mV: str
    raw_value: str
    unit: str
    pH: str
    temperature_C: str
    is_open_access: str
    epmc_url: str
    context: str
    status: str              # hit / no_hit_found


# ----------------------------
# Utility
# ----------------------------

def strip_xml_tags(s: str) -> str:
    return _TAG_RE.sub(" ", s)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def to_ascii_minus(s: str) -> str:
    return s.replace("−", "-").replace("–", "-")

def safe_float(x: str) -> Optional[float]:
    try:
        return float(to_ascii_minus(x))
    except Exception:
        return None

def context_window(text: str, start: int, end: int, window: int = 260) -> str:
    a = max(0, start - window)
    b = min(len(text), end + window)
    return normalize_ws(text[a:b])

def score_candidate(snippet: str) -> int:
    s = 0
    if MIDPOINT_WORDS.search(snippet):
        s += 5
    if FLAVIN_WORDS.search(snippet):
        s += 2
    if APPLIED_WORDS.search(snippet):
        s -= 5
    return s

def normalize_to_mV(value: float, unit: str) -> Optional[float]:
    u = unit.lower()
    if "mv" in u or "milliv" in u:
        return value
    if u.startswith("v") or "volt" in u:
        if abs(value) > 1.0:   # reject weird "5 V" type stuff
            return None
        return value * 1000.0
    return None

def sanity_ok(nmv: float) -> bool:
    return abs(nmv) <= 1200

def extract_local_ph_temp(snippet: str) -> Tuple[str, str]:
    ph = ""
    tC = ""
    m = PH_ANY_PATTERN.search(snippet)
    if m:
        ph = m.group(1)

    m = TEMP_C_PATTERN.search(snippet)
    if m:
        tC = m.group(1)
    else:
        mk = TEMP_K_PATTERN.search(snippet)
        if mk:
            try:
                k = float(mk.group(1))
                tC = f"{k - 273.15:.2f}"
            except Exception:
                pass
        elif ROOM_TEMP_PATTERN.search(snippet):
            tC = "25"

    return ph, tC

def extract_global_ph_temp(full_text: str) -> Tuple[str, str]:
    if not full_text:
        return "", ""

    ph_candidates: List[Tuple[int, int, str]] = []
    for m in PH_ANY_PATTERN.finditer(full_text):
        win = full_text[max(0, m.start()-300): min(len(full_text), m.end()+300)]
        sc = 2 if MIDPOINT_WORDS.search(win) else 0
        ph_candidates.append((sc, m.start(), m.group(1)))
    ph_candidates.sort(reverse=True)
    best_ph = ph_candidates[0][2] if ph_candidates else ""

    temp_candidates: List[Tuple[int, int, str]] = []
    for m in TEMP_C_PATTERN.finditer(full_text):
        win = full_text[max(0, m.start()-300): min(len(full_text), m.end()+300)]
        sc = 2 if MIDPOINT_WORDS.search(win) else 1
        temp_candidates.append((sc, m.start(), m.group(1)))

    for m in TEMP_K_PATTERN.finditer(full_text):
        try:
            k = float(m.group(1))
            c = k - 273.15
            win = full_text[max(0, m.start()-300): min(len(full_text), m.end()+300)]
            sc = 1 if MIDPOINT_WORDS.search(win) else 0
            temp_candidates.append((sc, m.start(), f"{c:.2f}"))
        except Exception:
            pass

    if not temp_candidates and ROOM_TEMP_PATTERN.search(full_text):
        temp_candidates.append((0, 0, "25"))

    temp_candidates.sort(reverse=True)
    best_temp = temp_candidates[0][2] if temp_candidates else ""
    return best_ph, best_temp


# ----------------------------
# UniProt: pull names, organism, PDB IDs, and redox potential annotation
# ----------------------------

def uniprot_fetch(uid: str, timeout: int = 20) -> Optional[Dict]:
    try:
        r = requests.get(UNIPROT_JSON.format(uid), headers=UA, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def uniprot_extract_metadata(u: Dict) -> Tuple[str, str, str, List[str]]:
    # recommended protein name
    pname = ""
    try:
        pd = u.get("proteinDescription", {})
        rec = pd.get("recommendedName", {})
        fname = rec.get("fullName", {})
        pname = fname.get("value", "") if isinstance(fname, dict) else (fname or "")
    except Exception:
        pass

    # gene
    gene = ""
    try:
        genes = u.get("genes", []) or []
        if genes:
            g = genes[0].get("geneName", {})
            gene = g.get("value", "") if isinstance(g, dict) else ""
    except Exception:
        pass

    # organism
    org = ""
    try:
        org = (u.get("organism", {}) or {}).get("scientificName", "") or ""
    except Exception:
        pass

    # PDB IDs
    pdbs: List[str] = []
    try:
        xrefs = u.get("uniProtKBCrossReferences", []) or []
        for ref in xrefs:
            if ref.get("database") == "PDB" and ref.get("id"):
                pdbs.append(str(ref["id"]).upper())
    except Exception:
        pass

    # de-dup preserve order
    seen = set()
    out = []
    for p in pdbs:
        if p not in seen:
            out.append(p)
            seen.add(p)

    return pname, gene, org, out

def uniprot_extract_redox_potential(u: Dict) -> List[Tuple[float, str]]:
    """
    Tries hard to find UniProt redox potential comments.
    Returns list of (mV, raw_text) candidates.
    """
    out: List[Tuple[float, str]] = []
    comments = u.get("comments", []) or []

    # UniProt JSON schema varies; this searches any comment blob for numbers + mV near "redox potential"
    blob = normalize_ws(str(comments))
    blob = to_ascii_minus(blob)

    # Very strict: only accept values with unit
    for m in NUM_UNIT_PATTERN.finditer(blob):
        unit = m.group("unit")
        raw = safe_float((m.group("sign") or "") + m.group("num"))
        if raw is None:
            continue
        nmv = normalize_to_mV(raw, unit)
        if nmv is None or not sanity_ok(nmv):
            continue

        # require "redox potential" somewhere nearby in the blob (this is coarse but works)
        near = blob[max(0, m.start()-200): min(len(blob), m.end()+200)]
        if "redox" not in near.lower() and "midpoint" not in near.lower() and "potential" not in near.lower():
            continue

        out.append((nmv, normalize_ws(m.group(0))))

    # de-dup by value
    uniq = {}
    for nmv, rawtxt in out:
        key = round(nmv, 3)
        if key not in uniq:
            uniq[key] = rawtxt
    return [(k, uniq[k]) for k in sorted(uniq.keys())]


# ----------------------------
# Europe PMC: search + fulltext for OA
# ----------------------------

def epmc_search(query: str, page_size: int = 25, timeout: int = 20) -> Dict:
    params = {"query": query, "format": "json", "pageSize": str(page_size)}
    r = requests.get(EPMC_SEARCH, params=params, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r.json()

def epmc_get_fulltext_xml(source: str, ident: str, timeout: int = 25) -> Optional[str]:
    url = f"{EPMC_ARTICLE}/{source}/{ident}/fullTextXML"
    r = requests.get(url, timeout=timeout, headers=UA)
    if r.status_code != 200:
        return None
    return r.text

def build_queries(pname: str, gene: str, org: str, pdbs: List[str]) -> List[str]:
    """
    Build a few progressively broader queries.
    Europe PMC search supports boolean operators.
    """
    # Keep queries reasonably short; long queries can reduce recall
    name_q = f'"{pname}"' if pname else ""
    gene_q = f'"{gene}"' if gene else ""
    org_q = f'"{org}"' if org else ""
    pdb_q = " OR ".join([f'"{p}"' for p in pdbs[:3]])  # only top 3 to avoid bloating

    redox_core = '(midpoint OR "redox potential" OR "reduction potential" OR potentiometr* OR "redox titrat*" OR Em OR "E0\'")'

    queries = []
    if name_q and org_q:
        queries.append(f"({name_q} AND {org_q}) AND {redox_core}")
    if gene_q and org_q:
        queries.append(f"({gene_q} AND {org_q}) AND {redox_core}")
    if name_q:
        queries.append(f"{name_q} AND {redox_core}")
    if gene_q:
        queries.append(f"{gene_q} AND {redox_core}")
    if pdb_q:
        queries.append(f"({pdb_q}) AND {redox_core}")

    # final broad fallback: just redox_core with org (risky, huge)
    if org_q:
        queries.append(f"{org_q} AND {redox_core}")

    # de-dup
    seen = set()
    out = []
    for q in queries:
        qn = normalize_ws(q)
        if qn and qn not in seen:
            out.append(qn)
            seen.add(qn)
    return out

def extract_em_candidates(full_text: str) -> List[Tuple[float, float, str, str, int]]:
    """
    Returns list of (nmv, raw, unit, snippet, score)
    """
    t = to_ascii_minus(full_text)
    candidates: List[Tuple[float, float, str, str, int]] = []

    for m in NUM_UNIT_PATTERN.finditer(t):
        raw = safe_float((m.group("sign") or "") + m.group("num"))
        if raw is None:
            continue
        unit = m.group("unit")
        nmv = normalize_to_mV(raw, unit)
        if nmv is None or not sanity_ok(nmv):
            continue

        snippet = context_window(t, m.start(), m.end())
        sc = score_candidate(snippet)

        # strong gating: needs midpoint/redox words; avoid applied potentials
        if sc < 5:
            continue

        candidates.append((nmv, raw, unit, snippet, sc))

    candidates.sort(key=lambda x: (x[4], abs(x[0])), reverse=True)
    return candidates

def search_epmc_for_uid(uid: str, pname: str, gene: str, org: str, pdbs: List[str],
                        stop_after: int, sleep_s: float) -> List[EmRow]:
    rows: List[EmRow] = []
    queries = build_queries(pname, gene, org, pdbs)
    if not queries:
        return rows

    for q in queries:
        if sleep_s:
            time.sleep(sleep_s)
        try:
            data = epmc_search(q, page_size=25)
        except Exception:
            continue

        results = data.get("resultList", {}).get("result", []) or []
        for rec in results:
            pmid = rec.get("pmid") or ""
            source = rec.get("source") or ""
            ident = rec.get("id") or ""
            year = str(rec.get("pubYear") or "")
            title = rec.get("title") or ""
            is_oa = (rec.get("isOpenAccess") or "N").upper()
            abstract = rec.get("abstractText") or ""

            text_parts = [abstract]

            # add OA full text if possible
            if is_oa == "Y" and source and ident:
                try:
                    xml = epmc_get_fulltext_xml(source, ident)
                    if xml:
                        text_parts.append(strip_xml_tags(xml))
                except Exception:
                    pass

            full_text = normalize_ws(html.unescape(" ".join([p for p in text_parts if p])))

            if not full_text:
                continue

            cands = extract_em_candidates(full_text)
            if not cands:
                continue

            gph, gtemp = extract_global_ph_temp(full_text)

            url = EPMC_LANDING.format(source if source else "MED", ident if ident else pmid)

            for nmv, raw, unit, snippet, sc in cands[: max(1, stop_after)]:
                lph, ltemp = extract_local_ph_temp(snippet)
                ph = lph or gph
                tempC = ltemp or gtemp

                rows.append(
                    EmRow(
                        uniprot_id=uid,
                        source="europepmc",
                        pmid=pmid,
                        year=year,
                        title=title,
                        normalized_mV=f"{nmv:.3f}",
                        raw_value=f"{raw}",
                        unit=unit,
                        pH=ph,
                        temperature_C=tempC,
                        is_open_access=is_oa,
                        epmc_url=url,
                        context=(snippet[:497] + "...") if len(snippet) > 500 else snippet,
                        status="hit",
                    )
                )

                if stop_after and len(rows) >= stop_after:
                    return rows

        # if we already got something from a tighter query, stop early
        if rows:
            return rows

    return rows


# ----------------------------
# IO helpers
# ----------------------------

def read_uniprot_ids(path: str) -> List[str]:
    ids = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("["):
                continue
            ids.append(ln.split()[-1])
    # de-dup preserve order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def write_csv(path: str, rows: List[EmRow]) -> None:
    fieldnames = [
        "uniprot_id","source","pmid","year","title",
        "normalized_mV","raw_value","unit",
        "pH","temperature_C",
        "is_open_access","epmc_url","context","status"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            w.writerow({k: d.get(k, "") for k in fieldnames})


# ----------------------------
# Per UID worker
# ----------------------------

def process_uid(uid: str, stop_after: int, sleep_s: float) -> List[EmRow]:
    rows: List[EmRow] = []

    u = uniprot_fetch(uid)
    if not u:
        return [EmRow(uid, "uniprot", "", "", "", "", "", "", "", "", "", "", "", "no_hit_found")]

    pname, gene, org, pdbs = uniprot_extract_metadata(u)

    # 1) UniProt redox potential annotations
    uni_cands = uniprot_extract_redox_potential(u)
    for nmv, rawtxt in uni_cands[: max(1, stop_after)]:
        rows.append(
            EmRow(
                uniprot_id=uid,
                source="uniprot",
                pmid="",
                year="",
                title=f"{pname} ({org})".strip(),
                normalized_mV=f"{nmv:.3f}",
                raw_value=rawtxt,
                unit="mV",
                pH="",
                temperature_C="",
                is_open_access="",
                epmc_url=f"https://www.uniprot.org/uniprotkb/{uid}/entry",
                context=f"UniProt redox potential annotation near: {rawtxt}",
                status="hit",
            )
        )

    if rows:
        return rows

    # 2) Europe PMC by name/gene/org/pdbs
    epmc_rows = search_epmc_for_uid(uid, pname, gene, org, pdbs, stop_after=stop_after, sleep_s=sleep_s)
    if epmc_rows:
        return epmc_rows

    return [EmRow(uid, "europepmc", "", "", f"{pname} ({org})".strip(), "", "", "", "", "", "", "", "", "no_hit_found")]


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File with UniProt IDs, one per line.")
    ap.add_argument("--out", default="em_results.csv", help="Output CSV path.")
    ap.add_argument("--workers", type=int, default=16, help="Thread workers.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds before each Europe PMC query.")
    ap.add_argument("--stop-after", type=int, default=2, help="Max hits per UniProt (2 is a good start).")
    args = ap.parse_args()

    uids = read_uniprot_ids(args.input)
    if not uids:
        print("No UniProt IDs found in input.", file=sys.stderr)
        sys.exit(1)

    all_rows: List[EmRow] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_uid, uid, args.stop_after, args.sleep): uid for uid in uids}
        for fut in concurrent.futures.as_completed(futs):
            try:
                rows = fut.result()
            except Exception:
                rows = []
            all_rows.extend(rows)

    write_csv(args.out, all_rows)
    hits = sum(1 for r in all_rows if r.status == "hit")
    print(f"Wrote {len(all_rows)} rows to {args.out}. hits={hits}, no_hit={len(all_rows)-hits}")

if __name__ == "__main__":
    main()
