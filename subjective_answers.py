# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd

# ----------------- Config -----------------
events_dir = Path(r"P:\BIOSTAT\beh\events")
out_dir = Path(r"P:\BIOSTAT\beh\q_and_a")
question_prefix = "Showing"
skip_prefix = "Playing"     # if the item after a question starts with this, skip one more
out_name_tmpl = "answers_sub-{}.csv"
# ------------------------------------------

out_dir.mkdir(parents=True, exist_ok=True)

def _to_str_list(seq):
    """Coerce a sequence to a list of strings (None -> 'None')."""
    out = []
    for x in seq:
        if isinstance(x, str):
            out.append(x)
        else:
            # if dict with 'value'
            if isinstance(x, dict) and "value" in x:
                v = x["value"]
                out.append(v if isinstance(v, str) else str(v))
            else:
                out.append(str(x))
    return out

def parse_events_from_csv(p: Path):
    """
    Try to extract an ordered list of event strings from a CSV.
    Preferred: a column literally named 'value'.
    Fallback: look for a column whose cells are JSON and contain 'value'.
    """
    df = pd.read_csv(p)
    # Preferred: direct 'value' column
    if "value" in df.columns:
        return _to_str_list(df["value"].tolist())

    # Try common alternatives: 'event', 'label', 'type'
    for col in ["event", "label", "type"]:
        if col in df.columns and df[col].dtype == object:
            # If it looks like strings already, use it
            if df[col].map(lambda x: isinstance(x, str)).all():
                return _to_str_list(df[col].tolist())

    # Fallback: a column containing JSON per row (pick the first that parses)
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            parsed = []
            ok = True
            for cell in series:
                try:
                    d = json.loads(cell) if isinstance(cell, str) else cell
                except Exception:
                    ok = False
                    break
                if isinstance(d, dict) and "value" in d:
                    parsed.append(d["value"])
                else:
                    ok = False
                    break
            if ok and parsed:
                return _to_str_list(parsed)

    raise ValueError(f"Could not find a usable event 'value' in CSV: {p}")

def parse_events_from_json(p: Path):
    """
    Extract an ordered list of event strings from a JSON file.
    Accepts:
      - List[str]
      - List[dict{'value': str, ...}]
      - Dict with key 'eventStruct' -> list of {value: ...}
    """
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # List of strings
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return _to_str_list(obj)

    # List of dicts with 'value'
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj) and all("value" in x for x in obj):
        return _to_str_list([x["value"] for x in obj])

    # Dict with 'eventStruct'
    if isinstance(obj, dict):
        if "eventStruct" in obj and isinstance(obj["eventStruct"], list):
            es = obj["eventStruct"]
            if all(isinstance(x, dict) and "value" in x for x in es):
                return _to_str_list([x["value"] for x in es])

        # Sometimes JSON is { "data": [... like above ...] }
        for k in ["data", "events"]:
            if k in obj and isinstance(obj[k], list):
                inner = obj[k]
                if all(isinstance(x, str) for x in inner):
                    return _to_str_list(inner)
                if all(isinstance(x, dict) and "value" in x for x in inner):
                    return _to_str_list([x["value"] for x in inner])

    raise ValueError(f"Unsupported JSON structure: {p}")

def parse_event_values(p: Path):
    """Pick a parser based on extension; return a list[str] of event values in order."""
    ext = p.suffix.lower()
    if ext == ".csv":
        return parse_events_from_csv(p)
    if ext == ".json":
        return parse_events_from_json(p)
    # If your folder contains a different extension, add a handler here.
    raise ValueError(f"Unsupported file type '{ext}' for {p.name}")

# Gather files (non-directories)
all_files = [f for f in sorted(events_dir.iterdir()) if f.is_file()]

# Mimic MATLAB's 1-based subject indexing
for i_a, fpath in enumerate(all_files, start=1):
    try:
        values = parse_event_values(fpath)  # ordered list of strings
    except Exception as e:
        print(f"[{i_a}] {fpath.name}: could not parse events: {e}")
        continue

    # Find indices (0-based) of questions: strings starting with 'Showing'
    idx_q = [i for i, s in enumerate(values) if isinstance(s, str) and s.startswith(question_prefix)]

    if not idx_q:
        print(f"No match found for subject {i_a}. Skipping...")
        continue

    # Compute the answer index for each question
    idx_ans = []
    n = len(values)
    for q_idx in idx_q:
        cand = q_idx + 1
        if cand < n:
            nxt = values[cand]
            if isinstance(nxt, str) and nxt.startswith(skip_prefix):
                cand = q_idx + 2
        else:
            cand = None  # out of bounds

        if cand is None or cand >= n:
            idx_ans.append(None)
        else:
            idx_ans.append(cand)

    # Build Q/A rows
    questions = [values[i] for i in idx_q]
    answers = [values[a] if a is not None else "NO ANSWER FOUND" for a in idx_ans]

    df_out = pd.DataFrame({"Question": questions, "Answer": answers})

    out_path = out_dir / out_name_tmpl.format(i_a)
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[{i_a}] wrote {out_path}")
