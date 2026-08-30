"""Disk persistence for multi-start least-squares fit history.

Stores each multi-start fit run as a JSON record in a local file
(`multi_start_history.json`, next to this module), so past runs survive
app/browser restarts — unlike `st.session_state`, which resets whenever the
Streamlit process restarts or the session ends.
"""

import json
import os

import pandas as pd

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_start_history.json")


def _entry_to_record(entry):
    """Convert one in-memory history entry (with a DataFrame) to a JSON-safe dict."""
    return {
        "timestamp": entry["timestamp"],
        "fit_is_two_step": bool(entry["fit_is_two_step"]),
        "data_source": entry["data_source"],
        "n_total": int(entry["n_total"]),
        "n_converged": int(entry["n_converged"]),
        "param_names": list(entry["param_names"]),
        "derived_names": list(entry["derived_names"]),
        "true_values": entry["true_values"],
        "results_records": json.loads(entry["results_df"].to_json(orient="records")),
    }


def _record_to_entry(record):
    """Convert one on-disk JSON record back to an in-memory entry (with a real DataFrame)."""
    entry = dict(record)
    entry["results_df"] = pd.DataFrame(record["results_records"])
    del entry["results_records"]
    return entry


def _load_records():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_records(records):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_history():
    """Load all stored history entries from disk (returns [] if none exist yet).

    Each entry's `results_df` is reconstructed as a real pandas DataFrame.
    """
    return [_record_to_entry(r) for r in _load_records()]


def append_entry(entry):
    """Append one in-memory entry (with a DataFrame) to the on-disk history file."""
    records = _load_records()
    records.append(_entry_to_record(entry))
    _save_records(records)


def clear_history():
    """Delete all stored history entries."""
    _save_records([])


def delete_entry(index):
    """Delete the history entry at position `index` (0-based, oldest first)."""
    records = _load_records()
    if 0 <= index < len(records):
        del records[index]
        _save_records(records)
