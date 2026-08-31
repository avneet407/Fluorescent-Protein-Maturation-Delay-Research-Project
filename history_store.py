"""Disk persistence for run history: multi-start least-squares fits and
profile likelihood sweeps.

Each kind is stored as JSON records in its own local file next to this
module, so past runs survive app/browser restarts — unlike
`st.session_state`, which resets whenever the Streamlit process restarts or
the session ends.
"""

import json
import os

import pandas as pd

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_START_HISTORY_FILE = os.path.join(MODULE_DIR, "multi_start_history.json")
PROFILE_HISTORY_FILE = os.path.join(MODULE_DIR, "profile_likelihood_history.json")


def _load_records(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_records(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# ---------------------------------------------------------
# Multi-start fit history
# ---------------------------------------------------------

def _multi_start_entry_to_record(entry):
    """Convert one in-memory multi-start entry (with a DataFrame) to a JSON-safe dict."""
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


def _multi_start_record_to_entry(record):
    """Convert one on-disk multi-start JSON record back to an in-memory entry (with a real DataFrame)."""
    entry = dict(record)
    entry["results_df"] = pd.DataFrame(record["results_records"])
    del entry["results_records"]
    return entry


def load_multi_start_history():
    """Load all stored multi-start fit history entries from disk (returns [] if none exist yet)."""
    return [_multi_start_record_to_entry(r) for r in _load_records(MULTI_START_HISTORY_FILE)]


def append_multi_start_entry(entry):
    """Append one in-memory multi-start entry (with a DataFrame) to its on-disk history file."""
    records = _load_records(MULTI_START_HISTORY_FILE)
    records.append(_multi_start_entry_to_record(entry))
    _save_records(MULTI_START_HISTORY_FILE, records)


def clear_multi_start_history():
    """Delete all stored multi-start fit history entries."""
    _save_records(MULTI_START_HISTORY_FILE, [])


def delete_multi_start_entry(index):
    """Delete the multi-start history entry at position `index` (0-based, oldest first)."""
    records = _load_records(MULTI_START_HISTORY_FILE)
    if 0 <= index < len(records):
        del records[index]
        _save_records(MULTI_START_HISTORY_FILE, records)


# ---------------------------------------------------------
# Profile likelihood history
# ---------------------------------------------------------

def _profile_entry_to_record(entry):
    """Convert one in-memory profile likelihood entry (with a DataFrame) to a JSON-safe dict."""
    return {
        "timestamp": entry["timestamp"],
        "dataset_key": entry["dataset_key"],
        "dataset_label": entry["dataset_label"],
        "synthetic_params": entry["synthetic_params"],
        "noise_params": entry["noise_params"],
        "run_key": entry["run_key"],
        "fit_is_two_step": bool(entry["fit_is_two_step"]),
        "u_step": entry["u_step"],
        "time_start": entry["time_start"],
        "time_end": entry["time_end"],
        "baseline": entry["baseline"],
        "profile_target": entry["profile_target"],
        "is_derived": bool(entry["is_derived"]),
        "true_value": entry.get("true_value"),
        "profile_records": json.loads(entry["profile_df"].to_json(orient="records")),
    }


def _profile_record_to_entry(record):
    """Convert one on-disk profile likelihood JSON record back to an in-memory entry (with a real DataFrame)."""
    entry = dict(record)
    entry["profile_df"] = pd.DataFrame(record["profile_records"])
    del entry["profile_records"]
    return entry


def load_profile_history():
    """Load all stored profile likelihood history entries from disk (returns [] if none exist yet)."""
    return [_profile_record_to_entry(r) for r in _load_records(PROFILE_HISTORY_FILE)]


def append_profile_entry(entry):
    """Append one in-memory profile likelihood entry (with a DataFrame) to its on-disk history file."""
    records = _load_records(PROFILE_HISTORY_FILE)
    records.append(_profile_entry_to_record(entry))
    _save_records(PROFILE_HISTORY_FILE, records)


def clear_profile_history():
    """Delete all stored profile likelihood history entries."""
    _save_records(PROFILE_HISTORY_FILE, [])


def delete_profile_entry(index):
    """Delete the profile likelihood history entry at position `index` (0-based, oldest first)."""
    records = _load_records(PROFILE_HISTORY_FILE)
    if 0 <= index < len(records):
        del records[index]
        _save_records(PROFILE_HISTORY_FILE, records)
