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
BLEACH_HISTORY_FILE = os.path.join(MODULE_DIR, "bleach_fit_history.json")


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
    """Convert one in-memory profile likelihood entry (with a DataFrame) to a JSON-safe dict.

    `profile_type` distinguishes a single-quantity sweep ("1d", the default)
    from a joint (a, b) sweep ("2d"); the latter uses `true_a`/`true_b`
    instead of `true_value`.
    """
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
        "profile_type": entry.get("profile_type", "1d"),
        "profile_target": entry["profile_target"],
        "is_derived": bool(entry.get("is_derived", False)),
        "true_value": entry.get("true_value"),
        "true_a": entry.get("true_a"),
        "true_b": entry.get("true_b"),
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


# ---------------------------------------------------------
# Bleaching tab fit history (Bleaching Only Simulation tab)
# ---------------------------------------------------------
#
# Each entry bundles everything run in the Bleaching Only tab up to the
# moment the known-b fit's "Run Multi-Start Fit" button was clicked: the
# bleach-only fit ("bleach_only": M0, kb, alpha against a pure-decay trace,
# kd fixed at 0) and the known bleaching pole fit ("known_b": the full
# maturation model with b = kb + kd fixed and kd fixed at 0). `bleach_only`
# is whatever was last run this session, or None if it hadn't been run at
# all. `known_b`'s `extra_info` carries fit-specific context (which
# maturation model, the fixed b value). Older entries (saved before Bode
# plots were removed from the tab) may still carry "bleach_bode"/
# "known_b_bode" keys; these are preserved on disk but no longer read.

def _bleach_part_to_record(part):
    """Convert one in-memory fit-result part (with a DataFrame) to a JSON-safe dict, or None."""
    if part is None:
        return None
    return {
        "n_total": int(part["n_total"]),
        "n_converged": int(part["n_converged"]),
        "param_names": list(part["param_names"]),
        "derived_names": list(part["derived_names"]),
        "true_values": part["true_values"],
        "extra_info": part.get("extra_info", {}),
        "results_records": json.loads(part["results_df"].to_json(orient="records")),
    }


def _bleach_part_to_entry(record):
    """Convert one on-disk fit-result part back to an in-memory part (with a real DataFrame), or None."""
    if record is None:
        return None
    part = dict(record)
    part["results_df"] = pd.DataFrame(record["results_records"])
    del part["results_records"]
    return part


def _bleach_bode_to_record(part):
    """Convert one in-memory Bode-plot part (plain floats/lists already) to a JSON-safe dict, or None.

    `cutoff_syn`/`cutoff_fit` may themselves be None (the 2-step model's
    analytical cutoff has no closed form for every parameter combination).
    """
    if part is None:
        return None
    return {
        "w": list(part["w"]),
        "mag_syn": list(part["mag_syn"]),
        "phase_syn": list(part["phase_syn"]),
        "mag_fit": list(part["mag_fit"]),
        "phase_fit": list(part["phase_fit"]),
        "fit_label": part["fit_label"],
        "cutoff_syn": None if part["cutoff_syn"] is None else float(part["cutoff_syn"]),
        "cutoff_fit": None if part["cutoff_fit"] is None else float(part["cutoff_fit"]),
        "formula_latex": part.get("formula_latex"),
    }


def _bleach_bode_to_entry(record):
    """Convert one on-disk Bode-plot record back to an in-memory part, or None."""
    if record is None:
        return None
    return dict(record)


def _bleach_entry_to_record(entry):
    """Convert one in-memory bleaching-tab history entry to a JSON-safe dict."""
    return {
        "timestamp": entry["timestamp"],
        "bleach_only": _bleach_part_to_record(entry.get("bleach_only")),
        "bleach_bode": _bleach_bode_to_record(entry.get("bleach_bode")),
        "known_b": _bleach_part_to_record(entry.get("known_b")),
        "known_b_bode": _bleach_bode_to_record(entry.get("known_b_bode")),
    }


def _bleach_record_to_entry(record):
    """Convert one on-disk bleaching-tab JSON record back to an in-memory history entry."""
    return {
        "timestamp": record["timestamp"],
        "bleach_only": _bleach_part_to_entry(record.get("bleach_only")),
        "bleach_bode": _bleach_bode_to_entry(record.get("bleach_bode")),
        "known_b": _bleach_part_to_entry(record.get("known_b")),
        "known_b_bode": _bleach_bode_to_entry(record.get("known_b_bode")),
    }


def load_bleach_history():
    """Load all stored bleaching-tab fit history entries from disk (returns [] if none exist yet)."""
    return [_bleach_record_to_entry(r) for r in _load_records(BLEACH_HISTORY_FILE)]


def append_bleach_entry(entry):
    """Append one in-memory bleaching-tab fit entry (with a DataFrame) to its on-disk history file."""
    records = _load_records(BLEACH_HISTORY_FILE)
    records.append(_bleach_entry_to_record(entry))
    _save_records(BLEACH_HISTORY_FILE, records)


def clear_bleach_history():
    """Delete all stored bleaching-tab fit history entries."""
    _save_records(BLEACH_HISTORY_FILE, [])


def delete_bleach_entry(index):
    """Delete the bleaching-tab fit history entry at position `index` (0-based, oldest first)."""
    records = _load_records(BLEACH_HISTORY_FILE)
    if 0 <= index < len(records):
        del records[index]
        _save_records(BLEACH_HISTORY_FILE, records)
