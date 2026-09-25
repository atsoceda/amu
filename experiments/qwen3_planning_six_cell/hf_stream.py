"""Re-export of the canonical implementation in skills/hf-range-streaming/scripts/hf_stream.py.

Kept so existing imports (`from hf_stream import ...`) keep working; edit the skill copy, not this file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "skills/hf-range-streaming/scripts/hf_stream.py"
_spec = importlib.util.spec_from_file_location("hf_range_streaming", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HF = _mod.HF
url = _mod.url
fetch_range = _mod.fetch_range
SafetensorsRemote = _mod.SafetensorsRemote
load_feature_index = _mod.load_feature_index
fetch_feature_records = _mod.fetch_feature_records
