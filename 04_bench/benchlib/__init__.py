"""benchlib — local LOCOMO proxy harness (architecture layer 09, not part of
the competition service).

BENCH-ID-NEUTRALITY note: dataset-specific knowledge (names, IDs, categories)
is allowed HERE and ONLY here. 01_store / 02_retrieval / 03_adapter / 09_api /
10_config treat user_id / session_id / request_id as opaque strings.
"""

from .ledger import LEDGER_FIELDS, append_ledger_csv, append_ledger_jsonl, build_ledger_record
from .locomo_loader import Conversation, LocomoQuestion, load_locomo
from .runner import run_benchmark
from .scoring import exact_match, token_f1
from .surrogate import surrogate_answer

__all__ = [
    "LEDGER_FIELDS",
    "append_ledger_csv",
    "append_ledger_jsonl",
    "build_ledger_record",
    "Conversation",
    "LocomoQuestion",
    "load_locomo",
    "run_benchmark",
    "exact_match",
    "token_f1",
    "surrogate_answer",
]
