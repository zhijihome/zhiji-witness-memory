"""G8 (ledger schema) + mini end-to-end benchmark run through the real API
surface, producing a real ledger record (G7/G9/G22 evidence at bench level)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from benchlib.ledger import (
    LEDGER_FIELDS,
    append_ledger_csv,
    append_ledger_jsonl,
    build_ledger_record,
)
from benchlib.locomo_loader import Conversation, LocomoQuestion, LocomoTurn
from benchlib.runner import run_benchmark
from benchlib.scoring import exact_match, token_f1
from benchlib.surrogate import surrogate_answer

REPO = Path(__file__).resolve().parent.parent


def test_ledger_record_has_mandated_fields_in_order(tmp_path):
    metrics = {f: None for f in LEDGER_FIELDS}
    metrics.update({"RETURN_SIZE": "top_k", "ORDERING_POLICY": "relevance_desc", "NEIGHBOR_POLICY": "off"})
    record = build_ledger_record(metrics)
    assert list(record.keys())[: len(LEDGER_FIELDS)] == LEDGER_FIELDS
    append_ledger_jsonl(tmp_path / "l.jsonl", record)
    append_ledger_csv(tmp_path / "l.csv", record)
    jsonl_line = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert set(LEDGER_FIELDS).issubset(jsonl_line.keys())
    with open(tmp_path / "l.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["RETURN_SIZE"] == "top_k"


def test_ledger_rejects_missing_mandated_field():
    with pytest.raises(ValueError, match="missing mandated fields"):
        build_ledger_record({"VERSION": "x"})


def test_scoring_basics():
    assert token_f1("the dog", "a dog") == 1.0  # articles/normalization ignored
    assert token_f1("dog cat", "dog") == pytest.approx(2 / 3)
    assert exact_match("The Dog.", "dog") == 1.0
    assert exact_match("cat", "dog") == 0.0


def test_surrogate_is_deterministic_and_ignores_gold():
    ev = ["Barkley was adopted in May. He likes tennis."]
    a1 = surrogate_answer("When was Barkley adopted?", None, ev)
    a2 = surrogate_answer("When was Barkley adopted?", None, ev)
    assert a1 == a2
    assert "May" in a1
    # options path: picks by evidence overlap, never by gold
    pick = surrogate_answer("Which is right?", ["A. Barkley likes tennis", "B. Barkley hates tennis"], ev)
    assert pick.startswith("A")


def _mini_conversation() -> Conversation:
    t = 1683000000000
    turns1 = [
        LocomoTurn(dia_id="D1:1", speaker="alice", text="I adopted a dog named Barkley in May 2023.", ts_ms=t),
        LocomoTurn(dia_id="D1:2", speaker="bob", text="Wonderful! How old is Barkley?", ts_ms=t + 60_000),
        LocomoTurn(dia_id="D1:3", speaker="alice", text="Barkley is four years old.", ts_ms=t + 120_000),
    ]
    turns2 = [
        LocomoTurn(dia_id="D2:1", speaker="alice", text="I started piano lessons last month.", ts_ms=t + 86_400_000),
    ]
    questions = [
        LocomoQuestion(question="What is the name of the dog?", answer="Barkley", evidence=("D1:1",), category=1),
        LocomoQuestion(question="How old is the dog?", answer="four years old", evidence=("D1:3",), category=1),
        LocomoQuestion(
            question="When was the dog adopted?", answer="May 2023", evidence=("D1:1",), category=3
        ),
    ]
    return Conversation(sample_id="mini-1", sessions=[turns1, turns2], questions=questions)


def test_mini_benchmark_end_to_end(tmp_path, tmp_config):
    from aml_api import create_app
    from fastapi.testclient import TestClient

    cfg = tmp_config()
    app = create_app(cfg)
    with TestClient(app) as client:  # TestClient is an httpx.Client — same surface as the real thing
        record = run_benchmark(
            client,
            [_mini_conversation()],
            cfg,
            run_id="mini-test",
            results_root=tmp_path / "results",
            label="unit-mini",
            write_ledger=True,
        )

    for field in LEDGER_FIELDS:
        assert field in record, f"ledger missing {field}"
    assert record["COORDINATE"] == "LOCAL"
    assert record["N_QUESTIONS"] == 3
    assert record["FAILURE_CASES"] == 0
    assert record["MODEL_CALLS"] == 0 and record["TOKENS"] == 0 and record["COST"] == 0.0
    assert record["RETURN_SIZE"] == "top_k"
    assert record["ORDERING_POLICY"] == "relevance_desc"
    assert record["NEIGHBOR_POLICY"] == "off"
    assert (record["FACT"] or 0) > 0  # single-hop questions answered with real F1
    assert record["EVIDENCE_RECALL_AT_100"] > 0
    assert (tmp_path / "results" / "ledger" / "ledger.jsonl").exists()
    assert (tmp_path / "results" / "ledger" / "ledger.csv").exists()
    details = (tmp_path / "results" / "runs" / "mini-test" / "questions.jsonl").read_text().splitlines()
    assert len(details) == 3


@pytest.mark.skipif(
    not (REPO / "04_bench" / "data" / "locomo10.json").exists(), reason="LOCOMO data not fetched"
)
def test_real_locomo_loads():
    from benchlib.locomo_loader import load_locomo

    convs = load_locomo(REPO / "04_bench" / "data" / "locomo10.json")
    assert len(convs) == 10
    assert all(c.questions for c in convs)
    total = sum(len(c.questions) for c in convs)
    assert total == 1986
