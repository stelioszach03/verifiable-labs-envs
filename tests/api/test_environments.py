"""Tests for /v1/environments."""
from __future__ import annotations


def test_environments_returns_at_least_10(client):
    r = client.get("/v1/environments")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 10
    assert len(body["environments"]) == body["count"]


def test_environments_contains_known_ids(client):
    r = client.get("/v1/environments")
    ids = {row["id"] for row in r.json()["environments"]}
    expected = {
        "sparse-fourier-recovery",
        "super-resolution-div2k-x4",
        "lodopab-ct-simplified",
        "phase-retrieval",
        "mri-knee-reconstruction",
    }
    missing = expected - ids
    assert not missing, f"missing envs: {missing}"


def test_environments_qualified_id_uses_stelioszach_prefix(client):
    body = client.get("/v1/environments").json()
    for row in body["environments"]:
        assert row["qualified_id"] == f"stelioszach/{row['id']}"


def test_environments_multi_turn_flag_correct(client):
    body = client.get("/v1/environments").json()
    by_id = {row["id"]: row for row in body["environments"]}
    assert by_id["sparse-fourier-recovery"]["multi_turn"] is False
    assert by_id["sparse-fourier-recovery-multiturn"]["multi_turn"] is True


def test_environments_tool_use_flag_correct(client):
    body = client.get("/v1/environments").json()
    by_id = {row["id"]: row for row in body["environments"]}
    assert by_id["sparse-fourier-recovery-tools"]["tool_use"] is True
    assert by_id["sparse-fourier-recovery"]["tool_use"] is False


def test_environments_each_row_has_description(client):
    body = client.get("/v1/environments").json()
    for row in body["environments"]:
        assert isinstance(row["description"], str)
        assert len(row["description"]) > 10  # substantive copy
        assert row["domain"] != "unknown"
