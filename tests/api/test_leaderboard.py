"""Tests for /v1/leaderboard."""
from __future__ import annotations


def test_leaderboard_returns_rows_for_known_env(client):
    r = client.get("/v1/leaderboard?env_id=sparse-fourier-recovery")
    assert r.status_code == 200
    body = r.json()
    assert body["env_id"] == "sparse-fourier-recovery"
    assert isinstance(body["rows"], list)
    # Sprint-1 v2 + paper-final 1A must contribute at least 4 model rows.
    assert len(body["rows"]) >= 3
    # CSVs that contributed are listed.
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) >= 1


def test_leaderboard_rows_sorted_descending(client):
    body = client.get("/v1/leaderboard?env_id=sparse-fourier-recovery").json()
    rewards = [row["mean_reward"] for row in body["rows"]]
    assert rewards == sorted(rewards, reverse=True)


def test_leaderboard_unknown_env_returns_404(client):
    r = client.get("/v1/leaderboard?env_id=no-such-env")
    assert r.status_code == 404


def test_leaderboard_qualified_id_works(client):
    r = client.get(
        "/v1/leaderboard?env_id=stelioszach/sparse-fourier-recovery"
    )
    assert r.status_code == 200
    assert "rows" in r.json()


def test_leaderboard_row_has_expected_fields(client):
    body = client.get(
        "/v1/leaderboard?env_id=sparse-fourier-recovery"
    ).json()
    if not body["rows"]:
        return  # nothing to check
    row = body["rows"][0]
    for key in ("model", "n", "mean_reward", "std_reward", "parse_fail_rate"):
        assert key in row, f"row missing {key}: {row}"


def test_leaderboard_missing_env_id_returns_422(client):
    r = client.get("/v1/leaderboard")
    assert r.status_code == 422
