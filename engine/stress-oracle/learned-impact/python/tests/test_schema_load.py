from pathlib import Path

from learned_impact.schema import validate_event

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_event"


def test_fixture_validates():
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}; run learned_impact_smoke first"
    man = validate_event(FIXTURE)
    assert man["schema_version"] == "learned-impact/1"
    assert man["K"] == 4
    assert man["n_frames"] >= 1
