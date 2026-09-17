"""Parity fixture self-check, run as part of the normal test suite.

If this fails, the fusion core's behaviour changed. That may be
intentional - but it must be *noticed*, and the fixture regenerated in
its own commit rather than quietly drifting.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.parity.verify_fixture import compare, replay

FIXTURE_PATH = Path(__file__).resolve().parent / "fixture_ukf_reference.json"


def test_reference_implementation_reproduces_its_own_fixture():
    fixture = json.loads(FIXTURE_PATH.read_text())
    result = compare(fixture, replay(fixture))
    assert not result["failures"], (
        f"{len(result['failures'])} cycles out of tolerance - the fusion core's "
        "behaviour changed. If that was intentional, regenerate the fixture "
        "with `python -m tools.parity.generate_fixture` in its own commit."
    )


def test_fixture_covers_a_blackout_and_a_reacquisition():
    """A fixture that never loses GNSS would not exercise the part of
    the filter most likely to differ between implementations."""
    fixture = json.loads(FIXTURE_PATH.read_text())
    availability = [c["input"]["gnss_available"] for c in fixture["cycles"]]
    assert any(availability) and not all(availability), "fixture needs both regimes"
    # And a reacquisition, which is where the re-admission ramp lives.
    transitions = sum(
        1 for a, b in zip(availability, availability[1:]) if not a and b
    )
    assert transitions >= 1, "fixture must include at least one GNSS reacquisition"
