"""
Streamlit dashboard smoke tests.

These run the app under ``streamlit.testing.v1.AppTest`` — no browser,
no live data fetch. Their job is to catch the failure modes that won't
surface in the unit-test suite: import errors at module load, top-level
Streamlit-API misuse, broken sidebar widget construction, and stale
session-state assumptions in the no-result path. Anything beyond a
landing-page render is exercised end-to-end by ``run_valuation.py`` in
the headless CLI tests; we deliberately do not mock ``value_stock``
inside Streamlit, since the value-add of doing so is small and the
maintenance cost is large.
"""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _fresh_app(timeout: float = 30.0) -> AppTest:
    """Instantiate the app with a generous timeout — first run imports
    plotly + reportlab, which is heavier than the default 3s allowance."""
    return AppTest.from_file(str(_APP_PATH), default_timeout=timeout).run()


def test_app_loads_without_exceptions():
    """App must import and render once without raising."""
    at = _fresh_app()
    assert not at.exception, f"App raised on first render: {at.exception}"


def test_landing_page_shows_placeholder():
    """Before the user clicks Run, the placeholder copy is visible."""
    at = _fresh_app()
    info_blocks = [b.value for b in at.info]
    assert any("Run Valuation" in v for v in info_blocks), (
        "Landing-page placeholder text is missing — Streamlit layout has drifted."
    )


def test_sidebar_inputs_are_constructed():
    """All sidebar widgets the user needs must be present."""
    at = _fresh_app()
    selectbox_labels = [s.label for s in at.sidebar.selectbox]
    number_input_labels = [n.label for n in at.sidebar.number_input]
    button_labels = [b.label for b in at.sidebar.button]

    assert "Select an Indian stock" in selectbox_labels
    assert any("Sector" in lbl for lbl in selectbox_labels)
    assert any("Risk-free rate" in lbl for lbl in number_input_labels)
    assert any("Equity risk premium" in lbl for lbl in number_input_labels)
    assert any("Terminal growth" in lbl for lbl in number_input_labels)
    assert any("Run Valuation" in lbl for lbl in button_labels)


def test_custom_ticker_field_overrides_selectbox():
    """Typing a custom ticker should win over the selectbox default."""
    at = _fresh_app()
    custom = next(
        ti for ti in at.sidebar.text_input
        if "custom NSE ticker" in ti.label
    )
    custom.set_value("RELIANCE.NS")
    at.run()
    assert not at.exception, (
        f"Setting a custom ticker raised: {at.exception}"
    )
