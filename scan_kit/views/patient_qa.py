"""Patient-specific QA from DICOM: plan and logged deliveries through the Monte Carlo on the planning CT."""

from __future__ import annotations

from ..common import ViewSettings


def run(
    session_ids: list[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
) -> None:
    """Open the patient QA window; the selected sessions are the logged deliveries."""
    from .vispy_plot import ensure_gl_plus

    # vispy locks its GL wrapper on first gloo import; the ray march wants gl+.
    ensure_gl_plus()
    from .patient_qa_window import run_patient_qa_window

    run_patient_qa_window(session_ids, base_dir)
