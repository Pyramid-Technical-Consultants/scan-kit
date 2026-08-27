"""Live RCI smoke test (connect + upload only; no start).

Set SCAN_KIT_RCI_HOST to an RCI IP (e.g. 192.168.100.184) to run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scan_kit.workflows.plan_runner.service import PlanRunnerService

_MINIMAL_CSV = """#NO,ENERGY(MeV),CURRENT(A),BEAM_SIZE(mm),X_POSITION(mm),Y_POSITION(mm),CHARGE_REQ(MU),VELOCITY(mm/s)
1,100.0,1e-9,3.61,0.0,0.0,1.0,1000.0
2,99.0,1e-9,3.61,1.0,0.0,1.0,1000.0
"""


@pytest.mark.slow
def test_live_rci_connect_and_upload(tmp_path: Path) -> None:
    host = os.environ.get("SCAN_KIT_RCI_HOST", "").strip()
    if not host:
        pytest.skip("SCAN_KIT_RCI_HOST not set")

    csv_path = tmp_path / "input_map.csv"
    csv_path.write_text(_MINIMAL_CSV, encoding="utf-8")

    svc = PlanRunnerService()
    info = svc.connect(host)
    assert info.get("host") == host

    target = svc.upload_plan(csv_path, timeout_s=45.0)
    assert target.startswith("/")

    svc.disconnect()
