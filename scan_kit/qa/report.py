"""Patient QA report: one self-contained HTML page plus the DVHs as CSV."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

from ..dicom.dose import DISCLAIMER


def _table(head, rows) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def write_report(
    folder: str | Path, name: str, *, title: str, provenance: dict, dvhs: dict, goals=(), gamma=None,
    matches=(), png: bytes | None = None,
) -> Path:
    """Write ``name.html`` and ``name_dvh.csv`` into *folder*; returns the HTML path.

    *goals* are ``(Goal, value, passed)``; *gamma* a :class:`~.analysis.GammaReport`;
    *matches* the :class:`~.delivered.DeliveryMatch` of each logged beam.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    parts = [f"<h1>{html.escape(title)}</h1>", f"<p class=warn>{html.escape(DISCLAIMER)}</p>"]
    if png:
        parts.append(f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}">')
    if matches:
        parts += ["<h2>Delivery</h2>", _table(
            ("Beam", "Planned MU", "Delivered MU", "Ratio", "Unmatched spots", "Worst energy (MeV)", "Position RMS (mm)"),
            [(m.beam, f"{m.planned_mu:.2f}", f"{m.delivered_mu:.2f}",
              f"{m.delivered_mu / m.planned_mu:.4f}" if m.planned_mu > 0 else "",
              m.unmatched, f"{m.energy_error:.2f}", f"{m.position_rms:.2f}") for m in matches])]
    if gamma is not None:
        c = gamma.criteria
        parts += ["<h2>Gamma vs TPS</h2>", f"<p>{c.dose_pct:g} % / {c.dta_mm:g} mm, {c.cutoff_pct:g} % cutoff: "
                  f"<b>{100.0 * gamma.rate:.2f} %</b> of {gamma.evaluated} voxels pass</p>"]
    if goals:
        parts += ["<h2>Clinical goals</h2>", _table(
            ("Goal", "Value", "Result"),
            [(html.escape(g.text), f"{v:.2f} {html.escape(g.unit)}", "pass" if ok else "<b class=warn>FAIL</b>")
             for g, v, ok in goals])]
    if dvhs:
        parts += ["<h2>Dose statistics (Gy(RBE), whole course)</h2>", _table(
            ("Structure", "Volume (cc)", "Dmean", "D98%", "D95%", "D50%", "D2%", "Dmax"),
            [(html.escape(n), f"{h.volume_cc:.1f}", *(f"{v:.3f}" for v in (
                h.mean, h.dose_at(0.98), h.dose_at(0.95), h.dose_at(0.5), h.dose_at(0.02), h.dmax)))
             for n, h in dvhs.items()])]
    parts += ["<h2>Provenance</h2>", f"<pre>{html.escape(json.dumps(provenance, indent=2, default=str))}</pre>"]
    style = ("body{font-family:sans-serif;max-width:1100px;margin:2em auto}img{max-width:100%}"
             "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 8px;text-align:right}"
             ".warn{color:#b00}")
    out = folder / f"{name}.html"
    out.write_text(f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
                   f"<style>{style}</style>{''.join(parts)}", encoding="utf-8")
    if dvhs:
        first = next(iter(dvhs.values()))
        cols = np.column_stack([first.edges] + [100.0 * h.cumulative for h in dvhs.values()])
        header = "dose_gy_rbe," + ",".join(f"{n} volume %".replace(",", " ") for n in dvhs)
        np.savetxt(folder / f"{name}_dvh.csv", cols, delimiter=",", header=header, comments="", fmt="%.6g")
    return out
