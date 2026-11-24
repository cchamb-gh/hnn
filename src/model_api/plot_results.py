"""Generate presentation SVGs and a T3-themed HTML visual from evaluation_results.json.

Uses only the Python standard library and NumPy so it runs anywhere model_api is installed.

Usage:
    python -m model_api.plot_results
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _fmt_sci(val: float) -> str:
    if val == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mant = val / (10.0**exp)
    if -2 <= exp <= 2:
        return f"{val:.4g}"
    return f"{mant:.1f}×10^{exp}"


def _log_map(val: float, ymin: float, ymax: float, y_bottom: float, y_top: float) -> float:
    lv = math.log10(max(val, ymin))
    lmin = math.log10(ymin)
    lmax = math.log10(ymax)
    frac = (lv - lmin) / (lmax - lmin)
    return y_bottom - frac * (y_bottom - y_top)


def _lin_map(val: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    frac = (val - vmin) / (vmax - vmin)
    return out_min + frac * (out_max - out_min)


def build_symmetry_svg(
    data: dict[str, Any],
    *,
    themed: bool = False,
    width: int = 1160,
    height: int = 435,
) -> str:
    """Build 3-panel SVG: Phase-Space Orbit, Energy vs Time, and Log Divergence."""
    if themed:
        bg = "none"
        panel_bg = "var(--card)"
        border = "var(--border)"
        ink = "var(--foreground)"
        muted = "var(--muted-foreground)"
        grid = "var(--border)"
        c_hnn = "var(--chart-1)"
        c_base = "var(--warning)"
        c_true = "var(--muted-foreground)"
        font = "var(--font-sans)"
        mono = "var(--font-mono)"
    else:
        bg = "#FBFBFA"
        panel_bg = "#FFFFFF"
        border = "#DADDE2"
        ink = "#16181D"
        muted = "#4E5360"
        grid = "#E8EBEF"
        c_hnn = "#2557D6"
        c_base = "#D95F02"
        c_true = "#555B66"
        font = "'IBM Plex Sans', system-ui, -apple-system, sans-serif"
        mono = "'IBM Plex Mono', ui-monospace, monospace"

    sym_roll = data["symmetry"]["trained"]["symmetric"]["rollout"]["representative_rollout"]
    base_roll = data["symmetry"]["trained"]["baseline"]["rollout"]["representative_rollout"]

    ts = np.asarray(sym_roll["t"], dtype=float)
    sq, sp = np.asarray(sym_roll["q"]), np.asarray(sym_roll["p"])
    bq, bp = np.asarray(base_roll["q"]), np.asarray(base_roll["p"])
    se = np.asarray(sym_roll["energy"])
    be = np.asarray(base_roll["energy"])

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="display:block;font-family:{font};">'
    ]
    if bg != "none":
        parts.append(f'<rect width="{width}" height="{height}" fill="{bg}" rx="10"/>')

    # Layout: 3 panels across
    gap = 18
    pad = 14
    pw = (width - 2 * pad - 2 * gap) / 3.0
    ph = height - 2 * pad

    panels = [
        (pad, pad, pw, ph, "A. Phase-Space Orbit (t ∈ [0, 100])"),
        (pad + pw + gap, pad, pw, ph, "B. True Energy H = q² + p² Over Time"),
        (pad + 2 * (pw + gap), pad, pw, ph, "C. Liouville Divergence |∇·f| (Log Scale)"),
    ]

    for px, py, w, h, title in panels:
        parts.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="10" fill="{panel_bg}" stroke="{border}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px + 16:.1f}" y="{py + 28:.1f}" fill="{ink}" '
            f'font-size="14" font-weight="600">{html.escape(title)}</text>'
        )

    # --- Panel A: Phase-space orbit (q vs p) ---
    px, py, w, h, _ = panels[0]
    plot_size = min(w - 82, h - 130)
    cx0 = px + 50 + (w - 68 - plot_size) / 2.0
    cy0 = py + 48
    cx1 = cx0 + plot_size
    cy1 = cy0 + plot_size

    # Grid & axes for [-1.2, 1.2]
    for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
        tx = _lin_map(tick, -1.2, 1.2, cx0, cx1)
        ty = _lin_map(tick, -1.2, 1.2, cy1, cy0)
        sw = "1.2" if tick == 0.0 else "0.7"
        parts.append(f'<line x1="{tx:.1f}" y1="{cy0:.1f}" x2="{tx:.1f}" y2="{cy1:.1f}" stroke="{grid}" stroke-width="{sw}"/>')
        parts.append(f'<line x1="{cx0:.1f}" y1="{ty:.1f}" x2="{cx1:.1f}" y2="{ty:.1f}" stroke="{grid}" stroke-width="{sw}"/>')
        if tick in (-1.0, 0.0, 1.0):
            parts.append(f'<text x="{tx:.1f}" y="{cy1 + 15:.1f}" fill="{muted}" font-size="11" text-anchor="middle">{tick:g}</text>')
            parts.append(f'<text x="{cx0 - 8:.1f}" y="{ty + 4:.1f}" fill="{muted}" font-size="11" text-anchor="end">{tick:g}</text>')

    parts.append(f'<rect x="{cx0:.1f}" y="{cy0:.1f}" width="{plot_size:.1f}" height="{plot_size:.1f}" fill="none" stroke="{border}" stroke-width="1"/>')

    # True unit circle
    r_px = (1.0 / 2.4) * plot_size
    mid_x = (cx0 + cx1) / 2.0
    mid_y = (cy0 + cy1) / 2.0
    parts.append(
        f'<circle cx="{mid_x:.1f}" cy="{mid_y:.1f}" r="{r_px:.1f}" fill="none" '
        f'stroke="{c_true}" stroke-width="1.6" stroke-dasharray="5 4"/>'
    )

    # Subsample stride 2 for smooth path (dt=0.2, 500 pts)
    def _poly(qs: np.ndarray, ps: np.ndarray) -> str:
        pts = [
            f"{_lin_map(q, -1.2, 1.2, cx0, cx1):.1f},{_lin_map(p, -1.2, 1.2, cy1, cy0):.1f}"
            for q, p in zip(qs[::2], ps[::2])
        ]
        return " ".join(pts)

    parts.append(
        f'<polyline points="{_poly(bq, bp)}" fill="none" stroke="{c_base}" '
        f'stroke-width="1.35" stroke-opacity="0.82"/>'
    )
    parts.append(
        f'<polyline points="{_poly(sq, sp)}" fill="none" stroke="{c_hnn}" '
        f'stroke-width="1.5" stroke-opacity="0.9"/>'
    )
    # Start point (1, 0)
    sx0 = _lin_map(1.0, -1.2, 1.2, cx0, cx1)
    sy0 = _lin_map(0.0, -1.2, 1.2, cy1, cy0)
    parts.append(f'<circle cx="{sx0:.1f}" cy="{sy0:.1f}" r="3.8" fill="{ink}"/>')

    parts.append(f'<text x="{mid_x:.1f}" y="{cy1 + 31:.1f}" fill="{muted}" font-size="11.5" text-anchor="middle">Position q</text>')
    parts.append(
        f'<text x="{cx0 - 34:.1f}" y="{mid_y:.1f}" fill="{muted}" font-size="11.5" '
        f'text-anchor="middle" transform="rotate(-90 {cx0 - 34:.1f} {mid_y:.1f})">Momentum p</text>'
    )
    # Legend row at bottom of panel A
    ly = py + h - 14
    parts.append(f'<line x1="{px + 20:.1f}" y1="{ly - 4:.1f}" x2="{px + 38:.1f}" y2="{ly - 4:.1f}" stroke="{c_hnn}" stroke-width="2.5"/>')
    parts.append(f'<text x="{px + 43:.1f}" y="{ly:.1f}" fill="{ink}" font-size="11">HNN</text>')
    parts.append(f'<line x1="{px + 92:.1f}" y1="{ly - 4:.1f}" x2="{px + 110:.1f}" y2="{ly - 4:.1f}" stroke="{c_base}" stroke-width="2.5"/>')
    parts.append(f'<text x="{px + 115:.1f}" y="{ly:.1f}" fill="{ink}" font-size="11">Baseline MLP</text>')
    parts.append(f'<line x1="{px + 206:.1f}" y1="{ly - 4:.1f}" x2="{px + 224:.1f}" y2="{ly - 4:.1f}" stroke="{c_true}" stroke-width="1.8" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{px + 229:.1f}" y="{ly:.1f}" fill="{muted}" font-size="11">Exact (H=1)</text>')

    # --- Panel B: True Energy over time ---
    px, py, w, h, _ = panels[1]
    bx0, by0 = px + 54, py + 48
    bx1, by1 = px + w - 18, py + h - 82
    e_min, e_max = 0.50, 1.16

    for etick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        ey = _lin_map(etick, e_min, e_max, by1, by0)
        parts.append(f'<line x1="{bx0:.1f}" y1="{ey:.1f}" x2="{bx1:.1f}" y2="{ey:.1f}" stroke="{grid}" stroke-width="0.7"/>')
        if etick in (0.5, 0.7, 0.9, 1.0, 1.1):
            parts.append(f'<text x="{bx0 - 8:.1f}" y="{ey + 4:.1f}" fill="{muted}" font-size="11" text-anchor="end">{etick:.1f}</text>')

    for ttick in (0, 25, 50, 75, 100):
        tx = _lin_map(ttick, 0.0, 100.0, bx0, bx1)
        parts.append(f'<line x1="{tx:.1f}" y1="{by0:.1f}" x2="{tx:.1f}" y2="{by1:.1f}" stroke="{grid}" stroke-width="0.7"/>')
        parts.append(f'<text x="{tx:.1f}" y="{by1 + 16:.1f}" fill="{muted}" font-size="11" text-anchor="middle">{ttick}</text>')

    parts.append(f'<rect x="{bx0:.1f}" y="{by0:.1f}" width="{bx1 - bx0:.1f}" height="{by1 - by0:.1f}" fill="none" stroke="{border}" stroke-width="1"/>')

    # Training horizon marker at t=3
    tx3 = _lin_map(3.0, 0.0, 100.0, bx0, bx1)
    parts.append(f'<rect x="{bx0:.1f}" y="{by0:.1f}" width="{tx3 - bx0:.1f}" height="{by1 - by0:.1f}" fill="{c_hnn}" fill-opacity="0.10"/>')

    # Reference H = 1.0
    ey1 = _lin_map(1.0, e_min, e_max, by1, by0)
    parts.append(f'<line x1="{bx0:.1f}" y1="{ey1:.1f}" x2="{bx1:.1f}" y2="{ey1:.1f}" stroke="{c_true}" stroke-width="1.5" stroke-dasharray="5 4"/>')

    def _epoly(energies: np.ndarray) -> str:
        pts = [
            f"{_lin_map(t, 0.0, 100.0, bx0, bx1):.1f},{_lin_map(e, e_min, e_max, by1, by0):.1f}"
            for t, e in zip(ts[::3], energies[::3])
        ]
        return " ".join(pts)

    parts.append(f'<polyline points="{_epoly(be)}" fill="none" stroke="{c_base}" stroke-width="2.1"/>')
    parts.append(f'<polyline points="{_epoly(se)}" fill="none" stroke="{c_hnn}" stroke-width="2.1"/>')

    # Clean non-overlapping curve labels
    y_hnn_lbl = _lin_map(1.085, e_min, e_max, by1, by0)
    y_base_lbl = _lin_map(0.79, e_min, e_max, by1, by0)
    parts.append(f'<text x="{bx1 - 8:.1f}" y="{y_hnn_lbl:.1f}" fill="{c_hnn}" font-size="11.5" font-weight="600" text-anchor="end">HNN: {se[-1]:.2f} (flat)</text>')
    parts.append(f'<text x="{bx1 - 8:.1f}" y="{y_base_lbl:.1f}" fill="{c_base}" font-size="11.5" font-weight="600" text-anchor="end">Baseline: {be[-1]:.2f} (-41%)</text>')

    parts.append(f'<text x="{(bx0 + bx1)/2:.1f}" y="{by1 + 36:.1f}" fill="{muted}" font-size="11.5" text-anchor="middle">Time t (train horizon t ≤ 3)</text>')
    parts.append(
        f'<text x="{bx0 - 38:.1f}" y="{(by0 + by1)/2:.1f}" fill="{muted}" font-size="11.5" '
        f'text-anchor="middle" transform="rotate(-90 {bx0 - 38:.1f} {(by0 + by1)/2:.1f})">True Energy H(q, p)</text>'
    )
    parts.append(
        f'<text x="{px + 16:.1f}" y="{py + h - 14:.1f}" fill="{muted}" font-size="10.8">'
        f'20-traj E-MSE @ t=100: HNN 1.3e-4 vs Base 2.8e-2 (215×)</text>'
    )

    # --- Panel C: Log-scale divergence bars ---
    px, py, w, h, _ = panels[2]
    dx0, dy0 = px + 56, py + 48
    dx1, dy1 = px + w - 18, py + h - 82
    d_min, d_max = 1e-10, 1.0

    for exp in (-10, -8, -6, -4, -2, 0):
        val = 10.0**exp
        yy = _log_map(val, d_min, d_max, dy1, dy0)
        parts.append(f'<line x1="{dx0:.1f}" y1="{yy:.1f}" x2="{dx1:.1f}" y2="{yy:.1f}" stroke="{grid}" stroke-width="0.7"/>')
        lbl = f"10^{exp}" if exp != 0 else "1"
        parts.append(f'<text x="{dx0 - 7:.1f}" y="{yy + 4:.1f}" fill="{muted}" font-size="10.5" font-family="{mono}" text-anchor="end">{lbl}</text>')

    parts.append(f'<rect x="{dx0:.1f}" y="{dy0:.1f}" width="{dx1 - dx0:.1f}" height="{dy1 - dy0:.1f}" fill="none" stroke="{border}" stroke-width="1"/>')

    sym_u = data["symmetry"]["untrained"]["symmetric"]["divergence_in_dist"]["mean"]
    base_u = data["symmetry"]["untrained"]["baseline"]["divergence_in_dist"]["mean"]
    sym_id = data["symmetry"]["trained"]["symmetric"]["divergence_in_dist"]["mean"]
    base_id = data["symmetry"]["trained"]["baseline"]["divergence_in_dist"]["mean"]
    sym_ood = data["symmetry"]["trained"]["symmetric"]["divergence_ood"]["mean"]
    base_ood = data["symmetry"]["trained"]["baseline"]["divergence_ood"]["mean"]

    groups = [
        ("Untrained", sym_u, base_u),
        ("Trained\n(In-Dist)", sym_id, base_id),
        ("Trained\n(OOD)", sym_ood, base_ood),
    ]
    gw = (dx1 - dx0) / len(groups)
    bw = min(28.0, gw * 0.30)

    for gi, (glabel, hv, bv) in enumerate(groups):
        gx = dx0 + (gi + 0.5) * gw
        # HNN bar
        hy = _log_map(hv, d_min, d_max, dy1, dy0)
        parts.append(f'<rect x="{gx - bw - 3:.1f}" y="{hy:.1f}" width="{bw:.1f}" height="{dy1 - hy:.1f}" rx="3" fill="{c_hnn}"/>')
        parts.append(f'<text x="{gx - bw/2 - 3:.1f}" y="{hy - 5:.1f}" fill="{c_hnn}" font-size="9.5" font-family="{mono}" text-anchor="middle">{hv:.1e}</text>')
        # Base bar
        by = _log_map(bv, d_min, d_max, dy1, dy0)
        parts.append(f'<rect x="{gx + 3:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{dy1 - by:.1f}" rx="3" fill="{c_base}"/>')
        parts.append(f'<text x="{gx + bw/2 + 3:.1f}" y="{by - 5:.1f}" fill="{c_base}" font-size="9.5" font-family="{mono}" text-anchor="middle">{bv:.2f}</text>')
        # X-axis label
        lines = glabel.split("\n")
        for li, line in enumerate(lines):
            parts.append(f'<text x="{gx:.1f}" y="{dy1 + 15 + li * 13:.1f}" fill="{muted}" font-size="11" text-anchor="middle">{html.escape(line)}</text>')

    parts.append(
        f'<text x="{dx0 - 42:.1f}" y="{(dy0 + dy1)/2:.1f}" fill="{muted}" font-size="11.5" '
        f'text-anchor="middle" transform="rotate(-90 {dx0 - 42:.1f} {(dy0 + dy1)/2:.1f})">Mean |∇·f| (0 = exact)</text>'
    )
    parts.append(
        f'<text x="{px + 16:.1f}" y="{py + h - 12:.1f}" fill="{muted}" font-size="11">'
        f'HNN |dĤ/dt| = 0.00e+00 (exact by architecture)</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def build_data_efficiency_svg(
    data: dict[str, Any],
    *,
    themed: bool = False,
    width: int = 1160,
    height: int = 445,
) -> str:
    """Build 2-panel SVG: Rollout Energy MSE and Clean Vector-Field MSE vs training size."""
    if themed:
        bg = "none"
        panel_bg = "var(--card)"
        border = "var(--border)"
        ink = "var(--foreground)"
        muted = "var(--muted-foreground)"
        grid = "var(--border)"
        c_hnn = "var(--chart-1)"
        c_base = "var(--warning)"
        c_accent = "var(--success)"
        font = "var(--font-sans)"
        mono = "var(--font-mono)"
    else:
        bg = "#FBFBFA"
        panel_bg = "#FFFFFF"
        border = "#DADDE2"
        ink = "#16181D"
        muted = "#4E5360"
        grid = "#E8EBEF"
        c_hnn = "#2557D6"
        c_base = "#D95F02"
        c_accent = "#1E7E4A"
        font = "'IBM Plex Sans', system-ui, -apple-system, sans-serif"
        mono = "'IBM Plex Mono', ui-monospace, monospace"

    rows = data["data_efficiency"]["fractions"]
    pcts = [r["percent"] for r in rows]
    pts_list = [r["n_points"] for r in rows]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="display:block;font-family:{font};">'
    ]
    if bg != "none":
        parts.append(f'<rect width="{width}" height="{height}" fill="{bg}" rx="10"/>')

    pad = 14
    gap = 20
    pw = (width - 2 * pad - gap) / 2.0
    ph = height - 2 * pad

    panels = [
        (pad, pad, pw, ph, "A. Long-Horizon Rollout Energy MSE (t ∈ [0, 30], 5 Seeds)"),
        (pad + pw + gap, pad, pw, ph, "B. Single-Step Clean Vector-Field MSE (5 Seeds)"),
    ]

    for px, py, w, h, title in panels:
        parts.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="10" fill="{panel_bg}" stroke="{border}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px + 18:.1f}" y="{py + 29:.1f}" fill="{ink}" '
            f'font-size="14.5" font-weight="600">{html.escape(title)}</text>'
        )

    def _draw_curve_panel(
        panel_rect: tuple[float, float, float, float, str],
        metric_key: str,
        ymin: float,
        ymax: float,
        yticks: list[int],
        ylabel: str,
        footnote: str,
        show_efficiency_callout: bool = False,
    ) -> None:
        px, py, w, h, _ = panel_rect
        x0, y0 = px + 68, py + 52
        x1, y1 = px + w - 24, py + h - 88

        # Grid Y (log)
        for exp in yticks:
            val = 10.0**exp
            yy = _log_map(val, ymin, ymax, y1, y0)
            parts.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="{grid}" stroke-width="0.8"/>')
            lbl = f"10^{exp}" if exp != 0 else "1"
            parts.append(f'<text x="{x0 - 8:.1f}" y="{yy + 4:.1f}" fill="{muted}" font-size="11" font-family="{mono}" text-anchor="end">{lbl}</text>')

        # X positions (evenly spaced categorical positions for clean reading of 4%, 10%, 25%, 50%, 100%)
        xs = [x0 + (i + 0.35) * (x1 - x0) / (len(rows) - 0.3) for i in range(len(rows))]
        for xi, pct, npts in zip(xs, pcts, pts_list):
            parts.append(f'<line x1="{xi:.1f}" y1="{y0:.1f}" x2="{xi:.1f}" y2="{y1:.1f}" stroke="{grid}" stroke-width="0.7" stroke-dasharray="3 3"/>')
            parts.append(f'<text x="{xi:.1f}" y="{y1 + 17:.1f}" fill="{ink}" font-size="11.5" font-weight="500" text-anchor="middle">{pct}%</text>')
            parts.append(f'<text x="{xi:.1f}" y="{y1 + 31:.1f}" fill="{muted}" font-size="10" text-anchor="middle">({npts} pts)</text>')

        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" fill="none" stroke="{border}" stroke-width="1"/>')

        # Compute medians and IQR (25th-75th) per fraction
        for kind, color, offset in [("baseline", c_base, 3.5), ("symmetric", c_hnn, -3.5)]:
            meds, q25s, q75s, raw_vals = [], [], [], []
            for r in rows:
                vals = np.asarray(r[kind][metric_key]["values"], dtype=float)
                meds.append(float(np.median(vals)))
                q25s.append(float(np.percentile(vals, 25)))
                q75s.append(float(np.percentile(vals, 75)))
                raw_vals.append(vals)

            # IQR shaded band (clamped to plot box)
            upper_pts = [
                f"{xi:.1f},{_log_map(min(v, ymax), ymin, ymax, y1, y0):.1f}"
                for xi, v in zip(xs, q75s)
            ]
            lower_pts = [
                f"{xi:.1f},{_log_map(min(v, ymax), ymin, ymax, y1, y0):.1f}"
                for xi, v in zip(reversed(xs), reversed(q25s))
            ]
            parts.append(
                f'<polygon points="{" ".join(upper_pts + lower_pts)}" '
                f'fill="{color}" fill-opacity="0.14"/>'
            )

            # Individual seed dots (subtle)
            for xi, vals in zip(xs, raw_vals):
                for v in vals:
                    cy = _log_map(min(max(v, ymin), ymax), ymin, ymax, y1, y0)
                    parts.append(f'<circle cx="{xi + offset:.1f}" cy="{cy:.1f}" r="2.3" fill="{color}" fill-opacity="0.45"/>')

            # Median polyline
            line_pts = [
                f"{xi:.1f},{_log_map(min(v, ymax), ymin, ymax, y1, y0):.1f}"
                for xi, v in zip(xs, meds)
            ]
            parts.append(
                f'<polyline points="{" ".join(line_pts)}" fill="none" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            for xi, v in zip(xs, meds):
                cy = _log_map(min(v, ymax), ymin, ymax, y1, y0)
                parts.append(f'<circle cx="{xi:.1f}" cy="{cy:.1f}" r="4.2" fill="{color}" stroke="{panel_bg}" stroke-width="1.3"/>')

        if show_efficiency_callout:
            # Horizontal dashed reference line at Baseline 100% median (0.01376)
            base_100 = float(np.median(rows[-1]["baseline"][metric_key]["values"]))
            y_ref = _log_map(base_100, ymin, ymax, y1, y0)
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y_ref:.1f}" x2="{x1:.1f}" y2="{y_ref:.1f}" '
                f'stroke="{c_base}" stroke-width="1.3" stroke-dasharray="5 4" stroke-opacity="0.85"/>'
            )
            parts.append(
                f'<text x="{x1 - 8:.1f}" y="{y_ref + 14:.1f}" fill="{c_accent}" '
                f'font-size="11" font-weight="600" text-anchor="end">HNN @ 10% beats Baseline @ 100% (5.4×)</text>'
            )

        # Axis titles
        parts.append(
            f'<text x="{(x0 + x1)/2:.1f}" y="{y1 + 52:.1f}" fill="{muted}" '
            f'font-size="11.5" text-anchor="middle">Training Set Size (% of 750 noisy samples)</text>'
        )
        parts.append(
            f'<text x="{x0 - 50:.1f}" y="{(y0 + y1)/2:.1f}" fill="{muted}" font-size="11.5" '
            f'text-anchor="middle" transform="rotate(-90 {x0 - 50:.1f} {(y0 + y1)/2:.1f})">{html.escape(ylabel)}</text>'
        )

        # Legend top-right inside plot
        lx = x1 - 154
        ly = y0 + 18
        parts.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx + 20:.1f}" y2="{ly:.1f}" stroke="{c_hnn}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{lx + 10:.1f}" cy="{ly:.1f}" r="3.5" fill="{c_hnn}"/>')
        parts.append(f'<text x="{lx + 26:.1f}" y="{ly + 4:.1f}" fill="{ink}" font-size="11.5">HNN (median + IQR)</text>')
        parts.append(f'<line x1="{lx:.1f}" y1="{ly + 18:.1f}" x2="{lx + 20:.1f}" y2="{ly + 18:.1f}" stroke="{c_base}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{lx + 10:.1f}" cy="{ly + 18:.1f}" r="3.5" fill="{c_base}"/>')
        parts.append(f'<text x="{lx + 26:.1f}" y="{ly + 22:.1f}" fill="{ink}" font-size="11.5">Baseline (median + IQR)</text>')

        parts.append(
            f'<text x="{px + 18:.1f}" y="{py + h - 14:.1f}" fill="{muted}" font-size="10.8">'
            f'{html.escape(footnote)}</text>'
        )

    _draw_curve_panel(
        panels[0],
        "rollout_t30_energy_mse",
        ymin=1e-4,
        ymax=1e4,
        yticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4],
        ylabel="Rollout Energy MSE (lower is better)",
        footnote="Dots show individual seeds (0–4); HNN stays 39×–121× lower from 10% to 100%.",
        show_efficiency_callout=True,
    )
    _draw_curve_panel(
        panels[1],
        "clean_vf_mse",
        ymin=1e-3,
        ymax=1.0,
        yticks=[-3, -2, -1, 0],
        ylabel="Clean Vector-Field MSE (lower is better)",
        footnote="Evaluated on 750 noiseless test points; HNN beats Baseline at 100% (1.95e-3 vs 3.30e-3).",
        show_efficiency_callout=False,
    )

    parts.append("</svg>")
    return "\n".join(parts)


def build_t3_html_dashboard(data: dict[str, Any]) -> str:
    """Build self-contained, T3-theme-compatible HTML summary for html_render."""
    sym_svg = build_symmetry_svg(data, themed=True, width=1080, height=420)
    eff_svg = build_data_efficiency_svg(data, themed=True, width=1080, height=430)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    color: var(--foreground);
    font-family: var(--font-sans);
    font-size: 13px;
    line-height: 1.45;
  }}
  .stack {{
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
  }}
  .kpi {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
  }}
  .kpi-label {{
    font-size: 11px;
    color: var(--muted-foreground);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }}
  .kpi-value {{
    font-size: 20px;
    font-weight: 600;
    font-family: var(--font-mono);
    color: var(--foreground);
  }}
  .kpi-sub {{
    font-size: 11.5px;
    color: var(--muted-foreground);
    margin-top: 3px;
  }}
  .section-label {{
    font-size: 12px;
    font-weight: 600;
    color: var(--muted-foreground);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 2px 0 6px 2px;
  }}
  .chart-wrap {{
    width: 100%;
  }}
  @media (max-width: 560px) {{
    .kpi-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="stack">
    <div class="kpi-row">
      <div class="kpi">
        <div class="kpi-label">Symmetry: Phase-Space Divergence</div>
        <div class="kpi-value">7.4e-8 vs 5.4e-2</div>
        <div class="kpi-sub">HNN is ~730,000× closer to exact ∇·f = 0</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Symmetry: Energy MSE @ t = 100</div>
        <div class="kpi-value">215× lower</div>
        <div class="kpi-sub">HNN 1.30e-4 (flat) vs Baseline 2.79e-2</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Data Efficiency: Energy @ t = 30</div>
        <div class="kpi-value">10% beats 100%</div>
        <div class="kpi-sub">HNN @ 10% (2.53e-3) &lt; Base @ 100% (1.38e-2)</div>
      </div>
    </div>

    <div>
      <div class="section-label">1. Symmetry Test — Saved Checkpoints (Exact Conservation vs Drift)</div>
      <div class="chart-wrap">{sym_svg}</div>
    </div>

    <div>
      <div class="section-label">2. Data Efficiency — 4% to 100% of Training Data (5 Seeds)</div>
      <div class="chart-wrap">{eff_svg}</div>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HNN evaluation charts")
    parser.add_argument(
        "--input",
        type=str,
        default="doc/presentation/assets/evaluation_results.json",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="doc/presentation/assets",
    )
    parser.add_argument(
        "--html_out",
        type=str,
        default="",
        help="Optional path for T3-themed HTML visual",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    sym_path = assets_dir / "symmetry_test.svg"
    eff_path = assets_dir / "data_efficiency.svg"
    sym_path.write_text(build_symmetry_svg(data, themed=False))
    eff_path.write_text(build_data_efficiency_svg(data, themed=False))
    print(f"Wrote {sym_path} and {eff_path}")

    if args.html_out:
        html_path = Path(args.html_out)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(build_t3_html_dashboard(data))
        print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
