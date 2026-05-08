"""
PDF research note generator using ReportLab.

The output is a 4-5 page equity research note with:
  * Cover page (title, author, supervisor, institution)
  * Executive summary + recommendation
  * Cost-of-equity derivation
  * DDM workings
  * Peer set + relative valuation
  * Quality scoring
  * Monte Carlo + sensitivity table
  * Disclaimer
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

from config import (
    AUTHOR_NAME, INSTITUTION, PROJECT_SUBTITLE, PROJECT_TITLE,
    REPORT_DIR, SUPERVISOR_NAME,
)

if TYPE_CHECKING:
    from src.integrated_valuation import ValuationResult


_PRIMARY = colors.HexColor("#1f4e79")
_BUY = colors.HexColor("#2e7d32")
_SELL = colors.HexColor("#c62828")
_HOLD = colors.HexColor("#f9a825")
_LIGHT = colors.HexColor("#f0f4f8")


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        "TitleBig", parent=s["Title"], fontSize=24, textColor=_PRIMARY,
        leading=28, spaceAfter=12,
    ))
    s.add(ParagraphStyle(
        "Subtitle", parent=s["Normal"], fontSize=13, textColor=colors.grey,
        leading=16, spaceAfter=8, alignment=1,
    ))
    s.add(ParagraphStyle(
        "H2", parent=s["Heading2"], textColor=_PRIMARY, spaceBefore=10,
    ))
    s.add(ParagraphStyle(
        "Body", parent=s["BodyText"], fontSize=10, leading=14,
    ))
    s.add(ParagraphStyle(
        "BuyTag", parent=s["Heading1"], textColor=colors.white,
        backColor=_BUY, alignment=1, borderPadding=10,
    ))
    s.add(ParagraphStyle(
        "SellTag", parent=s["Heading1"], textColor=colors.white,
        backColor=_SELL, alignment=1, borderPadding=10,
    ))
    s.add(ParagraphStyle(
        "HoldTag", parent=s["Heading1"], textColor=colors.white,
        backColor=_HOLD, alignment=1, borderPadding=10,
    ))
    return s


def _money(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"₹{x:,.2f}"


def _pct(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.2%}"


def _table_style(header_color=_PRIMARY) -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ])


# ---------------------------------------------------------------------------
def generate_pdf(result: "ValuationResult") -> Path:
    """Build the PDF research note and return the on-disk path.

    Lays out the cover page, executive summary, cost-of-equity workings,
    DDM detail, peer multiples, quality scoring, Monte Carlo + tornado,
    and disclaimer using ReportLab. The output filename is timestamped
    (UTC) so successive runs do not overwrite each other.

    Parameters
    ----------
    result : ValuationResult
        Output of :func:`integrated_valuation.value_stock`. Every section
        is rendered from this object — the function is a pure formatter
        and never re-runs the underlying valuation.

    Returns
    -------
    Path
        Path to the written PDF, under ``REPORT_DIR``.
    """
    safe = result.target.ticker.replace("&", "AND").replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = REPORT_DIR / f"{safe}_valuation_{ts}.pdf"

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    s = _styles()
    story = []

    # --- Cover page ---
    story.append(Spacer(1, 4*cm))
    story.append(Paragraph(PROJECT_TITLE, s["TitleBig"]))
    story.append(Paragraph(PROJECT_SUBTITLE, s["Subtitle"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        f"<b>Equity:</b> {result.target.name} ({result.target.ticker})",
        s["Body"]))
    story.append(Paragraph(f"<b>Sector:</b> {result.target.sector}", s["Body"]))
    story.append(Paragraph(
        f"<b>Report Date:</b> {datetime.now(timezone.utc).strftime('%d %B %Y')}",
        s["Body"]))
    story.append(Spacer(1, 4*cm))

    cover_meta = Table([
        ["Author", AUTHOR_NAME],
        ["Supervisor", SUPERVISOR_NAME],
        ["Programme", INSTITUTION],
    ], colWidths=[5*cm, 10*cm])
    cover_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _LIGHT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(cover_meta)
    story.append(PageBreak())

    # --- Executive summary ---
    story.append(Paragraph("1. Executive Summary", s["H2"]))
    rec = result.recommendation
    rec_style = "BuyTag" if rec == "BUY" else ("SellTag" if rec == "SELL" else "HoldTag")
    story.append(Paragraph(f"Recommendation: <b>{rec}</b> &nbsp; "
                           f"Margin of Safety: {_pct(result.margin_of_safety)}",
                           s[rec_style]))
    story.append(Spacer(1, 0.4*cm))

    summary_tbl = Table([
        ["Metric", "Value", "Notes"],
        ["Current Market Price", _money(result.target.price), "Last close (NSE)"],
        ["DDM Intrinsic Value", _money(result.ddm.value_per_share)
         if result.ddm.valid else "Not applicable", result.ddm.model],
        ["Relative Intrinsic Value", _money(result.relative.weighted_value),
         "Multi-multiple weighted"],
        ["Blended Intrinsic Value", _money(result.blended_value),
         f"DDM {result.weight_ddm:.0%} / Rel {result.weight_relative:.0%}"],
        ["Cost of Equity (Ke)", _pct(result.coe.ke), result.coe.method],
        ["Quality Composite (0-100)", f"{result.quality.composite:.1f}",
         f"F={result.quality.piotroski_f}/9, DQ={result.quality.dividend_quality}/10"],
        ["Monte Carlo Median", _money(result.monte_carlo.p50),
         f"P10 {_money(result.monte_carlo.p10)} – P90 {_money(result.monte_carlo.p90)}"],
    ], colWidths=[5*cm, 4*cm, 7.5*cm])
    summary_tbl.setStyle(_table_style())
    story.append(summary_tbl)

    # --- Methodology recap ---
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("2. Methodology", s["H2"]))
    story.append(Paragraph(
        "Two independent valuation tracks are computed and blended. "
        "The Dividend Discount Model auto-selects between Gordon, Two-Stage, "
        "Three-Stage and the H-Model depending on the firm's payout history "
        "and growth profile. The relative-valuation track builds a peer set "
        "via K-Means clustering on standardised financial ratios (with "
        "Mahalanobis-distance ranking), then aggregates trimmed harmonic-mean "
        "P/E, P/B, P/S, EV/EBITDA and PEG multiples. Sector-aware weights "
        "are applied — for example, banks lean on P/B; capital-intensive "
        "sectors lean on EV/EBITDA. The two tracks are then blended with a "
        "default 50/50 weight that tilts up to ±15 pp based on a quality "
        "score combining Piotroski F-Score, Altman Z'' (emerging-market "
        "variant) and a dividend-quality score.",
        s["Body"]))

    # --- Cost of Equity ---
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("3. Cost of Equity", s["H2"]))
    coe_tbl = Table([
        ["Component", "Value"],
        ["Risk-free rate (10Y G-Sec)", _pct(result.coe.risk_free)],
        ["Equity risk premium (India)", _pct(result.coe.erp)],
        ["Raw Beta", f"{result.coe.beta_raw:.2f}" if result.coe.beta_raw is not None else "—"],
        ["Adjusted Beta (Bloomberg)", f"{result.coe.beta_adjusted:.2f}"],
        ["Small-cap premium", _pct(result.coe.small_cap_premium)],
        ["Cost of Equity (CAPM)", _pct(result.coe.ke)],
    ], colWidths=[8*cm, 6*cm])
    coe_tbl.setStyle(_table_style())
    story.append(coe_tbl)
    story.append(Paragraph(f"Method: {result.coe.method}", s["Body"]))

    # --- DDM ---
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("4. Dividend Discount Model", s["H2"]))
    if result.ddm.valid:
        rows = [["Input", "Value"]]
        for k, v in result.ddm.inputs.items():
            if isinstance(v, float):
                rows.append([k, f"{v:,.4f}"])
            else:
                rows.append([k, str(v)])
        rows.append(["DDM Intrinsic Value (per share)",
                     _money(result.ddm.value_per_share)])
        ddm_tbl = Table(rows, colWidths=[8*cm, 6*cm])
        ddm_tbl.setStyle(_table_style())
        story.append(ddm_tbl)
        story.append(Paragraph(f"Variant selected: <b>{result.ddm.model}</b>", s["Body"]))
        if result.ddm.note:
            story.append(Paragraph(f"<i>{result.ddm.note}</i>", s["Body"]))
    else:
        story.append(Paragraph(
            f"DDM not applicable — {result.ddm.note}",
            s["Body"]))

    story.append(PageBreak())

    # --- Relative valuation ---
    story.append(Paragraph("5. Relative Valuation", s["H2"]))
    story.append(Paragraph(
        f"Peer set ({len(result.peer_set.peers)} firms) — "
        f"selection method: <i>{result.peer_set.method}</i>", s["Body"]))
    story.append(Spacer(1, 0.2*cm))

    rel_rows = [["Multiple", "Peer Aggregate", "Implied Price (₹)", "Weight"]]
    for k, mr in result.relative.multiples.items():
        rel_rows.append([
            k,
            f"{mr.aggregated_multiple:,.2f}" if np.isfinite(mr.aggregated_multiple) else "—",
            _money(mr.implied_price) if mr.valid else "—",
            f"{result.relative.weights.get(k, 0):.0%}",
        ])
    rel_rows.append([
        "Weighted Relative Value", "—", _money(result.relative.weighted_value), "—"])
    rel_tbl = Table(rel_rows, colWidths=[3.5*cm, 4*cm, 4*cm, 3*cm])
    rel_tbl.setStyle(_table_style())
    story.append(rel_tbl)

    # --- Quality ---
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("6. Quality Diagnostics", s["H2"]))
    q_tbl = Table([
        ["Metric", "Value", "Interpretation"],
        ["Piotroski F-Score", f"{result.quality.piotroski_f}/9",
         "≥ 7 = strong; ≤ 3 = weak"],
        ["Altman Z'' (EM)", f"{result.quality.altman_z:.2f}",
         "> 2.6 safe; 1.1–2.6 grey; < 1.1 distress"],
        ["Dividend Quality", f"{result.quality.dividend_quality}/10",
         "Track record + payout sustainability"],
        ["Composite", f"{result.quality.composite:.1f}/100", ""],
    ], colWidths=[5*cm, 3.5*cm, 7*cm])
    q_tbl.setStyle(_table_style())
    story.append(q_tbl)

    # --- Monte Carlo + tornado ---
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("7. Risk & Sensitivity", s["H2"]))
    mc = result.monte_carlo
    story.append(Paragraph(
        f"<b>Monte Carlo</b> (10,000 paths) — Mean ₹{mc.mean:,.0f}, "
        f"Std ₹{mc.std:,.0f}, P10/P50/P90: ₹{mc.p10:,.0f} / "
        f"₹{mc.p50:,.0f} / ₹{mc.p90:,.0f}.", s["Body"]))

    if result.tornado:
        t_rows = [["Lever", "Low (₹)", "High (₹)", "Swing (₹)"]]
        for b in result.tornado:
            t_rows.append([b.lever, f"{b.low_value:,.2f}",
                           f"{b.high_value:,.2f}", f"{b.swing:,.2f}"])
        t_tbl = Table(t_rows, colWidths=[6*cm, 3*cm, 3*cm, 3*cm])
        t_tbl.setStyle(_table_style())
        story.append(Spacer(1, 0.2*cm))
        story.append(t_tbl)

    # --- Disclaimer ---
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("8. Disclaimer", s["H2"]))
    story.append(Paragraph(
        "This report has been prepared for academic purposes as part of a "
        "Semester IV student project under the supervision of "
        f"{SUPERVISOR_NAME}. It does not constitute investment advice, an "
        "offer, or a solicitation to buy or sell any security. All financial "
        "data is sourced from publicly available filings via Yahoo Finance "
        "and may contain errors or omissions. Forward-looking statements are "
        "estimates produced by the valuation engine and are subject to change.",
        s["Body"]))

    doc.build(story)
    return path
