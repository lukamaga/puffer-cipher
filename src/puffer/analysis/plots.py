"""Charts for the statistics report, rendered twice: for light and for dark pages.

Every chart reads only the JSON-ready results mapping that
:func:`puffer.analysis.report.run_report` writes to ``results.json``, so the
charts can be re-rendered from a saved file (``python -m
puffer.analysis.plots docs/results/results.json docs/media``) without
collecting anything again.

Visual rules (the method of the project's dataviz guide, applied here)
---------------------------------------------------------------------
* **Colour follows the entity, as on the dashboard.** Sensor noise is
  blue, fish motion orange and the OS-keyed PufferRNG output aqua (the
  dashboard's "system" colour), in the dashboard's validated light and dark
  steps (``dashboard/src/styles/tokens.css``). The combined stream, a mix of
  both channels, is neutral gray rather than a fourth hue. The two liveness
  runs take two further slots of the same palette (violet, magenta) so they
  are never mistaken for a channel. Each pair and the trio were checked with the
  palette validator against the exact surfaces used here (white and GitHub's
  ``#0d1117``); light-mode aqua and magenta sit below 3:1 on white, so every
  series that uses them is also labelled directly.
* **One axis per plot.** Channels whose rates differ by three orders of
  magnitude get separate panels (small multiples), never a second y-axis.
* **Quiet chrome.** Hairline solid gridlines one step off the surface, no
  boxes, 2 px lines, markers at least 8 px with a ring in the surface colour,
  text in ink tokens (never in a series colour).
* **Every chart states its data source** in its subtitle (simulated
  aquarium, seed, resolution, frames) and its exact numbers are in
  ``RESULTS.md``, which is the table view of every chart.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
from matplotlib import font_manager, rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

from puffer.analysis.report import (
    ESTIMATOR_LABELS,
    STREAM_LABELS,
    TEST_LABELS,
    describe_source,
)

__all__ = ["CHARTS", "DARK", "LIGHT", "THEMES", "Theme", "render_all"]

WIDTH_IN: Final = 8.0
DPI: Final = 200  # 1600 px wide
FONT_FAMILIES: Final = ("Inter", "Helvetica Neue", "Arial", "DejaVu Sans")


@dataclass(frozen=True, slots=True)
class Theme:
    """Surface, ink and series colours of one rendering (light or dark)."""

    name: str
    surface: str
    ink: str  # titles, values
    ink2: str  # subtitles, labels
    ink3: str  # ticks, footers
    grid: str
    axis: str
    wash: str  # night band, acceptance band
    series: Mapping[str, str]


LIGHT: Final = Theme(
    name="light",
    surface="#ffffff",
    ink="#0a1a24",
    ink2="#435561",
    ink3="#5d7080",
    grid="#e3eaed",
    axis="#c3cfd5",
    wash="#eef2f6",
    series={
        "sensor": "#2a78d6",
        "fish": "#eb6834",
        "system": "#1baf7a",
        "live": "#6250d6",
        "replay": "#e87ba4",
        "neutral": "#7a8b95",
    },
)
DARK: Final = Theme(
    name="dark",
    surface="#0d1117",
    ink="#e6edf3",
    ink2="#a7bac4",
    ink3="#7d8b99",
    grid="#1f2731",
    axis="#30363d",
    wash="#151d29",
    series={
        "sensor": "#3987e5",
        "fish": "#d95926",
        "system": "#199e70",
        "live": "#9085e9",
        "replay": "#d55181",
        "neutral": "#6f8794",
    },
)
THEMES: Final = (LIGHT, DARK)

_STREAM_COLOUR = {"sensor": "sensor", "fish": "fish", "combined": "neutral", "pufferrng": "system"}


# ======================================================================== layout helpers


def _font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in FONT_FAMILIES if name in available), "DejaVu Sans")


def _style(theme: Theme) -> dict[str, Any]:
    return {
        "font.family": _font(),
        "font.size": 9.0,
        "text.color": theme.ink,
        "axes.edgecolor": theme.axis,
        "axes.labelcolor": theme.ink3,
        "axes.facecolor": theme.surface,
        "figure.facecolor": theme.surface,
        "xtick.color": theme.ink3,
        "ytick.color": theme.ink3,
        "xtick.labelcolor": theme.ink3,
        "ytick.labelcolor": theme.ink3,
        "axes.linewidth": 0.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }


PAD_IN: Final = 0.34  # outer margin left and right
TITLE_PT: Final = 13.5
SUBTITLE_PT: Final = 9.0
LABEL_PT: Final = 8.2
SMALL_PT: Final = 7.4
_NBSP = "\u00a0"
_KEEP_TOGETHER = (
    re.compile(r"\b(SP) (800-\d+[AB]?)"),
    re.compile(r"(\d) (s|ms|%|bits?|fps|px|frames|sigma)\b"),
)


def _wrap(fig: Figure, text: str, fontsize: float, width_in: float) -> list[str]:
    """Greedy word wrap measured with the real font, so lines use the full width."""
    for pattern in _KEEP_TOGETHER:
        text = pattern.sub(lambda m: f"{m.group(1)}{_NBSP}{m.group(2)}", text)
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    probe = fig.text(0, 0, "", fontsize=fontsize)
    limit = width_in * fig.dpi
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            trial = f"{current} {word}" if current else word
            probe.set_text(trial)
            if current and probe.get_window_extent(renderer).width > limit:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    probe.remove()
    return lines


class _Canvas:
    """A figure laid out top-down in inches: title, subtitle, legend row, content, footer."""

    def __init__(
        self,
        theme: Theme,
        title: str,
        subtitle: str,
        footer: str,
        content_in: float,
        *,
        legend: bool = False,
    ) -> None:
        self.theme = theme
        self.fig = Figure(figsize=(WIDTH_IN, 4.0), dpi=DPI)
        FigureCanvasAgg(self.fig)  # a canvas that can measure text (no pyplot state)
        lines = _wrap(self.fig, subtitle, SUBTITLE_PT, WIDTH_IN - 2 * PAD_IN)
        self.header_in = 0.48 + 0.19 + 0.205 * len(lines) + 0.12
        self.legend_in = 0.30 if legend else 0.0
        self.footer_in = 0.48
        self.height = self.header_in + self.legend_in + content_in + self.footer_in
        self.fig.set_size_inches(WIDTH_IN, self.height)
        self.fig.patch.set_facecolor(theme.surface)
        x = PAD_IN / WIDTH_IN
        self.fig.text(
            x,
            self.y(0.48),
            title,
            ha="left",
            va="baseline",
            fontsize=TITLE_PT,
            fontweight="semibold",
            color=theme.ink,
        )
        self.fig.text(
            x,
            self.y(0.48 + 0.19),
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=SUBTITLE_PT,
            color=theme.ink2,
            linespacing=1.5,
        )
        self.fig.text(
            x, 0.17 / self.height, footer, ha="left", va="bottom", fontsize=7.2, color=theme.ink3
        )
        self.content_top_in = self.header_in + self.legend_in

    def y(self, inches_from_top: float) -> float:
        return 1.0 - inches_from_top / self.height

    def x(self, inches_from_left: float) -> float:
        return inches_from_left / WIDTH_IN

    def axes(self, left_in: float, top_in: float, width_in: float, height_in: float) -> Axes:
        """Axes placed in inches: ``top_in`` measured from the top of the content area."""
        top = self.content_top_in + top_in
        return self.fig.add_axes(
            (self.x(left_in), self.y(top + height_in), width_in / WIDTH_IN, height_in / self.height)
        )

    def legend(self, items: Sequence[tuple[str, dict[str, Any]]], left_in: float) -> None:
        """One row of ``(label, marker style)`` keys just above the content area."""
        y = self.y(self.header_in + self.legend_in * 0.45)
        x_in = left_in
        for text, style in items:
            self.fig.lines.append(
                Line2D([self.x(x_in + 0.07)], [y], transform=self.fig.transFigure, **style)
            )
            label = self.fig.text(
                self.x(x_in + 0.22),
                y,
                text,
                fontsize=LABEL_PT,
                color=self.theme.ink2,
                va="center",
                ha="left",
            )
            x_in += 0.22 + 0.075 * len(label.get_text()) + 0.3


def _key(theme: Theme, colour: str, *, hollow: bool = False) -> dict[str, Any]:
    """Legend marker style: a filled dot, or a hollow ring for the second series."""
    return {
        "marker": "o",
        "markersize": 6.5,
        "linestyle": "none",
        "markerfacecolor": theme.surface if hollow else colour,
        "markeredgecolor": colour,
        "markeredgewidth": 1.4,
    }


def _clean(ax: Axes, theme: Theme, *, grid_axis: Literal["x", "y"] = "y") -> None:
    """Hairline grid on one axis, no frame except a hairline baseline."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.axis)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0, pad=5, labelsize=8.0)
    ax.grid(True, axis=grid_axis, color=theme.grid, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_facecolor(theme.surface)


def _panel_title(ax: Axes, theme: Theme, title: str, note: str = "") -> None:
    ax.text(
        0.0,
        1.0,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="semibold",
        color=theme.ink,
    )
    if note:
        ax.text(
            1.0,
            1.0,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=SMALL_PT,
            color=theme.ink3,
        )


def _footer(results: Mapping[str, Any]) -> str:
    prov = results.get("provenance", {})
    date = str(results.get("generated_utc", ""))[:10]
    return (
        f"puffer analyze, report v{results.get('report_version', 1)}, {date}, code "
        f"{prov.get('commit', 'unknown')}. Every number: docs/results/RESULTS.md"
    )


def _first_run(results: Mapping[str, Any], night: bool | None = None) -> str | None:
    for label, run in results["runs"].items():
        if night is None or (run["night_fraction"] > 0.5) == night:
            return str(label)
    return None


def _dots(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    colour: str,
    theme: Theme,
    *,
    hollow: bool = False,
    size: float = 6.6,
    zorder: int = 4,
) -> None:
    """Markers >= 8 px: filled with a surface-coloured ring, or hollow rings of the series."""
    ax.plot(
        xs,
        ys,
        linestyle="none",
        marker="o",
        markersize=size,
        markerfacecolor=theme.surface if hollow else colour,
        markeredgecolor=colour if hollow else theme.surface,
        markeredgewidth=1.6 if hollow else 1.2,
        zorder=zorder,
        clip_on=False,
    )


def _nice_ceiling(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * exp >= value:
            return step * exp
    return 10 * exp


def _short(value: float) -> str:
    return f"{value:,.0f}" if value >= 10 else f"{value:.1f}"


def _thousands(value: float, _pos: object = None) -> str:
    return f"{value:,.0f}"


# ======================================================================== 1. min-entropy


def _estimate(assessment: Mapping[str, Any], name: str, track: str) -> float | None:
    for result in assessment.get("results", []):
        if result["name"] == name and result["details"].get("track") == track:
            return float(result["min_entropy"])
    return None


def chart_min_entropy(results: Mapping[str, Any], theme: Theme) -> Figure | None:
    """Every SP 800-90B estimator, sensor and fish side by side, day versus night."""
    day, night = _first_run(results, night=False), _first_run(results, night=True)
    if day is None:
        return None
    assessed = results["sp800_90b"]
    groups: list[tuple[str, list[tuple[str, str, int]]]] = []
    for track, scale, header in (
        ("original", 1, "SYMBOLS (original track)"),
        ("bitstring", 8, "BITS, x 8 (bitstring track)"),
    ):
        names = [
            n
            for n in ESTIMATOR_LABELS
            if any(_estimate(assessed[day][c], n, track) is not None for c in ("sensor", "fish"))
        ]
        groups.append((header, [(n, track, scale) for n in names]))
    # y positions (row units, top = 0): header, rows ..., gap, header, rows ..., gap, assessed
    rows: list[tuple[float, str, str, int]] = []
    headers: list[tuple[float, str]] = []
    y = 0.0
    for header, items in groups:
        headers.append((y, header))
        y += 0.85
        for name, track, scale in items:
            rows.append((y, name, track, scale))
            y += 1.0
        y += 0.35
    y_final = y + 0.35
    total = y_final + 0.75
    row_in = 0.235
    src = results["runs"][day]["source"]
    frames = results["runs"][day]["frames"]
    modes = "one day run and one night run" if night else "one day run"
    subtitle = (
        "SP 800-90B section 6.3 estimators on the non-IID track, in bits per 8-bit sample. "
        "Lower means more predictable; the assessed min-entropy is the lowest value.\n"
        f"Data: {describe_source(src)}, {frames:,} frames per run, {modes}."
    )
    content_in = total * row_in + 0.75
    canvas = _Canvas(
        theme,
        "Min-entropy of the raw samples, estimator by estimator",
        subtitle,
        _footer(results),
        content_in,
        legend=night is not None,
    )
    label_in = 1.95
    gap_in = 0.34
    panel_in = (WIDTH_IN - 2 * PAD_IN - label_in - gap_in) / 2
    plot_top, plot_h = 0.32, total * row_in
    if night is not None:
        canvas.legend(
            [
                ("day", _key(theme, theme.ink2)),
                ("night (IR camera)", _key(theme, theme.ink2, hollow=True)),
            ],
            PAD_IN + label_in,
        )
    for k, channel in enumerate(("sensor", "fish")):
        ax = canvas.axes(PAD_IN + label_in + k * (panel_in + gap_in), plot_top, panel_in, plot_h)
        _clean(ax, theme, grid_axis="x")
        colour = theme.series[channel]
        ax.set_xlim(0, 8)
        ax.set_ylim(total - 0.2, -0.5)
        ax.set_xticks(range(0, 9))
        ax.set_yticks([])
        ax.axhline(y_final - 0.7, color=theme.axis, linewidth=0.8)
        cap = results.get("engine", {}).get("h_cap", {}).get(channel)
        if cap:
            ax.axvline(cap, color=theme.ink3, linewidth=1.0, alpha=0.55, zorder=1)
            ax.text(
                cap + 0.1,
                headers[0][0],
                f"credit cap {cap:g}",
                fontsize=SMALL_PT,
                color=theme.ink3,
                va="center",
                ha="left",
            )
        offsets = {day: -0.14, night: 0.14} if night else {day: 0.0}
        for yv, name, track, scale in rows:
            pair = []
            for label in (day, night):
                if label is None:
                    continue
                value = _estimate(assessed[label][channel], name, track)
                if value is not None:
                    pair.append((value * scale, yv + offsets[label]))
            if len(pair) == 2:
                ax.plot(*zip(*pair, strict=True), color=colour, linewidth=1.2, alpha=0.4, zorder=2)
        for label in (day, night):
            if label is None:
                continue
            a = assessed[label][channel]
            pts = [
                (v * scale, yv + offsets[label])
                for yv, name, track, scale in rows
                if (v := _estimate(a, name, track)) is not None
            ]
            if a.get("min_entropy") is not None:
                pts.append((float(a["min_entropy"]), y_final + offsets[label]))
            if pts:
                xs, ys = zip(*pts, strict=True)
                _dots(ax, xs, ys, colour, theme, hollow=label == night)
        finals = [
            (lbl, float(assessed[lbl][channel]["min_entropy"]))
            for lbl in (day, night)
            if lbl is not None and assessed[lbl][channel].get("min_entropy") is not None
        ]
        if finals:
            # Each assessed value next to its own dot (filled = day, ring = night, see legend):
            # the lower one on its left, the higher one on its right.
            ordered = sorted(finals, key=lambda item: item[1])
            for rank, (lbl, v) in enumerate(ordered):
                left = rank == 0 and len(ordered) > 1 and v > 1.2
                ax.text(
                    v - 0.28 if left else v + 0.28,
                    y_final + offsets[lbl],
                    f"{v:.2f}",
                    fontsize=LABEL_PT,
                    va="center",
                    ha="right" if left else "left",
                    color=theme.ink,
                    fontweight="semibold",
                )
        n_used = assessed[day][channel].get("n_used", 0)
        title = {"sensor": "Sensor noise", "fish": "Fish motion"}[channel]
        _panel_title(ax, theme, title, f"{n_used:,} samples per run")
        ax.set_xlabel("bits per sample", fontsize=LABEL_PT, labelpad=5)
    lab = canvas.axes(PAD_IN, plot_top, label_in, plot_h)
    lab.set_axis_off()
    lab.set_xlim(0, 1)
    lab.set_ylim(total - 0.2, -0.5)
    for yv, text in headers:
        lab.text(0.0, yv, text, fontsize=6.9, color=theme.ink3, va="center", ha="left")
    for yv, name, _, _ in rows:
        lab.text(
            0.0,
            yv,
            ESTIMATOR_LABELS[name],
            fontsize=LABEL_PT,
            va="center",
            ha="left",
            color=theme.ink2,
        )
    lab.text(
        0.0,
        y_final,
        "Assessed (minimum)",
        fontsize=8.6,
        va="center",
        ha="left",
        color=theme.ink,
        fontweight="semibold",
    )
    return canvas.fig


# ======================================================================== 2. day/night cycle


def _night_spans(
    t: npt.NDArray[np.float64], night: npt.NDArray[np.bool_]
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for ti, ni in zip(t, night, strict=True):
        if ni and start is None:
            start = float(ti)
        elif not ni and start is not None:
            spans.append((start, float(ti)))
            start = None
    if start is not None:
        spans.append((start, float(t[-1])))
    return spans


def chart_cycle(results: Mapping[str, Any], theme: Theme) -> Figure | None:
    """Credited bits per second over day -> night -> day, one panel per channel."""
    cyc = results.get("cycle")
    if not cyc or not cyc.get("bins", {}).get("t"):
        return None
    b = cyc["bins"]
    t = np.asarray(b["t"], dtype=float)
    night = np.asarray(b["night"], dtype=bool)
    src = cyc["source"]
    t_end = float(t[-1]) + float(b["bin_s"])
    subtitle = (
        f"Bits the engine credited in each {b['bin_s']:g} s of simulated time, exactly as "
        "computed per frame (no smoothing). Shaded: night, camera in IR mode, fish asleep.\n"
        f"Data: {describe_source(src, cyc['frames'])}, mode=cycle with "
        f"{src.get('cycle_day_s'):g} s day and {src.get('cycle_night_s'):g} s night."
    )
    panel_h, gap = 1.55, 0.62
    canvas = _Canvas(
        theme,
        "Day, night, day: what each channel earns",
        subtitle,
        _footer(results),
        2 * panel_h + gap + 0.62,
    )
    left_in = 0.62
    width_in = WIDTH_IN - PAD_IN - left_in - PAD_IN
    edges = np.append(t, t_end)
    spans = _night_spans(edges, np.append(night, False))
    phases = cyc.get("phases", [])
    panels = (("sensor", 1e-3, "Sensor noise, kbit/s"), ("fish", 1.0, "Fish motion, bit/s"))
    ax = None
    for k, (channel, scale, title) in enumerate(panels):
        ax = canvas.axes(PAD_IN + left_in, 0.28 + k * (panel_h + gap), width_in, panel_h)
        _clean(ax, theme)
        values = np.asarray(b["bits_per_s"][channel], dtype=float) * scale
        steps = np.append(values, values[-1])
        peak = float(np.max(values)) if values.size else 0.0
        top = _nice_ceiling(max(peak, 1e-9) * 1.3)
        for a, c in spans:
            ax.axvspan(a, c, color=theme.wash, zorder=0, linewidth=0)
        colour = theme.series[channel]
        ax.fill_between(
            edges, 0, steps, step="post", color=colour, alpha=0.10, linewidth=0, zorder=1
        )
        ax.step(edges, steps, where="post", color=colour, linewidth=1.6, zorder=3)
        ax.set_ylim(0, top)
        ax.set_xlim(float(t[0]), t_end)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
        _panel_title(ax, theme, title)
        late = next((p for p in phases if p["phase"] == "night, second half"), None)
        for phase in phases:
            if phase is late:
                continue  # folded into the night label below
            mean = phase[channel]["bits_per_s"] * scale
            mid = (phase["start_s"] + min(phase["end_s"], t_end)) / 2
            text = f"mean {_short(mean)}"
            if phase["phase"] == "night" and late is not None:
                late_mean = late[channel]["bits_per_s"] * scale
                if abs(late_mean - mean) > 0.1 * max(mean, 1e-9):
                    text += f", last {late['end_s'] - late['start_s']:g} s: {_short(late_mean)}"
            ax.text(
                mid, top * 0.96, text, ha="center", va="top", fontsize=LABEL_PT, color=theme.ink
            )
    if ax is not None:
        ax.set_xlabel("simulated time, s", fontsize=LABEL_PT, labelpad=5)
    return canvas.fig


# ======================================================================== 3. liveness


def chart_liveness(results: Mapping[str, Any], theme: Theme) -> Figure | None:
    """Liveness confidence of the live tank and of a bit-exact replay."""
    live = results.get("liveness")
    if not live:
        return None
    src = live["source"]
    subtitle = (
        "Confidence that the video answers the LED challenge, in bits: -log2 of the probability "
        "that a video unaware of the challenge matches this well.\n"
        f"Data: {describe_source(src, live['live']['frames'])}. The replay is the same footage "
        "recorded losslessly (FFV1) and played against a fresh challenge."
    )
    plot_h = 2.55
    canvas = _Canvas(
        theme,
        "Anti-replay: the live tank verifies, its recording never does",
        subtitle,
        _footer(results),
        plot_h + 0.75,
    )
    left_in, label_in = 0.6, 1.25
    ax = canvas.axes(PAD_IN + left_in, 0.12, WIDTH_IN - 2 * PAD_IN - left_in - label_in, plot_h)
    _clean(ax, theme)
    threshold = float(live["threshold_bits"])
    series = {k: live[k]["series"] for k in ("live", "replay")}
    t_end = max(max(sr["t"]) for sr in series.values())
    y_max = max(max(sr["confidence_bits"]) for sr in series.values())
    top = _nice_ceiling(max(y_max, threshold) * 1.15)
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, top)
    window_s = live["window_bits"] * live["symbol_s"]
    ax.axvspan(0, min(window_s, t_end), color=theme.wash, zorder=0, linewidth=0)
    ax.text(
        min(window_s, t_end) / 2,
        top * 0.965,
        f"first {live['window_bits']}-bit window",
        ha="center",
        va="top",
        fontsize=SMALL_PT,
        color=theme.ink3,
    )
    ax.axhline(threshold, color=theme.ink2, linewidth=1.0, zorder=2)
    ax.text(
        t_end * 0.99,
        threshold + top * 0.02,
        f"VERIFIED needs {threshold:g} bits",
        ha="right",
        va="bottom",
        fontsize=SMALL_PT,
        color=theme.ink2,
    )
    for key in ("replay", "live"):
        sr = series[key]
        ax.plot(
            sr["t"],
            sr["confidence_bits"],
            color=theme.series[key],
            linewidth=2.0,
            zorder=4 if key == "live" else 3,
        )
        _dots(ax, [sr["t"][-1]], [sr["confidence_bits"][-1]], theme.series[key], theme, zorder=6)
    lv, rp = live["live"], live["replay"]
    ends = {
        "live": (
            series["live"]["confidence_bits"][-1],
            "Live tank",
            f"up to {lv['max_confidence_bits']:.0f} bits",
        ),
        "replay": (
            series["replay"]["confidence_bits"][-1],
            "Replay",
            f"up to {rp['max_confidence_bits']:.1f} bits,\nnever verified",
        ),
    }
    for _key_name, (value, name, detail) in ends.items():
        ax.annotate(
            f"{name}\n{detail}",
            (t_end, value),
            xytext=(12, 0 if value > top * 0.1 else 8),
            textcoords="offset points",
            fontsize=LABEL_PT,
            color=theme.ink,
            va="center" if value > top * 0.1 else "bottom",
            ha="left",
            annotation_clip=False,
            linespacing=1.35,
        )
    if lv.get("time_to_verified_s") is not None:
        tv = float(lv["time_to_verified_s"])
        _dots(ax, [tv], [threshold], theme.series["live"], theme, zorder=7)
        ax.annotate(
            f"verified at {tv:.1f} s",
            (tv, threshold),
            xytext=(9, -6),
            textcoords="offset points",
            fontsize=LABEL_PT,
            color=theme.ink2,
            ha="left",
            va="top",
        )
    ax.set_xlabel("simulated time, s", fontsize=LABEL_PT, labelpad=5)
    ax.set_ylabel("confidence, bits", fontsize=LABEL_PT, labelpad=6)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    return canvas.fig


# ======================================================================== 4. SP 800-22 proportions

_SHORT_STREAM = {
    "sensor": "Sensor only",
    "fish": "Fish only",
    "combined": "Combined",
    "pufferrng": "PufferRNG output",
}


def chart_nist_proportions(results: Mapping[str, Any], theme: Theme) -> Figure | None:
    """Lowest pass proportion per SP 800-22 test, per stream, against the acceptance band."""
    nist = results["sp800_22"]
    groups: dict[str, list[str]] = {}
    for key, res in nist.items():
        if res.get("mode") == "multi":
            groups.setdefault(key.partition("/")[2], []).append(key)
    order = [g for g in ("sensor", "fish", "combined", "pufferrng") if g in groups]
    if not order:
        return None
    tests = [
        t for t in TEST_LABELS if any(t in nist[k]["per_test"] for g in order for k in groups[g])
    ]
    n_rows = len(tests)
    runs = results["runs"]

    def is_night(key: str) -> bool:
        return bool(runs.get(key.partition("/")[0], {}).get("night_fraction", 0) > 0.5)

    first = _first_run(results)
    src = runs[first]["source"] if first else {}
    counts = "; ".join(
        f"{_SHORT_STREAM[g]} {' and '.join(str(nist[k]['sequences']) for k in groups[g])}"
        for g in order
    )
    n_bits = nist[groups[order[0]][0]]["n_bits"]
    subtitle = (
        "For each test, the lowest share of sequences with p >= 0.01 over its sub-tests. Shaded: "
        "the SP 800-22 acceptance region for the number of sequences the test ran on (excursion "
        "tests apply to fewer). A dot left of its band means a sub-test fell outside.\n"
        f"Sequences of {n_bits:,} bits ({counts}). Conditioned streams use no OS randomness. "
        f"Data: {describe_source(src)}."
    )
    row_in = 0.27
    plot_h = n_rows * row_in + 0.1
    has_night = any(is_night(k) for g in order for k in groups[g])
    canvas = _Canvas(
        theme,
        "SP 800-22 pass proportions per test",
        subtitle,
        _footer(results),
        plot_h + 0.95,
        legend=has_night,
    )
    label_in, gap_in = 1.75, 0.42
    k_panels = len(order)
    panel_in = (WIDTH_IN - 2 * PAD_IN - label_in - gap_in * (k_panels - 1)) / k_panels
    lowest = min(
        min(p["proportion_interval"][0], p["min_proportion"])
        for g in order
        for k in groups[g]
        for p in nist[k]["per_test"].values()
    )
    x_min = math.floor((lowest - 0.025) * 20) / 20
    if has_night:
        canvas.legend(
            [
                ("day run", _key(theme, theme.ink2)),
                ("night run", _key(theme, theme.ink2, hollow=True)),
            ],
            PAD_IN + label_in,
        )
    y = np.arange(n_rows, dtype=float)
    for i, group in enumerate(order):
        ax = canvas.axes(PAD_IN + label_in + i * (panel_in + gap_in), 0.3, panel_in, plot_h)
        _clean(ax, theme, grid_axis="x")
        ax.set_xlim(x_min, 1.0)
        ax.set_ylim(n_rows - 0.4, -0.6)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MultipleLocator(0.1 if 1.0 - x_min > 0.16 else 0.05))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
        colour = theme.series[_STREAM_COLOUR[group]]
        keys = groups[group]
        for j, key in enumerate(keys):
            offset = 0.0 if len(keys) == 1 else (-0.17 if j == 0 else 0.17)
            half = 0.36 if len(keys) == 1 else 0.17
            per_test = nist[key]["per_test"]
            pts = []
            for r, test in enumerate(tests):
                pt = per_test.get(test)
                if pt is None:
                    continue
                lo = pt["proportion_interval"][0]
                yc = y[r] + offset
                ax.fill_between(
                    [lo, 1.0], yc - half, yc + half, color=theme.wash, linewidth=0, zorder=0
                )
                ax.plot(
                    [lo, lo],
                    [yc - half, yc + half],
                    color=theme.axis,
                    linewidth=1.0,
                    zorder=1,
                    solid_capstyle="butt",
                )
                pts.append((pt["min_proportion"], yc))
            if pts:
                _dots(
                    ax,
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    colour,
                    theme,
                    hollow=is_night(key),
                    size=6.2,
                )
        for r, test in enumerate(tests):
            if all(test not in nist[k]["per_test"] for k in keys):
                ax.text(
                    (x_min + 1.0) / 2,
                    y[r],
                    "not applicable",
                    fontsize=SMALL_PT,
                    color=theme.ink3,
                    ha="center",
                    va="center",
                )
        _panel_title(ax, theme, _SHORT_STREAM[group])
    lab = canvas.axes(PAD_IN, 0.3, label_in, plot_h)
    lab.set_axis_off()
    lab.set_ylim(n_rows - 0.4, -0.6)
    lab.set_xlim(0, 1)
    for r, test in enumerate(tests):
        lab.text(0.0, y[r], TEST_LABELS[test], fontsize=LABEL_PT, color=theme.ink2, va="center")
    canvas.fig.text(
        canvas.x(PAD_IN + label_in + (WIDTH_IN - 2 * PAD_IN - label_in) / 2),
        canvas.y(canvas.content_top_in + 0.3 + plot_h + 0.42),
        "share of sequences with p >= 0.01",
        ha="center",
        va="top",
        fontsize=LABEL_PT,
        color=theme.ink3,
    )
    return canvas.fig


# ======================================================================== 5. p-value uniformity


def chart_pvalues(results: Mapping[str, Any], theme: Theme) -> Figure | None:
    """Histogram of every SP 800-22 p-value of the PufferRNG output (should be flat)."""
    key = next((k for k in results["sp800_22"] if k.endswith("/pufferrng")), None)
    if key is None or results["sp800_22"][key].get("mode") != "multi":
        return None
    res = results["sp800_22"][key]
    values = np.asarray([p for ps in res["p_values"].values() for p in ps], dtype=float)
    if not values.size:
        return None
    bins = 20
    counts, edges = np.histogram(np.clip(values, 0, 1), bins=bins, range=(0.0, 1.0))
    n = int(values.size)
    expected = n / bins
    sigma = math.sqrt(n * (1 / bins) * (1 - 1 / bins))
    uniform_ok = sum(1 for sub in res["subtests"] if sub["uniform"] is not False)
    assessed = (
        "assessed" if res.get("uniformity_assessed") else "not assessed (fewer than 55 sequences)"
    )
    subtitle = (
        f"All {n:,} p-values of {res['subtest_count']} sub-tests on {res['sequences']} sequences "
        f"of {res['n_bits']:,} bits, in 20 bins. A good generator gives a flat histogram; the "
        f"band is the expected count +- 2.58 sigma.\n"
        f"Per sub-test uniformity (SP 800-22 4.2.2, {assessed}): {uniform_ok} of "
        f"{res['subtest_count']} sub-tests pass."
    )
    plot_h = 2.3
    canvas = _Canvas(
        theme,
        "PufferRNG output: SP 800-22 p-values are uniform",
        subtitle,
        _footer(results),
        plot_h + 0.75,
    )
    left_in, right_in = 0.62, 0.95
    ax = canvas.axes(PAD_IN + left_in, 0.12, WIDTH_IN - 2 * PAD_IN - left_in - right_in, plot_h)
    _clean(ax, theme)
    colour = theme.series["system"]
    width = 1 / bins
    ax.bar(edges[:-1] + width / 2, counts, width=width * 0.82, color=colour, zorder=3, linewidth=0)
    band_lo, band_hi = expected - 2.58 * sigma, expected + 2.58 * sigma
    ax.axhspan(band_lo, band_hi, color=theme.wash, zorder=1, linewidth=0)
    for bound in (band_lo, band_hi):
        ax.axhline(bound, color=theme.ink3, linewidth=0.8, alpha=0.8, zorder=5)
    ax.axhline(expected, color=theme.ink, linewidth=1.1, zorder=5)
    ax.annotate(
        f"expected\n{expected:,.0f} per bin",
        (1.0, expected),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=LABEL_PT,
        color=theme.ink2,
        annotation_clip=False,
        linespacing=1.35,
    )
    ax.set_xlim(0, 1)
    top = _nice_ceiling(max(float(np.max(counts)), band_hi) * 1.12)
    ax.set_ylim(0, top)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("p-value", fontsize=LABEL_PT, labelpad=5)
    ax.set_ylabel("p-values per bin", fontsize=LABEL_PT, labelpad=6)
    return canvas.fig


# ======================================================================== 6. ent raw vs conditioned


_EntRow = tuple[str, str, str, Mapping[str, Any], float | None]


def chart_ent(results: Mapping[str, Any], theme: Theme) -> Figure | None:
    """Shannon entropy per byte (ent) of raw samples versus hashed streams."""
    label = _first_run(results)
    if label is None:
        return None
    ent = results["ent"]
    rows: list[_EntRow] = []
    for channel, name in (("sensor", "Sensor noise samples"), ("fish", "Fish motion samples")):
        stats = ent["raw"].get(f"{label}/{channel}")
        if stats:
            h = results["sp800_90b"][label][channel].get("min_entropy")
            rows.append(("RAW SAMPLES", name, channel, stats, h))
    for stream in ("sensor", "fish", "combined"):
        stats = ent["streams"].get(f"{label}/{stream}")
        if stats:
            rows.append(
                (
                    "HASHED, NO OS RANDOMNESS",
                    STREAM_LABELS[stream],
                    _STREAM_COLOUR[stream],
                    stats,
                    None,
                )
            )
    stats = ent["streams"].get(f"{label}/pufferrng")
    if stats:
        rows.append(("KEYED WITH OS.URANDOM", "PufferRNG output", "system", stats, None))
    if not rows:
        return None
    slots: list[float] = []
    headers: list[tuple[float, str]] = []
    pos, prev = 0.0, None
    for row in rows:
        if row[0] != prev:
            if prev is not None:
                pos += 0.45
            headers.append((pos, row[0]))
            pos += 0.8
            prev = row[0]
        slots.append(pos)
        pos += 1.0
    total = pos - 0.4
    row_in = 0.33
    src = results["runs"][label]["source"]
    subtitle = (
        "Bars: Shannon entropy per byte measured by ent (8.0 means a perfectly flat byte "
        "histogram). Diamonds: SP 800-90B min-entropy per sample of the same raw samples.\n"
        f"Data: {describe_source(src, results['runs'][label]['frames'])}, one day run."
    )
    canvas = _Canvas(
        theme,
        "Hashing makes anything look random; min-entropy does not",
        subtitle,
        _footer(results),
        total * row_in + 0.8,
    )
    label_in, verdict_in = 1.95, 1.5
    plot_w = WIDTH_IN - 2 * PAD_IN - label_in - verdict_in
    ax = canvas.axes(PAD_IN + label_in, 0.12, plot_w, total * row_in)
    _clean(ax, theme, grid_axis="x")
    ax.set_xlim(0, 8.0)
    ax.set_ylim(total, -0.5)
    ax.set_yticks([])
    ax.set_xticks(range(0, 9))
    ax.set_xlabel("bits per byte (bars) or per sample (diamonds)", fontsize=LABEL_PT, labelpad=5)
    ax.text(
        1.0 + 0.12 / plot_w,
        -0.5,
        "ent chi-square test",
        transform=ax.get_yaxis_transform(),
        fontsize=SMALL_PT,
        color=theme.ink3,
        ha="left",
        va="center",
    )
    bar_h = 0.56
    for (_, name, colour_key, stats, h), yv in zip(rows, slots, strict=True):
        colour = theme.series[colour_key]
        e = float(stats["entropy"])
        ax.barh(yv, e, height=bar_h, color=colour, zorder=3, linewidth=0)
        text = f"{e:.4f}" if e >= 7.9 else f"{e:.2f}"
        if e > 1.2:
            ax.text(
                e - 0.08,
                yv,
                text,
                va="center",
                ha="right",
                fontsize=7.8,
                color="#ffffff",
                fontweight="semibold",
                zorder=5,
            )
        else:
            ax.text(
                e + 0.08, yv, text, va="center", ha="left", fontsize=7.8, color=theme.ink, zorder=5
            )
        if h is not None:
            ax.plot(
                [h],
                [yv],
                marker="D",
                markersize=7.2,
                color=theme.ink,
                markeredgecolor=theme.surface,
                markeredgewidth=1.4,
                zorder=6,
            )
            ax.annotate(
                f"min-entropy {h:.2f}",
                (h, yv),
                xytext=(0, 9),
                textcoords="offset points",
                va="bottom",
                ha="center",
                fontsize=SMALL_PT,
                color=theme.ink2,
                zorder=6,
            )
        ax.text(
            -0.015,
            yv,
            name,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=LABEL_PT,
            color=theme.ink2,
        )
        pct = float(stats["chi_square_percent"])
        ok = 1.0 <= pct <= 99.0
        detail = "< 0.01 %" if pct < 0.01 else ("> 99.99 %" if pct > 99.99 else f"{pct:.1f} %")
        ax.text(
            1.0 + 0.12 / plot_w,
            yv,
            f"{'pass' if ok else 'FAIL'}  ({detail})",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=LABEL_PT,
            color=theme.ink if not ok else theme.ink2,
            fontweight="semibold" if not ok else "normal",
        )
    for yv, text in headers:
        ax.text(
            -0.015,
            yv,
            text,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.9,
            color=theme.ink3,
        )
    return canvas.fig


# ======================================================================== driver

ChartFn = Callable[[Mapping[str, Any], Theme], Figure | None]

CHARTS: Final[dict[str, tuple[ChartFn, str]]] = {
    "min-entropy": (
        chart_min_entropy,
        "SP 800-90B min-entropy per sample by estimator, sensor and fish, day and night",
    ),
    "day-night-cycle": (chart_cycle, "Credited bits per second over a day, night, day cycle"),
    "liveness-replay": (
        chart_liveness,
        "Liveness confidence over time: the live tank verifies, its replay never does",
    ),
    "sp800-22-proportions": (
        chart_nist_proportions,
        "SP 800-22 pass proportions per test for conditioned streams and PufferRNG output",
    ),
    "sp800-22-pvalues": (chart_pvalues, "Histogram of SP 800-22 p-values of PufferRNG output"),
    "ent-raw-vs-conditioned": (
        chart_ent,
        "ent entropy per byte of raw samples versus hashed streams, with SP 800-90B min-entropy",
    ),
}
"""Chart name -> (renderer, alt text). Files are ``<name>-light.png`` and ``<name>-dark.png``."""


def render_all(results: Mapping[str, Any], media_dir: Path) -> dict[str, str]:
    """Render every chart the results support, in both themes; return ``{name: alt text}``."""
    media_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, (render, alt) in CHARTS.items():
        for theme in THEMES:
            with rc_context(cast(Any, _style(theme))):
                fig = render(results, theme)
                if fig is None:
                    break
                fig.savefig(
                    media_dir / f"{name}-{theme.name}.png", dpi=DPI, facecolor=theme.surface
                )
        else:
            written[name] = alt
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m puffer.analysis.plots results.json media_dir``: re-render the charts."""
    import json

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m puffer.analysis.plots RESULTS_JSON MEDIA_DIR", file=sys.stderr)
        return 2
    results = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    for name in render_all(results, Path(args[1])):
        print(f"{args[1]}/{name}-light.png, {name}-dark.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
