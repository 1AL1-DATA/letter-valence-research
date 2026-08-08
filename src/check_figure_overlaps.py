"""Reproducible figure-quality gate: no two text artists may overlap.

Regenerates every figure through its own module and, at savefig time, checks
the window extents of all rendered text (labels, tick labels, titles, legend
texts, annotations) for pairwise bounding-box collisions. Exits non-zero if any
figure has an overlap above the threshold, so it can gate the pipeline.

Run from the research repo root:
    /home/a/esg-dashboard/.venv/bin/python -m src.check_figure_overlaps

Threshold note: a pair counts as an overlap when both axes overlap by more than
2 px in each direction (boxes of ordinary neighbouring labels touch without
merging). Pure touching is allowed; merging is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MIN_OVERLAP_PX = 2.0
failures: list[str] = []


def _texts(ax):
    out = list(ax.texts) + list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    out += [ax.title, ax.xaxis.label, ax.yaxis.label]
    legend = ax.get_legend()
    if legend is not None:
        out += legend.get_texts()
    return [t for t in out if t.get_text().strip()]


def check(fig, out_path) -> None:
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        renderer = FigureCanvasAgg(fig).get_renderer()
    total = 0
    for ax in fig.axes:
        ts = _texts(ax)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                ea = ts[i].get_window_extent(renderer=renderer)
                eb = ts[j].get_window_extent(renderer=renderer)
                ix = min(ea.x1, eb.x1) - max(ea.x0, eb.x0)
                iy = min(ea.y1, eb.y1) - max(ea.y0, eb.y0)
                if ix > MIN_OVERLAP_PX and iy > MIN_OVERLAP_PX:
                    total += 1
                    print(f"  OVERLAP {ts[i].get_text()[:40]!r} x {ts[j].get_text()[:40]!r}")
    status = "OK" if total == 0 else f"FAIL ({total} overlaps)"
    print(f"{out_path}: {status}")
    if total:
        failures.append(str(out_path))


def _wrap_and_run(module_name: str) -> None:
    orig = matplotlib.figure.Figure.savefig

    def checked(fig, *a, **k):
        check(fig, a[0])
        return orig(fig, *a, **k)

    matplotlib.figure.Figure.savefig = checked
    try:
        __import__(module_name, fromlist=["main"]).main()
    finally:
        matplotlib.figure.Figure.savefig = orig


def main() -> None:
    for mod in ("src.figures", "src.figures_cascade", "src.figures_general"):
        print(f"== {mod} ==")
        _wrap_and_run(mod)
    print("== src.visualise ==")
    _wrap_and_run("src.visualise")
    print()
    if failures:
        print("FIGURE QUALITY GATE FAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("FIGURE QUALITY GATE PASSED: no text overlaps in any figure.")


if __name__ == "__main__":
    main()
