"""The four outputs: which score to trust, the reliability diagram, the threshold sweep,
and whether a single global threshold survives document quality.

Reads every cached response in `results/extractions/`, scores each field against its ground
truth with `compare.py`, and writes the combined table plus four figures to `results/`. This
is the only module a reader needs to rerun to reproduce every number in the notebook, once
`run.py` has populated the cache.

Run: python -m calibration.analyse          (from confidence-calibration/)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.compare import score_rows  # noqa: E402
from corpus.cord import FIELD_TYPES as CORD_FIELD_TYPES  # noqa: E402
from corpus.degrade import TIERS  # noqa: E402
from corpus.schemas import FIELD_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
EXTRACTIONS_DIR = RESULTS_DIR / "extractions"
FIGURES_DIR = RESULTS_DIR / "figures"
DOCS_DIR = ROOT / "documents"
CORD_DIR = DOCS_DIR / "cord"

TIER_LABELS = {"t0": "T0 clean PDF", "t1": "T1 scanned", "t2": "T2 scanned+noisy", "t3": "T3 photographed"}
TIER_COLOR = {"t0": "#1f7a3d", "t1": "#2f6fbd", "t2": "#b8860b", "t3": "#c0392b"}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "font.size": 11,
        "axes.grid": True,
        "grid.color": "#e4e4e4",
        "grid.linewidth": 0.7,
    }
)


def load_rows() -> pd.DataFrame:
    """Score every cached extraction against its ground truth and return one long table."""
    records = []

    for truth_file in sorted(DOCS_DIR.glob("*.truth.json")):
        stem = truth_file.name.removesuffix(".truth.json")
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        for tier in TIERS:
            cache = EXTRACTIONS_DIR / f"{stem}__{tier}.json"
            if not cache.exists():
                continue
            result = json.loads(cache.read_text(encoding="utf-8"))["response"]["result"]
            context = {
                "corpus": "generated",
                "document": stem,
                "doc_type": truth["doc_type"],
                "tier": tier,
            }
            records.extend(
                score_rows(result, truth["fields"], FIELD_TYPES[truth["doc_type"]], context)
            )

    for truth_file in sorted(CORD_DIR.glob("*.truth.json")):
        stem = truth_file.name.removesuffix(".truth.json")
        cache = EXTRACTIONS_DIR / f"cord_{stem}.json"
        if not cache.exists():
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        result = json.loads(cache.read_text(encoding="utf-8"))["response"]["result"]
        context = {"corpus": "cord", "document": stem, "doc_type": "receipt", "tier": "real"}
        records.extend(score_rows(result, truth["fields"], CORD_FIELD_TYPES, context))

    frame = pd.DataFrame(records)
    frame = frame[frame["correct"].notna()].copy()
    frame["correct"] = frame["correct"].astype(bool)
    frame["min_score"] = frame[["grounding_score", "extraction_score"]].min(axis=1)
    return frame


# ---------------------------------------------------------------------------
# Output 1: which score predicts correctness
# ---------------------------------------------------------------------------


def score_discrimination(frame: pd.DataFrame) -> pd.DataFrame:
    """AUC-style separation: for each candidate score, P(correct row scores higher than
    incorrect row). 0.5 is a coin flip, 1.0 perfectly separates the two."""
    rows = []
    for label, column in [
        ("grounding_score", "grounding_score"),
        ("extraction_score", "extraction_score"),
        ("min(grounding, extraction)", "min_score"),
    ]:
        correct = frame.loc[frame["correct"], column].dropna().to_numpy()
        wrong = frame.loc[~frame["correct"], column].dropna().to_numpy()
        counts = {"n_correct": len(correct), "n_wrong": len(wrong)}
        if len(correct) == 0 or len(wrong) == 0:
            rows.append({"score": label, "separation": float("nan"), **counts})
            continue
        # Vectorised Mann-Whitney U / n1*n2: the AUC of using this score as a
        # correct-vs-wrong classifier.
        wins = (correct[:, None] > wrong[None, :]).mean()
        ties = (correct[:, None] == wrong[None, :]).mean()
        rows.append({"score": label, "separation": wins + 0.5 * ties, **counts})
    return pd.DataFrame(rows)


def plot_score_discrimination(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#2f6fbd", "#b8860b", "#1f7a3d"]
    bars = ax.barh(table["score"], table["separation"], color=colors)
    ax.axvline(0.5, color="#999999", linewidth=1, linestyle="--")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Separation: P(correct field scores higher than an incorrect one)")
    ax.set_title("Which score actually predicts correctness")
    for bar, value in zip(bars, table["separation"], strict=True):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output 2: reliability diagram
# ---------------------------------------------------------------------------


def reliability_table(frame: pd.DataFrame, column: str, n_bins: int = 10) -> pd.DataFrame:
    working = frame.dropna(subset=[column]).copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    working["bin"] = pd.cut(working[column], edges, include_lowest=True)
    grouped = working.groupby(["tier", "bin"], observed=True).agg(
        confidence=(column, "mean"), accuracy=("correct", "mean"), n=("correct", "size")
    )
    return grouped.reset_index()


def expected_calibration_error(frame: pd.DataFrame, column: str, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for tier, group in frame.dropna(subset=[column]).groupby("tier"):
        table = reliability_table(group.assign(tier=tier), column, n_bins)
        total = table["n"].sum()
        if total == 0:
            continue
        ece = (table["n"] * (table["confidence"] - table["accuracy"]).abs()).sum() / total
        rows.append({"tier": tier, "score": column, "ece": ece, "n": total})
    return pd.DataFrame(rows)


def plot_reliability_diagram(frame: pd.DataFrame, column: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="perfect calibration")

    for tier in [t for t in TIERS if t in frame["tier"].unique()]:
        table = reliability_table(frame[frame["tier"] == tier], column)
        table = table[table["n"] > 0]
        if table.empty:
            continue
        ax.plot(
            table["confidence"], table["accuracy"],
            marker="o", color=TIER_COLOR[tier], label=TIER_LABELS[tier],
        )
        for _, row in table.iterrows():
            ax.annotate(str(int(row["n"])), (row["confidence"], row["accuracy"]),
                        fontsize=7, color=TIER_COLOR[tier], xytext=(3, 3), textcoords="offset points")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"Mean {column.replace('_', ' ')} in bucket")
    ax.set_ylabel("Observed accuracy in bucket")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output 3: threshold sweep
# ---------------------------------------------------------------------------


def threshold_sweep(frame: pd.DataFrame, column: str, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.5, 1.0, 0.02), 2)
    rows = []
    for tier in [t for t in TIERS if t in frame["tier"].unique()]:
        tier_frame = frame[frame["tier"] == tier].dropna(subset=[column])
        total = len(tier_frame)
        if total == 0:
            continue
        for threshold in thresholds:
            accepted = tier_frame[tier_frame[column] >= threshold]
            error_rate = 1.0 - accepted["correct"].mean() if len(accepted) else float("nan")
            rows.append(
                {
                    "tier": tier,
                    "threshold": threshold,
                    "coverage": len(accepted) / total,
                    "error_rate": error_rate,
                    "n_accepted": len(accepted),
                }
            )
    return pd.DataFrame(rows)


def sub_1pct_thresholds(sweep: pd.DataFrame) -> pd.DataFrame:
    """For each tier, the lowest threshold whose accepted set has under 1% error."""
    rows = []
    for tier, group in sweep.groupby("tier"):
        safe = group[(group["error_rate"] <= 0.01) & (group["n_accepted"] >= 5)]
        if safe.empty:
            note = "not reached in this corpus"
            rows.append({"tier": tier, "threshold": float("nan"), "coverage": float("nan"), "note": note})
        else:
            best = safe.sort_values("threshold").iloc[0]
            rows.append(
                {"tier": tier, "threshold": best["threshold"], "coverage": best["coverage"], "note": ""}
            )
    return pd.DataFrame(rows)


def plot_threshold_sweep(sweep: pd.DataFrame, path: Path) -> None:
    fig, (ax_cov, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))
    for tier in [t for t in TIERS if t in sweep["tier"].unique()]:
        table = sweep[sweep["tier"] == tier]
        label, color = TIER_LABELS[tier], TIER_COLOR[tier]
        ax_cov.plot(table["threshold"], table["coverage"], color=color, label=label)
        ax_err.plot(table["threshold"], table["error_rate"] * 100, color=color, label=label)

    ax_cov.set_xlabel("Threshold")
    ax_cov.set_ylabel("Coverage (fraction auto-accepted)")
    ax_cov.set_title("How much gets auto-accepted")
    ax_cov.set_ylim(0, 1.02)

    ax_err.axhline(1.0, color="#999999", linestyle="--", linewidth=1)
    ax_err.set_xlabel("Threshold")
    ax_err.set_ylabel("Error rate among accepted, %")
    ax_err.set_title("Error rate among what gets through")
    ax_err.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output 4: does one global threshold survive quality
# ---------------------------------------------------------------------------


def fixed_threshold_table(frame: pd.DataFrame, column: str, thresholds=(0.85, 0.97)) -> pd.DataFrame:
    rows = []
    for tier in [t for t in TIERS if t in frame["tier"].unique()]:
        tier_frame = frame[frame["tier"] == tier].dropna(subset=[column])
        for threshold in thresholds:
            accepted = tier_frame[tier_frame[column] >= threshold]
            error_rate = 1.0 - accepted["correct"].mean() if len(accepted) else float("nan")
            rows.append(
                {
                    "tier": tier,
                    "threshold": threshold,
                    "coverage": len(accepted) / len(tier_frame) if len(tier_frame) else float("nan"),
                    "error_rate": error_rate,
                    "n": len(tier_frame),
                }
            )
    return pd.DataFrame(rows)


def plot_fixed_thresholds(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tiers = [t for t in TIERS if t in table["tier"].unique()]
    x = np.arange(len(tiers))
    width = 0.35
    offsets = (-width / 2, width / 2)
    for offset, threshold in zip(offsets, sorted(table["threshold"].unique()), strict=True):
        values = [
            table[(table["tier"] == t) & (table["threshold"] == threshold)]["error_rate"].mean() * 100
            for t in tiers
        ]
        ax.bar(x + offset, values, width, label=f"threshold {threshold:g}")
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1, label="1% error")
    ax.set_xticks(x, [TIER_LABELS[t] for t in tiers], rotation=15, ha="right")
    ax.set_ylabel("Error rate among accepted, %")
    ax.set_title("Do the shipped thresholds (0.85 / 0.97) hold up as quality drops?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_rows()
    frame.to_csv(RESULTS_DIR / "scored_fields.csv", index=False)
    print(f"Scored {len(frame)} fields across {frame['document'].nunique()} documents")
    print(frame.groupby("tier")["correct"].agg(["mean", "size"]))

    discrimination = score_discrimination(frame)
    discrimination.to_csv(RESULTS_DIR / "score_discrimination.csv", index=False)
    plot_score_discrimination(discrimination, FIGURES_DIR / "01_score_discrimination.png")
    print("\n[1] score discrimination:\n", discrimination)

    generated = frame[frame["corpus"] == "generated"]
    plot_reliability_diagram(
        generated, "grounding_score", FIGURES_DIR / "02_reliability_grounding.png",
        "Reliability diagram: grounding_score (generated corpus, by tier)",
    )
    ece = expected_calibration_error(generated, "grounding_score")
    ece.to_csv(RESULTS_DIR / "calibration_error.csv", index=False)
    print("\n[2] expected calibration error (grounding_score):\n", ece)

    sweep = threshold_sweep(generated, "grounding_score")
    sweep.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    plot_threshold_sweep(sweep, FIGURES_DIR / "03_threshold_sweep.png")
    safe = sub_1pct_thresholds(sweep)
    safe.to_csv(RESULTS_DIR / "sub_1pct_thresholds.csv", index=False)
    print("\n[3] sub-1% threshold per tier:\n", safe)

    fixed = fixed_threshold_table(generated, "grounding_score")
    fixed.to_csv(RESULTS_DIR / "fixed_thresholds.csv", index=False)
    plot_fixed_thresholds(fixed, FIGURES_DIR / "04_fixed_thresholds.png")
    print("\n[4] shipped thresholds (0.85 / 0.97) across tiers:\n", fixed)

    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
