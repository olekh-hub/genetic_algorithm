from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

REP_PALETTE = {"binary": "#0072B2", "real": "#D55E00"}
ENGINE_PALETTE = {
    "custom": "#4C72B0",
    "pygad": "#DD8452",
}
IMPL_ORDER = ["custom_binary", "pygad_none_binary", "custom_real", "pygad_none_real"]


def _style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.88)
    plt.rcParams.update({
        "figure.facecolor": "#f6f7fb",
        "axes.facecolor": "#f6f7fb",
        "axes.edgecolor": "#d0d4e4",
        "axes.labelcolor": "#1c1f26",
        "text.color": "#1c1f26",
        "grid.color": "#e3e6ef",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "600",
    })


def _nice_axis(
    ax,
    rotate_x: float | None = None,
    *,
    categorical_x: bool = False,
    categorical_y: bool = False,
):
    if not categorical_y:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    if rotate_x:
        ax.tick_params(axis="x", rotation=rotate_x, labelsize=9)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    elif not categorical_x:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))


def _normalize_categoricals(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for col in ("selection", "crossover", "mutation", "engine", "representation"):
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip().str.lower()
    return d


SELECTION_ORDER = ["best", "roulette", "tournament"]
SELECTION_LABELS = {"best": "Best", "roulette": "Roulette", "tournament": "Tournament"}
ENGINE_LABELS = {"custom": "Custom (P1/P2)", "pygad": "PyGAD"}
REP_LABELS = {"binary": "Binary", "real": "Real"}
CROSSOVER_LABELS = {
    "single_point": "Single-point",
    "two_point": "Two-point",
    "uniform": "Uniform",
    "granular": "Granular",
    "arithmetic": "Arithmetic",
    "blend_alpha": "BLX-α",
    "blend_alpha_beta": "BLX-αβ",
    "averaging": "Averaging",
    "linear": "Linear",
}


def _apply_labels(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "selection" in d.columns:
        d["selection"] = d["selection"].map(lambda x: SELECTION_LABELS.get(x, x.title()))
    if "representation" in d.columns:
        d["representation"] = d["representation"].astype(str).str.lower().map(
            lambda x: REP_LABELS.get(x, x.title()),
        )
    if "crossover" in d.columns:
        d["crossover"] = d["crossover"].map(
            lambda x: CROSSOVER_LABELS.get(x, x.replace("_", " ").title()),
        )
    return d


def _agg(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    agg_dict = {}
    for m in metrics:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    out = df.groupby(keys, observed=True).agg(**agg_dict).reset_index()
    for m in metrics:
        out[f"{m}_std"] = out[f"{m}_std"].fillna(0)
    return out


def _agg_lookup(agg: pd.DataFrame, cat_col: str, cat: str, engine: str, val_col: str) -> float:
    mask = (agg[cat_col] == cat) & (agg["engine"] == engine)
    if not mask.any():
        return float("nan")
    return float(agg.loc[mask, val_col].iloc[0])


def _grouped_barv(
    ax,
    agg: pd.DataFrame,
    cat_col: str,
    val_col: str,
    categories: list[str],
    engines: list[str] | None = None,
) -> None:
    """Grouped vertical bars — one tick per category, engines side by side."""
    engines = engines or ["custom", "pygad"]
    x = np.arange(len(categories))
    width = 0.36
    for i, eng in enumerate(engines):
        vals = [_agg_lookup(agg, cat_col, c, eng, val_col) for c in categories]
        offset = (i - (len(engines) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=ENGINE_PALETTE[eng],
            label=ENGINE_LABELS[eng],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.xaxis.set_major_locator(mticker.FixedLocator(x))


def _grouped_barh(
    ax,
    agg: pd.DataFrame,
    cat_col: str,
    val_col: str,
    categories: list[str],
    engines: list[str] | None = None,
) -> None:
    """Grouped horizontal bars — one tick per category, engines side by side."""
    engines = engines or ["custom", "pygad"]
    y = np.arange(len(categories))
    height = 0.36
    for i, eng in enumerate(engines):
        vals = [_agg_lookup(agg, cat_col, c, eng, val_col) for c in categories]
        offset = (i - (len(engines) - 1) / 2) * height
        ax.barh(
            y + offset,
            vals,
            height=height,
            color=ENGINE_PALETTE[eng],
            label=ENGINE_LABELS[eng],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.yaxis.set_major_locator(mticker.FixedLocator(y))
    ax.invert_yaxis()


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _label_row(row: pd.Series) -> str:
    impl = row.get("implementation", "")
    if pd.isna(impl):
        eng = row.get("engine", "custom")
        rep = row.get("representation", row.get("variant", ""))
        par = row.get("parallel", "none")
        impl = f"{eng}_{par}_{rep}" if eng == "pygad" else f"custom_{rep}"
    return str(impl)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    d = _normalize_categoricals(df.copy())
    if "engine" not in d.columns and "implementation" in d.columns:
        d["engine"] = d["implementation"].astype(str).str.contains("pygad", regex=False).map(
            {True: "pygad", False: "custom"},
        )
    if "parallel" not in d.columns:
        d["parallel"] = "none"
    if "use_custom_ops" not in d.columns:
        d["use_custom_ops"] = False
    if "implementation" not in d.columns:
        d["implementation"] = d.apply(_label_row, axis=1)
    if "error" in d.columns:
        d = d[d["error"].isna() | (d["error"].astype(str).str.len() == 0)]
    if "best_fitness" in d.columns:
        d = d[d["best_fitness"].notna()]
    d["representation"] = pd.Categorical(
        d["representation"], categories=["binary", "real"], ordered=True,
    )
    return d


def filter_baseline_runs(d: pd.DataFrame) -> pd.DataFrame:
    """Serial PyGAD/custom runs without custom operator overrides."""
    d = d.copy()
    if "parallel" in d.columns:
        d = d[d["parallel"].fillna("none").astype(str) == "none"]
    if "use_custom_ops" in d.columns:
        d = d[~_as_bool(d["use_custom_ops"])]
    return d


def plot_encoding_overview(df: pd.DataFrame, out: Path) -> plt.Figure:
    raw = filter_baseline_runs(prepare_df(df))
    agg = _agg(_apply_labels(raw), ["representation", "engine"], ["best_fitness", "elapsed"])
    rep_order = [REP_LABELS["binary"], REP_LABELS["real"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=140)
    fig.subplots_adjust(wspace=0.32, top=0.88, bottom=0.14)

    for ax, metric, ylabel, title in zip(
        axes,
        ["best_fitness", "elapsed"],
        ["Median best fitness", "Median time (s)"],
        ["Best Fitness", "Wall Time"],
    ):
        _grouped_barv(ax, agg, "representation", f"{metric}_mean", rep_order)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Encoding")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.55)
        ax.legend(fontsize=8, frameon=True, loc="upper right")
        _nice_axis(ax, categorical_x=True)

    fig.suptitle("Custom (P1/P2) vs PyGAD", fontsize=14, weight="650")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_selection_effect(df: pd.DataFrame, out: Path) -> plt.Figure:
    raw = filter_baseline_runs(prepare_df(df))
    sel_order = [SELECTION_LABELS[s] for s in SELECTION_ORDER]
    rep_keys = [("binary", REP_LABELS["binary"]), ("real", REP_LABELS["real"])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.90, bottom=0.12)

    for col, (rep_key, rep_label) in enumerate(rep_keys):
        sub = _apply_labels(raw[raw["representation"] == rep_key])
        for row, (metric, ylabel) in enumerate([
            ("best_fitness", "Median best fitness"),
            ("elapsed", "Median time (s)"),
        ]):
            ax = axes[row, col]
            agg = _agg(sub, ["selection", "engine"], [metric])
            _grouped_barv(ax, agg, "selection", f"{metric}_mean", sel_order)
            title = "Best Fitness" if metric == "best_fitness" else "Wall Time"
            ax.set_title(f"{rep_label} — {title}", fontsize=11)
            ax.set_xlabel("Selection" if row == 1 else "")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.grid(True, axis="y", linestyle="--", alpha=0.55)
            _nice_axis(ax, rotate_x=20 if row == 1 else None, categorical_x=True)
            if col == 1 and row == 0:
                ax.legend(fontsize=7, frameon=True, loc="upper right")

    fig.suptitle("Selection Method — by Encoding", fontsize=14, weight="650")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_crossover_effect(df: pd.DataFrame, out: Path) -> plt.Figure:
    raw = filter_baseline_runs(prepare_df(df))
    rep_keys = [("binary", REP_LABELS["binary"]), ("real", REP_LABELS["real"])]

    n_rows = max(
        len(_agg(_apply_labels(raw[raw["representation"] == k]), ["crossover", "engine"], ["best_fitness"]))
        for k, _ in rep_keys
    ) // 2 or 1
    fig_h = max(5.0, 0.95 * n_rows + 2.2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, fig_h), dpi=140, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.06, wspace=0.38)

    for ax, (rep_key, rep_label) in zip(axes, rep_keys):
        sub = _apply_labels(raw[raw["representation"] == rep_key])
        agg = _agg(sub, ["crossover", "engine"], ["best_fitness"])
        cross_order = sorted(agg["crossover"].unique(), key=str)
        _grouped_barh(ax, agg, "crossover", "best_fitness_mean", cross_order)
        ax.set_title(f"{rep_label} — Crossover", pad=8)
        ax.set_xlabel("Median best fitness")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, axis="x", linestyle="--", alpha=0.55)
        _nice_axis(ax, categorical_y=True)
        ax.legend(fontsize=8, frameon=True, loc="lower right")

    fig.suptitle("Crossover Operator Comparison", fontsize=14, weight="650", y=1.02)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return fig


def plot_early_stop(df: pd.DataFrame, out: Path) -> plt.Figure:
    d = prepare_df(df)
    if "early_stop" not in d.columns or d["early_stop"].nunique() < 2:
        return None
    d = filter_baseline_runs(d)
    d = d.copy()
    d["early_stop"] = _as_bool(d["early_stop"]).map({True: "On", False: "Off"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140, constrained_layout=True)
    for ax, metric in zip(axes, ["best_fitness", "elapsed"]):
        sns.pointplot(
            data=d, x="early_stop", y=metric, hue="representation",
            palette=REP_PALETTE, dodge=0.35, errorbar=("ci", 95), ax=ax,
        )
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Early Stopping")
        ax.grid(True, axis="y", linestyle="--", alpha=0.55)
    fig.suptitle("Early Stopping Effect", fontsize=15, weight="650", y=1.03)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_parallel_speedup(df: pd.DataFrame, out: Path) -> plt.Figure:
    d = prepare_df(df)
    d = d[d["engine"] == "pygad"]
    if d["parallel"].nunique() < 2:
        return None

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140, constrained_layout=True)
    sns.barplot(
        data=d, x="parallel", y="elapsed", hue="representation",
        palette=REP_PALETTE, errorbar=("ci", 95), ax=ax,
    )
    ax.set_title("PyGAD Parallel Processing — Wall Time")
    ax.set_xlabel("Parallel Mode")
    ax.set_ylabel("Elapsed (s)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.55)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_population_scaling(df: pd.DataFrame, out: Path) -> plt.Figure:
    d = prepare_df(df)
    if d["pop_size"].nunique() < 2:
        return None
    d = filter_baseline_runs(d)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140, constrained_layout=True)
    sns.lineplot(
        data=d, x="pop_size", y="best_fitness", hue="implementation",
        marker="o", errorbar=("ci", 95), ax=ax,
    )
    ax.set_title("Best Fitness vs Population Size")
    ax.set_xlabel("Population Size")
    ax.grid(True, linestyle="--", alpha=0.55)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_convergence_panel(hist_dir: Path, df: pd.DataFrame, out: Path, max_lines: int = 8) -> plt.Figure:
    d = prepare_df(df)
    d = filter_baseline_runs(d)
    if d.empty:
        return None

    # pick best run per engine+representation
    picks = (
        d.sort_values("best_fitness")
        .groupby(["representation", "engine"], observed=True)
        .first()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140, constrained_layout=True)
    for ax, rep in zip(axes, ["binary", "real"]):
        sub = picks[picks["representation"] == rep].head(max_lines // 2)
        for _, row in sub.iterrows():
            eid = row["experiment_id"]
            path = hist_dir / f"{eid}.csv"
            if not path.exists():
                continue
            hist = pd.read_csv(path)
            label = f"{row['engine']} ({row.get('crossover', '')})"
            color = ENGINE_PALETTE.get(row["engine"], "#333")
            ax.plot(hist["epoch"], hist["best"], label=label, linewidth=2, color=color)
        ax.set_title(f"{rep.title()} — Convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Best Objective")
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.55)

    fig.suptitle("Objective Value Over Iterations", fontsize=15, weight="650", y=1.03)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_heatmap(df: pd.DataFrame, out: Path) -> plt.Figure:
    d = prepare_df(df)
    d = filter_baseline_runs(d)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=140, constrained_layout=True)
    for ax, (rep, eng) in zip(axes.flat, [
        ("binary", "custom"), ("binary", "pygad"),
        ("real", "custom"), ("real", "pygad"),
    ]):
        sub = d[(d["representation"] == rep) & (d["engine"] == eng)]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(
            values="best_fitness", index="crossover", columns="mutation", aggfunc="mean",
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu_r", ax=ax, cbar_kws={"label": "fitness"})
        ax.set_title(f"{rep.title()} / {eng}")

    fig.suptitle("Mean Best Fitness — Crossover × Mutation", fontsize=15, weight="650", y=1.02)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return fig


def generate_all_plots(project_root: Path | None = None) -> list[Path]:
    _style()
    root = project_root or Path(__file__).resolve().parents[2]
    results_dir = root / "results"
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = out_dir / "histories"

    summaries = sorted(results_dir.glob("benchmark_*.csv"))
    if not summaries:
        raise FileNotFoundError(f"No benchmark_*.csv in {results_dir}. Run benchmark first.")
    df = pd.read_csv(summaries[-1])

    saved: list[Path] = []
    for name, fn in [
        ("01_encoding_overview.png", plot_encoding_overview),
        ("02_selection_effect.png", plot_selection_effect),
        ("03_crossover_effect.png", plot_crossover_effect),
        ("04_early_stop.png", plot_early_stop),
        ("05_parallel_speedup.png", plot_parallel_speedup),
        ("06_population_scaling.png", plot_population_scaling),
        ("08_heatmap_crossover_mutation.png", plot_heatmap),
    ]:
        path = out_dir / name
        result = fn(df, path)
        if result is not None:
            saved.append(path)
            print(f"  wrote {path}")

    conv_path = out_dir / "07_convergence_panel.png"
    if plot_convergence_panel(hist_dir, df, conv_path) is not None:
        saved.append(conv_path)
        print(f"  wrote {conv_path}")

    return saved
