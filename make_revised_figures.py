#!/usr/bin/env python3
"""Create manuscript figures from the completed v3 aggregate outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


BLUE = "#2B6CA3"
ORANGE = "#D05B14"
GREEN = "#1B9E77"
GREY = "#6D6D6D"


def save_all(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(stem.with_suffix(".tif")) as tif:
        rgb = tif.convert("RGB")
    rgb.save(stem.with_suffix(".tif"), compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)


def pooled_row(data: pd.DataFrame, **filters) -> pd.Series:
    selected = data.copy()
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one pooled row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def figure_1(results: Path, output: Path) -> None:
    pooled = pd.read_csv(results / "pooled_model_results_reml_hk_v3.csv")
    continuous = []
    for label, term, color in [
        ("Cognitive worsening", "cmm4_change_raw[cognitive]", GREY),
        ("Functional worsening", "cmm4_change_raw[functional]", BLUE),
        ("Functional minus cognitive", "cmm4_change_raw[functional-minus-cognitive]", ORANGE),
    ]:
        row = pooled_row(
            pooled, analysis="continuous_domain_interaction", adjustment_set="mental_health",
            standardization="fixed", outcome="stacked_worsening",
            exposure="cmm4_change_raw", term=term,
        )
        continuous.append((label, row, color))
    transitions = []
    for label, outcome, color in [
        ("Any cognitive", "any_cognitive", GREY),
        ("Any functional", "any_functional", BLUE),
        ("Isolated cognitive-only", "cognitive_only", GREY),
        ("Functional-only", "functional_only", BLUE),
        ("Joint", "joint", GREEN),
    ]:
        row = pooled_row(
            pooled, analysis="transition_gee", adjustment_set="mental_health",
            state_definition="matched_0.20", standardization="fixed", persistent=False,
            outcome=outcome, exposure="cmm4_start",
        )
        transitions.append((label, row, color))

    fig = plt.figure(figsize=(8.8, 6.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 5], width_ratios=[2.7, 2.4, 4.6], hspace=0.58, wspace=0.05)
    forest_axes = []
    for index, (rows, title, xlabel, reference) in enumerate([
        (continuous, "A  Continuous paired-domain GEE", "Beta per unit CMM4 increase", 0.0),
        (transitions, "B  20% target-occupancy transitions", "Odds ratio per additional CMM4 condition", 1.0),
    ]):
        label_ax = fig.add_subplot(grid[index, 0])
        forest_ax = fig.add_subplot(grid[index, 1])
        value_ax = fig.add_subplot(grid[index, 2], sharey=forest_ax)
        forest_axes.append(forest_ax)
        y = np.arange(len(rows))[::-1]
        for yi, (label, row, color) in zip(y, rows):
            forest_ax.errorbar(
                float(row.pooled), yi,
                xerr=[[float(row.pooled - row.ci_low)], [float(row.ci_high - row.pooled)]],
                fmt="o", color=color, ecolor=color, markersize=5, capsize=0,
            )
            label_ax.text(0, yi, label, ha="left", va="center", fontsize=7.4)
            value_ax.text(
                0, yi,
                f"{row.pooled:.3f} ({row.ci_low:.3f}, {row.ci_high:.3f}); I²={row.i2:.1f}%; k={int(row.k)}; N={int(row.n_total):,}",
                ha="left", va="center", fontsize=6.25,
            )
        forest_ax.axvline(reference, color="#888888", linestyle="--", linewidth=0.8)
        label_ax.set_title(title, fontsize=8.2, fontweight="bold", loc="left", pad=7)
        forest_ax.set_xlabel(xlabel, fontsize=7)
        forest_ax.set_yticks([])
        forest_ax.tick_params(axis="x", labelsize=6.5)
        forest_ax.spines[["top", "right"]].set_visible(False)
        for text_ax in [label_ax, value_ax]:
            text_ax.set_xlim(0, 1)
            text_ax.set_ylim(-0.65, len(rows) - 0.35)
            text_ax.axis("off")
        forest_ax.set_ylim(-0.65, len(rows) - 0.35)
    forest_axes[0].set_xlim(-0.005, 0.12)
    forest_axes[1].set_xlim(0.86, 1.48)
    fig.suptitle("CMM4 associations across cognitive and functional outcomes", x=0.025, ha="left", fontsize=10, fontweight="bold")
    fig.text(
        0.025, 0.01,
        "Fixed-reference composites; exact interval adjustment; participant-clustered GEE; REML/Hartung–Knapp pooling. Binary states are distribution-defined, not clinical diagnoses.",
        fontsize=6.4, color="#555555",
    )
    save_all(fig, output / "Figure_1")


def figure_2(results: Path, output: Path) -> None:
    cohort = pd.read_csv(results / "cohort_model_results_v3.csv")
    selected = cohort[
        cohort["analysis"].eq("transition_gee")
        & cohort["adjustment_set"].eq("mental_health")
        & cohort["state_definition"].eq("matched_0.20")
        & cohort["standardization"].eq("fixed")
        & cohort["persistent"].eq(False)
        & cohort["exposure"].eq("cmm4_start")
        & cohort["outcome"].isin(["any_cognitive", "any_functional"])
    ].copy()
    selected["or"] = np.exp(selected["estimate"])
    selected["low"] = np.exp(selected["estimate"] - 1.96 * selected["std_error"])
    selected["high"] = np.exp(selected["estimate"] + 1.96 * selected["std_error"])
    cohort_order = ["charls", "elsa", "hrs", "klosa", "mhas", "share"]
    names = ["CHARLS", "ELSA", "HRS", "KLoSA", "MHAS", "SHARE"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharey=True, gridspec_kw={"wspace": 0.48})
    for ax, outcome, title, color in [
        (axes[0], "any_cognitive", "Any cognitive transition", GREY),
        (axes[1], "any_functional", "Any functional transition", BLUE),
    ]:
        d = selected[selected["outcome"].eq(outcome)].set_index("cohort").reindex(cohort_order)
        y = np.arange(len(cohort_order))[::-1]
        ax.errorbar(d["or"], y, xerr=[d["or"] - d["low"], d["high"] - d["or"]], fmt="o", color=color, ecolor=color, markersize=4.5)
        ax.axvline(1, color="#888888", linestyle="--", linewidth=0.8)
        for yi, estimate, low, high in zip(y, d["or"], d["low"], d["high"]):
            ax.text(1.02, yi, f"{estimate:.2f} ({low:.2f}, {high:.2f})", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.1)
        ax.set_xlim(0.78, 1.78)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel("Odds ratio per additional CMM4 condition", fontsize=6.8)
        ax.tick_params(axis="both", labelsize=6.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(names))[::-1], names, fontsize=6.8)
    fig.suptitle("Cohort-specific 20% target-occupancy transition estimates", x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.text(0.02, 0.01, "Mental-health-adjusted participant-clustered GEE; fixed-reference scores; survivor-conditional domain outcomes.", fontsize=6.4, color="#555555")
    save_all(fig, output / "Figure_2")


def occupancy_table(data: pd.DataFrame, definition: str) -> pd.DataFrame:
    return (
        data[
            data["standardization"].eq("fixed")
            & data["state_definition"].eq(definition)
            & data["persistent"].eq(False)
            & data["state_from"].isin(["cognitive_only", "functional_only"])
        ]
        .pivot(index="cohort", columns="state_from", values="origin_percent")
    )


def figure_3(results: Path, output: Path) -> None:
    occupancy = pd.read_csv(results / "interval_origin_state_occupancy_v3.csv")
    reversion = pd.read_csv(results / "state_reversion_v3.csv")
    cohort_order = ["charls", "elsa", "hrs", "klosa", "mhas", "share"]
    names = ["CHARLS", "ELSA", "HRS", "KLoSA", "MHAS", "SHARE"]
    legacy = occupancy_table(occupancy, "legacy").reindex(cohort_order)
    matched = occupancy_table(occupancy, "matched_0.20").reindex(cohort_order)
    rev = (
        reversion[
            reversion["standardization"].eq("fixed")
            & reversion["state_definition"].eq("matched_0.20")
            & reversion["state_from"].isin(["cognitive_only", "functional_only", "joint"])
        ]
        .pivot(index="cohort", columns="state_from", values="reversion_percent")
        .reindex(cohort_order)
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), gridspec_kw={"wspace": 0.42})
    x = np.arange(len(cohort_order)); width = 0.35
    for ax, table, title in [(axes[0], legacy, "A  Historical ±0.43"), (axes[1], matched, "B  20% target occupancy")]:
        ax.bar(x - width / 2, table["cognitive_only"], width, color=GREY, label="Cognitive-only")
        ax.bar(x + width / 2, table["functional_only"], width, color=BLUE, label="Functional-only")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_ylabel("Interval-origin occupancy (%)", fontsize=6.8)
        ax.set_xticks(x, names, rotation=38, ha="right", fontsize=6.2)
        ax.legend(frameon=False, fontsize=6.0)
    shared_occupancy_max = float(np.nanmax([legacy.to_numpy().max(), matched.to_numpy().max()]))
    shared_occupancy_max = max(5.0, math.ceil(shared_occupancy_max / 5.0) * 5.0)
    axes[0].set_ylim(0, shared_occupancy_max)
    axes[1].set_ylim(0, shared_occupancy_max)
    for offset, state, color, label in [
        (-0.18, "cognitive_only", GREY, "Cognitive-only"),
        (0.0, "functional_only", BLUE, "Functional-only"),
        (0.18, "joint", GREEN, "Joint"),
    ]:
        axes[2].scatter(x + offset, rev[state], color=color, s=22, label=label, zorder=3)
    axes[2].set_title("C  20% one-scheduled-interval reversion", fontsize=8, fontweight="bold")
    axes[2].set_ylabel("Returned to unimpaired (%)", fontsize=6.8)
    axes[2].set_xticks(x, names, rotation=38, ha="right", fontsize=6.2)
    axes[2].legend(frameon=False, fontsize=6.0)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=6.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    fig.suptitle("Threshold choice changes state occupancy and stability", x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.text(0.02, 0.005, "Occupancy is based on interval origins; 20% thresholds do not split tied scores, so realised functional occupancy can differ from the nominal target.", fontsize=6.3, color="#555555")
    save_all(fig, output / "Figure_3")


def supplementary_figure_1(results: Path, additional: Path, output: Path) -> None:
    pooled = pd.read_csv(results / "pooled_model_results_reml_hk_v3.csv")
    extra = pd.read_csv(additional / "additional_sensitivity_pooled_results.csv")
    scenarios = [
        ("Primary CMM4", "primary", "matched_0.20", "cmm4_start"),
        ("Reference-locked 20%", "transition_gee", "reference_locked_0.20", "cmm4_start"),
        ("Diabetes + stroke", "diabetes_stroke_transition_gee", "matched_0.20", "diabetes_stroke_start"),
        ("CMM3 excluding stroke", "cmm3_no_stroke_transition_gee", "matched_0.20", "cmm3_no_stroke_start"),
        ("Exclude ELSA", "exclude_elsa_transition_gee", "matched_0.20", "cmm4_start"),
        ("Exclude KLoSA", "exclude_klosa_transition_gee", "matched_0.20", "cmm4_start"),
        ("KLoSA ADL-only", "klosa_adl_only_transition_gee", "matched_0.20", "cmm4_start"),
    ]
    rows = {"any_cognitive": [], "any_functional": []}
    for label, analysis, definition, exposure in scenarios:
        for outcome in rows:
            if analysis == "primary":
                r = pooled_row(
                    pooled, analysis="transition_gee", adjustment_set="mental_health",
                    state_definition=definition, standardization="fixed", persistent=False,
                    outcome=outcome, exposure=exposure,
                )
            else:
                r = pooled_row(
                    extra, analysis=analysis, state_definition=definition,
                    standardization="fixed", outcome=outcome, exposure=exposure,
                )
            rows[outcome].append((label, r))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=True, gridspec_kw={"wspace": 0.42})
    y = np.arange(len(scenarios))[::-1]
    for ax, outcome, title, color in [
        (axes[0], "any_cognitive", "Any cognitive transition", GREY),
        (axes[1], "any_functional", "Any functional transition", BLUE),
    ]:
        for yi, (label, r) in zip(y, rows[outcome]):
            ax.errorbar(
                float(r.pooled), yi,
                xerr=[[float(r.pooled - r.ci_low)], [float(r.ci_high - r.pooled)]],
                fmt="o", color=color, ecolor=color, markersize=4.5,
            )
            ax.text(1.01, yi, f"{r.pooled:.2f} ({r.ci_low:.2f}, {r.ci_high:.2f})",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.2)
        ax.axvline(1, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_xlim(0.88, 1.68)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Odds ratio per additional condition", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    axes[0].set_yticks(y, [s[0] for s in scenarios], fontsize=6.8)
    fig.suptitle("Key transition sensitivity analyses", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.02, 0.01, "Participant-clustered GEE with REML/Hartung–Knapp pooling. Restricted disease counts differ in range from CMM4 and should be interpreted within exposure definition.", fontsize=6.3, color="#555555")
    save_all(fig, output / "Figure_S1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--additional", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure_1(args.results, args.output)
    figure_2(args.results, args.output)
    figure_3(args.results, args.output)
    if args.additional is not None:
        supplementary_figure_1(args.results, args.additional, args.output)
    print(f"Wrote completed-v3 figures to {args.output}")


if __name__ == "__main__":
    main()
