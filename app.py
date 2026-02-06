"""
Power Analysis for Avian Analgesiometric Studies
=================================================
Streamlit application for calculating sample sizes for future avian pain
studies using repeated-measures designs analysed with linear mixed models.

Variance components and effect sizes are pre-populated from the published
avian thermal / mechanical antinociception literature and can be adjusted
for any new drug or species.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from study_data import (
    SPECIES_MODELS,
    get_species_model_keys,
    get_species_model_params,
    get_studies_for_species_model,
    get_studies_dataframe,
)
from power_engine import (
    generate_effect_profile,
    analytical_power_crossover,
    analytical_power_parallel,
    find_sample_size,
    sensitivity_table,
    simulate_power,
)

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Avian Pain Study Power Analysis",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .big-number {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1f77b4;
        line-height: 1.1;
    }
    .result-box {
        background: #f0f7ff;
        border: 2px solid #1f77b4;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e8f4f8;
        border: 1px solid #17a2b8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("Power Analysis for Avian Analgesiometric Studies")
st.markdown(
    "Calculate sample sizes for repeated-measures pain studies in birds "
    "analysed with **linear mixed models**. Variance parameters are "
    "pre-populated from the published avian thermal/mechanical "
    "antinociception literature."
)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR – study design configuration
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Study design")

    # Species / model
    species_keys = get_species_model_keys()
    selected_key = st.selectbox(
        "Species & nociceptive model",
        species_keys,
        index=0,
        help="Select a species and testing model. Variance parameters "
             "will be pre-filled from published data.",
    )
    params = get_species_model_params(selected_key)

    # Design type
    design = st.radio(
        "Study design",
        ["Crossover", "Parallel"],
        index=0,
        help="**Crossover**: each subject receives every treatment "
             "(within-subject comparison, smaller n required). "
             "**Parallel**: separate groups per treatment.",
    )

    st.divider()
    st.header("Time points")

    default_tp = params["typical_timepoints"]
    use_default_tp = st.checkbox(
        f"Use typical time points ({len(default_tp)} pts)",
        value=True,
    )
    if use_default_tp:
        timepoints = default_tp
        st.caption(f"Time points (h): {timepoints}")
    else:
        tp_str = st.text_input(
            "Enter time points (hours, comma-separated)",
            value=", ".join(str(t) for t in default_tp),
        )
        try:
            timepoints = [float(x.strip()) for x in tp_str.split(",") if x.strip()]
        except ValueError:
            timepoints = default_tp
            st.error("Invalid input — using defaults.")

    # post-baseline only for power calc
    post_baseline_tp = [t for t in timepoints if t > 0]
    n_timepoints = len(post_baseline_tp)

    st.divider()
    st.header("Statistical parameters")

    alpha = st.number_input(
        "Significance level (alpha)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.3f",
    )
    target_power = st.number_input(
        "Target power",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.05,
        format="%.2f",
    )
    test_type = st.selectbox(
        "Primary test",
        ["overall", "peak", "interaction"],
        format_func=lambda x: {
            "overall": "Overall treatment effect",
            "peak": "Peak treatment effect",
            "interaction": "Treatment × Time interaction",
        }[x],
        help="**Overall**: average treatment effect across all time-points. "
             "**Peak**: effect at the single most efficacious time-point. "
             "**Interaction**: whether the treatment effect varies over time.",
    )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA – tabs
# ═══════════════════════════════════════════════════════════════════════════
tab_power, tab_ref, tab_sensitivity, tab_sim, tab_methods = st.tabs([
    "Power Analysis",
    "Reference Data",
    "Sensitivity Analysis",
    "Simulation Validation",
    "Methods & Assumptions",
])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — Power analysis
# ───────────────────────────────────────────────────────────────────────────
with tab_power:
    st.markdown(
        "This page calculates **how many birds you need** in your study. "
        "It uses information from previously published studies (how much "
        "natural variability exists between birds and between measurements) "
        "combined with the drug effect you expect to see. A larger expected "
        "drug effect or less variability means fewer birds are needed; a "
        "smaller effect or more variability means more birds are needed."
    )

    st.subheader("Variance components")
    st.markdown(
        f"Pre-filled from **{params['species']}** – "
        f"*{params['model']}* literature. Adjust as needed."
    )

    # Use dynamic keys so widgets reset when species changes
    _sk = selected_key  # short alias for widget keys

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        within_sd = st.number_input(
            f"Within-subject SD (residual), {params['unit']}",
            min_value=0.1,
            value=params["within_subject_sd"],
            step=0.1,
            format="%.2f",
            key=f"within_sd_{_sk}",
            help="Residual standard deviation from the LMM (σ_ε). "
                 "Represents measurement-to-measurement variability "
                 "within the same bird.",
        )
    with col_v2:
        between_sd = st.number_input(
            f"Between-subject SD, {params['unit']}",
            min_value=0.1,
            value=params["between_subject_sd"],
            step=0.1,
            format="%.2f",
            key=f"between_sd_{_sk}",
            help="Between-bird SD (σ_b). In crossover designs this "
                 "cancels out; in parallel designs it inflates the "
                 "required sample size.",
        )

    icc = between_sd ** 2 / (between_sd ** 2 + within_sd ** 2)
    st.caption(f"Implied ICC = {icc:.2f}")

    st.divider()
    st.subheader("Expected drug effect")

    # Show reference drug effects for context
    ref_studies = get_studies_for_species_model(selected_key)
    if ref_studies:
        sig_studies = [s for s in ref_studies if s["significant"]]
        if sig_studies:
            effects = [s["peak_effect"] for s in sig_studies]
            st.info(
                f"Published significant peak effects for this species/model: "
                f"**{min(effects):.1f}** – **{max(effects):.1f}** {params['unit']} "
                f"(median {np.median(effects):.1f})"
            )

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        peak_effect = st.number_input(
            f"Expected peak effect ({params['unit']})",
            min_value=0.1,
            value=float(round(np.median(
                [s["peak_effect"] for s in ref_studies]
            ) if ref_studies else 3.0, 1)),
            step=0.5,
            format="%.1f",
            key=f"peak_effect_{_sk}",
            help="Maximum expected difference between treatment and "
                 "control at the time of peak drug activity.",
        )
    with col_e2:
        t_peak = st.number_input(
            "Time to peak (h)",
            min_value=0.1,
            value=float(round(np.median(
                [s["time_to_peak_h"] for s in ref_studies]
            ) if ref_studies else 1.0, 1)),
            step=0.5,
            format="%.1f",
            key=f"t_peak_{_sk}",
        )
    with col_e3:
        duration = st.number_input(
            "Duration of effect (h)",
            min_value=0.5,
            value=float(round(np.median(
                [s["duration_h"] for s in ref_studies]
            ) if ref_studies else 4.0, 1)),
            step=0.5,
            format="%.1f",
            key=f"duration_{_sk}",
        )

    # Generate effect profile
    profile = generate_effect_profile(post_baseline_tp, peak_effect, t_peak, duration)
    mean_effect = float(np.mean(profile))

    # Plot effect profile
    with st.expander("Expected effect profile over time", expanded=True):
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(
            x=post_baseline_tp,
            y=profile,
            mode="lines+markers",
            name="Treatment effect",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
        ))
        fig_profile.add_hline(
            y=mean_effect, line_dash="dash", line_color="gray",
            annotation_text=f"Mean effect = {mean_effect:.2f} {params['unit']}",
        )
        fig_profile.update_layout(
            xaxis_title="Time (h)",
            yaxis_title=f"Treatment effect ({params['unit']})",
            height=350,
            margin=dict(l=60, r=30, t=30, b=60),
        )
        st.plotly_chart(fig_profile, use_container_width=True)

    # ── Run power analysis ──
    st.divider()
    st.subheader("Results")

    # Determine the effect for the selected test type
    if test_type == "overall":
        test_effect = mean_effect
    elif test_type == "peak":
        test_effect = peak_effect
    else:
        test_effect = mean_effect  # not directly used for interaction

    result = find_sample_size(
        target_power=target_power,
        design=design,
        n_timepoints=n_timepoints,
        mean_effect=test_effect,
        within_subject_sd=within_sd,
        between_subject_sd=between_sd,
        alpha=alpha,
        test=test_type,
        effect_profile=profile,
    )

    # Display the result
    col_r1, col_r2, col_r3 = st.columns([1, 1, 1])

    with col_r1:
        if design == "Crossover":
            label = "subjects (each receives all treatments)"
        else:
            label = "subjects per group"

        st.markdown(
            f"""<div class="result-box">
                <div style="font-size:1rem; color:#555;">Required sample size</div>
                <div class="big-number">{result['n']}</div>
                <div style="font-size:0.9rem; color:#777;">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_r2:
        st.markdown(
            f"""<div class="result-box">
                <div style="font-size:1rem; color:#555;">Achieved power</div>
                <div class="big-number">{result['power']:.1%}</div>
                <div style="font-size:0.9rem; color:#777;">
                    at n = {result['n']}, α = {alpha}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_r3:
        test_labels = {
            "overall": "Overall treatment effect",
            "peak": "Peak treatment effect",
            "interaction": "Treatment × Time interaction",
        }
        st.markdown(
            f"""<div class="result-box">
                <div style="font-size:1rem; color:#555;">Test</div>
                <div style="font-size:1.3rem; font-weight:600; color:#1f77b4;">
                    {test_labels[test_type]}</div>
                <div style="font-size:0.9rem; color:#777;">
                    {design} design, {n_timepoints} time points</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if not result["target_achieved"]:
        st.warning(
            f"Target power of {target_power:.0%} could not be achieved with "
            f"n ≤ 200. The maximum power at n = 200 is {result['power']:.1%}. "
            f"Consider increasing the expected effect size or reducing variance."
        )

    # Power curve
    st.subheader("Power curve")
    curve_data = result["power_curve"]
    curve_df = pd.DataFrame(curve_data, columns=["n", "Power"])

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=curve_df["n"],
        y=curve_df["Power"],
        mode="lines",
        name="Power",
        line=dict(color="#1f77b4", width=3),
    ))
    # target power line
    fig_curve.add_hline(
        y=target_power, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"Target power = {target_power:.0%}",
    )
    # result marker
    fig_curve.add_trace(go.Scatter(
        x=[result["n"]],
        y=[result["power"]],
        mode="markers",
        marker=dict(size=14, color="#e74c3c", symbol="star"),
        name=f"n = {result['n']}",
    ))
    fig_curve.update_layout(
        xaxis_title="Sample size (n)",
        yaxis_title="Power",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=450,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        margin=dict(l=60, r=30, t=30, b=60),
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    # Interpretation
    with st.expander("Interpretation", expanded=True):
        unit = params["unit"]
        if design == "Crossover":
            st.markdown(f"""
**Design**: Crossover (each bird receives treatment and control with washout)

**Required**: **{result['n']} birds** to detect a {'mean' if test_type == 'overall' else 'peak'} \
treatment effect of **{test_effect:.1f} {unit}** with **{target_power:.0%} power** at \
**α = {alpha}**.

**Key parameters**:
- Within-subject SD (σ_ε): {within_sd:.2f} {unit}
- Number of post-baseline time points: {n_timepoints}
- Effect profile: peak of {peak_effect:.1f} {unit} at {t_peak:.1f}h, \
lasting {duration:.1f}h
- Mean effect across time points: {mean_effect:.2f} {unit}

**Note**: In a crossover design, between-subject variance (σ_b = {between_sd:.1f} {unit}) \
is eliminated from the treatment comparison, which substantially reduces the \
required sample size compared to a parallel design.
            """)
        else:
            st.markdown(f"""
**Design**: Parallel groups (separate treatment and control groups)

**Required**: **{result['n']} birds per group** ({2 * result['n']} total) to detect a \
{'mean' if test_type == 'overall' else 'peak'} treatment effect of \
**{test_effect:.1f} {unit}** with **{target_power:.0%} power** at **α = {alpha}**.

**Key parameters**:
- Within-subject SD (σ_ε): {within_sd:.2f} {unit}
- Between-subject SD (σ_b): {between_sd:.2f} {unit}
- Number of post-baseline time points: {n_timepoints}
- Effect profile: peak of {peak_effect:.1f} {unit} at {t_peak:.1f}h, \
lasting {duration:.1f}h
- Mean effect across time points: {mean_effect:.2f} {unit}

**Note**: In a parallel design, both within- and between-subject variance \
contribute to the standard error. A crossover design would require fewer animals.
            """)

# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — Reference data
# ───────────────────────────────────────────────────────────────────────────
with tab_ref:
    st.subheader("Published avian analgesiometric studies")
    st.markdown(
        "Summary of published studies providing the variance components and "
        "effect sizes used in this tool. Parameters are approximate values "
        "derived from the published literature."
    )

    # Full study table
    studies_df = get_studies_dataframe()
    st.dataframe(
        studies_df,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    st.divider()

    # Variance components table
    st.subheader("Variance components by species & model")
    vc_rows = []
    for key, p in SPECIES_MODELS.items():
        vc_rows.append({
            "Species": p["species"],
            "Model": p["model"],
            "Unit": p["unit"],
            "Baseline mean": p["baseline_mean"],
            "Between-subject SD": p["between_subject_sd"],
            "Within-subject SD": p["within_subject_sd"],
            "ICC": round(p["icc"], 2),
            "Typical n": p["typical_n"],
            "Time points": len(p["typical_timepoints"]),
        })
    vc_df = pd.DataFrame(vc_rows)
    st.dataframe(vc_df, use_container_width=True, hide_index=True)

    st.divider()

    # Effect size forest plot for selected species
    st.subheader(f"Effect sizes — {params['species']}")
    sel_studies = get_studies_for_species_model(selected_key)
    if sel_studies:
        fig_forest = go.Figure()
        labels = [f"{s['drug']} {s['dose']}" for s in sel_studies]
        effects = [s["peak_effect"] for s in sel_studies]
        colours = [
            "#2ca02c" if s["significant"] else "#d62728"
            for s in sel_studies
        ]
        fig_forest.add_trace(go.Bar(
            y=labels,
            x=effects,
            orientation="h",
            marker_color=colours,
            text=[f"{e:.1f}" for e in effects],
            textposition="outside",
        ))
        fig_forest.update_layout(
            xaxis_title=f"Peak effect ({params['unit']})",
            height=max(250, len(sel_studies) * 45),
            margin=dict(l=200, r=50, t=30, b=60),
        )
        st.plotly_chart(fig_forest, use_container_width=True)
        st.caption("Green = statistically significant; Red = not significant.")
    else:
        st.info("No reference studies available for this species/model.")

# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — Sensitivity analysis
# ───────────────────────────────────────────────────────────────────────────
with tab_sensitivity:
    st.subheader("Sensitivity analysis")
    st.markdown(
        "In practice, you may not know exactly how strong your new drug's "
        "effect will be. This page lets you explore a **range of scenarios** "
        "simultaneously: what if the drug effect is smaller or larger than "
        "expected? How does the required number of birds change? The table "
        "and heatmap below show the probability of detecting a drug effect "
        "(power) for every combination of effect size and sample size. "
        "Green cells mean you have enough birds; red cells mean you do not."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        es_min = st.number_input(
            "Min effect size", value=round(peak_effect * 0.3, 1), step=0.5,
            format="%.1f", key=f"es_min_{_sk}",
        )
        es_max = st.number_input(
            "Max effect size", value=round(peak_effect * 2.0, 1), step=0.5,
            format="%.1f", key=f"es_max_{_sk}",
        )
        n_es_steps = st.slider("Number of effect-size steps", 3, 15, 8,
                               key=f"n_es_{_sk}")
    with col_s2:
        ns_min = st.number_input("Min n", value=4, step=1, min_value=3,
                                 key=f"ns_min_{_sk}")
        ns_max = st.number_input("Max n", value=40, step=5, min_value=4,
                                 key=f"ns_max_{_sk}")
        n_ns_steps = st.slider("Number of n steps", 3, 15, 8,
                               key=f"n_ns_{_sk}")

    effect_sizes = list(np.linspace(es_min, es_max, n_es_steps).round(1))
    sample_sizes = sorted(set(
        int(x) for x in np.linspace(ns_min, ns_max, n_ns_steps).round()
    ))

    # For the interaction test we need to regenerate the effect profile
    # scaled to each effect size
    def _profile_gen(es):
        return generate_effect_profile(post_baseline_tp, es, t_peak, duration)

    # Decide what effect to pass for each test type
    if test_type == "interaction":
        sens_df = sensitivity_table(
            design=design,
            n_timepoints=n_timepoints,
            within_subject_sd=within_sd,
            between_subject_sd=between_sd,
            alpha=alpha,
            test=test_type,
            effect_sizes=effect_sizes,
            sample_sizes=sample_sizes,
            effect_profile_generator=_profile_gen,
        )
    elif test_type == "overall":
        # For "overall", the mean effect depends on peak & profile shape.
        # We'll compute the mean for each peak effect size.
        mean_effects = []
        for es in effect_sizes:
            prof = generate_effect_profile(post_baseline_tp, es, t_peak, duration)
            mean_effects.append(round(float(np.mean(prof)), 2))

        sens_df = sensitivity_table(
            design=design,
            n_timepoints=n_timepoints,
            within_subject_sd=within_sd,
            between_subject_sd=between_sd,
            alpha=alpha,
            test=test_type,
            effect_sizes=mean_effects,
            sample_sizes=sample_sizes,
        )
        sens_df.index = [f"{es} (mean {me})" for es, me in zip(effect_sizes, mean_effects)]
        sens_df.index.name = f"Peak effect (mean) [{params['unit']}]"
    else:  # peak
        sens_df = sensitivity_table(
            design=design,
            n_timepoints=n_timepoints,
            within_subject_sd=within_sd,
            between_subject_sd=between_sd,
            alpha=alpha,
            test=test_type,
            effect_sizes=effect_sizes,
            sample_sizes=sample_sizes,
        )

    # Colour-code the table
    def _color_power(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v >= target_power:
            return "background-color: #d4edda"
        elif v >= target_power - 0.1:
            return "background-color: #fff3cd"
        else:
            return "background-color: #f8d7da"

    styled = sens_df.style.map(_color_power)
    st.dataframe(styled, use_container_width=True, height=450)
    st.caption(
        f"Green: power ≥ {target_power:.0%}; "
        f"Yellow: {target_power - 0.1:.0%}–{target_power:.0%}; "
        f"Red: < {target_power - 0.1:.0%}."
    )

    # Heatmap
    st.subheader("Power heatmap")
    # Build numeric matrix for heatmap
    heat_data = []
    for i, es in enumerate(effect_sizes):
        row = []
        for n in sample_sizes:
            if test_type == "overall":
                prof = generate_effect_profile(post_baseline_tp, es, t_peak, duration)
                eff = float(np.mean(prof))
            elif test_type == "peak":
                eff = es
                prof = None
            else:
                prof = _profile_gen(es)
                eff = float(np.mean(prof))

            kw = dict(
                n_timepoints=n_timepoints, mean_effect=eff,
                within_subject_sd=within_sd, alpha=alpha,
                test=test_type, effect_profile=prof,
            )
            if design == "Crossover":
                kw["n_subjects"] = n
                pwr = analytical_power_crossover(**kw)
            else:
                kw["n_per_group"] = n
                kw["between_subject_sd"] = between_sd
                pwr = analytical_power_parallel(**kw)
            row.append(pwr)
        heat_data.append(row)

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_data,
        x=[str(n) for n in sample_sizes],
        y=[f"{es:.1f}" for es in effect_sizes],
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in heat_data],
        texttemplate="%{text}",
        colorbar=dict(title="Power"),
    ))
    fig_heat.update_layout(
        xaxis_title="Sample size (n)",
        yaxis_title=f"Peak effect ({params['unit']})",
        height=450,
        margin=dict(l=80, r=30, t=30, b=60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — Simulation validation
# ───────────────────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("Simulation-based power validation")
    st.markdown(
        "The Power Analysis tab uses a mathematical formula to estimate how "
        "many birds you need. This page **double-checks that formula** by "
        "actually simulating hundreds of fake experiments: it generates "
        "realistic data (with the variability and drug effect you specified), "
        "runs the same statistical analysis you would use on real data (a "
        "linear mixed model), and counts how often the drug effect is "
        "detected. If the formula and the simulation agree, you can be "
        "confident in the sample-size recommendation."
    )

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_n = st.number_input(
            "Sample size to simulate",
            min_value=3, max_value=200,
            value=result["n"],
            step=1,
            key=f"sim_n_{_sk}",
        )
    with col_sim2:
        n_sims = st.selectbox(
            "Number of simulations",
            [100, 200, 500, 1000],
            index=1,
            key=f"n_sims_{_sk}",
            help="More simulations = more precise estimate but slower.",
        )

    sim_seed = st.number_input("Random seed", value=42, step=1,
                               key=f"seed_{_sk}")

    if st.button("Run simulation", type="primary", key=f"run_sim_{_sk}"):
        with st.spinner(f"Running {n_sims} simulations..."):
            sim_result = simulate_power(
                n_subjects=sim_n,
                design=design,
                n_timepoints=n_timepoints,
                treatment_effect_profile=profile,
                between_subject_sd=between_sd,
                within_subject_sd=within_sd,
                n_simulations=n_sims,
                alpha=alpha,
                seed=int(sim_seed),
            )

        col_sr1, col_sr2, col_sr3 = st.columns(3)
        with col_sr1:
            st.metric("Simulated power", f"{sim_result['power']:.1%}")
        with col_sr2:
            st.metric(
                "95% CI",
                f"{sim_result['ci_lower']:.1%} – {sim_result['ci_upper']:.1%}",
            )
        with col_sr3:
            st.metric("Converged models", f"{sim_result['n_converged']}/{n_sims}")

        # Comparison
        if test_type == "overall":
            eff = mean_effect
        elif test_type == "peak":
            eff = peak_effect
        else:
            eff = mean_effect

        if design == "Crossover":
            analytical = analytical_power_crossover(
                sim_n, n_timepoints, eff, within_sd, alpha, test_type, profile,
            )
        else:
            analytical = analytical_power_parallel(
                sim_n, n_timepoints, eff, between_sd, within_sd, alpha, test_type, profile,
            )

        st.markdown(f"""
| Method | Power |
|--------|-------|
| Analytical | {analytical:.3f} |
| Simulation | {sim_result['power']:.3f} ({sim_result['ci_lower']:.3f} – {sim_result['ci_upper']:.3f}) |
        """)

        if abs(analytical - sim_result["power"]) < 0.1:
            st.success(
                "The analytical and simulation-based estimates are in good "
                "agreement, supporting the validity of the analytical result."
            )
        else:
            st.warning(
                "The analytical and simulation estimates differ by >10 "
                "percentage points. The simulation result may be more "
                "reliable, especially for small samples or non-standard "
                "designs. Consider increasing the number of simulations."
            )

# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — Methods & assumptions
# ───────────────────────────────────────────────────────────────────────────
with tab_methods:
    st.subheader("Statistical methods")
    st.markdown(
        "This page describes the math behind the power calculations for "
        "those who want to understand or report the methodology. In short: "
        "the tool assumes your data will be analysed with a **linear mixed "
        "model** (LMM) — the standard approach for repeated-measures studies "
        "where each bird is measured multiple times. The power calculation "
        "figures out how likely you are to find a statistically significant "
        "drug effect given the natural variability in your measurements, the "
        "expected strength of the drug, and the number of birds. The "
        "formulas below are what reviewers and statisticians would expect to "
        "see in a grant application or methods section."
    )

    st.markdown(r"""
### Linear mixed model

The assumed data-generating model is:

$$y_{kij} = \mu + \tau_i + \gamma_j + (\tau\gamma)_{ij} + b_k + \varepsilon_{kij}$$

| Symbol | Meaning |
|--------|---------|
| $y_{kij}$ | Outcome for subject $k$, treatment $i$, time $j$ |
| $\mu$ | Grand mean |
| $\tau_i$ | Fixed effect of treatment |
| $\gamma_j$ | Fixed effect of time |
| $(\tau\gamma)_{ij}$ | Treatment × time interaction |
| $b_k$ | Random intercept for subject $k$, $b_k \sim N(0, \sigma^2_b)$ |
| $\varepsilon_{kij}$ | Residual, $\varepsilon_{kij} \sim N(0, \sigma^2_\varepsilon)$ |

### Power formulae

**Crossover design — overall treatment effect:**

Each subject provides a paired comparison. The subject-level mean
difference has variance $2\sigma^2_\varepsilon / J$ (where $J$ = number of
post-baseline time-points). The test is equivalent to a paired $t$-test
with $n - 1$ degrees of freedom and non-centrality parameter:

$$\text{ncp} = \frac{\delta}{\sqrt{2\sigma^2_\varepsilon / (nJ)}}$$

Power is computed from the non-central $t$ distribution.

**Crossover design — peak effect:**

Same as above but with $J = 1$ (single time-point):

$$\text{ncp} = \frac{\delta_{\text{peak}}}{\sqrt{2\sigma^2_\varepsilon / n}}$$

**Crossover design — treatment × time interaction:**

$F$-test with $J - 1$ numerator df and $(n-1)(J-1)$ denominator df.
Non-centrality parameter:

$$\lambda = \frac{n \sum_j (\delta_j - \bar{\delta})^2}{2\sigma^2_\varepsilon}$$

**Parallel design — overall treatment effect:**

Two-sample $t$-test on subject means with variance
$2(\sigma^2_b + \sigma^2_\varepsilon/J) / n$ and $2(n-1)$ df.

### Effect profile model

The time-course of the treatment effect is modelled using a modified
Bateman function:

$$\delta(t) = \delta_{\text{peak}} \left(\frac{t}{t_{\text{peak}}}\right)^a \exp\!\left(a\!\left(1 - \frac{t}{t_{\text{peak}}}\right)\right)$$

where $a$ is chosen so that $\delta(t_{\text{duration}}) \approx 0.05 \times \delta_{\text{peak}}$.

### Assumptions

1. **Balanced design**: equal observations per subject across time-points
2. **Normality**: responses are normally distributed
3. **Compound symmetry**: residuals are IID conditional on random effects
4. **No carryover**: in crossover designs, adequate washout between periods
5. **No period effects**: treatment effects do not depend on sequence
6. **Missing data are minimal**: power calculations assume complete data

### Degrees of freedom

Degrees of freedom follow the Satterthwaite approximation for balanced
random-intercept models. For simple crossover designs, this equals $n - 1$
(identical to a paired $t$-test). For parallel designs, $2(n - 1)$.
These match the Kenward-Roger approximation for balanced data.

### Simulation validation

The simulation tab generates complete datasets under $H_1$, fits a
``MixedLM`` (random-intercept) model via REML, and records whether the
Wald test for the treatment coefficient achieves $p < \alpha$. The
proportion of significant results across replications estimates power
empirically.
    """)

    st.divider()
    st.subheader("References")
    st.markdown("""
- Beaufrère H, et al. Analgesic effects of tramadol hydrochloride in
  American kestrels (*Falco sparverius*). *Am J Vet Res*. 2011.
- Beaufrère H, et al. Pharmacokinetics of tramadol and analgesic effects in
  American kestrels. *Am J Vet Res*. 2014.
- Beaufrère H, et al. Analgesic effects of hydromorphone in American
  kestrels. *Am J Vet Res*. 2016.
- Paul-Murphy JR, et al. Analgesic effects of butorphanol and
  buprenorphine in conscious African grey parrots. *Am J Vet Res*. 2009.
- Sánchez-Migallón Guzmán D, et al. Analgesic effects of meloxicam and
  butorphanol in Hispaniolan Amazon parrots. *Am J Vet Res*.
- Sánchez-Migallón Guzmán D, et al. Evaluation of thermal antinociceptive
  effects of tramadol in cockatiels. *Am J Vet Res*. 2014.
- Keller DL, et al. Pharmacokinetics and antinociceptive effects of
  nalbuphine in avian species.
- Stegmann GF, Mukaratirwa S. Analgesic effects of carprofen in broiler
  chickens. *Vet Anaesth Analg*.
- Green PL, Machin KL. Avian analgesia. *Semin Avian Exot Pet Med*.
- Hawkins MG, Paul-Murphy JR. Avian analgesia. *Vet Clin North Am Exot
  Anim Pract*.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Power Analysis for Avian Analgesiometric Studies | "
    "Variance components derived from published avian thermal / mechanical "
    "nociceptive threshold studies. Adjust all parameters for your specific "
    "study. Simulation validation is recommended for final sample-size "
    "decisions."
)
