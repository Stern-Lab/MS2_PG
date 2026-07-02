import argparse
import pickle
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# ============================================================
# Hierarchical beta-binomial model pipeline:
# low-frequency mutations, MOI × protein effect
# ============================================================

# -----------------------------
# Preprocess data
# -----------------------------

def prepare_mutation_df(
    df,
    mut_type,
    freq_threshold=0.01,
    adaptive_col="adaptive_2plus_lines_def",
    exclude_adaptive=True,
    exclude_stop_codons=False,
    protein_order=("mat", "cp", "lys", "rep")
):
    """
    Prepare mutation-frequency dataframe for hierarchical beta-binomial model.

    Parameters
    ----------
    df : pd.DataFrame
        Full mutation-frequency dataframe.
    mut_type : str
        Either "nonsyn" or "syn".
    freq_threshold : float
        Maximum mutation frequency to include.
    adaptive_col : str
        Column defining adaptive/nonadaptive status.
    exclude_adaptive : bool
        Whether to keep only nonadaptive mutations.
    exclude_stop_codons : bool
        Whether to exclude premature stop codon mutations.
    protein_order : tuple/list
        Order of protein categories.

    Returns
    -------
    df_low : pd.DataFrame
        Processed dataframe.
    """

    df_low = df[
        (df["freq"] < freq_threshold) &
        (df["syn_nonsyn"] == mut_type)
    ].copy()

    if exclude_adaptive:
        df_low = df_low[df_low[adaptive_col] == "nonada"].copy()

    if exclude_stop_codons:
        df_low = df_low[~df_low["stop_codon"].astype(bool)].copy()

    df_low["y"] = df_low["read_count"].astype(int)

    df_low["n"] = np.round(df_low["read_count"] / df_low["freq"]).astype(int)

    df_low["MOI10"] = (df_low["MOI"] == 10).astype(int)

    df_low["experiment"] = (
        "MOI" + df_low["MOI"].astype(str) +
        "_line_" + df_low["line"].astype(str)
    )

    df_low["protein"] = pd.Categorical(
        df_low["protein"],
        categories=list(protein_order),
        ordered=True
    )

    df_low = df_low.dropna(
        subset=["protein", "y", "n", "MOI10", "experiment"]
    ).copy()

    df_low = df_low[(df_low["n"] > 0) & (df_low["y"] >= 0)].copy()
    df_low = df_low[df_low["y"] <= df_low["n"]].copy()

    return df_low


# -----------------------------
# Fit model
# -----------------------------

def fit_beta_binomial_moi_model(
    df_low,
    protein_order=("mat", "cp", "lys", "rep"),
    draws=2000,
    tune=2000,
    chains=4,
    target_accept=0.99,
    random_seed=42
):
    """
    Fit hierarchical beta-binomial model with:
    - protein baseline effects
    - protein-specific MOI10 effects
    - random experiment intercept
    """

    protein_idx = df_low["protein"].cat.codes.values
    experiment_codes, experiment_names = pd.factorize(df_low["experiment"])

    y = df_low["y"].values
    n = df_low["n"].values
    MOI10 = df_low["MOI10"].values

    coords = {
        "protein": list(protein_order),
        "experiment": experiment_names,
        "obs": np.arange(len(df_low))
    }

    with pm.Model(coords=coords) as model:

        protein_i = pm.Data("protein_i", protein_idx, dims="obs")
        experiment_i = pm.Data("experiment_i", experiment_codes, dims="obs")
        moi10_i = pm.Data("moi10_i", MOI10, dims="obs")
        n_i = pm.Data("n_i", n, dims="obs")

        alpha = pm.Normal("alpha", mu=-7, sigma=2)

        protein_baseline = pm.Normal(
            "protein_baseline",
            mu=0,
            sigma=1,
            dims="protein"
        )

        moi10_effect = pm.Normal(
            "moi10_effect",
            mu=0,
            sigma=1,
            dims="protein"
        )

        sigma_experiment = pm.Exponential("sigma_experiment", 1)

        experiment_re = pm.Normal(
            "experiment_re",
            mu=0,
            sigma=sigma_experiment,
            dims="experiment"
        )

        kappa = pm.Exponential("kappa", 1)

        eta = (
            alpha
            + protein_baseline[protein_i]
            + moi10_effect[protein_i] * moi10_i
            + experiment_re[experiment_i]
        )

        p = pm.Deterministic("p", pm.math.sigmoid(eta), dims="obs")

        a = p * kappa
        b = (1 - p) * kappa

        pm.BetaBinomial(
            "y_obs",
            n=n_i,
            alpha=a,
            beta=b,
            observed=y,
            dims="obs"
        )

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed
        )

    return model, trace


# -----------------------------
# Summarize results
# -----------------------------

def summarize_moi_model(trace, reference_protein="mat"):
    summary = az.summary(
        trace,
        var_names=["moi10_effect", "sigma_experiment", "kappa"],
        hdi_prob=0.95
    )

    moi_samples = trace.posterior["moi10_effect"]

    prob_positive = (
        (moi_samples > 0)
        .mean(dim=("chain", "draw"))
        .to_dataframe(name="P(MOI10 effect > 0)")
    )

    bayes_p_like = (
        (moi_samples <= 0)
        .mean(dim=("chain", "draw"))
        .to_dataframe(name="P(MOI10 effect <= 0)")
    )

    contrasts = {}

    proteins = list(moi_samples.protein.values)

    for protein in proteins:
        if protein == reference_protein:
            continue

        diff = (
            moi_samples.sel(protein=protein)
            - moi_samples.sel(protein=reference_protein)
        )

        diff_values = diff.stack(sample=("chain", "draw")).values

        contrasts[f"{protein} - {reference_protein}"] = {
            "mean": np.mean(diff_values),
            "ci_2.5%": np.quantile(diff_values, 0.025),
            "ci_97.5%": np.quantile(diff_values, 0.975),
            "P(diff > 0)": np.mean(diff_values > 0)
        }

    contrasts_df = pd.DataFrame(contrasts).T

    return summary, prob_positive, bayes_p_like, contrasts_df



def run_moi_mutation_model(
    df,
    mut_type,
    freq_threshold=0.01,
    exclude_adaptive=True,
    exclude_stop_codons=False,
    protein_order=("mat", "cp", "lys", "rep"),
    reference_protein="mat",
    draws=2000,
    tune=2000,
    chains=4,
    target_accept=0.99,
    random_seed=42
):
    """
    Complete pipeline:
    preprocessing -> model fitting -> summaries
    """

    df_model = prepare_mutation_df(
        df=df,
        mut_type=mut_type,
        freq_threshold=freq_threshold,
        exclude_adaptive=exclude_adaptive,
        exclude_stop_codons=exclude_stop_codons,
        protein_order=protein_order
    )

    print(f"\nRunning model for {mut_type} mutations")
    print(f"Number of observations: {len(df_model)}")
    print(df_model.groupby(["MOI", "protein"]).size())

    model, trace = fit_beta_binomial_moi_model(
        df_low=df_model,
        protein_order=protein_order,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed
    )

    summary, prob_positive, bayes_p_like, contrasts_df = summarize_moi_model(
        trace=trace,
        reference_protein=reference_protein
    )

    return {
        "df_model": df_model,
        "model": model,
        "trace": trace,
        "summary": summary,
        "prob_positive": prob_positive,
        "bayes_p_like": bayes_p_like,
        "contrasts_df": contrasts_df
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to pickled/csv dataframe")
    parser.add_argument("--mut_type", required=True, choices=["nonsyn", "syn"])
    parser.add_argument("--freq_threshold", type=float, default=0.01)
    parser.add_argument("--exclude_adaptive", type=int, default=1)
    parser.add_argument("--exclude_stop_codons", type=int, default=0)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="output pickle path")
    args = parser.parse_args()

    if args.data.endswith(".csv"):
        df = pd.read_csv(args.data)
    else:
        df = pd.read_pickle(args.data)

    results = run_moi_mutation_model(
        df=df,
        mut_type=args.mut_type,
        freq_threshold=args.freq_threshold,
        exclude_adaptive=bool(args.exclude_adaptive),
        exclude_stop_codons=bool(args.exclude_stop_codons),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed
    )


    az.to_netcdf(results["trace"], args.out.replace(".pkl", "_trace.nc"))
    light = {k: v for k, v in results.items() if k not in ("trace", "model")}
    with open(args.out, "wb") as f:
        pickle.dump(light, f)

    print(f"Done. Saved to {args.out}")

if __name__ == "__main__":
    main()