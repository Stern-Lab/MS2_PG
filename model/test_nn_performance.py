import torch
import numpy as np
import pandas as pd
import pickle
import arviz as az
import os
import argparse
from sbi import analysis as sbianalysis
from parameters_modelAB import modelA_priors, modelB_priors
import matplotlib.pyplot as plt
import seaborn as sns
from train_AB import passages_for_simple_sumstat, append_sims_from_batches_dir, get_total_sumstat_p10


# from sbi.analysis import check_sbc, run_sbc


def calculate_coverage_and_accuracy(
        density_estimator_path,
        test_theta,
        test_x,
        param_names,
        num_samples=1000,
        hdi_prob=0.95
):
    # TODO: add more hdi probs (turn hdi_prob to a list)
    """
    Args:
        test_theta: torch.Tensor or np.array of true parameters [n_sims, n_params]
        test_x: torch.Tensor or np.array of sumstats [n_sims, n_stat_dims]
    """
    with open(density_estimator_path, "rb") as handle:
        posterior_obj = pickle.load(handle)

    results = []

    n_sims = test_x.shape[0]
    print(f"Starting evaluation on {n_sims} simulations...", flush=True)

    for i in range(n_sims):
        true_val = test_theta[i].numpy() if torch.is_tensor(test_theta) else test_theta[i]
        x_obs = test_x[i].unsqueeze(0) if torch.is_tensor(test_x) else torch.tensor([test_x[i]])

        # 1. Get MAP Estimator
        posterior_obj.set_default_x(x_obs)
        map_estimate = posterior_obj.map().detach().cpu().numpy().flatten()

        # 2. Get Samples for Coverage/HDI
        samples = posterior_obj.sample((num_samples,), x=x_obs, show_progress_bars=True).numpy()

        sim_stats = []
        for p_idx, p_name in enumerate(param_names):
            p_true = true_val[p_idx]
            p_map = map_estimate[p_idx]
            p_samples = samples[:, p_idx]

            # Accuracy: Ratio and Absolute Error
            ratio = p_map / p_true if p_true != 0 else np.nan
            abs_error = np.abs(p_map - p_true)

            # Coverage: Check if true value is within HDI
            hdi = az.hdi(p_samples, hdi_prob=hdi_prob)
            is_covered = (p_true >= hdi[0]) and (p_true <= hdi[1])

            sim_stats.append({
                "sim_id": i,
                "parameter": p_name,
                "true_value": p_true,
                "MAP_estimate": p_map,
                "ratio_MAP_true": ratio,
                "abs_error": abs_error,
                "is_covered": int(is_covered),
                "hdi_low": hdi[0],
                "hdi_high": hdi[1]
            })

        results.extend(sim_stats)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_sims} simulations")

    return pd.DataFrame(results)


def calculate_coverage_lst_and_accuracy(
        density_estimator_path,
        test_theta,
        test_x,
        param_names,
        num_samples=1000,
        hdi_probs=[0.95]  # Changed to a list
):
    with open(density_estimator_path, "rb") as handle:
        posterior_obj = pickle.load(handle)

    results = []
    n_sims = test_x.shape[0]
    print(f"Starting evaluation on {n_sims} simulations...", flush=True)

    for i in range(n_sims):
        true_val = test_theta[i].numpy() if torch.is_tensor(test_theta) else test_theta[i]
        x_obs = test_x[i].unsqueeze(0) if torch.is_tensor(test_x) else torch.tensor([test_x[i]])

        posterior_obj.set_default_x(x_obs)
        map_estimate = posterior_obj.map().detach().cpu().numpy().flatten()
        samples = posterior_obj.sample((num_samples,), x=x_obs, show_progress_bars=True).numpy()

        sim_stats = []
        for p_idx, p_name in enumerate(param_names):
            p_true = true_val[p_idx]
            p_map = map_estimate[p_idx]
            p_samples = samples[:, p_idx]

            ratio = p_map / p_true if p_true != 0 else np.nan
            abs_error = np.abs(p_map - p_true)

            # Dictionary to store this parameter's stats
            stat_entry = {
                "sim_id": i,
                "parameter": p_name,
                "true_value": p_true,
                "MAP_estimate": p_map,
                "ratio_MAP_true": ratio,
                "abs_error": abs_error,
            }

            # Loop through all requested HDIs
            for prob in hdi_probs:
                hdi = az.hdi(p_samples, hdi_prob=prob)
                is_covered = (p_true >= hdi[0]) and (p_true <= hdi[1])

                # Dynamic column names based on probability
                suffix = int(prob * 100)
                stat_entry[f"is_covered_{suffix}"] = int(is_covered)
                stat_entry[f"hdi_{suffix}_low"] = hdi[0]
                stat_entry[f"hdi_{suffix}_high"] = hdi[1]

                # Maintain backward compatibility for original columns if 0.95 is present
                # if prob == 0.95:
                #     stat_entry["is_covered"] = int(is_covered)
                #     stat_entry["hdi_low"] = hdi[0]
                #     stat_entry["hdi_high"] = hdi[1]

            sim_stats.append(stat_entry)

        results.extend(sim_stats)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_sims} simulations")

    return pd.DataFrame(results)


def summarize_test_results2(df):
    """Aggregates accuracy and all coverage metrics found in the dataframe."""
    # Identify all coverage columns (e.g. is_covered_50, is_covered_95)
    coverage_cols = [c for c in df.columns if c.startswith("is_covered")]

    agg_dict = {
        "ratio_MAP_true": ["mean", "std"],
        "abs_error": "mean"
    }
    for col in coverage_cols:
        agg_dict[col] = "mean"

    summary = df.groupby("parameter").agg(agg_dict)
    return summary


def summarize_test_results(df):
    """Prints aggregate accuracy and coverage metrics."""
    summary = df.groupby("parameter").agg({
        "ratio_MAP_true": ["mean", "std"],
        "is_covered": "mean",  # This is the coverage probability
        "abs_error": "mean"
    })
    summary.columns = ["Ratio_Mean", "Ratio_Std", "Coverage_95", "MAE"]
    return summary


def plot_diagnostics_new(df, output_dir, param_to_name, num_of_sims):
    plt.rcParams.update({'mathtext.fontset': 'stix'})
    os.makedirs(output_dir, exist_ok=True)

    params = df['parameter'].unique()
    n_params = len(params)

    if num_of_sims > 1000:
        num_of_sims = num_of_sims // 1000
        k = True
    else:
        k = False

    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    axes_flat = axes.flatten()

    for i, p_name in enumerate(params):
        sub_df = df[df['parameter'] == p_name]
        display_name = param_to_name.get(p_name, p_name)

        ax_parity = axes_flat[2 * i]
        ax_hist = axes_flat[2 * i + 1]

        # --- Plot 1: (True vs. MAP) ---
        sns.scatterplot(data=sub_df, x="true_value", y="MAP_estimate", ax=ax_parity, alpha=0.6)

        min_val = min(sub_df["true_value"].min(), sub_df["MAP_estimate"].min())
        max_val = max(sub_df["true_value"].max(), sub_df["MAP_estimate"].max())
        ax_parity.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (1:1)')

        ax_parity.set_title(f"{display_name}", fontsize=12)
        ax_parity.set_xlabel("True Value")
        ax_parity.set_ylabel("MAP Estimate")

        if i == 0:
            ax_parity.legend(loc='upper left')

        # --- Plot 2: Accuracy Histogram (Ratio) ---
        ratios = sub_df["ratio_MAP_true"].dropna()
        # Filter outliers (1st to 99th percentile) for visual clarity
        ratios = ratios[ratios.between(ratios.quantile(0.01), ratios.quantile(0.99))]

        sns.histplot(ratios, kde=True, ax=ax_hist, color="teal", element="step", fill=True, alpha=0.4,
                     edgecolor="teal")
        ax_hist.axvline(1.0, color="red", linestyle="-", label="Target (1.0)")

        ax_hist.set_title(f"{display_name}", fontsize=12)
        ax_hist.set_xlabel("MAP / True Ratio")
        ax_hist.set_ylabel("Density")

        if i == 0:
            ax_hist.legend(loc='upper right')

    # 2. Hide unused subplots
    for j in range(2 * n_params, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f"Model Inference Diagnostics", fontsize=22, y=1.02, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"accuracy_diagnostics_plots_{num_of_sims}{'k' if k else ''}_sims.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved accuracy plots to {save_path}")


def plot_diagnostics_log_ratio(df, output_dir, param_to_name, num_of_sims):
    plt.rcParams.update({'mathtext.fontset': 'stix'})
    os.makedirs(output_dir, exist_ok=True)

    params = df['parameter'].unique()
    n_params = len(params)

    if num_of_sims > 1000:
        num_of_sims = num_of_sims // 1000
        k = True
    else:
        k = False

    fig, axes = plt.subplots(6, 4, figsize=(22, 24))
    axes_flat = axes.flatten()

    for i, p_name in enumerate(params):
        if i >= 12: break

        sub_df = df[df['parameter'] == p_name]

        display_name = param_to_name.get(p_name, p_name)

        ax_parity = axes_flat[2 * i]
        ax_hist = axes_flat[2 * i + 1]

        # --- Plot 1  ---
        sns.scatterplot(data=sub_df, x="true_value", y="MAP_estimate", ax=ax_parity, alpha=0.6)

        min_val = min(sub_df["true_value"].min(), sub_df["MAP_estimate"].min())
        max_val = max(sub_df["true_value"].max(), sub_df["MAP_estimate"].max())
        ax_parity.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (1:1)')

        ax_parity.set_title(f"{display_name}", fontsize=12)
        ax_parity.set_xlabel("True Value")
        ax_parity.set_ylabel("MAP Estimate")

        if i == 0:
            ax_parity.legend(loc='upper left')

        # --- Plot 2: Log10(MAP / True) Ratio Histogram ---
        log_ratio = np.log10(sub_df["ratio_MAP_true"] + 1e-9)

        sns.histplot(log_ratio, kde=True, ax=ax_hist, color="mediumpurple", element="step", fill=True, alpha=0.4,
                     edgecolor="mediumpurple")
        ax_hist.axvline(0.0, color="red", linestyle="-", label="Target (0)")

        ax_hist.set_title(f"{display_name}", fontsize=12)
        ax_hist.set_xlabel("log10(MAP Error Ratio)")
        ax_hist.set_ylabel("Density")

        if i == 0:
            ax_hist.legend(loc='upper right')

    for j in range(2 * n_params, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle("Model Inference Diagnostics",
                 fontsize=22, y=1.02, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"log_ratio_diagnostics_{num_of_sims}{'k' if k else ''}_sims.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved named diagnostic plots to {save_path}")


def main(test_data_path, posterior_path, model, output_dir, num_of_sims, total_sumstat=0):
    param_names = list(modelB_priors.keys()) if model == "B" else list(modelA_priors.keys())

    num_of_params = 12
    if num_of_sims > 1000:
        k = True
    else:
        k = False
    param_to_name2 = {
        'mu': r'Mutation Rate ($\mu$)',
        'w_syn': r'Syn effect ($\omega_{syn}$)',
        'w_nonsyn_mat': r'Mat non-syn effect ($\omega_{ns{,}mat}$)',
        'w_nonsyn_cp': r'Cp non-syn effect ($\omega_{ns{,}cp}$)',
        'w_nonsyn_lys': r'Lys non-syn effect ($\omega_{ns{,}lys}$)',
        'w_nonsyn_rep': r'Rep non-syn effect ($\omega_{ns{,}rep}$)',
        'w_ada': r'Adaptive effect ($\omega_{ada}$)',
        'p_ada_syn': r'Adaptive syn prob ($P_{ada{,}syn}$)',
        "p_ada_ns_mat": r'Adaptive mat non-syn prob ($P_{ada{,}ns{,}mat}$)',
        "p_ada_ns_cp": r'Adaptive cp non-syn prob ($P_{ada{,}ns{,}cp}$)',
        "p_ada_ns_lys": r'Adaptive lys non-syn prob ($P_{ada{,}ns{,}lys}$)',
        "p_ada_ns_rep": r'Adaptive rep non-syn prob ($P_{ada{,}ns{,}rep}$)',
        "p_mat_nonsyn_rec": r'Mat recessive prob ($P_{mat{,}ns{,}rec}$)',
        "p_cp_nonsyn_rec": r'Cp recessive prob ($P_{cp{,}ns{,}rec}$)',
        "p_lys_nonsyn_rec": r'Lys recessive prob ($P_{lys{,}ns{,}rec}$)',
        "p_rep_nonsyn_rec": r'Rep recessive prob ($P_{rep{,}ns{,}rec}$)',
    }
    # estimator_path = os.path.join(output_path, 'big_estimator.pkl')
    test_xs = []
    test_thetas = []
    test_xs, test_thetas = append_sims_from_batches_dir(test_xs, test_thetas, test_data_path)
    tensor_test_xs = torch.stack(
        test_xs
    )  # tensor torch.Size([ensemble_size, 1000, 110])  - 1000 num of sim in a batch
    real_num_of_batches, real_sim_num, _ = tensor_test_xs.shape
    print(f"{real_num_of_batches=} and {real_sim_num=}")
    tensor_test_thetas = torch.stack(
        test_thetas
    )  # tensor torch.Size([ensemble_size, 1000, 12]) - 12 num of params
    s = 3113 if total_sumstat else 110
    tensor_all_xs = tensor_test_xs.reshape(
        real_num_of_batches * real_sim_num, s
    )  # 80000: 8 batches, 10000 simulations per batch
    tensor_all_thetas = tensor_test_thetas.reshape(real_num_of_batches * real_sim_num,
                                                   num_of_params)  # 12 number of parameters of the model

    tensor_all_xs = tensor_all_xs[:num_of_sims, :]
    tensor_all_thetas = tensor_all_thetas[:num_of_sims, :]
    if total_sumstat:
        sumstat_test = get_total_sumstat_p10(tensor_all_xs)
    else:
        sumstat_test = passages_for_simple_sumstat(tensor_all_xs, passages=[10])

    # test_results_df = calculate_coverage_and_accuracy(
    #     posterior_path,
    #     tensor_all_thetas,
    #     sumstat_test,
    #     param_names
    # )

    hdi_to_test = [0.50, 0.80, 0.95]

    test_results_df = calculate_coverage_lst_and_accuracy(
        posterior_path,
        tensor_all_thetas,
        sumstat_test,
        param_names,
        hdi_probs=hdi_to_test  # Pass the list here
    )


    # Save detailed results
    test_results_df.to_csv(
        f"{output_dir}/nn_test_results_{num_of_sims // 1000 if k else num_of_sims}{'k' if k else ''}_sims.csv",
        index=False)

    # Show Summary
    performance_summary = summarize_test_results2(test_results_df)
    print("\n--- PERFORMANCE SUMMARY ---")
    print(performance_summary)
    performance_summary.to_csv(
        f"{output_dir}/nn_test_summary_metrics_{num_of_sims // 1000 if k else num_of_sims}{'k' if k else ''}_sims.csv")
    print(f"performance summary saved to {output_dir}")
    # plot_diagnostics_log_ratio(test_results_df, output_dir, param_to_name2, num_of_sims)
    # plot_diagnostics_new(test_results_df, output_dir, param_to_name2, num_of_sims)
    # perform_sbc_analysis(posterior_obj, test_data['thetas'], test_data['x'], param_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="path to simulations containing subdirs of test simulations",
    )

    parser.add_argument(
        "--posterior_path",
        type=str,
        required=True,
        help="path to trained NN posterior",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["A", "B"],
        required=True,
        help="model A or B",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output directory of test results, plots and dfs",
    )

    parser.add_argument(
        "--num_of_sims",
        type=int,
        required=True,
        help="number of simulations to test",
    )

    parser.add_argument(
        "--total_sumstat",
        type=int,
        choices=[0, 1],
        help="use total sumstat or not",
    )

    args = parser.parse_args()
    main(
        test_data_path=args.test_data_path,
        posterior_path=args.posterior_path,
        model=args.model,
        output_dir=args.output_dir,
        num_of_sims=args.num_of_sims,
        total_sumstat=args.total_sumstat,
    )
