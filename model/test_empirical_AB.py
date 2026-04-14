import argparse
import os
import pickle
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sbi import analysis as sbianalysis
import torch
from parameters_model_AB import modelA_priors, modelB_priors

class PosteriorWrapper:
    """Wrapper for a trained SNPE posterior (density estimator)."""

    def __init__(self, density_estimator_path, param_names_lst):
        with open(density_estimator_path, "rb") as handle:
            inference_from_disk = pickle.load(handle)
        self.post = inference_from_disk
        self.posterior = None
        self.labels = param_names_lst

    def sample_posterior(self, empirical_data_vector, num_samples=1000):
        """Sample posterior conditioned on empirical summary statistics."""
        print("sampling posterior!!")
        self.posterior = self.post.sample((num_samples,), x=empirical_data_vector)

    def compute_sbi_map(self, empirical_data_vector):
        if not isinstance(empirical_data_vector, torch.Tensor):
            x = torch.tensor(empirical_data_vector)
        else:
            x = empirical_data_vector

        if x.ndim == 1:
            x = x.unsqueeze(0)

        # sbi's map() returns the parameter vector that maximizes the log-probability
        self.post.set_default_x(x)
        map_estimate = self.post.map()
        return map_estimate.detach().cpu().numpy().flatten()

    def get_post(self):
        return pd.DataFrame(self.posterior.numpy(), columns=self.labels)


def get_emp_sumstats_dict(emp_smst_path):
    with open(emp_smst_path, 'rb') as f:
        emp_sumstat_by_line_dict = pickle.load(f)
    return emp_sumstat_by_line_dict


def get_emp_mean_sumstat(emp_sumstat_by_line_dict):
    return torch.tensor(list(emp_sumstat_by_line_dict.values())).mean(axis=0)


def summarize_posterior_with_sbi(post_df, map_vector, hdi_prob=0.95):
    """
    Args:
        post_df: The sampled dataframe.
        map_vector: The array returned by the SBI .map() method.
    """
    stats = []
    for i, col in enumerate(post_df.columns):
        vals = post_df[col].values
        # Use the specific index from the SBI optimization result
        map_est = map_vector[i]

        hdi_low, hdi_high = az.hdi(vals, hdi_prob=hdi_prob)
        stats.append({
            "param": col,
            "MAP": map_est,
            "HDI_low": hdi_low,
            "HDI_high": hdi_high
        })
    return pd.DataFrame(stats)



def plot_posteriors(post_df, stats, save_plot, plot_output_dir, plot_name, priors_dict=None, bins=50):
    """
    Plot posterior histograms with MAP + HDI and overlay the Uniform Prior.

    priors_dict: dict { 'param_name': (low, high) }
    """
    n_params = len(post_df.columns)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes_flat = axes.flatten()
    if n_params == 1:
        axes = [axes]

    for i, col in enumerate(post_df.columns):
        vals = post_df[col].values
        ax = axes_flat[i]

        # 1. Plot the Posterior Histogram
        ax.hist(vals, bins=bins, density=True, alpha=0.6, color="skyblue", label="Posterior")

        # 2. Add MAP and HDI
        row = stats.loc[stats["param"] == col]
        if not row.empty:
            map_val = row["MAP"].values[0]
            hdi_low = row["HDI_low"].values[0]
            hdi_high = row["HDI_high"].values[0]

            ax.axvline(map_val, color="red", linestyle="--", label="MAP")
            ax.axvspan(hdi_low, hdi_high, color="gray", alpha=0.2, label="95% HDI")

        # 3. Overlay the Prior and adjust X-limits
        if priors_dict and col in priors_dict:
            prior_low, prior_high = priors_dict[col]

            # Calculate height of the uniform prior
            prior_height = 1.0 / (prior_high - prior_low)

            # Plot prior as a horizontal line
            ax.hlines(y=prior_height, xmin=prior_low, xmax=prior_high,
                      color="orange", linewidth=2, label="Prior", linestyle='-')

            # Ensure x-axis covers the full prior range
            # We add a 5% margin so the line doesn't touch the plot edges
            padding = (prior_high - prior_low) * 0.05
            ax.set_xlim(prior_low - padding, prior_high + padding)

        ax.set_title(f"{col}", fontweight='bold')
        ax.set_ylabel("Density")
        if i==0:
            ax.legend(fontsize='small', frameon=False)


    plt.tight_layout()
    if save_plot:
        os.makedirs(plot_output_dir, exist_ok=True)
        save_path = os.path.join(plot_output_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved posterior plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main(
    empirical_data_path,
    density_estimator_path,
    model,
    num_of_samples,
    plot_output_dir,
    plot_name,
    emp_line_to_use='mean',
):
    emp_data_dict = get_emp_sumstats_dict(empirical_data_path)
    # mean_empirical_data = get_emp_mean_sumstat(emp_data_dict)
    print(f'got empirical data!\n{emp_data_dict=}', flush=True)
    MOI = '0.1' if emp_data_dict.get('E') is None else '10'
    if model == 'B':
        param_names = list(modelB_priors.keys())
    else:
        param_names = list(modelA_priors.keys())
    model = PosteriorWrapper(density_estimator_path, param_names)
    print(f'{MOI=}, {param_names=}')

    if emp_line_to_use!='mean':
        emp_line_to_use_lst = emp_line_to_use.split('_')

        if (MOI=='10' and not set(emp_line_to_use_lst).issubset(set(['A', 'G', 'E', 'H']))) or (MOI=='0.1' and not set(emp_line_to_use_lst).issubset(set(['1', '2', '3']))):
            raise ValueError("MOI10 must be used with line A, G, E, H and MOI0.1 must be used with line 1, 2, 3")
        if MOI=='0.1':
            emp_line_to_use_lst = [int(i) for i in emp_line_to_use_lst]

        if len(emp_line_to_use_lst) > 1:
            emp_data_dict_to_use = {key: emp_data_dict.get(key) for key in emp_line_to_use_lst}
            emp_data = get_emp_mean_sumstat(emp_data_dict_to_use)
        else: # single line
            emp_data = torch.tensor(emp_data_dict[emp_line_to_use_lst[0]])
    else:  # use all possible lines, and get the mean sumstat of them all
        emp_data = get_emp_mean_sumstat(emp_data_dict) if emp_data_dict.get('mean_sumstat') is None else  emp_data_dict.get('mean_sumstat')


    print(f'using empirical data line_{emp_line_to_use if emp_line_to_use else "mean"}\n{len(emp_data)=},{emp_data[:10]=}', flush=True)

    # 2. Get samples (for HDI calculation)
    model.sample_posterior(emp_data, num_samples=num_of_samples)
    post_df = model.get_post()

    # 3. Calculate MAP using SBI optimization
    # (Assuming self.post in your wrapper is the DirectPosterior object)
    sbi_map_values = model.compute_sbi_map(emp_data)

    # 4. Summarize using the high-accuracy MAP
    stats = summarize_posterior_with_sbi(post_df, sbi_map_values)

    print("SBI Optimized MAP Estimates:")
    print(stats)
    name = f'line_{emp_line_to_use.replace(",", "_")}_{num_of_samples//1000}k_samples' if emp_line_to_use else f'mean_smst_{num_of_samples//1000}k_samples'
    priors_dict = modelB_priors if MOI == '10' else modelA_priors
    plot_posteriors(post_df, stats, True, plot_output_dir, f'MOI{MOI}_{name}_{plot_name}', priors_dict)
    stats.to_csv(plot_output_dir + f'MOI{MOI}_{name}_{plot_name}_stats.csv', index=False)
    print(f'saved stats to {plot_output_dir}MOI{MOI}_{plot_name}_stats.csv')
    post_df.to_csv(plot_output_dir + f'MOI{MOI}_{name}_{plot_name}_posterior_samples.csv', index=False)
    print(f'saved posterior samples to {plot_output_dir}MOI{MOI}_{plot_name}_posterior_samples.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--empirical_data_path",
        type=str,
        required=True,
        help="path empirical data",
    )

    parser.add_argument(
        "--density_estimator_path",
        type=str,
        required=True,
        help="Path to density estimator",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['A', 'B'],
        help="model to use (A or B)",
    )
    parser.add_argument(
        "--num_of_samples",
        type=int,
        required=True,
        help="number of samples to be drawn from the posterior",
    )
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        required=True,
        help="output directory for the plot",
    )
    parser.add_argument(
        "--plot_name", type=str, required=True, help="name of plot"
    )
    parser.add_argument(
        "--emp_line_to_use", type=str, required=False, help="empirical line/s to use (1, 2, 3 for MOI=0.1 and A, G, E, H for MOI=10) separated by ',' only! given lines are averaged"
    )
    args = parser.parse_args()
    main(
        empirical_data_path=args.empirical_data_path,
        density_estimator_path=args.density_estimator_path,
        model=args.model,
        num_of_samples=args.num_of_samples,
        plot_output_dir=args.plot_output_dir,
        plot_name=args.plot_name,
        emp_line_to_use=args.emp_line_to_use,
    )