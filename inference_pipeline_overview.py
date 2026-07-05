"""
Inference pipeline overview for the MS2 public-goods model.

This file is intentionally NOT meant to be run as a normal script.
It documents the logical flow of the simulation-based inference workflow used in
this repository.

The real pipeline is executed through the following scripts:

    sbi_simulate_AB.py
    train_AB.py
    test_empirical_AB.py
    test_nn_performance_AB.py

The purpose of this file is to help readers understand how these pieces fit
together.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Example output directories.
# These are placeholders and should be changed for real runs.
MODEL_A_SIM_DIR = RESULTS_DIR / "stage1_simulations"
MODEL_A_POSTERIOR_DIR = RESULTS_DIR / "stage1_posterior"
MODEL_A_EMPIRICAL_DIR = RESULTS_DIR / "stage1_empirical_results"

MODEL_B_SIM_DIR = RESULTS_DIR / "stage2_simulations"
MODEL_B_POSTERIOR_DIR = RESULTS_DIR / "stage2_posterior"
MODEL_B_EMPIRICAL_DIR = RESULTS_DIR / "stage2_empirical_results"


# ---------------------------------------------------------------------
# Conceptual imports
# ---------------------------------------------------------------------
# These imports show the structure of the codebase.
# They are not needed unless this file is adapted into a real executable script.

# from parameters_model_AB import modelA_priors, modelB_priors
# from simulator_model_AB import simulate
# from evolutionary_model_AB import (
#     get_mutations,
#     simulate_next_passage_final,
#     wrangle_data_simplified,
#     get_expanded_sumstat_simplified,
#     get_full_geno_sumstat_all_passages,
# )


# ---------------------------------------------------------------------
# Generate Model A simulations
# ---------------------------------------------------------------------

def generate_stage1_simulations():
    """
    Generate simulations under stage 1.

    Stage 1 corresponds to the low-MOI condition.
    In this model, coinfection/complementation is assumed to be absent, so
    recessiveness probabilities are fixed to zero.

    The purpose of this stage is to infer baseline evolutionary parameters,
    especially the nonsynonymous fitness effects for each MS2 protein.

    Real command-line script:

        python model/sbi_simulate_AB.py \\
            --od path/to/stage1_simulations \\
            --sr 5e-5 \\
            --sample_size 1309 \\
            --model A \\
            --fixed_params_lst 0 0 0 0 \\
            --sim_seq_sampling 1 \\
            --e 1 \\
            --s 1000 \\
            --i 0 \\
            --long_sumstat 0

    Main outputs:

        model_A_parameters.txt
        batch_<index>/theta.pt
        batch_<index>/x_all_passages.pt

    where theta.pt contains sampled parameters and x_all_passages.pt contains
    simulated summary statistics.
    """
    pass


# ---------------------------------------------------------------------
# Train Model A posterior
# ---------------------------------------------------------------------

def train_stage1_posterior():
    """
    Train a neural posterior estimator for stage 1.

    The training script reads simulation batches containing theta.pt and
    x_all_passages.pt, extracts the desired summary statistics, and trains an
    SBI neural posterior estimator.

    Real command-line script:

        python model/train_AB.py \\
            --training_path path/to/stage1_simulations \\
            --output_path path/to/stage1_posterior \\
            --model A \\
            --big_or_ensemble big_estimator \\
            --stop_after_epochs 100 \\
            --num_sim 1000 \\
            --ensemble_size 1 \\
            --total_sumstat 0

    Main output:

        big_estimator_SR_<num_simulations>_<epochs>.pkl

    This posterior is later conditioned on empirical low-MOI summary statistics.
    """
    pass


# ---------------------------------------------------------------------
# Apply Stage 1 posterior to empirical low-MOI data
# ---------------------------------------------------------------------

def infer_low_moi_empirical_parameters():
    """
    Use the trained stage 1 posterior to infer low-MOI evolutionary parameters.

    The empirical input is a pickle file containing empirical summary statistics.
    The posterior is sampled conditioned on those empirical statistics.

    Real command-line script:

        python model/test_empirical_AB.py \\
            --empirical_data_path path/to/empirical_stage1_sumstats.pkl \\
            --density_estimator_path path/to/stage1_posterior.pkl \\
            --model A \\
            --num_of_samples 10000 \\
            --plot_output_dir path/to/stage1_empirical_results/ \\
            --plot_name stage1_empirical \\
            --emp_line_to_use <line>

    Main outputs:

        posterior samples CSV
        MAP and HDI statistics CSV
        posterior plots

    The key output of this stage is the set of inferred nonsynonymous fitness
    parameters:

        w_nonsyn_mat
        w_nonsyn_cp
        w_nonsyn_lys
        w_nonsyn_rep

    These are used as fixed parameters in Stage 2.
    """
    pass


# ---------------------------------------------------------------------
# Generate Stage 2 simulations
# ---------------------------------------------------------------------

def generate_stage2_simulations():
    """
    Generate simulations under stage 2.

    Stage 2 corresponds to the high-MOI condition, where coinfection is common
    and deleterious nonsynonymous mutations may be masked by complementation.

    In Stage 2, the four nonsynonymous fitness parameters inferred from
    Model A are fixed. The model then infers protein-specific recessiveness
    probabilities:

        p_mat_nonsyn_rec
        p_cp_nonsyn_rec
        p_lys_nonsyn_rec
        p_rep_nonsyn_rec

    These are the main public-goods parameters.

    Real command-line script:

        python model/sbi_simulate_AB.py \\
            --od path/to/modelB_simulations \\
            --sr 5e-5 \\
            --sample_size 1309 \\
            --model B \\
            --fixed_params_lst <w_mat> <w_cp> <w_lys> <w_rep> \\
            --sim_seq_sampling 1 \\
            --e 1 \\
            --s 1000 \\
            --i 0 \\
            --long_sumstat 0

    The values passed to --fixed_params_lst should come from the Stage 1
    empirical posterior, usually from the MAP estimates.
    """
    pass


# ---------------------------------------------------------------------
# Train Stage 2 posterior
# ---------------------------------------------------------------------

def train_modelB_posterior():
    """
    Train a neural posterior estimator for Stage 2.

    Real command-line script:

        python model/train_AB.py \\
            --training_path path/to/stage2_simulations \\
            --output_path path/to/stage2_posterior \\
            --model B \\
            --big_or_ensemble big_estimator \\
            --stop_after_epochs 100 \\
            --num_sim 1000 \\
            --ensemble_size 1 \\
            --total_sumstat 0

    Main output:

        big_estimator_SR_<num_simulations>_<epochs>.pkl

    This posterior is later conditioned on empirical high-MOI summary statistics.
    """
    pass


# ---------------------------------------------------------------------
# Apply Stage 2 posterior to empirical high-MOI data
# ---------------------------------------------------------------------

def infer_high_moi_public_goods_parameters():
    """
    Use the trained Stage 2 posterior to infer high-MOI parameters.

    Real command-line script:

        python model/test_empirical_AB.py \\
            --empirical_data_path path/to/empirical_stage2_sumstats.pkl \\
            --density_estimator_path path/to/stage2_posterior.pkl \\
            --model B \\
            --num_of_samples 10000 \\
            --plot_output_dir path/to/stage2_empirical_results/ \\
            --plot_name modelB_empirical \\
            --emp_line_to_use <line>

    The main biological parameters of interest are:

        p_mat_nonsyn_rec
        p_cp_nonsyn_rec
        p_lys_nonsyn_rec
        p_rep_nonsyn_rec

    A high value means that deleterious nonsynonymous mutations in that protein
    are often masked under coinfection, consistent with public-goods behavior.
    """
    pass


# ---------------------------------------------------------------------
#  Validate posterior performance on synthetic data
# ---------------------------------------------------------------------

def test_neural_posterior_performance():
    """
    Evaluate the trained posterior on held-out synthetic simulations.

    This step tests whether the posterior estimator can recover known simulated
    parameters. It computes MAP accuracy and posterior coverage.

    Real command-line script:

        python model/test_nn_performance_AB.py \\
            --test_data_path path/to/test_simulations \\
            --posterior_path path/to/trained_posterior.pkl \\
            --model B \\
            --output_dir path/to/performance_results \\
            --num_of_sims 2000 \\
            --total_sumstat 0

    Main outputs:

        nn_test_results_<N>_sims.csv
        nn_test_summary_metrics_<N>_sims.csv

    These files summarize how accurately the estimator recovers the true
    parameters used to generate synthetic data.
    """
    pass


# ---------------------------------------------------------------------
# Full conceptual workflow
# ---------------------------------------------------------------------

def full_pipeline_overview():
    """
    Conceptual order of the full inference workflow.

    This function is not meant to execute the analysis. It only lists the
    logical order of operations.
    """

    # Low-MOI baseline inference
    generate_stage1_simulations()
    train_stage1_posterior()
    infer_low_moi_empirical_parameters()

    # High-MOI public-goods inference
    generate_stage2_simulations()
    train_stage2_posterior()
    infer_high_moi_public_goods_parameters()

    # Synthetic validation
    test_neural_posterior_performance()


if __name__ == "__main__":
    raise RuntimeError(
        "This file documents the inference pipeline and is not intended to be run. "
        "Use sbi_simulate_AB.py, train_AB.py, test_empirical_AB.py, and "
        "test_nn_performance_AB.py for real analyses."
    )