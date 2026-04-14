import argparse
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
    check_sbi_inputs,
)
from sbi.inference import simulate_for_sbi
from sbi import utils as sbiutils
import torch
import os
import json
import multiprocessing as mp
from functools import partial
from simulator_model_AB import simulate
from parameters_model_AB import (
    passages,
    pop_size_A,
    gene_probs,
    syn_probs_by_gene,
    modelA_priors,
    modelB_priors,
)
import psutil
import time

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024**2  # Convert bytes to MB


def get_allocated_cpus(default=1) -> int:
    """Return CPUs allocated by Slurm to this task, falling back sensibly."""
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(key)
        if v and v.isdigit():
            return int(v)
    # fallback when not running under slurm (e.g., local run)
    return mp.cpu_count() if default is None else default


# TODO: save passage_dict to pickle file (change needed in simulator_model_AB.py)
def main(
    output_dir,
    seq_err_rate,
    seq_sample_size,
    model,
    fixed_params_lst,  # change to dict??
    simulate_sequence_sampling=True,
    ensemble_size=1,
    long_sumstat=0,
    simulations_per_batch=1000,
    index=0,
):
    start = time.time()
    if model == "A":
        parameters = modelA_priors.copy()
        params_priors = modelA_priors.copy()
        pop_size = pop_size_A
    elif model == "B":
        parameters = modelB_priors.copy()
        params_priors = modelB_priors.copy()
        pop_size = (
            pop_size_A  # not using pop_size_B since it's the same for both models
        )
    else:
        raise ValueError(f"Invalid model: {model}")

    parameters.update(syn_probs_by_gene)
    parameters.update(gene_probs)
    parameters["passages"] = passages
    parameters["pop_size"] = pop_size
    parameters["simulations_per_batch"] = simulations_per_batch
    parameters["sequencing_error_rate"] = seq_err_rate
    parameters["seq_sample_size"] = seq_sample_size
    parameters["simulate_seq_sampling"] = simulate_sequence_sampling
    parameters["fixed_params"] = fixed_params_lst
    parameters["model"] = model
    parameters["sumstat"] = "LRG" if long_sumstat else "SR"
    parameters["output_dir"] = output_dir
    parameters["batches"] = f"0-{index}"

    # save the parameters used for the run
    with open(f"{output_dir}/model_{model}_parameters.txt", "w") as outfile:
        json.dump(parameters, outfile)

    syn_probs_by_gene_lst = [
        syn_probs_by_gene["p_mat_syn"],
        syn_probs_by_gene["p_cp_syn"],
        syn_probs_by_gene["p_lys_syn"],
        syn_probs_by_gene["p_rep_syn"],
    ]
    gene_probs_lst = [
        gene_probs["p_mat"],
        gene_probs["p_cp"],
        gene_probs["p_lys"],
        gene_probs["p_rep"],
    ]

    simulator = partial(
        simulate,
        syn_probs_by_gene=syn_probs_by_gene_lst,
        gene_probs=gene_probs_lst,
        model=model,
        passages=passages,
        seq_error_rate=seq_err_rate,
        pop_size=pop_size,
        fixed_params_lst=fixed_params_lst,
        sample_size=seq_sample_size,
        simulate_sequence_sampling=simulate_sequence_sampling,
        long_sumstat=long_sumstat,
    )

    prior = sbiutils.BoxUniform(
        low=torch.as_tensor([val[0] for val in params_priors.values()]),
        high=torch.as_tensor([val[1] for val in params_priors.values()]),
    )
    # print(f"\nprior after sbiutils.BoxUniform: {prior}\n", flush=True)

    # create output directory paths for each batch
    if ensemble_size > 1:
        dir_paths = [
            os.path.join(output_dir, f"ensemble_batch_{x}")
            for x in range(ensemble_size)
        ]
    else:
        dir_paths = [os.path.join(output_dir, f"batch_{index}")]

    processed_prior, num_parameters, prior_returns_numpy = process_prior(prior)
    # print(f"\nprior after process_prior: {processed_prior}", flush=True)
    simulator = process_simulator(simulator, processed_prior, prior_returns_numpy)
    # print(f"\nsimulator after process_simulator: {simulator}", flush=True)
    check_sbi_inputs(simulator, processed_prior)
    print("\nprepared for sbi!", flush=True)

    for _path in dir_paths:
        print(f"Starting simulation for {_path}...", flush=True)
        os.makedirs(
            _path, exist_ok=True
        )  # was False, but now the dir exists and empty..
        print(f"Memory before simulation: {get_memory_usage()} MB", flush=True)

        try:
            allocated_cpus = get_allocated_cpus(
                default=None
            )  # None => use mp.cpu_count() when not in Slurm
            num_of_workers_slurm = min(max(1, allocated_cpus - 1), 20)

            print(
                f"Slurm allocated CPUs: {allocated_cpus}. Using num_workers={num_of_workers_slurm}",
                flush=True,
            )
            theta, x = simulate_for_sbi(
                simulator,
                proposal=processed_prior,
                num_simulations=simulations_per_batch,
                num_workers=num_of_workers_slurm,
            )  # num_workers=mp.cpu_count()-1
            print(f"Memory after simulation: {get_memory_usage()} MB", flush=True)
            print(f"Simulation completed for {_path}")
            torch.save(x, f"{_path}/x_all_passages.pt")
            torch.save(theta, f"{_path}/theta.pt")
            print(f"Saved simulation data to {_path}")
        except Exception as e:
            print(f"Error occurred in simulation for {_path}: {e}")
    end = time.time()
    print(f"Inference time: {end - start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--od", type=str, required=True, help="Path to output directory of simulations"
    )
    parser.add_argument(
        "--sr",
        type=float,
        required=True,
        help="Sequencing error rate, for loop it is usually 10^-5",
    )
    parser.add_argument(
        "--sample_size", type=int, required=True, help="sequencing sample size"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["A", "B"],
        required=True,
        help="'A' is for model A-MOI01, 'B' is for model B-MOI10",
    )
    parser.add_argument(
        "--fixed_params_lst",
        nargs="+",
        type=float,
        required=True,
        help="list of fixed parameters",
    )

    parser.add_argument(
        "--sim_seq_sampling",
        type=int,
        choices=[0, 1],
        help="simulate sequence sampling and add noise (error) to simulations. use 1 for true and 0 for false",
    )
    parser.add_argument(
        "--e",
        type=int,
        help="number of separate simulation batches \
                              and size of ensemble density estimator trained on the simulations",
    )
    parser.add_argument("--s", type=int, help="simulations per batch")
    parser.add_argument(
        "--i", help="index of batch: when running each batch separately"
    )  # type=int
    parser.add_argument(
        "--long_sumstat", type=int, choices=[0, 1], help="use long sumstat or not"
    )
    args = parser.parse_args()

    if not os.path.exists(args.od):
        os.mkdir(args.od)

    main(
        output_dir=args.od,
        seq_err_rate=args.sr,
        seq_sample_size=args.sample_size,
        model=args.model,
        fixed_params_lst=args.fixed_params_lst,
        simulate_sequence_sampling=args.sim_seq_sampling,
        ensemble_size=args.e,
        long_sumstat=args.long_sumstat,
        simulations_per_batch=args.s,
        index=args.i,
    )
