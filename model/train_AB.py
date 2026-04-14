# inference with NPE
import os
import torch
import json
import pickle
import time
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior as Ensemble
import argparse
from sbi import utils as sbiutils
import torch.nn as nn

PARAM_NAMES_A = [
    "mu",
    "w_syn",
    "w_nonsyn_mat",
    "w_nonsyn_cp",
    "w_nonsyn_lys",
    "w_nonsyn_rep",
    "w_ada",
    "p_ada_syn",
    "p_ada_ns_mat",
    "p_ada_ns_cp",
    "p_ada_ns_lys",
    "p_ada_ns_rep",
]

PARAM_NAMES_B = [
    "mu",
    "w_syn",
    "w_ada",
    "p_ada_syn",
    "p_ada_ns_mat",
    "p_ada_ns_cp",
    "p_ada_ns_lys",
    "p_ada_ns_rep",
    "p_mat_nonsyn_rec",
    "p_cp_nonsyn_rec",
    "p_lys_nonsyn_rec",
    "p_rep_nonsyn_rec",
]


def append_sims_from_batches_dir(xs, thetas, batches_dir):
    """
    returns a list of tensors
    """
    for batch_name in os.listdir(batches_dir):
        if ("parameters" in batch_name) or ("README" in batch_name):
            continue
        # if ((int(batch_name[6:]) > 40)):
        #     break
        # (int(batch_name[6:]) < 38) and (int(batch_name[6:])!=24)
        # JUST FOR TRAINING ON A SMALL DATA SET, SINCE NOT ALL DIRS ARE READY. batch 24 is empty.
        print(f"{batch_name=}", flush=True)
        batch_path = os.path.join(batches_dir, batch_name)
        xs.append(torch.load(os.path.join(batch_path, "x_all_passages.pt")))
        thetas.append(torch.load(os.path.join(batch_path, "theta.pt")))
    return xs, thetas


def get_prior_from_params(params, model):
    params_names = PARAM_NAMES_A if model == "A" else PARAM_NAMES_B
    prior_ = {k: v for k, v in params.items() if k in params_names}

    prior = sbiutils.BoxUniform(
        low=torch.as_tensor([val[0] for val in prior_.values()]),
        high=torch.as_tensor([val[1] for val in prior_.values()]),
    )
    return prior


def passages_for_simple_sumstat(xs, passages=[5, 8, 10]):
    """
    default sumstat is of length 110 (passages 0 to 10 nonsyn-ada/nonada for each protein and total syn ada/nonada).
    the sumstat we want to use is of passages 5,8 and 10. (CHECK THIS)
    args:
    xs -- tensor with num_simulations rows, each row is 110 elements long. shape(num_sims, 110)
    passages -- list of passages for the final sumstat

    returns a tensor with num_simulations rows and 10*len(passages) elements in each row -
    final sumstat of each simulation. shape(num_sims, 10*len(passages))

    """

    avg_muts_for_passages = []
    for i in passages:
        avg_muts = xs[:, 10 * i : 10 * (i + 1)]
        avg_muts_for_passages.append(avg_muts)

    sumstat = torch.cat(avg_muts_for_passages, dim=1)
    return sumstat


def get_total_sumstat_p10(xs):
    """
    default sumstat is of length 110+3003 (passages 0 to 10 nonsyn-ada/nonada for each protein and total syn ada/nonada)+genotype sumstat.
    the sumstat we want to use is of passage 10.
    args:
    xs -- tensor with num_simulations rows, each row is 110 elements long. shape(num_sims, 110+3003)
    passages -- list of passages for the final sumstat

    returns a tensor with num_simulations rows and 10*len(passages)+3003 elements in each row -
    final sumstat of each simulation. shape(num_sims, 10*len(passages)+3003)

    """
    sumstat = xs[:, 100:]

    return sumstat


def assign_embedding_net(sumstat):
    if sumstat == "LRG":
        embedding_net = nn.Sequential(
            nn.Linear(3013, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
    elif sumstat == "SR":
        embedding_net = nn.Identity()
    else:
        raise Exception(f"sumstat should be one of [LRG, SR] not {sumstat}!")
    return embedding_net


def train_ensemble(
    i,
    posterior_list,
    prior,
    x,
    theta,
    stop_after_epochs,
    num_sim,
    ensemble_size,
    output_path,
    sumstat="SR",
):
    # assert ensemble_mode in ['vanilla', 'bagging', 'ultra_bagging','big_estimator'], "invalid ensemble mode"
    # bagging = False
    # if (ensemble_mode == 'bagging') or (ensemble_mode == 'ultra_bagging'):
    #     bagging = True

    #### run inference ####
    if i == "ensemble":
        inference = Ensemble(posterior_list)
        posterior = inference
    else:
        inference = NPE(prior, density_estimator="maf")
        density_estimator = inference.append_simulations(theta, x).train(
            stop_after_epochs=stop_after_epochs, validation_fraction=0.1
        )
        posterior = inference.build_posterior(
            density_estimator, sample_with="mcmc", mcmc_method="slice_np"
        )
        posterior_list.append(posterior)

    #### save posterior ####
    if i == "ensemble":
        ending = f"ensemble_{sumstat}_{ensemble_size * num_sim}_{stop_after_epochs}_{i}"  # 8 is ensemble size
    else:
        ending = f"ensemble_{sumstat}_{num_sim}_{stop_after_epochs}_{i}"
    with open(f"{output_path}/{ending}.pkl", "wb") as handle:
        pickle.dump(posterior, handle)


def train_big_estimator(
    prior,
    x,
    theta,
    stop_after_epochs,
    num_sim,
    ensemble_size,
    output_path,
    sumstat="SR",
):
    embedding_net = assign_embedding_net(sumstat)
    # If my_embedding_net is None, sbiutils creates a standard MAF
    density_estimator_build_fun = posterior_nn(model="maf", embedding_net=embedding_net)
    inference = NPE(prior, density_estimator=density_estimator_build_fun)
    density_estimator = inference.append_simulations(theta, x).train(
        stop_after_epochs=stop_after_epochs, validation_fraction=0.1
    )
    posterior = inference.build_posterior(density_estimator)
    ending = f"big_estimator_{sumstat}_{ensemble_size * num_sim}_{stop_after_epochs}"
    with open(f"{output_path}/{ending}.pkl", "wb") as handle:
        pickle.dump(posterior, handle)


def main(
    training_path,
    output_path,
    model,
    big_or_ensemble,
    stop_after_epochs=100,
    num_sim=1000,
    ensemble_size=10,
    total_sumstat=0,
):
    start = time.time()
    assert big_or_ensemble in ["big_estimator", "ensemble"], (
        "invalid mode, could be one big_estimator or ultra-bagging ensemble"
    )
    k = (num_sim * ensemble_size) // 1000
    os.makedirs(output_path, exist_ok=True)
    with open(
        os.path.join(training_path, f"model_{model}_parameters.txt"), "r"
    ) as infile:
        params = json.load(infile)
    params["sims_per_model"] = num_sim
    params["ensemble_size"] = ensemble_size
    params["batches_dir"] = training_path
    params["big_or_ensemble"] = big_or_ensemble
    with open(
        os.path.join(output_path, f"parameters_train_model{model}_{k}k.txt"), "w"
    ) as outfile:
        json.dump(params, outfile)

    prior = get_prior_from_params(params, model)
    num_of_params = 12
    # estimator_path = os.path.join(output_path, 'big_estimator.pkl')
    xs = []
    thetas = []
    xs, thetas = append_sims_from_batches_dir(xs, thetas, training_path)
    tensor_xs = torch.stack(
        xs
    )  # tensor torch.Size([ensemble_size, 1000, 110])  - 1000 num of sim in a batch
    real_num_of_batches, real_sim_num, _ = tensor_xs.shape
    print(f"{real_num_of_batches=} and {real_sim_num=}")
    tensor_thetas = torch.stack(
        thetas
    )  # tensor torch.Size([ensemble_size, 1000, 12]) - 12 num of params
    s = 3113 if total_sumstat else 110
    tensor_all_xs = tensor_xs.reshape(
        real_num_of_batches * real_sim_num, s
    )  # 80000: 8 batches, 10000 simulations per batch
    tensor_all_thetas = tensor_thetas.reshape(
        real_num_of_batches * real_sim_num, num_of_params
    )  # 12 number of parameters of the model

    if num_sim * ensemble_size < real_sim_num * real_num_of_batches:
        tensor_all_xs = tensor_all_xs[: num_sim * ensemble_size, :]
        tensor_all_thetas = tensor_all_thetas[: num_sim * ensemble_size, :]
        print(
            f"{real_sim_num=}, {real_num_of_batches=} \nbut\n {num_sim=}, {ensemble_size=}"
        )
    print(f"{len(xs)=} and {len(xs[0])=}")

    if total_sumstat:
        x_sumstat = get_total_sumstat_p10(tensor_all_xs)
        name = "LRG"
    else:
        x_sumstat = passages_for_simple_sumstat(tensor_all_xs, [10])
        name = "SR"
    print(f"{total_sumstat=},sumstat: {name}, {x_sumstat.shape=}", flush=True)

    if big_or_ensemble == "big_estimator":
        train_big_estimator(
            prior,
            x_sumstat,
            tensor_all_thetas,
            stop_after_epochs,
            num_sim,
            ensemble_size,
            output_path,
            name,
        )

    elif big_or_ensemble == "ensemble":
        posterior_list = []
        print(f"{ensemble_size=}\n {type(ensemble_size)=}\n")
        for i in range(ensemble_size):
            x_sumstat_i = x_sumstat[i * num_sim : (i + 1) * num_sim, :]
            thetas_i = tensor_all_thetas[i * num_sim : (i + 1) * num_sim, :]
            train_ensemble(
                str(i),
                posterior_list,
                prior,
                x_sumstat_i,
                thetas_i,
                stop_after_epochs,
                num_sim,
                ensemble_size,
                output_path,
                name,
            )

        train_ensemble(
            "ensemble",
            posterior_list,
            prior,
            None,
            None,
            stop_after_epochs,
            num_sim,
            ensemble_size,
            output_path,
            name,
        )

    else:
        raise ValueError(
            f"Invalid big_or_ensemble value: {big_or_ensemble}. should be 'big_estimator' or 'ensemble'"
        )

    end = time.time()
    print(f"Inference time: {end - start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_path",
        type=str,
        required=True,
        help="path to simulations containing subdirs train and test",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory of simulations",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["A", "B"],
        required=True,
        help="model A or B",
    )

    parser.add_argument(
        "--big_or_ensemble",
        type=str,
        help="ensemble (ultra bagging)/big_estimator",
    )
    parser.add_argument(
        "--stop_after_epochs", type=int, default=100, help="num of epochs"
    )
    parser.add_argument(
        "--num_sim", type=int, default=1000, help="num of simulations per batch"
    )
    parser.add_argument("--ensemble_size", type=int, default=10, help="ensemble size")
    parser.add_argument(
        "--total_sumstat",
        type=int,
        choices=[0, 1],
        default=0,
        help="use total sumstat. use 1 for true and 0 for false",
    )

    args = parser.parse_args()
    main(
        training_path=args.training_path,
        output_path=args.output_path,
        model=args.model,
        big_or_ensemble=args.big_or_ensemble,
        stop_after_epochs=args.stop_after_epochs,
        num_sim=args.num_sim,
        ensemble_size=args.ensemble_size,
        total_sumstat=args.total_sumstat,
    )
