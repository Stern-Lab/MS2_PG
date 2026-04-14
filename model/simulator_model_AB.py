import numpy as np
import gc
from evolutionary_model_AB import (
    get_mut_type_probs_per_gene_lst_simp_model,
    get_mutations,
    wrangle_data_simplified,
    get_expanded_sumstat_simplified,
    normalize_freqs_dict,
    simulate_next_passage_final,
    get_full_geno_sumstat_all_passages,
)


def simulate_sequencing(
    passages_dict,
    sample_size,
    seq_error_rate,
    mutation_type_probs_per_gene_lst,
    gene_probs,
    simplified=True,
):
    print("simulating sequence sampling:", flush=True)
    sequenced_passages = dict()
    fitness_effects = np.ones(6)  # fake fitness effects
    # generate errors using error rate
    for i in passages_dict.keys():
        seq_errors = get_mutations(
            sample_size * seq_error_rate,
            mutation_type_probs_per_gene_lst,
            gene_probs,
            1 / (100 * sample_size),
            simplified,
        )

        sequenced_passages[i] = normalize_freqs_dict(passages_dict[i], sample_size)
        # add errors to the sequenced passage
        if len(sequenced_passages[i]) != 0:
            sequenced_passages[i] = simulate_next_passage_final(
                fitness_effects,
                sequenced_passages[i],
                seq_errors,
                sample_size,
                simplified,
                100,
            )
    return sequenced_passages


def get_modelA_priors(sampled_params):
    model_priors_dict = {
        "mu": sampled_params[0],
        "w_syn": sampled_params[1],
        "w_nonsyn_mat": sampled_params[2],
        "w_nonsyn_cp": sampled_params[3],
        "w_nonsyn_lys": sampled_params[4],
        "w_nonsyn_rep": sampled_params[5],
        "w_ada": sampled_params[6],
        "p_ada_syn": sampled_params[7],
        "p_ada_ns_mat": sampled_params[8],
        "p_ada_ns_cp": sampled_params[9],
        "p_ada_ns_lys": sampled_params[10],
        "p_ada_ns_rep": sampled_params[11],
        "p_mat_nonsyn_rec": 0,
        "p_cp_nonsyn_rec": 0,
        "p_lys_nonsyn_rec": 0,
        "p_rep_nonsyn_rec": 0,
    }
    return model_priors_dict


def get_modelB_priors(sampled_params, nonsyn_fitness_lst):
    model_priors_dict = {
        "mu": sampled_params[0],
        "w_syn": sampled_params[1],
        "w_nonsyn_mat": nonsyn_fitness_lst[0],
        "w_nonsyn_cp": nonsyn_fitness_lst[1],
        "w_nonsyn_lys": nonsyn_fitness_lst[2],
        "w_nonsyn_rep": nonsyn_fitness_lst[3],
        "w_ada": sampled_params[2],
        "p_ada_syn": sampled_params[3],
        "p_ada_ns_mat": sampled_params[4],
        "p_ada_ns_cp": sampled_params[5],
        "p_ada_ns_lys": sampled_params[6],
        "p_ada_ns_rep": sampled_params[7],
        "p_mat_nonsyn_rec": sampled_params[8],
        "p_cp_nonsyn_rec": sampled_params[9],
        "p_lys_nonsyn_rec": sampled_params[10],
        "p_rep_nonsyn_rec": sampled_params[11],
    }
    return model_priors_dict


def simulate(
    sampled_params,
    syn_probs_by_gene,
    gene_probs,
    model,  # (NEW ARGUMENT)'A' or 'B' where A infers w_nonsyn params using MOI01 and B infers p_rec using MOI10
    passages,
    seq_error_rate,
    pop_size,
    fixed_params_lst,
    # should be a list of length 4 [w_nonsyn_mat, w_nonsyn_cp, w_nonsyn_lys, w_nonsyn_rep]
    sample_size=1309,
    simulate_sequence_sampling=True,
    simplified=True,
    batch_size=100,
    long_sumstat=0,
):
    try:
        if model == "A":
            model_priors_dict = get_modelA_priors(sampled_params)
            print(f"model A priors: {model_priors_dict=}", flush=True)
        elif model == "B":
            model_priors_dict = get_modelB_priors(sampled_params, fixed_params_lst)
            print(f"model B priors: {model_priors_dict=}", flush=True)
        else:
            raise ValueError(f"Invalid model: {model}")

        p_mat_syn = syn_probs_by_gene[0]
        p_cp_syn = syn_probs_by_gene[1]
        p_lys_syn = syn_probs_by_gene[2]
        p_rep_syn = syn_probs_by_gene[3]

        print("\nprocessed params")
        print(f"{simulate_sequence_sampling=}, params: {sampled_params}", flush=True)
        # def get_mut_type_probs_per_gene_lst_simp_model(p_protein_syn, p_ada_syn,p_ada_nonsyn,p_protein_nonsyn_rec):
        mat_probs_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_mat_syn,
            model_priors_dict["p_ada_syn"],
            model_priors_dict["p_ada_ns_mat"],
            model_priors_dict["p_mat_nonsyn_rec"],
        )
        cp_probs_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_cp_syn,
            model_priors_dict["p_ada_syn"],
            model_priors_dict["p_ada_ns_cp"],
            model_priors_dict["p_cp_nonsyn_rec"],
        )
        lys_probs_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_lys_syn,
            model_priors_dict["p_ada_syn"],
            model_priors_dict["p_ada_ns_lys"],
            model_priors_dict["p_lys_nonsyn_rec"],
        )
        rep_probs_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_rep_syn,
            model_priors_dict["p_ada_syn"],
            model_priors_dict["p_ada_ns_rep"],
            model_priors_dict["p_rep_nonsyn_rec"],
        )
        # print(f"{mat_probs_lst=}\n", flush=True)
        # print(f"{cp_probs_lst=}\n", flush=True)
        # print(f"{lys_probs_lst=}\n", flush=True)
        # print(f"{rep_probs_lst=}\n", flush=True)

        mat_probs_p0_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_mat_syn, 0, 0, model_priors_dict["p_mat_nonsyn_rec"]
        )
        cp_probs_p0_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_cp_syn, 0, 0, model_priors_dict["p_cp_nonsyn_rec"]
        )
        lys_probs_p0_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_lys_syn, 0, 0, model_priors_dict["p_lys_nonsyn_rec"]
        )
        rep_probs_p0_lst = get_mut_type_probs_per_gene_lst_simp_model(
            p_rep_syn, 0, 0, model_priors_dict["p_rep_nonsyn_rec"]
        )

        print("got prot probs p0", flush=True)

        mutation_type_probs_per_gene_lst = [
            mat_probs_lst,
            cp_probs_lst,
            lys_probs_lst,
            rep_probs_lst,
        ]
        mutation_type_probs_per_gene_for_p0 = [
            mat_probs_p0_lst,
            cp_probs_p0_lst,
            lys_probs_p0_lst,
            rep_probs_p0_lst,
        ]

        freq_threshold = (
            1 / pop_size  # N=2*10**7
        )
        fitness_effects = np.array(
            [
                model_priors_dict["w_syn"],
                model_priors_dict["w_nonsyn_mat"],
                model_priors_dict["w_nonsyn_cp"],
                model_priors_dict["w_nonsyn_lys"],
                model_priors_dict["w_nonsyn_rep"],
                model_priors_dict["w_ada"],
            ]
        )  # MAP_w_syn, MAP_w_nonsyn

        passage = dict()

        passage[0] = get_mutations(
            model_priors_dict["mu"],
            mutation_type_probs_per_gene_for_p0,
            gene_probs,
            freq_threshold,
            simplified,
        )
        print("\nsimulated passage 0", flush=True)
        mutations = get_mutations(
            model_priors_dict["mu"],
            mutation_type_probs_per_gene_lst,
            gene_probs,
            freq_threshold,
            simplified,
        )
        print("\ngot mutations", flush=True)

        for i in range(passages):
            # print(f"\npassage={i} (from simulator)", flush=True)
            passage[i + 1] = simulate_next_passage_final(
                fitness_effects, passage[i], mutations, pop_size, simplified, batch_size
            )
            print(f"\ncreated passage {i + 1}!", flush=True)

        if simulate_sequence_sampling:
            print("\nreached simulate sequencing!!\n", flush=True)
            sequenced_passages = simulate_sequencing(
                passage,
                sample_size,
                seq_error_rate,
                mutation_type_probs_per_gene_lst,
                gene_probs,
                simplified,
            )
            data = wrangle_data_simplified(sequenced_passages)

        else:
            data = wrangle_data_simplified(passage)
        print("\n wrangled data \n", flush=True)

        if long_sumstat:
            # get_expanded_sumstat_simplified -- for a sumstat that distinguished between adaptive and non-adaptive muts
            # can also be get_sumstat_simplified (length 5 sumstat, total values only)
            data = get_full_geno_sumstat_all_passages(data)
            # [i for i in range(0, 11)]  # get_total_sumstat(data, [i for i in range(0, passages + 1)))
            print(f"\n got long for all passages: \n{len(data)=}", flush=True)
        else:
            data = get_expanded_sumstat_simplified(
                data, [i for i in range(0, passages + 1)]
            )
            print(f"\n got short sumstat for all passages: \n{len(data)=}", flush=True)

        del passage, mutations
        gc.collect()

    except Exception as e:
        gc.collect()
        raise Exception(f"Exception: '{e}' occurred with params: {sampled_params}")

    return data
