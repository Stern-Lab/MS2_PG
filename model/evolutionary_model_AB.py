import itertools
from itertools import combinations
from math import comb
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import poisson
import torch
import gc


def multinomial_sampling(freqs_dict, pop_size):
    """
    sample from multinomial distribution.

    arguments:
    freqs_dict -- {genotype:freq_of_genotype}

    return a new frequency dict - dict of freqs after sampling.
    """
    freqs_after_sample = (
        np.random.multinomial(pop_size, list(freqs_dict.values())) / pop_size
    )
    freqs_dict = {
        key: val for key, val in zip(freqs_dict.keys(), freqs_after_sample) if val > 0
    }
    return freqs_dict


def get_poisson_probs(mutation_rate, min_freq):
    """
    P(k,lambda=mutation_rate)
    get probabilities according to poisson distribution
    of k mutations given the mutation_rate - mutations along the whole genome.
    iterate (each time bigger k) until hit the min_freq.

    retrun a dict {k=num_of_muts:prob}
    """
    probs = dict()
    for num_of_muts in range(
        1000
    ):  # just a large number coz while loops are slow and awkward
        prob = poisson.pmf(num_of_muts, mutation_rate)  # pmf(k,lambda (mu))
        if prob <= min_freq:
            break
        probs[num_of_muts] = prob
    return probs


"""
These functions generate the full probability distribution over all possible ways a given number of mutations can be 
allocated across four genes, based on a mutation count distribution (Poisson) and known per-gene mutation probabilities.
"""


def compositions(n, k):
    """
    Generate all k-length tuples of non-negative integers that sum to n.
    Used to enumerate all possible distributions of n mutations across k genes.
    Yields:
    tuple: A k-length tuple summing to n.
    """
    if k == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for tail in compositions(n - i, k - 1):
                yield (i,) + tail


def multinomial_coeff(counts):
    """Computes multinomial coefficient for a given tuple of counts where n=sum(counts): n! / (k1! * k2! * ... * kr!)
    Args:
    counts (tuple of int): Counts in each category (e.g., mutations per gene).

    Returns:
    int: Multinomial coefficient
    """
    total = sum(counts)
    coeff = 1
    for c in counts:
        coeff *= comb(total, c)
        total -= c
    return coeff


def multinomial_expansion(n, mut_type_probs):
    """
    Expands a total count 'n' over categories with given probabilities.

    Args:
        n (int): Total number of events (mutations).
        mut_type_probs (list of float): Probabilities for each mutation type.

    Returns:
        dict: {composition (tuple): probability}
            Where composition is a tuple of counts per category.
    """
    result = {}
    k = len(mut_type_probs)
    for combo in compositions(n, k):
        coeff = multinomial_coeff(combo)
        prob = coeff * np.prod([p**c for p, c in zip(mut_type_probs, combo)])
        result[combo] = prob
    return result


# from total mutation number in genome to distribution over genes
def distribute_mutations_across_genes(mut_num_dict, gene_probs):
    """
    Computes distribution of mutation counts per gene using multinomial expansion.

    Args:
        mut_num_dict (dict): {total_mutations_in_genome: probability (poisson)}
        gene_probs (list of float): length-4 vector of per-gene mutation assignment probs

    Returns:
        dict: {(m1, m2, m3, m4): probability}
    """
    mut_distribution_across_genes = {}
    for total_muts, p_total in mut_num_dict.items():
        for combo, p_combo in multinomial_expansion(total_muts, gene_probs).items():
            mut_distribution_across_genes[combo] = p_total * p_combo
    return mut_distribution_across_genes


def gene_mutation_type_combinations(mut_count, mutation_type_probs, min_freq):
    """
    Wrapper for multinomial_expansion for 6 mutation types in one gene.

    Args:
        mut_count (int): number of mutations in one gene
        mutation_type_probs (list): probabilities to observe each type of mutation (length 6)

    Returns:
        dict: {composition (tuple): probability}
            Where composition is a tuple of counts per mutation type.
    """
    d = multinomial_expansion(mut_count, mutation_type_probs)
    return {GC: prob for GC, prob in d.items() if prob > min_freq}



#############################################################
# Simplified model:
# genotype [protein_nonsyn_rec, protein_nonsyn_dom] X4
#          + [total_nonsyn_adaptive, total_syn_adaptive, total_syn_nonadaptive]
# first: [nonsyn_rec, nonsyn_dom, nonsyn_ada, syn_ada, syn_nonada] per protein
# after: sum syn_ada, syn_nonada, nonsyn_ada across proteins and get the final genotype.
#############################################################

"""
probability structure:
- chain rule 
- p(ada|protein, syn/nonsyn) = p(ada|syn/nonsyn) -- ada is independent of the protein
- protein -- type (syn/non) -- adaptive or not -- if not adaptive: recessive/dominant

(mat can be replaced by each one of the proteins) 
p(mat, syn, !ada) = p(mat)*p(syn|mat)*p(!ada|mat, syn) = 
 = p(mat)*p(syn|mat)*p(!ada|syn) = p(mat)*p(syn|mat)*(1- p(ada|syn))
 = p_mat*p_mat_syn*(1-p_ada_syn)

p(mat, nonsyn, !ada, rec) = p(mat)*p(nonsyn|mat)*p(!ada|mat, nonsyn)*p(rec|mat,nonsyn,!ada) = 
 = p(mat)*(1-p(syn|mat))*(1-p(ada|nonsyn))*p(rec|mat,nonsyn,!ada) = 
 = p_mat*(1-p_mat_syn)*(1-p_ada_nonsyn)*p_mat_nonsyn_rec

p(mat, nonsyn, !ada, dom) = p(mat)*p(nonsyn|mat)*p(!ada|mat, nonsyn)*p(dom|mat, nonsyn, !ada) = 
 = p(mat)*(1-p(syn|mat)*(1-p(ada|nonsyn)*(1-p(rec|mat,nonsyn,!ada)) = 
 = p_mat*(1-p_mat_syn)*(1-p_ada_nonsyn)*(1-p_mat_nonsyn_rec)

p(mat, syn, ada) = p(mat)*p(syn|mat)*p(ada|mat, syn) = p(mat)*p(syn|mat)*p(ada|syn) = 
 = p_mat*p_mat_syn*p_ada_syn

p(mat, nonsyn, ada) = p(mat)*p(nonsyn|mat)*p(ada|mat, nonsyn) = p(mat)*(1-p(syn|mat))*p(ada|nonsyn) = 
 = p_mat*(1-p_mat_syn)*p_ada_nonsyn
"""


def get_mut_type_probs_per_gene_lst_simp_model(
    p_protein_syn, p_ada_syn, p_ada_nonsyn, p_protein_nonsyn_rec
):
    """
    gets the probability parameters and returns the probability to see each type of mutation
    Args:
        p_protein (float): p(protein)
        p_protein_syn (float): p(syn|protein)
        p_ada_syn (float): p(ada|syn)
        p_ada_nonsyn (float): p(ada|nonsyn)
        p_protein_nonsyn_rec (float): p(rec|protein,!ada, nonsyn)
    Returns:
        list: [p(protein, nonsyn, !ada, rec), p(protein, nonsyn, !ada, dom), p(protein, nonsyn, ada),
              p(protein, syn, ada),  p(protein, syn, !ada)]
    """

    P_nsr = (1 - p_protein_syn) * (1 - p_ada_nonsyn) * p_protein_nonsyn_rec
    P_nsd = (1 - p_protein_syn) * (1 - p_ada_nonsyn) * (1 - p_protein_nonsyn_rec)
    P_nsb = (1 - p_protein_syn) * p_ada_nonsyn
    P_sb = p_protein_syn * p_ada_syn
    P_s = p_protein_syn * (1 - p_ada_syn)
    return [P_nsr, P_nsd, P_nsb, P_sb, P_s]




def collapse_into_simplified_geno_structure_fast(full_genotype_dict):
    df = pd.DataFrame(list(full_genotype_dict.keys()))
    df["prob"] = list(full_genotype_dict.values())

    nonsyn_cols = [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]
    syn_ada_cols = [3, 8, 13, 18]
    syn_nonada_cols = [4, 9, 14, 19]

    simplified = pd.DataFrame()
    for i, col_idx in enumerate(nonsyn_cols):
        simplified[i] = df[col_idx]

    simplified["tot_syn_ada"] = df[syn_ada_cols].sum(axis=1)
    simplified["tot_syn_nonada"] = df[syn_nonada_cols].sum(axis=1)
    simplified["prob"] = df["prob"]

    final_df = simplified.groupby(
        list(range(len(nonsyn_cols))) + ["tot_syn_ada", "tot_syn_nonada"]
    ).sum()

    return final_df["prob"].to_dict()




def gather_muts_by_fitness_simplified_by_gene(simplified_genotypes):
    """
    create a sub genotype matrix with columns for the dominant mutations (recessive mutations have fitness 1 and
    therefore we exclude them) and a column for the sum of beneficial (syn+nonsyn) mutations.
    the output of this function is used for the fitness calculation
    (w prameters are raised to the power of mutation numbers - rows of the matrix)
    genotype sturcture:
    [mat_ns_rec, mat_ns_dom, mat_ns_ada, cp_ns_rec, cp_ns_dom,cp_ns_ada, lys_ns_rec, lys_ns_dom, lys_ns_ada, rep_ns_rec,
    rep_ns_dom, rep_ns_ada, tot_syn_ada, tot_syn_nonada]

    arguments:
    genotyps -- 2D array of shape (N,tup_size=11) of all genotypes

    return a matrix of partial genotypes - (K_sd | Kmat_nsd | Kcp_nsd| Klys_nsd | Krep_nsd | sum(K_sb, K_nsb))
    """
    syn_nonada = simplified_genotypes[:, 13].reshape(-1, 1)
    mat_nonsyn_dom = simplified_genotypes[:, 1].reshape(-1, 1)
    cp_nonsyn_dom = simplified_genotypes[:, 4].reshape(-1, 1)
    lys_nonsyn_dom = simplified_genotypes[:, 7].reshape(-1, 1)
    rep_nonsyn_dom = simplified_genotypes[:, 10].reshape(-1, 1)
    all_beneficial_muts = np.sum(
        simplified_genotypes[:, [2, 5, 8, 11, 12]], axis=1
    ).reshape(-1, 1)  # nonsyn per gene + total syn
    gathered_df = np.concatenate(
        [
            syn_nonada,
            mat_nonsyn_dom,
            cp_nonsyn_dom,
            lys_nonsyn_dom,
            rep_nonsyn_dom,
            all_beneficial_muts,
        ],
        axis=1,
    )
    return gathered_df



def genotype_probabilities_with_audit(
    gene_mutation_distribution, mutation_type_probs_per_gene, min_freq
):
        """
    Compute the probability distribution over full mutation-type genotype vectors.
    For each possible allocation of mutations across genes, this function enumerates
    all possible assignments of those mutations into mutation classes within each gene
    and combines them into full genotype vectors. Genotypes whose joint probability is
    below `min_freq` are discarded to keep the state space computationally tractable.

    Args:
        gene_mutation_distribution (dict):
            Keys are 4-tuples representing the number of mutations in each gene.
            Values are the probability of that total mutation distribution.
            (output of expand_mutation_combinations)

        mutation_type_probs_per_gene : list of list of float
            For each gene, a list of probabilities for the possible mutation classes
            used by the model.
    
        min_freq : float
            Minimum probability threshold. Mutation-type combinations or full genotype
            vectors with probability below this value are excluded.]

    Returns:
        dict:
            Keys are 20-length tuples: (5 mutation types × 4 genes)
            Values are the total probabilities of observing each genotype.
    """
    genotype_dict = {}
    total_leaked_prob = 0.0

    for gene_counts, p_gene_counts in gene_mutation_distribution.items():
        gene_type_options = []
        for i, mut_count in enumerate(gene_counts):
            combos = gene_mutation_type_combinations(
                mut_count, mutation_type_probs_per_gene[i], min_freq
            )
            if not combos:
                # If we skip this, we lose the entire p_gene_counts mass
                total_leaked_prob += p_gene_counts
                break
            gene_type_options.append(list(combos.items()))
        else:
            for combination in itertools.product(*gene_type_options):
                joint_prob = p_gene_counts * np.prod([c[1] for c in combination])
                if joint_prob > min_freq:
                    genotype_vector = sum([c[0] for c in combination], ())
                    genotype_dict[genotype_vector] = (
                        genotype_dict.get(genotype_vector, 0) + joint_prob
                    )
                else:
                    total_leaked_prob += joint_prob

    # print(f"Audit: Total probability captured: {sum(genotype_dict.values()):.6f}", flush=True)
    # print(f"Audit: Total probability leaked: {total_leaked_prob:.6f}", flush=True)
    return genotype_dict


def get_mutations(
    mutation_rate, mutation_type_probs_per_gene, gene_probs, min_freq, simplified
):
    """
    creates dictionary of genotype:freq.
    first uses mutation rate to get {total_num_of_muts:pois_porb}, then creates dict of all possible allocations of
    total number of mutations to the 4 genes. then allocates the number of mutations in each gene to the possible types
    of mutations.
    Args:
        mutation_rate (float): sampled mutation rate
        mutation_type_probs_per_gene (list): list of joint probabilities of each type of mutation
        gene_probs (list): list of probability of each protein
        min_freq (float): genotype frequency threshold
        simplified (bool): if true collapse the genotype into the simplified genotype (no syn_rec param)
    Returns:
        mutations (dict): {genotype: freq)
    """
    mut_poisson_prob = get_poisson_probs(
        mutation_rate, min_freq
    )  # {mut_num:prob_of_mut_num}
    # print(f"{mut_poisson_prob=}")
    distribution_over_genes = distribute_mutations_across_genes(
        mut_poisson_prob, gene_probs
    )  # {(k1,k2,k3,k4):prob}
    mutations = genotype_probabilities_with_audit(
        distribution_over_genes, mutation_type_probs_per_gene, min_freq
    )
    if simplified:
        mutations = collapse_into_simplified_geno_structure_fast(mutations)
    gc.collect()
    return mutations


def gather_muts_by_fitness_new(genotypes):
    """
    create a sub genotype matrix with columns for the dominant mutations (recessive mutations have fitness 1 and
    therefore we exclude them) and a column for the sum of beneficial (syn+nonsyn) mutations.
    the output of this function is used for the fitness calculation
    (w prameters are raised to the power of mutation numbers - rows of the matrix)

    arguments:
    genotyps -- 2D array of shape (N,tup_size=24) of all genotypes

    return a matrix of partial genotypes - (K_sd | K_nsd | sum(K_sb, K_nsb))
    """
    syn_dominant = np.sum(genotypes[:, [i for i in range(1, 24, 6)]], axis=1).reshape(
        -1, 1
    )
    nonsyn_dominant = np.sum(
        genotypes[:, [i for i in range(3, 24, 6)]], axis=1
    ).reshape(-1, 1)
    all_beneficial_muts = np.sum(
        genotypes[:, [4, 5, 10, 11, 16, 17, 22, 23]], axis=1
    ).reshape(-1, 1)  # Ksb, Knsb
    return np.concatenate([syn_dominant, nonsyn_dominant, all_beneficial_muts], axis=1)




def get_raw_fitness(fitness_effects, genotypes, simplified=True):
    """Calculates non-normalized fitness for a matrix of genotypes."""
    if simplified:
        mbf = gather_muts_by_fitness_simplified_by_gene(genotypes)
    else:
        mbf = gather_muts_by_fitness_new(genotypes)

    # Calculate w1^k1 * w2^k2 * w3^k3
    fitness = np.product(np.power(fitness_effects, mbf), axis=1)
    return fitness





def normalize_freqs_dict(freqs_dict, pop_size):
    """
    normalize frequencies

    arguments:
    freqs_dict -- dict of genotypes and their frequencies {genotype:freq}
    """
    min_freq = 10 * pop_size if pop_size < 100000 else pop_size
    freqs_dict = {
        key: val for key, val in freqs_dict.items() if val > (1 / min_freq)
    }  # to prevent occasional bugs
    freqs_sum = sum(freqs_dict.values())
    freqs_dict = {key: val / freqs_sum for key, val in freqs_dict.items()}
    return freqs_dict



def simulate_next_passage_final(
    fitness_effects, passage, mutations, pop_size, simplified, chunk_size=500
):
    """
    Complete passage simulation: Mutation -> Selection -> Chunked Aggregation -> Drift.
    """
    tuple_size = len(list(passage.keys())[0])

    # Existing genotypes in the population
    genotypes = np.array(list(passage.keys()), dtype=np.uint16)  # (N, tuple_size)
    genotypes_freqs = np.array(list(passage.values()), dtype=np.float32)  # (N,)

    # Possible mutation combinations for this passage
    mutation_keys = np.array(list(mutations.keys()), dtype=np.uint16)  # (M, tuple_size)
    mutation_freqs = np.array(list(mutations.values()), dtype=np.float32)  # (M,)

    all_chunks = []

    for i in range(0, len(genotypes), chunk_size):
        g_chunk = genotypes[i : i + chunk_size]
        f_chunk = genotypes_freqs[i : i + chunk_size]

        # Broadcasting: (C, 1, 24) + (M, 24) -> (C*M, 24)
        new_genos = (g_chunk[:, np.newaxis, :] + mutation_keys).reshape(-1, tuple_size)

        # Frequency calculation: (C, 1) * (M,) -> (C*M,)
        new_freqs = (f_chunk[:, np.newaxis] * mutation_freqs).reshape(-1)

        # Apply raw fitness (Selection)
        fitness_raw = get_raw_fitness(fitness_effects, new_genos, simplified)
        new_freqs *= fitness_raw

        # Combine identical genotypes within this chunk
        df_chunk = pd.DataFrame(new_genos)
        df_chunk["f"] = new_freqs
        # Group by all genotype columns and sum frequencies
        collapsed = (
            df_chunk.groupby(list(range(tuple_size)), sort=False).sum().reset_index()
        )
        all_chunks.append(collapsed)

    # Combine all chunk-summaries into one master DataFrame
    full_df = pd.concat(all_chunks).groupby(list(range(tuple_size)), sort=False).sum()

    # Normalization
    total_mass = full_df["f"].sum()
    full_df["f"] /= total_mass

    # Convert to dictionary for the Drift/Sampling function
    freqs_dict = {tuple(k): v for k, v in zip(full_df.index, full_df["f"])}

    del genotypes, mutation_keys, all_chunks, full_df
    gc.collect()

    # Drift (Multinomial Sampling)
    # Prune extreme rare events and sample based on pop_size
    freqs_dict = normalize_freqs_dict(freqs_dict, pop_size)
    freqs_dict = multinomial_sampling(freqs_dict, pop_size)

    return freqs_dict



def wrangle_data_simplified(passage):
    """
    creates a dataframe containing all genotypes and frequencies in each passage.

    arguments:
        passage -- nested dict. inner dict - genotype:freq. outer dict - passage_number {passage_i:{genotype_j:freq_j,..},..}

    returns data frame that summarizes the genotypes and frequencies by mut type and passage. columns are:
        non-syn_recessive | non-syn_dominant | X 4 (per gene) + non-syn_beneficial | syn_beneficial | syn_non-beneficial
        passage_i | all_muts | protein_total_nonsyn_nonada | syn_total | non_syn_total
    """
    data = pd.DataFrame(passage)
    data["all_muts"] = [sum(x) for x in data.index]
    data = (
        data.reset_index()
        .rename(
            columns={
                "level_0": "mat_nonsyn_recessive",
                "level_1": "mat_nonsyn_dominant",
                "level_2": "mat_nonsyn_ada",
                "level_3": "cp_nonsyn_recessive",
                "level_4": "cp_nonsyn_dominant",
                "level_5": "cp_nonsyn_ada",
                "level_6": "lys_nonsyn_recessive",
                "level_7": "lys_nonsyn_dominant",
                "level_8": "lys_nonsyn_ada",
                "level_9": "rep_nonsyn_recessive",
                "level_10": "rep_nonsyn_dominant",
                "level_11": "rep_nonsyn_ada",
                "level_12": "total_syn_ada",
                "level_13": "total_syn_nonada",
            }
        )
        .fillna(0)
    )

    data["mat_total_nonsyn"] = (
        data["mat_nonsyn_recessive"]
        + data["mat_nonsyn_dominant"]
        + data["mat_nonsyn_ada"]
    )
    data["cp_total_nonsyn"] = (
        data["cp_nonsyn_recessive"] + data["cp_nonsyn_dominant"] + data["cp_nonsyn_ada"]
    )
    data["lys_total_nonsyn"] = (
        data["lys_nonsyn_recessive"]
        + data["lys_nonsyn_dominant"]
        + data["lys_nonsyn_ada"]
    )
    data["rep_total_nonsyn"] = (
        data["rep_nonsyn_recessive"]
        + data["rep_nonsyn_dominant"]
        + data["rep_nonsyn_ada"]
    )

    data["mat_nonsyn_nonada"] = (
        data["mat_nonsyn_recessive"] + data["mat_nonsyn_dominant"]
    )
    data["cp_nonsyn_nonada"] = data["cp_nonsyn_recessive"] + data["cp_nonsyn_dominant"]
    data["lys_nonsyn_nonada"] = (
        data["lys_nonsyn_recessive"] + data["lys_nonsyn_dominant"]
    )
    data["rep_nonsyn_nonada"] = (
        data["rep_nonsyn_recessive"] + data["rep_nonsyn_dominant"]
    )

    data["syn_total"] = data["total_syn_ada"] + data["total_syn_nonada"]
    data["nonsyn_total"] = (
        data["mat_total_nonsyn"]
        + data["cp_total_nonsyn"]
        + data["lys_total_nonsyn"]
        + data["rep_total_nonsyn"]
    )
    data["ada_total"] = (
        data["mat_nonsyn_ada"]
        + data["cp_nonsyn_ada"]
        + data["lys_nonsyn_ada"]
        + data["rep_nonsyn_ada"]
        + data["total_syn_ada"]
    )

    return data




def get_sumstat_simplified(df, passages_lst, tensor=True):
    """
    creates the SR summary statistic.

    arguments:
    df -- the output df from 'wrangle_data'
    passages_lst -- list of the sequenced passages [5,8,10] for MOI10
    tensor -- bool, determines weather to return a torch.Tenor or a numpy array

    retruns a vector of size 10*len(passages) with the sum of nonsyn_ada/nonada_num*freq per protein and total_syn*freq for each passage
    """
    ret = list()
    for passage in passages_lst:
        print(f"{passage=}\n")
        ret.append(
            sum(df["mat_total_nonsyn"] * df[passage])
        )  # sum of: total num of mat nonsyn muts * freq of this genotype in the passage
        ret.append(sum(df["cp_total_nonsyn"] * df[passage]))
        ret.append(sum(df["lys_total_nonsyn"] * df[passage]))
        ret.append(sum(df["rep_total_nonsyn"] * df[passage]))
        ret.append(sum(df["syn_total"] * df[passage]))
    if tensor:
        return torch.Tensor(ret)
    else:
        return np.array(ret)


def get_expanded_sumstat_simplified(df, passages_lst):
    """
    creates expanded summary statistic - that distinguishes between adaptive and nonadaptive mutations (length = 10):
        (mat_ns_nonada, mat_ns_ada, cp_ns_nonada, cp_ns_ada, lys_ns_nonada, lys_ns_ada, rep_ns_nonada, rep_ns_ada,
        total_syn_nonada, total_syn_ada)

    arguments:
    df -- the output df from 'wrangle_data'
    passages_lst -- list of the sequenced passages [5,8,10] for MOI10
    tensor -- bool, determines weather to return a torch.Tenor or a numpy array

    retruns a vector of size 5*len(passages) with the sum of nonsyn_num*freq per protein and total_syn*freq for each passage
    """
    ret = list()
    for passage in passages_lst:
        print(f"{passage=}\n")
        ret.append(
            sum(df["mat_nonsyn_nonada"] * df[passage])
        )  # sum of: total num of mat nonsyn muts * freq of this genotype in the passage
        ret.append(sum(df["mat_nonsyn_ada"] * df[passage]))
        ret.append(sum(df["cp_nonsyn_nonada"] * df[passage]))
        ret.append(sum(df["cp_nonsyn_ada"] * df[passage]))
        ret.append(sum(df["lys_nonsyn_nonada"] * df[passage]))
        ret.append(sum(df["lys_nonsyn_ada"] * df[passage]))
        ret.append(sum(df["rep_nonsyn_nonada"] * df[passage]))
        ret.append(sum(df["rep_nonsyn_ada"] * df[passage]))
        ret.append(sum(df["total_syn_nonada"] * df[passage]))
        ret.append(sum(df["total_syn_ada"] * df[passage]))

    return torch.Tensor(ret)



def get_full_geno_sumstat(df):
    binned_categories = [
        "mat_nonsyn_nonada",
        "mat_nonsyn_ada",
        "cp_nonsyn_nonada",
        "cp_nonsyn_ada",
        "lys_nonsyn_nonada",
        "lys_nonsyn_ada",
        "rep_nonsyn_nonada",
        "rep_nonsyn_ada",
        "total_syn_nonada",
        "total_syn_ada",
    ]

    # Apply Transformation: 0->0, 1-2->1, 3-4->2, 5-6->3, 7-8->4
    # Logic: (val + 1) // 2 for val > 0
    for col in binned_categories:
        new_col_name = f"{col}_binned"
        df[new_col_name] = df[col].apply(lambda x: (int(x) + 1) // 2 if x > 0 else 0)

    def generate_indexing(n_categories, max_total):
        # This generates all tuples (k1, k2, ..., kn) where sum(ki) <= max_total
        # Adding a slack variable (n + 1) handles the "<" constraint
        for combo in combinations(range(n_categories + max_total), n_categories):
            res = []
            prev = -1
            for val in combo:
                res.append(val - prev - 1)
                prev = val
            yield tuple(res)

    new_index = list(generate_indexing(10, 6 - 1))
    grouped = df.groupby(
        [
            "mat_nonsyn_nonada_binned",
            "mat_nonsyn_ada_binned",
            "cp_nonsyn_nonada_binned",
            "cp_nonsyn_ada_binned",
            "lys_nonsyn_nonada_binned",
            "lys_nonsyn_ada_binned",
            "rep_nonsyn_nonada_binned",
            "rep_nonsyn_ada_binned",
            "total_syn_nonada_binned",
            "total_syn_ada_binned",
        ]
    )[10].sum()
    return torch.Tensor(grouped.reindex(new_index).fillna(0).values.flatten())


def get_full_geno_sumstat_all_passages(df):
    binned_categories = [
        "mat_nonsyn_nonada",
        "mat_nonsyn_ada",
        "cp_nonsyn_nonada",
        "cp_nonsyn_ada",
        "lys_nonsyn_nonada",
        "lys_nonsyn_ada",
        "rep_nonsyn_nonada",
        "rep_nonsyn_ada",
        "total_syn_nonada",
        "total_syn_ada",
    ]

    binned_names = [f"{col}_binned" for col in binned_categories]

    for col in binned_categories:
        new_col_name = f"{col}_binned"
        if col in df.columns:
            df[new_col_name] = df[col].apply(
                lambda x: (int(x) + 1) // 2 if x > 0 else 0
            )

    # Reconstruct the static indexing (3003 entries)
    def generate_indexing(n_categories, max_total):
        for combo in combinations(range(n_categories + max_total), n_categories):
            res = []
            prev = -1
            for val in combo:
                res.append(val - prev - 1)
                prev = val
            yield tuple(res)

    # 3003 genotypes based on stars and bars: bin_sum <= 5
    new_index = list(generate_indexing(10, 5))

    passage_stats = []
    for p in range(1, 11):
        # p_col = int(p)  # Using string if column names are '1', '2', etc.

        if p not in df.columns:
            # If a passage column is missing, append 3003 zeros
            passage_stats.append(torch.zeros(len(new_index)))
            continue

        # Group by binned categories and sum the frequencies in the specific passage column
        # We use .sum() because multiple rows might fall into the same binned genotype
        grouped = df.groupby(binned_names)[p].sum()

        # Reindex to ensure the vector is exactly 3003 and in the correct combinatorial order
        p_vector = torch.Tensor(grouped.reindex(new_index).fillna(0).values.flatten())
        passage_stats.append(p_vector)

    return torch.cat(passage_stats)


def get_total_sumstat(df, passages_lst):
    return torch.cat(
        (get_expanded_sumstat_simplified(df, passages_lst), get_full_geno_sumstat(df))
    )
