# Known probabilities:

syn_probs_by_gene = {
    "p_mat_syn": 0.23970671178793007,  # P(syn|mat)  0.23970671178793004, 0.23970671178793007
    "p_cp_syn": 0.21140350877192982,  # P(syn|cp)  0.20441051738761662, 0.23748939779474132
    "p_lys_syn": 0.035381750465549346,  # P(syn|lys)  0.03216374269005848, 0.24269005847953215
    "p_rep_syn": 0.22259136212624583,  # P(syn|rep)  0.2187627187627188, 0.23809523809523808
}

gene_probs = {
    "p_mat": 0.343505,  # probability of landing in region of mat = len(mat)/len(coding genome)
    "p_cp": 0.1142110,
    "p_lys": 0.066260,
    "p_rep": 0.476024,
}
passages = 10  # 10
pop_size_A = 2 * 10**7
pop_size_B = 10**11

modelA_priors = {
    "mu": (0.1, 0.3),  # mutation rate  -- I narrowed the prior even more
    "w_syn": (0.85, 1),  # avg syn mutation fitness
    "w_nonsyn_mat": (0.6, 0.85),  # avg nonsyn mutation fitness
    "w_nonsyn_cp": (0.6, 0.85),  # avg nonsyn mutation fitness
    "w_nonsyn_lys": (0.6, 0.85),  # avg nonsyn mutation fitness
    "w_nonsyn_rep": (0.6, 0.85),  # avg nonsyn mutation fitness
    "w_ada": (1, 3),  # mat avg adaptive mutation fitness
    "p_ada_syn": (0, 0.01),  # P(adaptive | syn)
    "p_ada_ns_mat": (0, 0.1),  # P(adaptive | nonsyn, mat)
    "p_ada_ns_cp": (0, 0.1),  # P(adaptive | nonsyn, cp)
    "p_ada_ns_lys": (0, 0.1),  # P(adaptive | nonsyn, lys)
    "p_ada_ns_rep": (0, 0.1),  # P(adaptive | nonsyn, rep)
}

modelB_priors = {
    "mu": (0.2, 0.3),  # mutation rate  -- I narrowed the prior even more
    "w_syn": (0.9, 1),  # avg syn mutation fitness
    "w_ada": (1.25, 2),  # mat avg adaptive mutation fitness
    "p_ada_syn": (0, 0.01),  # P(adaptive | syn)
    "p_ada_ns_mat": (0, 0.1),  # P(adaptive | nonsyn, mat)
    "p_ada_ns_cp": (0, 0.1),  # P(adaptive | nonsyn, cp)
    "p_ada_ns_lys": (0, 0.1),  # P(adaptive | nonsyn, lys)
    "p_ada_ns_rep": (0, 0.1),  # P(adaptive | nonsyn, rep)
    "p_mat_nonsyn_rec": (0, 1),  # P(rec | nonsyn, mat)
    "p_cp_nonsyn_rec": (0, 1),  # P(rec | nonsyn, cp)
    "p_lys_nonsyn_rec": (0, 1),  # P(rec | nonsyn, lys)
    "p_rep_nonsyn_rec": (0, 1),  # P(rec | nonsyn, rep)
}
