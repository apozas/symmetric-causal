# Code for
# Symmetric observations without symmetric causal explanations
# arXiv:2502.XXXXX
#
# Authors: Alejandro Pozas-Kerstjens, Christian William
#
# Requires: itertools
#           numpy for array operations
#           scipy for sparse matrix operations
#           sympy for extracting the certificate
#
# Last modified: Feb, 2025

from itertools import product, permutations
from scipy.sparse import dok_array
from sympy import Symbol

import numpy as np

################################################################################
# Constraints
################################################################################
def get_symmetry_constraints(n_parties):
    """This function returns all the symmetry constraints for a probability
    distrubution with n_parties parties. These are just the equality of all
    cyclic permutations of a given string of outcomes."""
    probs = [''.join(p) for p in product('01', repeat=n_parties)]

    used = ['0'*n_parties, '1'*n_parties]
    symm_eqs = []

    for p in probs:
        if p not in used:
            used.append(p)
            equivs = [p[i:] + p[:i] for i in range(len(p))]
            for equiv in equivs:
                if equiv not in used:
                    symm_eqs.append({p: 1, equiv: -1, 'c': 0})
                    used.append(equiv)
    return symm_eqs

def get_lpi_constraints(n_parties, E1, E2):
    """This function returns the constraints of the type of Eq. 4 in the
    manuscript for an arbitrary number of parties."""
    iden_eqs = []
    # With a single-body identifiable marginal
    p0 = (1 + E1) / 2
    p1 = (1 - E1) / 2
    probs = [''.join(p) for p in product('01', repeat=n_parties-3)]
    iden_eqs += [{'000' + p: 1-p0, '001' + p: 1-p0, '100' + p: 1-p0, '101' + p: 1-p0,
                                                    '010' + p: -p0,  '011' + p: -p0,
                                                    '110' + p: -p0,  '111' + p: -p0}
                for p in probs]
    iden_eqs += [{'010' + p: 1-p1, '011' + p: 1-p1, '110' + p: 1-p1, '111' + p: 1-p1,
                  '000' + p: -p1,  '001' + p: -p1,
                  '100' + p: -p1,  '101' + p: -p1}
                for p in probs]

    # With a two-body identifiable marginal
    p00 = (1 + 2*E1 + E2) / 4
    p01 = (1 - E2) / 4
    p10 = (1 - E2) / 4
    p11 = (1 - 2*E1 + E2) / 4
    probs = [''.join(p) for p in product('01', repeat=n_parties-4)]
    iden_eqs += [{'0000' + p: 1 - p00, '0001' + p: 1 - p00, '1000' + p: 1 - p00, '1001' + p: 1 - p00,
                                                            '0010' + p: -p00,    '0011' + p: -p00,
                  '0100' + p: -p00,    '0101' + p: -p00,    '0110' + p: -p00,    '0111' + p: -p00,
                                                            '1010' + p: -p00,    '1011' + p: -p00,
                  '1100' + p: -p00,    '1101' + p: -p00,    '1110' + p: -p00,    '1111' + p: -p00}
                for p in probs]
    iden_eqs += [{'0010' + p: 1 - p01, '0011' + p: 1 - p01, '1010' + p: 1 - p01, '1011' + p: 1 - p01,
                  '0000' + p: -p01,    '0001' + p: -p01,
                  '0100' + p: -p01,    '0101' + p: -p01,    '0110' + p: -p01,    '0111' + p: -p01,
                  '1000' + p: -p01,    '1001' + p: -p01,
                  '1100' + p: -p01,    '1101' + p: -p01,    '1110' + p: -p01,    '1111' + p: -p01}
                for p in probs]
    iden_eqs += [{'0100' + p: 1 - p10, '0101' + p: 1 - p10, '1100' + p: 1 - p10, '1101' + p: 1 - p10,
                  '0000' + p: -p10,    '0001' + p: -p10,    '0010' + p: -p10,    '0011' + p: -p10,
                                                            '0110' + p: -p10,    '0111' + p: -p10,
                  '1000' + p: -p10,    '1001' + p: -p10,    '1010' + p: -p10,    '1011' + p: -p10,
                                                            '1110' + p: -p10,    '1111' + p: -p10}
                for p in probs]
    iden_eqs += [{'0110' + p: 1 - p11, '0111' + p: 1 - p11, '1110' + p: 1 - p11, '1111' + p: 1 - p11,
                  '0000' + p: -p11,    '0001' + p: -p11,    '0010' + p: -p11,    '0011' + p: -p11,
                  '0100' + p: -p11,    '0101' + p: -p11,
                  '1000' + p: -p11,    '1001' + p: -p11,    '1010' + p: -p11,    '1011' + p: -p11,
                  '1100' + p: -p11,    '1101' + p: -p11}
                for p in probs]
    return iden_eqs

def get_norm_constraint(n_parties):
    """This function returns the normalization constraint. Namely, that the sum
    of all probabilities is 1."""
    return {**{''.join(p): 1 for p in product('01', repeat=n_parties)},
            **{'c': 1, 's': 1}}

def get_norm_constraint_ineq(n_parties):
    """This function returns the normalization constraint (that the sum of all
    probabilities is 1) in the form of two inequalities, sum >= 1 and
    -sum >= -1. This is necessary for a proper interpretation of certificates of
    infeasibility."""
    return [{**{''.join(p): 1 for p in product('01', repeat=n_parties)},
             **{'c': 1, 's': 1}},
            {**{''.join(p): -1 for p in product('01', repeat=n_parties)},
             **{'c': -1, 's': -1}}]


def get_merge_constraints(n_parties):
    """This function returns the merge constraints between distributions
    corresponding to different inflations. Namely, if we marginalize over the
    two last parties in an inflation, the distribution should be the same as
    that of marginalizing the last party in the inflation with one fewer party.
    This is Eq. 6 in the manuscript."""
    merge_eqs = []
    probs = [''.join(p) for p in product('01', repeat=n_parties-2)]
    for p in probs:
        merge_eqs.append({p + '00': 1, p + '01': 1, p + '10': 1, p + '11': 1,
                          p + '0': -1, p + '1': -1})
    return merge_eqs

def get_nlpi_constraints_ineq(n_parties, E1, E2):
    """This function returns the constraints of the type of Eq. 7 in the
    manuscript for an arbitrary number of parties. They are given in the form of
    two inequalities, lhs >= rhs and -lhs >= -rhs. This is necessary for a
    proper interpretation of certificates of infeasibility."""
    # First we need to know how many 1-body and 2-body marginals fit in the
    # network. This is a Diophantine equation (2x + 3y = n_parties), but we
    # probably are better off just trying all combinations
    oneb = 2*np.arange(n_parties)
    twob = 3*np.arange(n_parties)
    oneb, twob = np.meshgrid(oneb, twob)
    # We create lists of tuples with the valid combinations. The first element
    # is the number of 2-body marginals, and the second the number of 1-body
    # marginals
    valid1 = zip(*np.where(oneb + twob == n_parties))
    # We can always marginalize one more
    valid2 = zip(*np.where(oneb + twob == n_parties-1))

    p00 = (1 + 2*E1 + E2) / 4
    p01 = (1 - E2) / 4
    p10 = (1 - E2) / 4
    p11 = (1 - 2*E1 + E2) / 4
    s00 = Symbol('p00')
    s01 = Symbol('p01')
    s10 = Symbol('p10')
    s11 = Symbol('p11')
    twoblock = [['000', '100', p00, s00],
                ['001', '101', p01, s01],
                ['010', '110', p10, s10],
                ['011', '111', p11, s11]]
    p0 = (1 + E1) / 2
    p1 = (1 - E1) / 2
    s0 = Symbol('p0')
    s1 = Symbol('p1')
    oneblock = [['00', '10', p0, s0],
                ['01', '11', p1, s1]]
    blocks = [twoblock, oneblock]

    eqs = []
    for n_each in valid1:
        allprobs = np.unique(list(permutations([0]*n_each[0] + [1]*n_each[1])),
                             axis=0).tolist()
        if len(allprobs) > 1:
            realprobs = [allprobs[0]]
            for p in allprobs[1:]:
                is_cyclic = 0
                for rp in realprobs:
                    is_cyclic += p in [rp[i:] + rp[:i] for i in range(len(rp))]
                if not is_cyclic:
                    realprobs.append(p)
            allprobs = realprobs

        for prob in allprobs:
            eqs_components = product(*[blocks[i] for i in prob])
            for comp in eqs_components:
                c = np.prod([comp[i][2] for i in range(len(comp))])
                s = np.prod([comp[i][3] for i in range(len(comp))])
                coeff_list = product(*[comp[i][:2] for i in range(len(comp))])
                ineq1 = {**{''.join(coeffs): 1 for coeffs in coeff_list},
                         **{'c': c, 's': s}}
                ineq2 = {**{''.join(coeffs): -1 for coeffs in coeff_list},
                         **{'c': -c, 's': -s}}
                eqs.append(ineq1)
                eqs.append(ineq2)
    
    # Next, we consider all assignments for one fewer party than the maximum.
    # Thus, we have to add an additional party to marginalize over. We can do
    # the same as before but also adding a marginalized party in any of the
    # positions
    blocks += [[['0', '1', 1, 1]]]
    for n_each in valid2:
        allprobs = np.unique(list(permutations([0]*n_each[0]
                                               + [1]*n_each[1]
                                               + [2])),
                             axis=0).tolist()
        if len(allprobs) > 1:
            realprobs = [allprobs[0]]
            for p in allprobs[1:]:
                is_cyclic = 0
                for rp in realprobs:
                    is_cyclic += p in [rp[i:] + rp[:i] for i in range(len(rp))]
                if not is_cyclic:
                    realprobs.append(p)
            allprobs = realprobs
        for prob in allprobs:
            eqs_components = product(*[blocks[i] for i in prob])
            for comp in eqs_components:
                c = np.prod([comp[i][2] for i in range(len(comp))])
                s = np.prod([comp[i][3] for i in range(len(comp))])
                coeff_list = product(*[comp[i][:2] for i in range(len(comp))])
                ineq1 = {**{''.join(coeffs): 1 for coeffs in coeff_list},
                         **{'c': c, 's': s}}
                ineq2 = {**{''.join(coeffs): -1 for coeffs in coeff_list},
                         **{'c': -c, 's': -s}}
                eqs.append(ineq1)
                eqs.append(ineq2)
    return eqs

################################################################################
# Transformations to sparse arrays for efficient loading to Mosek
################################################################################
def to_sparse_single(constraints):
    """Function for the constraints that apply to only one inflation, i.e.,
    normalization, symmetry (Eq. 4 in the paper), and identification (Eq. 5 in
    the paper) constraints."""
    parties = np.unique([len(list(c[0])[0]) for c in constraints])
    n_vars = sum(2**ii for ii in parties)
    n_eqs = sum(len(c) for c in constraints)
    cons_mat = dok_array((n_eqs, n_vars))
    b = np.zeros((n_eqs))
    bsymb = np.zeros((n_eqs), dtype=object)
    eq_buffer = 0
    party_buffer = 0
    prob_to_idx = {}
    for p, cons in zip(parties, constraints):
        for i, eq in enumerate(cons):
            for k, v in eq.items():
                if k == 'c':
                    b[i+eq_buffer] = v
                    continue
                if k == 's':
                    bsymb[i+eq_buffer] = v
                    continue
                try:
                    idx = prob_to_idx[k]
                except KeyError:
                    idx = int(k, 2) + party_buffer
                    prob_to_idx[k] = idx
                cons_mat[i+eq_buffer, idx] = v
        eq_buffer += len(cons)
        party_buffer += 2**p
    return cons_mat.tocsc(), b, bsymb, prob_to_idx

def to_sparse_merge(constraints, prob_to_idx):
    """Function for the constraints that relate two inflation distributions,
    i.e., Eq. 6 in the paper."""
    n_vars = prob_to_idx[max(prob_to_idx,
                             key=prob_to_idx.get)] + 1  # Counting starts with 0
    n_eqs = sum(len(c) for c in constraints)
    cons_mat = dok_array((n_eqs, n_vars))
    eq_buffer = 0
    for cons in constraints:
        for i, eq in enumerate(cons):
            for k, v in eq.items():
                cons_mat[i+eq_buffer, prob_to_idx[k]] = v
        eq_buffer += len(cons)
    return cons_mat.tocsc()
