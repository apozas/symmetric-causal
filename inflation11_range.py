# Code for
# Symmetric observations without symmetric causal explanations
# arXiv:2502.14950
#
# Authors: Alejandro Pozas-Kerstjens, Christian William
#
# Requires: mosek for solving the linear programs
#           numpy for array operations
#           scipy for sparse matrix operations
#
# Last modified: Feb, 2025

import mosek
import numpy as np

from scipy.sparse import vstack
from utils import get_symmetry_constraints, get_lpi_constraints, \
                  get_norm_constraint, get_merge_constraints, to_sparse_single,\
                  to_sparse_merge

from tqdm import tqdm

n_parties = 14    # 3 original parties + inflation level 11
E1s = np.linspace(0.15, 0.5, 101)
E2s = []

for E1 in tqdm(E1s):
    E2l = -0.4
    E2r = 0
    while E2r-E2l > 1e-5:
        E2 = (E2l + E2r) / 2

        # Constraints corresponding to Eqs. 4 and 5 in the paper
        eqs_single = [get_symmetry_constraints(parties)
                      + get_lpi_constraints(parties, E1, E2)
                        for parties in range(4, n_parties+1)]

        # Normalization constraints
        eqs_norm = [[get_norm_constraint(parties)]
                    for parties in range(4, n_parties+1)]

        eqs = [sum(it, start=[]) for it in zip(eqs_single, eqs_norm)]

        # Rewrite in sparse form for Mosek
        cons_single, b, _, probs_to_idx = to_sparse_single(eqs)

        # Constraints corresponding to Eq. 6 in the paper
        eqs_merge  = [get_merge_constraints(parties)
                      for parties in range(5, n_parties+1)]
        cons_merge = to_sparse_merge(eqs_merge, probs_to_idx)

        b = np.concatenate((np.zeros(cons_merge.shape[0]), b))

        cons_mat = vstack((cons_merge, cons_single)).tocsc()

        n_eqs, n_vars = cons_mat.shape

        # Feed to Mosek
        env = mosek.Env()
        task = mosek.Task(env)
        task.inputdata(
            maxnumcon=n_eqs,
            maxnumvar=n_vars,
            c=np.zeros(n_vars),
            cfix=0,
            aptrb=cons_mat.indptr[:-1],
            aptre=cons_mat.indptr[1:],
            asub=cons_mat.indices,
            aval=cons_mat.data,
            # All constraints are input as equality constraints
            bkc=np.broadcast_to(mosek.boundkey.fx, n_eqs),
            blc=b,
            buc=b,
            # Constraints that the probabilities are >= 0 and <= 1
            bkx=np.broadcast_to(mosek.boundkey.ra, n_eqs),
            blx=np.zeros(n_vars),
            bux=np.ones(n_vars)
            )

        trmcode = task.optimize()
        basic   = mosek.soltype.bas
        (problemsta,
            solutionsta,
            skc,
            skx,
            skn,
            xc,
            xx,
            yy,
            slc,
            suc,
            slx,
            sux,
            snx) = task.getsolution(basic)

        if problemsta == mosek.solsta.optimal:
            E2r = E2
        else:
            E2l = E2
    E2s.append(E2)

np.savetxt('results/inflation11_range.txt', [E1s, E2s])