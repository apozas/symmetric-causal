# Code for
# Symmetric observations without symmetric causal explanations
# arXiv:2502.XXXXX
#
# Authors: Alejandro Pozas-Kerstjens, Christian William
#
# Requires: mosek for solving the linear programs
#           numpy for array operations
#           scipy for sparse matrix operations
#           sympy for handling of the certificate
#
# Last modified: Feb, 2025

import mosek
import numpy as np

from scipy.sparse import vstack
from sympy import symbols, simplify
from utils import get_symmetry_constraints, get_nlpi_constraints_ineq, \
                  get_norm_constraint_ineq, \
                  get_merge_constraints, to_sparse_single, to_sparse_merge

n_parties = 18    # 3 original parties + inflation level 15
E1        = 0.1656
E2        = -1/3

eqs_single = [get_symmetry_constraints(parties)
              for parties in range(4, n_parties+1)]

eqs_norm = [get_norm_constraint_ineq(parties)
            for parties in range(4, n_parties+1)]

eqs_iden = [get_nlpi_constraints_ineq(parties, E1, E2)
            for parties in range(4, n_parties+1)]

eqs = [sum(it, start=[]) for it in zip(eqs_single, eqs_norm, eqs_iden)]

# Rewrite in sparse form for Mosek
cons_single, b, bsymb, probs_to_idx = to_sparse_single(eqs)

# Constraints corresponding to Eq. 6 in the paper
eqs_merge = [get_merge_constraints(parties)
             for parties in range(5, n_parties+1)]

cons_merge = to_sparse_merge(eqs_merge, probs_to_idx)

b = np.concatenate((np.zeros(cons_merge.shape[0]), b))
bsymb = np.concatenate((np.zeros(cons_merge.shape[0]), bsymb))

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
    # Equality constraints for the constraints between different distributions
    # (Eq. 6 in the paper), greater-or-equal constraints for the normalization
    # and identification constraints (Eq. 7 in the paper, converted to pairs of
    # inequality constraints lhs >= rhs and -lhs >= -rhs), also equality
    # constraints for the symmetry constraints (Eq. 4 in the paper)
    bkc=np.concatenate(
        (np.broadcast_to(mosek.boundkey.fx, cons_merge.shape[0]),
         [mosek.boundkey.lo if 's' in eq.keys() else mosek.boundkey.fx
          for eq in np.array(sum(eqs, start=[]))]
          )
        ),
    blc=b,
    buc=b,
    # Constraints that the probabilities are non-negative
    bkx=np.broadcast_to(mosek.boundkey.lo, n_eqs),
    blx=np.zeros(n_vars),
    # Ridiculously large number to make the point that it is irrelevant
    bux=100000*np.ones(n_vars)
    )

trmcode = task.optimize()
basic = mosek.soltype.bas
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

if solutionsta == mosek.solsta.prim_infeas_cer:
    print("Infeasible distribution found. Calculating and saving certificate.")
    cert = np.array(yy) @ bsymb

    # Write certificate in terms of E1 and E2
    E1, E2 = symbols('E1 E2')
    ps = cert.free_symbols
    subs = {}
    for p in ps:
        pname = str(p)
        if len(pname) == 2:
            subs[p] = 1/2*(1+(-1)**int(pname[1])*E1)
        else:
            subs[p] = 1/4*(1 + ((-1)**int(pname[1])+(-1)**int(pname[2]))*E1
                                + (-1)**(int(pname[1])+int(pname[2]))*E2)
    cert = simplify(cert.subs(subs))

    lines = []
    for term in cert.as_ordered_terms():
        lines.append([term.as_coeff_mul()[0]*term.as_coeff_mul()[1][0],
                        term.as_coeff_exponent(E1)[1],
                        term.as_coeff_exponent(E2)[1]])

    np.savetxt('results/inflation15_cert.txt', lines)
