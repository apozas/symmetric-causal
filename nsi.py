# Code for
# Symmetric observations without symmetric causal explanations
# arXiv:2502.XXXXX
#
# Authors: Alejandro Pozas-Kerstjens
#
# Requires: inflation for setting up and solving the inflation problem
#           numpy for defining the probability distribution
#
# Last modified: Feb, 2025

import numpy as np
from inflation import InflationProblem, InflationLP

ip = InflationProblem(dag={"gamma": ["A", "B"],
                           "beta":  ["A", "C"],
                           "alpha": ["B", "C"]
                           },
                      outcomes_per_party=[2, 2, 2],
                      inflation_level_per_source=[2, 2, 2]
                      )

lp = InflationLP(ip)
# Below we prevent the program expecting to use E3 internally
del lp.knowable_atoms[-1]

prob = np.zeros((2, 2, 2, 1, 1, 1), dtype=object)

E1s = np.linspace(0.15, 0.5, 101)
E2s = []
for E1 in E1s:
    E2l = -0.4
    E2r = 0
    while E2r-E2l > 1e-5:
        E2 = (E2l + E2r) / 2

        for a,b,c in np.ndindex(2, 2, 2):
            prob[a, b, c, 0, 0, 0] = \
                1/8 * (1 + ((-1)**(a)+(-1)**(b)+(-1)**(c))*E1
                         + ((-1)**(a+b)+(-1)**(a+c)+(-1)**(b+c))*E2)

        lp.set_distribution(prob, use_lpi_constraints=True)
        lp.solve(feas_as_optim=True)

        if lp.primal_objective > -1e-10:
            E2r = E2
        else:
            E2l = E2
    E2s.append(E2)

np.savetxt('results/nsi.txt', [E1s, E2s])
