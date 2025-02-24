# Code for
# Symmetric observations without symmetric causal explanations
# arXiv:2502.14950
#
# Authors: Alejandro Pozas-Kerstjens
#
# Requires: matplotlib for plots
#           numpy for array operations
#           sympy for plotting the certificate
#
# Last modified: Feb, 2025

import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, real_roots

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18

# Orange area: region rejected by inflation level 11
[E1s_11, E2s_11] = np.loadtxt('results/inflation11_range.txt')
plt.fill_between(E1s_11, [-1/3]*len(E1s_11), E2s_11,
                 color='orange', label='Rejected by $n=11$')

# Dark green area: region rejected by the certificate for inflation level 15
cert_data = np.loadtxt('results/inflation15_cert.txt')

# From machine-readable form to sympy expression
E1, E2 = symbols('E1 E2')
cert = 0
for line in cert_data:
    cert += line[0] * E1**int(line[1]) * E2**int(line[2])

E1s_15 = E1s_11
E2s_15 = []
for e1 in E1s_15:
    roots = [r.n(5) for r in real_roots(cert.subs({E1: e1}))]
    roots = [r for r in roots if -1/3 - 1e-4 < r < 0]
    if len(roots) == 0:
        E2s_15.append(-1/3)
    else:
        E2s_15.append(float(roots[0]))

plt.fill_between(E1s_15, [-0.4]*len(E1s_15), E2s_15,
                 color='olive', label='Rejected by $n=15$ cert.')

# Yellow area: Boundary of the region for which a local model is known. Private
# emails with the authors of Nat. Commun. 11, 2378 (2020) [arXiv:1906.06495]
E1LM = np.arange(0.175, 0.5125, 0.0125)
E2LM = [-3.33333333e-01, -3.19659055e-01, -3.05925365e-01, -2.92205698e-01,
        -2.78747859e-01, -2.65342926e-01, -2.52950581e-01, -2.40673388e-01,
        -2.29472786e-01, -2.18933024e-01, -2.08318774e-01, -1.96822165e-01,
        -1.87247622e-01, -1.76528844e-01, -1.62336664e-01, -1.48248986e-01,
        -1.34014435e-01, -1.19780379e-01, -1.05666008e-01, -9.17544873e-02,
        -7.80700339e-02, -6.45965142e-02, -5.13208155e-02, -3.82320057e-02,
        -2.53207881e-02, -1.25791301e-02, -6.03319070e-08]

plt.fill_between(E1LM, [-0.4]*len(E1LM), E2LM,
                 color='#ffdd69', alpha=0.75, edgecolor='#ffe68f')

# Blue area: Region of distributions that do not admit any realization in the
# triangle. Reproduces the boundary of Nat. Commun. 11, 2378 (2020)
# [arXiv:1906.06495], but we use Quantum 7, 996 (2023) [arXiv:2211.04483]
# (see nsi.py)
[E1s_NSI, E2s_NSI] = np.loadtxt('results/nsi.txt')
plt.fill_between(E1s_NSI, [-0.4]*len(E1s_NSI), E2s_NSI,
                 color='#69ccd0', label='Rejected by NSI')

# Gray area: values of E1 and E2 for which some probabilities are negative
plt.fill_between(E1s_NSI, [-0.4]*len(E1s_NSI), [-1/3]*len(E1s_NSI),
                 color='#bfbfbf', label='Rejected by positivity')
plt.fill_between(E1s_NSI, [-0.4]*len(E1s_NSI), 2*E1s_NSI-1, color='#bfbfbf')

# Special points: the distribution for smallest E1 and E2=-1/3 that we can
# identify as incompatible using inflation level 15, and the distribution that
# with largest E1 that we know it admits a triangle-local model
plt.plot([0.158], [-1/3], 'ro', fillstyle='none', markersize=3)
plt.plot([0.1753], [-1/3], 'b^', fillstyle='none', markersize=3)

plt.ylim(-0.35, 0.02)
plt.xlim(0.15, 0.5)
plt.xlabel(r'$E_1$')
plt.ylabel(r'$E_2$')
plt.legend(loc='upper left', fontsize=16)

plt.savefig('results.pdf', bbox_inches='tight')
