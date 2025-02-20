## Code to accompany *[Symmetric observations without symmetric causal explanations](https://www.arxiv.org/abs/2502.XXXXX)*
#### Christian William, Patrick Remy, Jean-Daniel Bancal, Yu Cai, Nicolas Brunner, and Alejandro Pozas-Kerstjens

This is a repository containing the computational appendix of the article "*Symmetric observations without symmetric causal explanations*. Christian William, Patrick Remy, Jean-Daniel Bancal, Yu Cai, Nicolas Brunner, and Alejandro Pozas-Kerstjens. [arXiv:2502.XXXXX](https://www.arxiv.org/abs/2502.XXXXX)." It provides the codes for obtaining the results depicted in Figure 4 and Appendix C in the manuscript.

The code is written in Python.

Python libraries required:
- [inflation](https://www.github.com/ecboghiu/inflation) (and its requirements) for setting up and solving the inflation problems
- [matplotlib](https://matplotlib.org) for plots
- [mosek](https://www.mosek.com) for solving the linear programs
- [numpy](https://www.numpy.org) for math operations
- [scipy](https://scipy.org) for sparse matrix operations
- [sympy](https://www.sympy.org) for symbolic operations

Files:

  - [inflation11_range.py](https://github.com/apozas/symmetric-causal/blob/main/inflation11_range.py): Code for producing the orange region in Figure 4 in the manuscript. For each value of ![](https://latex.codecogs.com/svg.latex?E_1), it computes what is the highest value of ![](https://latex.codecogs.com/svg.latex?E_2) for which the corresponding distribution is identified to be incompatible with symmetric realizations in the triangle scenario via the 11th level of the hierarchy defined in the manuscript.

  - [inflation15_single.py](https://github.com/apozas/symmetric-causal/blob/main/inflation15_single.py): Code for producing the orange region in Figure 4 in the manuscript. For given values of ![](https://latex.codecogs.com/svg.latex?E_1) and ![](https://latex.codecogs.com/svg.latex?E_2), it determines whether the corresponding distribution is incompatible with symmetric realizations in the triangle scenario via the 15th level of the hierarchy defined in the manuscript. Upon finding an incompatible distribution, it saves the corresponding certificate of infeasibility.

  - [nsi.py](https://github.com/apozas/symmetric-causal/blob/main/nsi.py): Code for producing the blue region in Figure 4 in the manuscript. It reproduces the boundary of Fig. 2 in [Nat. Commun. 11, 2378 (2020)](https://doi.org/10.1038/s41467-020-16137-4) ([arXiv:1906.06495](https://arxiv.org/abs/1906.06495)) using the [inflation](https://www.github.com/ecboghiu/inflation) library.

  - [plot_results.py](https://github.com/apozas/symmetric-causal/blob/main/plot_results.py): Code to generate Figure 4 in the manuscript.

  - [utils.py](https://github.com/apozas/symmetric-causal/blob/main/utils.py): Helper functions to define the constraints of the linear programs and to process everything as sparse arrays.

  - [results](https://github.com/apozas/symmetric-causal/blob/main/results/): Folder where the results of the calculations are stored. In particular, it contains the certificate of infeasibility of Appendix C in machine-readable form. It is given as a matrix, where each row represents a monomial, and the columns represent, correspondingly, the prefactor, the power of ![](https://latex.codecogs.com/svg.latex?E_1), and the power of ![](https://latex.codecogs.com/svg.latex?E_2). Example code for reading it can be found in L30-L33 in [plot_results.py](https://github.com/apozas/symmetric-causal/blob/main/plot_results.py#L30).

If you would like to cite this work, please use the following format:

C. William, P. Remy, J.-D. Bancal, Y. Cai, N. Brunner, and A. Pozas-Kerstjens, _Symmetric observations without symmetric causal explanations_, arXiv:2502.XXXXX

```
@misc{william2025symmetric,
  author = {Christian William and Patrick Remy and Jean-Danial Bancal and Yu Cai and Nicolas Brunner and and Alejandro Pozas-Kerstjens},
  title = {Symmetric observations without symmetric causal explanations},
  archivePrefix = {arXiv},
  eprint = {2502.XXXXX},
  year = {2025}
}
```

