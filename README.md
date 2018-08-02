# Restricted-Boltzmann-Machine-on-Spin-systems

This repository uses the Restricted Boltzmann Machine (RBM) to study two spin systems: Ising model and XY model. For Ising model, the spins take value within 1 or -1, thus regular binary RBM (rbm_ising.py) is good enough. For the XY model, the spins can take any value from 0 to 2 pi. In this case, both Gaussian-Bernoulli (rbm_xy.py) and Gaussian-Gaussian RBM (rbm_xy_gaussian.py) are applied. The training data is sampled using MCMC for both Ising (mc_ising.py) and XY (mc_xy.py) systems. "sample_rbm_xy.py" samples spin configurations from the RBM.

A blog post will be available shortly.
