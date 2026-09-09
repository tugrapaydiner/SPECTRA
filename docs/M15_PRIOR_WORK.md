# M15 Prior-Work and Novelty Boundary

M15 does not claim novelty for the broad mechanisms below. These references were checked before final interpretation of the experiment.

- **Learned outcome verifiers / reranking:** Karl Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021, arXiv:2110.14168. The work trains verifiers to judge generated solutions and uses them to select among test-time candidates.
- **Process supervision / process reward models:** Hunter Lightman et al., *Let's Verify Step by Step*, 2023, arXiv:2305.20050. The work studies process supervision and releases step-level feedback for mathematical reasoning.
- **Proxy/reward-model overoptimization:** Leo Gao, John Schulman, Jacob Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023, PMLR 202:10835–10866. It directly studies optimization against an imperfect learned proxy and the resulting degradation in a separate gold objective.
- **Ensemble/conservative mitigation of reward-model overoptimization:** Thomas Coste, Usman Anwar, Robert Kirk, David Krueger, *Reward Model Ensembles Help Mitigate Overoptimization*, ICLR 2024. It studies ensemble-based conservative objectives as mitigations; this means SPECTRA's use of ensemble disagreement/LCB is not itself a novelty claim.
- **Vector-quantized latent representations:** Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu, *Neural Discrete Representation Learning*, 2017, arXiv:1711.00937. Nearest-code latent quantization and straight-through/commitment training are established VQ-VAE machinery.
- **Variance/covariance anti-collapse regularization:** Adrien Bardes, Jean Ponce, Yann LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, ICLR 2022. Variance/covariance regularizers are established anti-collapse objectives; applying related terms does not prove rank preservation in a recurrent latent trajectory.
- **Dynamical isometry:** Jeffrey Pennington, Samuel Schoenholz, Surya Ganguli, *Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice*, NeurIPS 2017. Dynamical isometry concerns the Jacobian singular-value distribution under specific architectural/initialization assumptions; a sampled norm penalty is not by itself a proof that SPECTRA satisfies it.

## Bounded M15 contribution

M15's contribution is system-specific rather than a broad algorithmic novelty claim:

> In a controlled SPECTRA latent-search pilot, a grounded verifier trained to predict *one-cycle structural improvement* remained meaningfully predictive of that target while its score was negatively related to *absolute Sudoku state quality*. Treating this target-specific probability as an absolute MCTS state value caused reproducible search-induced regressions. An independent absolute symbolic scorer nearly removed the regressions, a task-specific terminal-validity guard removed most of them, and FP32 search-state storage did not remove them.

This is evidence about the interaction of **target semantics × search optimization × recursive state distribution** in the measured SPECTRA setup. It is not a claim to have discovered Goodhart's law, reward hacking, verifier reranking, ensemble uncertainty, VQ, VICReg, or dynamical isometry.
