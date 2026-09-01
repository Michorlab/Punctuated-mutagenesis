This folder contains minimal simulation code underlying the figures. The files expose reusable functions only; plotting, file output, and cluster submission scripts are not included.

- sims_2.py: Wright-Fisher simulations for Fig. 2. Main function: simulate_wright_fisher_evolution_with_rates.
- sims_S2_1.py: branching-process simulations for Fig. S2_1. Main function: simulate_branching_evolution_with_rates.
- sweep_valleys_S2_2.jl: Wright-Fisher valley-crossing sweeps for Fig. S2_2D-G. Main functions: wright_fisher_with_fixation and parameter_sweep.
- sweep_mu_k.jl: Wright-Fisher simulations on a changing fitness landscape for Fig. 3. Main functions: run_landscape_wright_fisher_simulation and parameter_sweep_mu_k.
- sims_5.py: Wright-Fisher genome simulations varying one-step and two-step loci for Fig. 5 and Fig. S5_1. Main functions: run_simulations and parameter_sweep_varying_n1.
- slurm_sims_4.jl: population-wide burst birth-death simulations used for Fig. 4B-C. Main functions: run_populationwide_burst_simulation and summarize_populationwide_burst_result.

Dependencies: Python scripts require numpy. Julia scripts use standard libraries Random and Statistics; slurm_sims_4.jl also requires Distributions.