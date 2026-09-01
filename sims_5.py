import numpy as np


class WrightFisherProcess:
    def __init__(self, population_size, n1, n2, b1, c2, b2, nm, k, mu_a, seed=None):
        validate_inputs(population_size, n1, n2, b1, c2, b2, nm, k, mu_a, generations=1)
        self.population_size = population_size
        self.n1 = n1
        self.n2 = n2
        self.num_loci = n1 + n2
        self.b1 = b1
        self.c2 = c2
        self.b2 = b2
        self.nm = nm
        self.k = k
        self.mu_a = mu_a
        self.rng = np.random.default_rng(seed)

        self.population = np.zeros((population_size, self.num_loci), dtype=np.int8)
        self.fitness = np.ones(population_size, dtype=float)
        self.start_shift = self.rng.integers(1, nm + 1)
        self.flipped_n1_prop = []
        self.flipped_n2_1_prop = []
        self.flipped_n2_2_prop = []
        self.mu_values = []

    def mu_function(self, generation):
        adjusted_generation = generation + self.start_shift
        high_phase_length = self.nm / self.k
        low_phase_length = ((self.k - 1) / self.k) * self.nm
        phase_in_cycle = adjusted_generation % (high_phase_length + low_phase_length)

        if phase_in_cycle < high_phase_length:
            return self.k * self.mu_a
        return 0.0

    def mutate(self, genome, fitness, mu):
        if self.num_loci == 0:
            return genome, fitness

        mutation_count = self.rng.poisson(mu * self.num_loci)
        mutation_count = min(mutation_count, self.num_loci)
        if mutation_count == 0:
            return genome, fitness

        for locus in self.rng.choice(self.num_loci, size=mutation_count, replace=False):
            if locus < self.n1:
                if genome[locus] == 0:
                    genome[locus] = 1
                    fitness += self.b1
            else:
                if genome[locus] == 0:
                    genome[locus] = 1
                    fitness -= self.c2
                elif genome[locus] == 1:
                    genome[locus] = 2
                    fitness += self.c2 + self.b2

        return genome, fitness

    def track_mutations(self):
        if self.n1 > 0:
            flipped_n1 = np.mean(np.sum(self.population[:, : self.n1] == 1, axis=1) / self.n1)
        else:
            flipped_n1 = np.nan

        if self.n2 > 0:
            n2_population = self.population[:, self.n1 :]
            flipped_n2_1 = np.mean(np.sum(n2_population == 1, axis=1) / self.n2)
            flipped_n2_2 = np.mean(np.sum(n2_population == 2, axis=1) / self.n2)
        else:
            flipped_n2_1 = np.nan
            flipped_n2_2 = np.nan

        return flipped_n1, flipped_n2_1, flipped_n2_2

    def simulate(self, generations):
        validate_inputs(
            self.population_size,
            self.n1,
            self.n2,
            self.b1,
            self.c2,
            self.b2,
            self.nm,
            self.k,
            self.mu_a,
            generations,
        )

        for generation in range(generations):
            mu = self.mu_function(generation)
            self.mu_values.append(mu)

            total_fitness = np.sum(self.fitness)
            if total_fitness > 0:
                fitness_probs = self.fitness / total_fitness
            else:
                fitness_probs = np.full(self.population_size, 1 / self.population_size)

            parent_indices = self.rng.choice(
                self.population_size,
                size=self.population_size,
                p=fitness_probs,
            )
            new_population = self.population[parent_indices].copy()
            new_fitness = self.fitness[parent_indices].copy()

            for i in range(self.population_size):
                new_population[i], new_fitness[i] = self.mutate(
                    new_population[i],
                    new_fitness[i],
                    mu,
                )

            self.population = new_population
            self.fitness = new_fitness

            flipped_n1, flipped_n2_1, flipped_n2_2 = self.track_mutations()
            self.flipped_n1_prop.append(flipped_n1)
            self.flipped_n2_1_prop.append(flipped_n2_1)
            self.flipped_n2_2_prop.append(flipped_n2_2)

        return {
            "flipped_n1_prop": np.array(self.flipped_n1_prop),
            "flipped_n2_1_prop": np.array(self.flipped_n2_1_prop),
            "flipped_n2_2_prop": np.array(self.flipped_n2_2_prop),
            "mu_values": np.array(self.mu_values),
            "final_population": self.population,
            "final_fitness": self.fitness,
        }


def validate_inputs(population_size, n1, n2, b1, c2, b2, nm, k, mu_a, generations):
    if population_size < 1:
        raise ValueError("population_size must be at least 1")
    if n1 < 0 or n2 < 0:
        raise ValueError("n1 and n2 must be non-negative")
    if n1 + n2 < 1:
        raise ValueError("n1 + n2 must be at least 1")
    if b1 < 0:
        raise ValueError("b1 must be non-negative")
    if c2 < 0:
        raise ValueError("c2 must be non-negative")
    if b2 < 0:
        raise ValueError("b2 must be non-negative")
    if nm <= 0:
        raise ValueError("nm must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if mu_a < 0:
        raise ValueError("mu_a must be non-negative")
    if k * mu_a > 1:
        raise ValueError("k * mu_a must be a probability in [0, 1]")
    if generations < 1:
        raise ValueError("generations must be at least 1")


def nanmean_by_generation(values):
    if np.isnan(values).all(axis=0).any():
        result = np.full(values.shape[1], np.nan)
        valid_columns = ~np.isnan(values).all(axis=0)
        result[valid_columns] = np.nanmean(values[:, valid_columns], axis=0)
        return result
    return np.nanmean(values, axis=0)


def nanpercentile_by_generation(values, percentile):
    result = np.full(values.shape[1], np.nan)
    valid_columns = ~np.isnan(values).all(axis=0)
    result[valid_columns] = np.nanpercentile(values[:, valid_columns], percentile, axis=0)
    return result


def summarize_simulations(simulation_outputs, percentile_gap):
    lower_percentile = percentile_gap
    upper_percentile = 100 - percentile_gap

    flipped_n1 = np.array([output["flipped_n1_prop"] for output in simulation_outputs])
    flipped_n2_1 = np.array([output["flipped_n2_1_prop"] for output in simulation_outputs])
    flipped_n2_2 = np.array([output["flipped_n2_2_prop"] for output in simulation_outputs])
    mu_values = np.array([output["mu_values"] for output in simulation_outputs])

    return {
        "mean_flipped_n1": nanmean_by_generation(flipped_n1),
        "mean_flipped_n2_1": nanmean_by_generation(flipped_n2_1),
        "mean_flipped_n2_2": nanmean_by_generation(flipped_n2_2),
        "mean_mu": np.mean(mu_values, axis=0),
        "lower_flipped_n1": nanpercentile_by_generation(flipped_n1, lower_percentile),
        "upper_flipped_n1": nanpercentile_by_generation(flipped_n1, upper_percentile),
        "lower_flipped_n2_1": nanpercentile_by_generation(flipped_n2_1, lower_percentile),
        "upper_flipped_n2_1": nanpercentile_by_generation(flipped_n2_1, upper_percentile),
        "lower_flipped_n2_2": nanpercentile_by_generation(flipped_n2_2, lower_percentile),
        "upper_flipped_n2_2": nanpercentile_by_generation(flipped_n2_2, upper_percentile),
    }


def run_simulations(
    population_size,
    n1,
    n2,
    b1,
    c2,
    b2,
    nm,
    k,
    mu_a,
    generations,
    num_simulations,
    percentile_gap,
    seed=None,
):
    validate_inputs(population_size, n1, n2, b1, c2, b2, nm, k, mu_a, generations)
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    if not 0 <= percentile_gap <= 50:
        raise ValueError("percentile_gap must be in [0, 50]")

    seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(num_simulations)
    simulation_outputs = []

    for simulation_seed in child_seeds:
        process = WrightFisherProcess(
            population_size,
            n1,
            n2,
            b1,
            c2,
            b2,
            nm,
            k,
            mu_a,
            seed=simulation_seed,
        )
        simulation_outputs.append(process.simulate(generations))

    return summarize_simulations(simulation_outputs, percentile_gap)


def parameter_sweep_varying_n1(
    population_size,
    b1,
    c2,
    b2,
    nm,
    k_values,
    mu_a,
    generations,
    num_simulations,
    percentile_gap,
    total_loci,
    n1_step,
    seed=None,
):
    if total_loci < 1:
        raise ValueError("total_loci must be at least 1")
    if n1_step < 1:
        raise ValueError("n1_step must be at least 1")

    results = []
    seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    n1_values = list(range(0, total_loci + 1, n1_step))
    child_seeds = seed_sequence.spawn(len(k_values) * len(n1_values))
    seed_index = 0

    for k in k_values:
        for n1 in n1_values:
            n2 = total_loci - n1
            summary = run_simulations(
                population_size,
                n1,
                n2,
                b1,
                c2,
                b2,
                nm,
                k,
                mu_a,
                generations,
                num_simulations,
                percentile_gap,
                seed=child_seeds[seed_index],
            )
            seed_index += 1
            results.append(
                {
                    "n1": n1,
                    "n2": n2,
                    "k": k,
                    **summary,
                }
            )

    return results
