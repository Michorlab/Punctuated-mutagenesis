import random


class Cell:
    next_lineage_id = 0

    def __init__(self, fitness=1.0, mutations=0, lineage_id=None, parent_lineage_id=None):
        self.fitness = fitness
        self.mutations = mutations
        if lineage_id is None:
            self.lineage_id = Cell.next_lineage_id
            Cell.next_lineage_id += 1
        else:
            self.lineage_id = lineage_id
        self.parent_lineage_id = parent_lineage_id


def reset_lineage_ids():
    Cell.next_lineage_id = 0


def next_lineage_id():
    lineage_id = Cell.next_lineage_id
    Cell.next_lineage_id += 1
    return lineage_id


def mutate(cell, c, b):
    if cell.mutations % 2 == 0:
        cell.fitness *= 1 - c
    else:
        cell.fitness = (1 + b - c) * (cell.fitness / (1 - c))
    cell.mutations += 1


def reproduce_offspring(parent, c, b, mutation_rate, rng):
    offspring = Cell(
        parent.fitness,
        parent.mutations,
        lineage_id=parent.lineage_id,
        parent_lineage_id=parent.parent_lineage_id,
    )

    if rng.random() < mutation_rate:
        mutate(offspring, c, b)
        offspring.parent_lineage_id = parent.lineage_id
        offspring.lineage_id = next_lineage_id()

    return offspring


def update_population_wright_fisher(population, c, b, mutation_rate, rng):
    total_fitness = sum(cell.fitness for cell in population)
    if total_fitness > 0:
        weights = [cell.fitness / total_fitness for cell in population]
    else:
        weights = [1 / len(population)] * len(population)

    population[:] = [
        reproduce_offspring(
            rng.choices(population, weights=weights, k=1)[0],
            c,
            b,
            mutation_rate,
            rng,
        )
        for _ in range(len(population))
    ]


def snapshot_population(population):
    return [
        (cell.fitness, cell.mutations, cell.lineage_id, cell.parent_lineage_id)
        for cell in population
    ]


def validate_inputs(initial_population_size, num_generations, c, mu, mu_increase, nm, k, k_snapshot):
    if initial_population_size < 1:
        raise ValueError("initial_population_size must be at least 1")
    if num_generations < 0:
        raise ValueError("num_generations must be non-negative")
    if c == 1:
        raise ValueError("c cannot equal 1 because the compensatory mutation divides by 1 - c")
    if not 0 <= mu <= 1:
        raise ValueError("mu must be a probability in [0, 1]")
    if not 0 <= mu * mu_increase <= 1:
        raise ValueError("mu * mu_increase must be a probability in [0, 1]")
    if nm <= 0:
        raise ValueError("nm must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if k_snapshot < 1:
        raise ValueError("k_snapshot must be at least 1")


def simulate_wright_fisher_evolution_with_rates(
    initial_population_size,
    num_generations,
    c,
    b,
    mu,
    mu_increase,
    nm,
    k,
    k_snapshot=1,
    seed=None,
):
    validate_inputs(initial_population_size, num_generations, c, mu, mu_increase, nm, k, k_snapshot)
    reset_lineage_ids()
    rng = random.Random(seed)

    population = [Cell() for _ in range(initial_population_size)]
    population_trajectory = []
    mutation_rates = []

    high_mutation_phase_duration = max(1, int(round(nm / k)))
    low_mutation_phase_duration = max(1, int(round(nm)))
    phase_counter = 0
    is_high_mutation_phase = False

    for generation in range(1, num_generations + 1):
        mutation_rate = mu * mu_increase if is_high_mutation_phase else mu
        phase_duration = (
            high_mutation_phase_duration
            if is_high_mutation_phase
            else low_mutation_phase_duration
        )

        update_population_wright_fisher(population, c, b, mutation_rate, rng)
        mutation_rates.append(mutation_rate)

        if generation % k_snapshot == 0:
            population_trajectory.append(snapshot_population(population))

        phase_counter += 1
        if phase_counter >= phase_duration:
            is_high_mutation_phase = not is_high_mutation_phase
            phase_counter = 0

    return population, population_trajectory, Cell.next_lineage_id, mutation_rates
