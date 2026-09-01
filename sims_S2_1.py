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


def reproduce(parent, c, b, mutation_rate, rng):
    daughters = [
        Cell(
            parent.fitness,
            parent.mutations,
            lineage_id=parent.lineage_id,
            parent_lineage_id=parent.parent_lineage_id,
        ),
        Cell(
            parent.fitness,
            parent.mutations,
            lineage_id=parent.lineage_id,
            parent_lineage_id=parent.parent_lineage_id,
        ),
    ]

    for daughter in daughters:
        if rng.random() < mutation_rate:
            mutate(daughter, c, b)
            daughter.parent_lineage_id = parent.lineage_id
            daughter.lineage_id = next_lineage_id()

    return daughters


def update_population_branching(population, c, b, mutation_rate, division, rng):
    parent = rng.choices(population, weights=[cell.fitness for cell in population], k=1)[0]
    population.remove(parent)
    population.extend(reproduce(parent, c, b, mutation_rate, rng))

    if division % 2 == 0:
        population.remove(rng.choice(population))


def branching_phase_durations(cell_number, nm, k):
    high_duration = int(round((4 / 3) * cell_number * (2 ** (nm / k) - 1)))
    low_duration = int(round((4 / 3) * cell_number * (2 ** nm - 1)))
    return max(1, high_duration), max(1, low_duration)


def snapshot_population(population):
    return [
        (cell.fitness, cell.mutations, cell.lineage_id, cell.parent_lineage_id)
        for cell in population
    ]


def validate_inputs(initial_population_size, num_divisions, c, mu, mu_increase, nm, k):
    if initial_population_size < 1:
        raise ValueError("initial_population_size must be at least 1")
    if num_divisions < 0:
        raise ValueError("num_divisions must be non-negative")
    if not 0 <= c < 1:
        raise ValueError("c must be in [0, 1) because the compensatory mutation divides by 1 - c")
    if not 0 <= mu <= 1:
        raise ValueError("mu must be a probability in [0, 1]")
    if not 0 <= mu * mu_increase <= 1:
        raise ValueError("mu * mu_increase must be a probability in [0, 1]")
    if nm <= 0:
        raise ValueError("nm must be positive")
    if k <= 0:
        raise ValueError("k must be positive")


def simulate_branching_evolution_with_rates(
    initial_population_size,
    num_divisions,
    c,
    b,
    mu,
    mu_increase,
    nm,
    k,
    snapshot_log_step=1 / 3,
    seed=None,
):
    validate_inputs(initial_population_size, num_divisions, c, mu, mu_increase, nm, k)
    reset_lineage_ids()
    rng = random.Random(seed)

    population = [Cell() for _ in range(initial_population_size)]
    population_trajectory = []
    mutation_rates = []

    high_duration, low_duration = branching_phase_durations(len(population), nm, k)
    phase_counter = 0
    is_high_mutation_phase = False
    next_snapshot_power = 0

    for division in range(1, num_divisions + 1):
        mutation_rate = mu * mu_increase if is_high_mutation_phase else mu
        phase_duration = high_duration if is_high_mutation_phase else low_duration

        update_population_branching(population, c, b, mutation_rate, division, rng)
        mutation_rates.append(mutation_rate)

        if division >= 2 ** next_snapshot_power:
            population_trajectory.append(snapshot_population(population))
            next_snapshot_power += snapshot_log_step

        phase_counter += 1
        if phase_counter >= phase_duration:
            is_high_mutation_phase = not is_high_mutation_phase
            phase_counter = 0
            high_duration, low_duration = branching_phase_durations(len(population), nm, k)

    return population, population_trajectory, Cell.next_lineage_id, mutation_rates
