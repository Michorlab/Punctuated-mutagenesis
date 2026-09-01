using Random
using Statistics


function initialize_population(nl::Int, population_size::Int, rng::AbstractRNG)
    return [rand(rng, 1:nl, 2) for _ in 1:population_size]
end


function fitness(position, landscape)
    x, y = position
    return landscape[x, y]
end


function update_landscape!(landscape, rng::AbstractRNG)
    landscape .= rand(rng, size(landscape)...).^4 .+ 0.5
end


function mutate_moore_neighborhood(position, nl::Int, rng::AbstractRNG)
    x, y = position
    moves = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    )
    dx, dy = moves[rand(rng, 1:length(moves))]
    return [clamp(x + dx, 1, nl), clamp(y + dy, 1, nl)]
end


function weighted_parent_index(fitnesses::Vector{Float64}, rng::AbstractRNG)
    total_fitness = sum(fitnesses)
    if total_fitness <= 0
        return rand(rng, eachindex(fitnesses))
    end

    threshold = rand(rng) * total_fitness
    cumulative_fitness = 0.0
    for i in eachindex(fitnesses)
        cumulative_fitness += fitnesses[i]
        if cumulative_fitness >= threshold
            return i
        end
    end

    return lastindex(fitnesses)
end


function evolve_population_wright_fisher!(population, landscape, mutation_rate::Float64, rng::AbstractRNG)
    nl = size(landscape, 1)
    population_size = length(population)
    fitnesses = [fitness(position, landscape) for position in population]
    new_population = Vector{typeof(population[1])}(undef, population_size)

    for i in eachindex(population)
        parent = population[weighted_parent_index(fitnesses, rng)]
        offspring = copy(parent)
        if rand(rng) < mutation_rate
            offspring = mutate_moore_neighborhood(offspring, nl, rng)
        end
        new_population[i] = offspring
    end

    population .= new_population
end


function average_fitness(population, landscape)
    return mean(fitness(position, landscape) for position in population)
end


function time_varying_mutation_rate(generation::Int, mu::Float64, k::Float64, nm::Int)
    high_period = max(1, Int(round(nm / 10)))
    low_period = max(1, Int(round(0.9 * nm)))
    cycle_position = (generation - 1) % (high_period + low_period)

    if cycle_position < high_period
        return k * mu
    end
    return ((10 - k) / 9) * mu
end


function validate_inputs(
    nl::Int,
    population_size::Int,
    mu::Float64,
    k::Float64,
    nm::Int,
    generations::Int,
    landscape_update_interval::Int,
)
    if nl < 1
        throw(ArgumentError("nl must be at least 1"))
    end
    if population_size < 1
        throw(ArgumentError("population_size must be at least 1"))
    end
    if mu < 0
        throw(ArgumentError("mu must be non-negative"))
    end
    if !(1 <= k <= 10)
        throw(ArgumentError("k must be in [1, 10] for this mutation-rate schedule"))
    end
    if k * mu > 1
        throw(ArgumentError("k * mu must be a probability in [0, 1]"))
    end
    if nm < 1
        throw(ArgumentError("nm must be at least 1"))
    end
    if generations < 1
        throw(ArgumentError("generations must be at least 1"))
    end
    if landscape_update_interval < 1
        throw(ArgumentError("landscape_update_interval must be at least 1"))
    end
end


function run_landscape_wright_fisher_simulation(
    nl::Int,
    population_size::Int,
    mu::Float64,
    k::Float64,
    nm::Int,
    generations::Int,
    landscape_update_interval::Int;
    seed=nothing,
)
    validate_inputs(nl, population_size, mu, k, nm, generations, landscape_update_interval)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    landscape = rand(rng, nl, nl)
    update_landscape!(landscape, rng)
    population = initialize_population(nl, population_size, rng)

    average_fitness_history = Float64[]
    mutation_rate_history = Float64[]
    landscape_update_generations = Int[]

    for generation in 1:generations
        mutation_rate = time_varying_mutation_rate(generation, mu, k, nm)
        push!(mutation_rate_history, mutation_rate)

        evolve_population_wright_fisher!(population, landscape, mutation_rate, rng)

        if generation % landscape_update_interval == 0
            update_landscape!(landscape, rng)
            push!(landscape_update_generations, generation)
        end

        push!(average_fitness_history, average_fitness(population, landscape))
    end

    return (
        mean_average_fitness=mean(average_fitness_history),
        average_fitness_history=average_fitness_history,
        mutation_rate_history=mutation_rate_history,
        landscape_update_generations=landscape_update_generations,
        final_population=population,
        final_landscape=landscape,
    )
end


function parameter_sweep_mu_k(
    nl::Int,
    population_size::Int,
    nm::Int,
    generations::Int,
    landscape_update_interval::Int,
    mu_values::Vector{Float64},
    k_values::Vector{Float64};
    seed=nothing,
)
    average_fitness_grid = zeros(length(mu_values), length(k_values))

    for (i, mu) in enumerate(mu_values)
        for (j, k) in enumerate(k_values)
            simulation_seed = isnothing(seed) ? nothing : Int(seed) + 1_000_003 * i + j
            result = run_landscape_wright_fisher_simulation(
                nl,
                population_size,
                mu,
                k,
                nm,
                generations,
                landscape_update_interval;
                seed=simulation_seed,
            )
            average_fitness_grid[i, j] = result.mean_average_fitness
        end
    end

    return average_fitness_grid
end
