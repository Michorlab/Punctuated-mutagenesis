using Random


mutable struct Cell
    num_indirect::Int
    fitness::Float64
end


function indirect_mutation!(cell::Cell, c::Float64, b::Float64)
    cell.num_indirect += 1
    if isodd(cell.num_indirect)
        cell.fitness *= 1 - c
    else
        cell.fitness *= (1 + b) / (1 - c)
    end
end


function normalize_fitness!(population::Vector{Cell})
    mean_fitness = sum(cell.fitness for cell in population) / length(population)
    if mean_fitness > 0
        for cell in population
            cell.fitness /= mean_fitness
        end
    end
end


function weighted_parent_index(population::Vector{Cell}, rng::AbstractRNG)
    total_fitness = sum(cell.fitness for cell in population)
    if total_fitness <= 0
        return rand(rng, eachindex(population))
    end

    threshold = rand(rng) * total_fitness
    cumulative_fitness = 0.0
    for i in eachindex(population)
        cumulative_fitness += population[i].fitness
        if cumulative_fitness >= threshold
            return i
        end
    end

    return lastindex(population)
end


function reproduce_offspring(parent::Cell, c::Float64, b::Float64, mutation_rate::Float64, rng::AbstractRNG)
    offspring = Cell(parent.num_indirect, parent.fitness)
    if rand(rng) < mutation_rate
        indirect_mutation!(offspring, c, b)
    end
    return offspring
end


function update_population_wright_fisher!(
    population::Vector{Cell},
    c::Float64,
    b::Float64,
    mutation_rate::Float64,
    rng::AbstractRNG,
)
    next_population = Vector{Cell}(undef, length(population))
    for i in eachindex(population)
        parent = population[weighted_parent_index(population, rng)]
        next_population[i] = reproduce_offspring(parent, c, b, mutation_rate, rng)
    end
    population .= next_population
end


function validate_inputs(
    population_size::Int,
    generations::Int,
    c::Float64,
    b::Float64,
    mutation_rate::Float64,
    normalization_interval::Int,
)
    if population_size < 1
        throw(ArgumentError("population_size must be at least 1"))
    end
    if generations < 0
        throw(ArgumentError("generations must be non-negative"))
    end
    if c >= 1
        throw(ArgumentError("c must be less than 1 because the compensatory mutation divides by 1 - c"))
    end
    if b <= -1
        throw(ArgumentError("b must be greater than -1 to keep compensatory mutation fitness positive"))
    end
    if !(0 <= mutation_rate <= 1)
        throw(ArgumentError("mutation_rate must be a probability in [0, 1]"))
    end
    if normalization_interval < 1
        throw(ArgumentError("normalization_interval must be at least 1"))
    end
end


function wright_fisher_with_fixation(
    population_size::Int,
    c::Float64,
    b::Float64,
    mutation_rate::Float64,
    generations::Int,
    normalization_interval::Int;
    seed=nothing,
)
    validate_inputs(population_size, generations, c, b, mutation_rate, normalization_interval)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    population = [Cell(0, 1.0) for _ in 1:population_size]
    fixed_indirect_counts = Int[]

    for generation in 1:generations
        update_population_wright_fisher!(population, c, b, mutation_rate, rng)

        fixed_count = population[1].num_indirect
        if all(cell.num_indirect == fixed_count for cell in population) && !(fixed_count in fixed_indirect_counts)
            push!(fixed_indirect_counts, fixed_count)
        end

        if generation % normalization_interval == 0
            normalize_fitness!(population)
        end
    end

    mean_indirect_final = sum(cell.num_indirect for cell in population) / length(population)
    odd_fixations = sum(isodd, fixed_indirect_counts; init=0)
    fraction_odd_fixations = odd_fixations / length(fixed_indirect_counts)
    odd_fixations_per_indirect_mutation = odd_fixations / mean_indirect_final

    return fixed_indirect_counts, fraction_odd_fixations, odd_fixations_per_indirect_mutation, mean_indirect_final
end


function parameter_sweep(
    population_size::Int,
    b::Float64,
    generations::Int,
    normalization_interval::Int,
    mutation_rates::Vector{Float64},
    c_values::Vector{Float64};
    seed=nothing,
)
    fraction_odd_fixations = zeros(length(mutation_rates), length(c_values))
    odd_fixations_per_indirect_mutation = zeros(length(mutation_rates), length(c_values))
    mean_indirect_final = zeros(length(mutation_rates), length(c_values))

    for (i, mutation_rate) in enumerate(mutation_rates)
        for (j, c) in enumerate(c_values)
            simulation_seed = isnothing(seed) ? nothing : hash((seed, i, j))
            _, quantity1, quantity2, quantity3 = wright_fisher_with_fixation(
                population_size,
                c,
                b,
                mutation_rate,
                generations,
                normalization_interval;
                seed=simulation_seed,
            )
            fraction_odd_fixations[i, j] = quantity1
            odd_fixations_per_indirect_mutation[i, j] = quantity2
            mean_indirect_final[i, j] = quantity3
        end
    end

    return fraction_odd_fixations, odd_fixations_per_indirect_mutation, mean_indirect_final
end
