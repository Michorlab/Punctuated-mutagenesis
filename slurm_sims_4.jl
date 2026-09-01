using Distributions
using Random
using Statistics


function pick_weighted_index(weights::AbstractVector{Float64}, threshold::Float64)::Int
    cumulative_weight = 0.0
    for i in eachindex(weights)
        cumulative_weight += weights[i]
        if cumulative_weight >= threshold
            return i
        end
    end

    return lastindex(weights)
end


function draw_random_events(
    N::Int,
    n_special_pathways::Int,
    theta::Float64,
    mutation_probability::Float64,
    burst_rate_change::Float64,
    normal_mutation_mean::Float64,
    normal_mutation_sd::Float64,
    death_probability::Float64,
    rng::AbstractRNG,
)
    multiplier = ceil(Int, 1.5 * (log(N) / log(2 - 2 * death_probability)))
    event_count = multiplier * N
    division_events = rand(rng, event_count)
    birth_events = ifelse.(rand(rng, event_count) .> death_probability, 1, -1)
    birth_events[1:min(40, event_count)] .= 1

    total_events = findfirst(cumsum(birth_events) .== N - 1)
    if isnothing(total_events)
        throw(ErrorException("preallocated event sequence did not reach the requested population size"))
    end

    birth_events = birth_events[1:total_events]
    n_out_burst_mutations = rand(rng, Binomial(10000, mutation_probability), 2 * total_events)
    out_burst_is_special = rand(rng, sum(n_out_burst_mutations)) .< theta
    out_burst_pathways = rand(rng, 1:n_special_pathways, sum(out_burst_is_special))

    n_in_burst_mutations = rand(rng, Binomial(10000, burst_rate_change * mutation_probability), 2 * total_events)
    in_burst_is_special = rand(rng, sum(n_in_burst_mutations)) .< theta
    in_burst_pathways = rand(rng, 1:n_special_pathways, sum(in_burst_is_special))

    normal_effect_count = sum(n_in_burst_mutations) + sum(n_out_burst_mutations)
    normal_effects = rand(rng, Normal(normal_mutation_mean, normal_mutation_sd), normal_effect_count)
    normal_effects[rand(rng, normal_effect_count) .< 0.9] .= 0.0
    burst_entry_exit_events = rand(rng, 2 * total_events)

    return (
        division_events=division_events,
        birth_events=birth_events,
        n_out_burst_mutations=n_out_burst_mutations,
        out_burst_is_special=out_burst_is_special,
        out_burst_pathways=out_burst_pathways,
        n_in_burst_mutations=n_in_burst_mutations,
        in_burst_is_special=in_burst_is_special,
        in_burst_pathways=in_burst_pathways,
        normal_effects=normal_effects,
        burst_entry_exit_events=burst_entry_exit_events,
    )
end


function mutation(
    is_special,
    special_pathways,
    normal_effects,
    daughter,
    cell_fitness::Float64,
    valley_size::Float64,
    peak_size::Float64,
)
    cell_fitness += sum(normal_effects[is_special .== 0])

    for i in 1:sum(is_special)
        pathway = special_pathways[i]
        daughter[pathway] += 1
        if daughter[pathway] == 1
            cell_fitness -= valley_size
        elseif daughter[pathway] == 2
            cell_fitness += valley_size + peak_size
        end
    end

    return daughter, max(cell_fitness, 1e-10)
end


function run_populationwide_burst_simulation(
    N::Int,
    n_special_pathways::Int,
    theta::Float64,
    mutation_probability::Float64,
    burst_rate_change::Float64,
    normal_mutation_mean::Float64,
    normal_mutation_sd::Float64,
    death_probability::Float64,
    prob_enter_burst::Float64,
    prob_exit_burst::Float64,
    valley_size::Float64,
    peak_size::Float64;
    seed=nothing,
)
    validate_simulation_inputs(
        N,
        n_special_pathways,
        theta,
        mutation_probability,
        burst_rate_change,
        normal_mutation_sd,
        death_probability,
        prob_enter_burst,
        prob_exit_burst,
    )

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    population = zeros(Int, N, n_special_pathways + 3)
    fitnesses = zeros(Float64, N)
    fitnesses[1] = 1.0
    cell_ids = zeros(Int, N)
    active_cells = 1
    id_counter = 1
    total_fitness = 1.0
    random_indices = ones(Int, 6)

    events = draw_random_events(
        N,
        n_special_pathways,
        theta,
        mutation_probability,
        burst_rate_change,
        normal_mutation_mean,
        normal_mutation_sd,
        death_probability,
        rng,
    )

    tree = zeros(Int, 2 * length(events.birth_events), n_special_pathways + 3)
    burst_or_not = zeros(Int, length(events.birth_events))
    burst = Int(rand(rng) < (1 / prob_exit_burst) / ((1 / prob_exit_burst) + (1 / prob_enter_burst)))

    for event_index in eachindex(events.birth_events)
        if events.birth_events[event_index] == 1
            dividing_index = pick_weighted_index(
                fitnesses[1:active_cells],
                events.division_events[event_index] * total_fitness,
            )

            daughter1 = copy(population[dividing_index, :])
            daughter2 = copy(population[dividing_index, :])

            if burst >= 1
                burst_or_not[event_index] = 1
                n_muts_daughter1 = events.n_in_burst_mutations[2 * (event_index - 1) + 1]
                n_muts_daughter2 = events.n_in_burst_mutations[2 * event_index]
                daughter1[n_special_pathways + 2] += n_muts_daughter1
                daughter2[n_special_pathways + 2] += n_muts_daughter2

                special_daughter1 = events.in_burst_is_special[
                    random_indices[1]:(random_indices[1] + n_muts_daughter1 - 1)
                ]
                random_indices[1] += n_muts_daughter1
                special_daughter2 = events.in_burst_is_special[
                    random_indices[1]:(random_indices[1] + n_muts_daughter2 - 1)
                ]
                random_indices[1] += n_muts_daughter2

                pathways_daughter1 = events.in_burst_pathways[
                    random_indices[2]:(random_indices[2] + sum(special_daughter1) - 1)
                ]
                random_indices[2] += sum(special_daughter1)
                pathways_daughter2 = events.in_burst_pathways[
                    random_indices[2]:(random_indices[2] + sum(special_daughter2) - 1)
                ]
                random_indices[2] += sum(special_daughter2)

                if events.burst_entry_exit_events[2 * (event_index - 1) + 1] < prob_exit_burst / active_cells
                    burst = 0
                end
            else
                n_muts_daughter1 = events.n_out_burst_mutations[2 * (event_index - 1) + 1]
                n_muts_daughter2 = events.n_out_burst_mutations[2 * event_index]
                daughter1[n_special_pathways + 3] += n_muts_daughter1
                daughter2[n_special_pathways + 3] += n_muts_daughter2

                special_daughter1 = events.out_burst_is_special[
                    random_indices[3]:(random_indices[3] + n_muts_daughter1 - 1)
                ]
                random_indices[3] += n_muts_daughter1
                special_daughter2 = events.out_burst_is_special[
                    random_indices[3]:(random_indices[3] + n_muts_daughter2 - 1)
                ]
                random_indices[3] += n_muts_daughter2

                pathways_daughter1 = events.out_burst_pathways[
                    random_indices[4]:(random_indices[4] + sum(special_daughter1) - 1)
                ]
                random_indices[4] += sum(special_daughter1)
                pathways_daughter2 = events.out_burst_pathways[
                    random_indices[4]:(random_indices[4] + sum(special_daughter2) - 1)
                ]
                random_indices[4] += sum(special_daughter2)

                if events.burst_entry_exit_events[2 * (event_index - 1) + 1] > 1 - prob_enter_burst / active_cells
                    burst = 1
                end
            end

            effects_daughter1 = events.normal_effects[
                random_indices[5]:(random_indices[5] + n_muts_daughter1 - 1)
            ]
            random_indices[5] += n_muts_daughter1
            effects_daughter2 = events.normal_effects[
                random_indices[5]:(random_indices[5] + n_muts_daughter2 - 1)
            ]
            random_indices[5] += n_muts_daughter2

            tree[id_counter:(id_counter + 1), n_special_pathways + 2] .= cell_ids[dividing_index]
            tree[id_counter, n_special_pathways + 3] = id_counter
            tree[id_counter + 1, n_special_pathways + 3] = id_counter + 1
            cell_ids[dividing_index] = id_counter
            cell_ids[active_cells + 1] = id_counter + 1

            tree[id_counter, n_special_pathways + 1] = sum(special_daughter1 .== 0)
            tree[id_counter + 1, n_special_pathways + 1] = sum(special_daughter2 .== 0)
            for pathway in pathways_daughter1
                tree[id_counter, pathway] += 1
            end
            for pathway in pathways_daughter2
                tree[id_counter + 1, pathway] += 1
            end

            mother_fitness = fitnesses[dividing_index]
            population[dividing_index, :], fitnesses[dividing_index] = mutation(
                special_daughter1,
                pathways_daughter1,
                effects_daughter1,
                daughter1,
                mother_fitness,
                valley_size,
                peak_size,
            )
            population[active_cells + 1, :], fitnesses[active_cells + 1] = mutation(
                special_daughter2,
                pathways_daughter2,
                effects_daughter2,
                daughter2,
                mother_fitness,
                valley_size,
                peak_size,
            )

            total_fitness += fitnesses[dividing_index] + fitnesses[active_cells + 1] - mother_fitness
            active_cells += 1
            id_counter += 2
        else
            dead_cell = Int(ceil(events.division_events[event_index] * active_cells))
            total_fitness -= fitnesses[dead_cell]
            fitnesses[dead_cell] = fitnesses[active_cells]
            population[dead_cell, :] = population[active_cells, :]
            cell_ids[dead_cell] = cell_ids[active_cells]
            cell_ids[active_cells] = 0
            fitnesses[active_cells] = 0.0
            population[active_cells, :] .= 0
            active_cells -= 1
        end

        if active_cells > 0 && active_cells % 1000 == 0
            total_fitness = sum(fitnesses[1:active_cells])
        end
    end

    return (
        population=population,
        fitnesses=fitnesses,
        tree=tree,
        cell_ids=cell_ids,
        burst_or_not=burst_or_not,
    )
end


function vaf_from_tree(tree, cell_ids)
    n_special_pathways = size(tree, 2) - 3
    leaf_path_to_top = Int[]

    for cell_id in cell_ids
        ancestor = cell_id
        while ancestor != 0
            push!(leaf_path_to_top, ancestor)
            ancestor = tree[ancestor, n_special_pathways + 2]
        end
    end

    nodes = unique(leaf_path_to_top)
    descendant_counts = zeros(Int, maximum(nodes))
    for ancestor in leaf_path_to_top
        descendant_counts[ancestor] += 1
    end

    vaf_spectrum = [Int[] for _ in 1:(n_special_pathways + 1)]
    for node in nodes
        for mutation_type in 1:(n_special_pathways + 1)
            for _ in 1:tree[node, mutation_type]
                push!(vaf_spectrum[mutation_type], descendant_counts[node])
            end
        end
    end

    return sort!.(vaf_spectrum)
end


function mutation_counts_above_threshold(vaf_spectrum, N::Int, threshold::Float64, n_special_pathways::Int)
    outcomes = zeros(Int, n_special_pathways + 1)
    for pathway in 1:n_special_pathways
        outcomes[pathway] = sum(vaf_spectrum[pathway] .> threshold * N)
        outcomes[n_special_pathways + 1] += outcomes[pathway]
    end
    outcomes[n_special_pathways + 1] += sum(vaf_spectrum[n_special_pathways + 1] .> threshold * N)
    return outcomes
end


function summarize_threshold_outcomes(outcomes)
    return (
        detectable_tsg_deactivation=sum(outcomes[1:(end - 1)] .> 1),
        detectable_mutations=outcomes[end],
    )
end


function summarize_populationwide_burst_result(result, N::Int, n_special_pathways::Int; thresholds=(0.0, 0.001, 0.01))
    population = result.population
    tsg_mutations_avg = mean(sum(population[:, 1:n_special_pathways] .>= 2, dims=2))
    in_burst_avg, out_burst_avg = mean(population[:, (n_special_pathways + 2):(n_special_pathways + 3)], dims=1)
    vaf_spectrum = vaf_from_tree(result.tree, result.cell_ids)

    threshold_summaries = Dict{Float64, NamedTuple}()
    for threshold in thresholds
        outcomes = mutation_counts_above_threshold(vaf_spectrum, N, threshold, n_special_pathways)
        threshold_summaries[threshold] = summarize_threshold_outcomes(outcomes)
    end

    return (
        total_mu_avg=in_burst_avg + out_burst_avg,
        in_burst_avg=in_burst_avg,
        out_burst_avg=out_burst_avg,
        tsg_mutations_avg=tsg_mutations_avg,
        threshold_summaries=threshold_summaries,
    )
end


function validate_simulation_inputs(
    N::Int,
    n_special_pathways::Int,
    theta::Float64,
    mutation_probability::Float64,
    burst_rate_change::Float64,
    normal_mutation_sd::Float64,
    death_probability::Float64,
    prob_enter_burst::Float64,
    prob_exit_burst::Float64,
)
    if N < 2
        throw(ArgumentError("N must be at least 2"))
    end
    if n_special_pathways < 1
        throw(ArgumentError("n_special_pathways must be at least 1"))
    end
    if !(0 <= theta <= 1)
        throw(ArgumentError("theta must be a probability in [0, 1]"))
    end
    if !(0 <= mutation_probability <= 1)
        throw(ArgumentError("mutation_probability must be a probability in [0, 1]"))
    end
    if burst_rate_change < 0 || burst_rate_change * mutation_probability > 1
        throw(ArgumentError("burst_rate_change * mutation_probability must be a probability in [0, 1]"))
    end
    if normal_mutation_sd < 0
        throw(ArgumentError("normal_mutation_sd must be non-negative"))
    end
    if !(0 <= death_probability < 0.5)
        throw(ArgumentError("death_probability must be in [0, 0.5) so the population can grow"))
    end
    if prob_enter_burst <= 0
        throw(ArgumentError("prob_enter_burst must be positive"))
    end
    if prob_exit_burst <= 0
        throw(ArgumentError("prob_exit_burst must be positive"))
    end
end
