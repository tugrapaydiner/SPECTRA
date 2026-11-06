// Add explicit engine controls without sharing any SPECTRA solver code.
// The included file supplies the standalone parser, SCC construction, proof
// machinery, observer, default closure policy, and C ABI.
#include "direct_twosat_native.cpp"

namespace {

void append_answer(
    std::vector<std::uint8_t>& output,
    const std::vector<std::uint8_t>& labels,
    const std::vector<std::uint8_t>& certificate,
    U n,
    W cap,
    StatsInternal& stats
) {
    const bool sat = !labels.empty() || (n == 0 && certificate.empty());
    if (output.size() >= cap) throw Resource("SCQ2 output exceeds max_bytes");
    output.push_back(sat ? 1 : 0);
    if (sat) {
        if (labels.size() != n)
            throw std::logic_error("SAT label count differs");
        if (output.size() > cap || W(labels.size()) > cap - output.size())
            throw Resource("SCQ2 output exceeds max_bytes");
        output.insert(output.end(), labels.begin(), labels.end());
        put32(output, 0, cap);
        ++stats.sat;
    } else {
        if (output.size() > cap || W(n) > cap - output.size())
            throw Resource("SCQ2 output exceeds max_bytes");
        output.insert(output.end(), n, 0);
        put32(output, static_cast<U>(certificate.size()), cap);
        put_bytes(output, certificate, cap);
        ++stats.unsat;
        stats.proof_bytes += certificate.size();
    }
}

void initialize_stats(
    const Packet& packet,
    const Solver& solver,
    U engine,
    StatsInternal& stats
) {
    stats.n = packet.n;
    stats.edges = packet.edge_count;
    stats.queries = packet.query_count;
    stats.pins = packet.pin_count;
    stats.components = solver.component_count;
    stats.dag_edges = static_cast<U>(solver.dag_arcs.size());
    stats.engine = engine;
    stats.closure_bytes = engine == 1U ? 8 * W(solver.closure.size()) : 0;
}

std::vector<std::uint8_t> begin_output(const Packet& packet, W cap) {
    W minimum = 16 + W(packet.query_count) * (1 + W(packet.n) + 4);
    if (minimum > cap) throw Resource("minimum SCQ2 output exceeds max_bytes");
    std::vector<std::uint8_t> output;
    output.reserve(static_cast<std::size_t>(std::min<W>(minimum + 4096, cap)));
    output.insert(output.end(), {'S', 'C', 'Q', '2'});
    put32(output, packet.n, cap);
    put64(output, packet.query_count, cap);
    return output;
}

std::vector<std::uint8_t> solve_packet_dag(
    const std::uint8_t* data,
    std::size_t size,
    W cap,
    StatsInternal& stats
) {
    Packet packet(data, size, cap);
    Solver solver(packet, cap);
    // The default challenger may build dense component reachability. This entry
    // point deliberately disables it after common construction so the benchmark
    // can choose the stronger independent engine rather than assuming closure wins.
    solver.closure.clear();
    solver.closure.shrink_to_fit();
    solver.closure_enabled = false;

    Observer observer(data, size);
    initialize_stats(packet, solver, 2U, stats);
    std::vector<std::uint8_t> output = begin_output(packet, cap);

    std::vector<U> touched;
    std::vector<U> marks(packet.n, 0);
    std::vector<W> effective(packet.n, 0);
    U generation = 1;
    for (U row = 0; row < packet.query_count; ++row, ++generation) {
        if (generation == 0) {
            std::fill(marks.begin(), marks.end(), 0);
            generation = 1;
        }
        std::vector<std::uint8_t> labels;
        std::vector<std::uint8_t> certificate;
        solver.solve_row(
            row, labels, certificate, touched, marks, effective, generation
        );
        append_answer(output, labels, certificate, packet.n, cap, stats);
    }
    if (!observer.verify(output))
        throw std::logic_error(
            "independent observer rejected direct 2-SAT DAG output"
        );
    stats.output_bytes = output.size();
    return output;
}

std::vector<std::uint8_t> solve_packet_tile(
    const std::uint8_t* data,
    std::size_t size,
    W cap,
    StatsInternal& stats
) {
    Packet packet(data, size, cap);
    Solver solver(packet, cap);
    solver.closure.clear();
    solver.closure.shrink_to_fit();
    solver.closure_enabled = false;
    Observer observer(data, size);
    initialize_stats(packet, solver, 3U, stats);
    std::vector<std::uint8_t> output = begin_output(packet, cap);

    // A fixed contradiction still receives fresh proof bytes per row. There is no
    // component DAG to tile in this case, so use the exact scalar proof path while
    // retaining the explicit tile engine identity in the receipt.
    if (solver.base_unsat) {
        std::vector<U> touched;
        std::vector<U> marks(packet.n, 0);
        std::vector<W> effective(packet.n, 0);
        U generation = 1;
        for (U row = 0; row < packet.query_count; ++row, ++generation) {
            if (generation == 0) {
                std::fill(marks.begin(), marks.end(), 0);
                generation = 1;
            }
            std::vector<std::uint8_t> labels;
            std::vector<std::uint8_t> certificate;
            solver.solve_row(
                row, labels, certificate, touched, marks, effective, generation
            );
            append_answer(output, labels, certificate, packet.n, cap, stats);
        }
        if (!observer.verify(output))
            throw std::logic_error(
                "independent observer rejected direct 2-SAT tile output"
            );
        stats.output_bytes = output.size();
        return output;
    }

    std::vector<U> marks(packet.n, 0);
    std::vector<W> effective(packet.n, 0);
    std::vector<U> touched;
    std::vector<U> roots;
    std::vector<U> root_offsets(65, 0);
    std::vector<U> empty(64, NIL);
    std::vector<U> conflict(64, NIL);
    std::vector<W> reached(solver.component_count, 0);
    std::vector<W> high_choices(packet.n, 0);
    U generation = 1;

    std::vector<U> proof_marks(packet.n, 0);
    std::vector<W> proof_effective(packet.n, 0);
    std::vector<U> proof_touched;
    U proof_generation = 1;

    for (U first_row = 0; first_row < packet.query_count; first_row += 64) {
        U lanes = std::min<U>(64, packet.query_count - first_row);
        roots.clear();
        root_offsets[0] = 0;
        std::fill(empty.begin(), empty.begin() + lanes, NIL);
        std::fill(conflict.begin(), conflict.begin() + lanes, NIL);
        std::fill(reached.begin(), reached.end(), 0);

        for (U lane = 0; lane < lanes; ++lane) {
            if (generation == 0) {
                std::fill(marks.begin(), marks.end(), 0);
                generation = 1;
            }
            U empty_vertex = NIL;
            std::vector<U> lane_roots = solver.seeds_for_row(
                first_row + lane,
                empty_vertex,
                touched,
                marks,
                effective,
                generation++
            );
            empty[lane] = empty_vertex;
            roots.insert(roots.end(), lane_roots.begin(), lane_roots.end());
            root_offsets[lane + 1] = static_cast<U>(roots.size());
            if (empty_vertex == NIL) {
                W bit = W(1) << lane;
                for (U root : lane_roots)
                    reached[solver.component[root]] |= bit;
            }
        }

        // Process all lanes through one independently implemented component-DAG
        // traversal. The SCC and DAG were derived by the challenger itself.
        for (U component : solver.topological) {
            W active = reached[component];
            if (!active) continue;
            for (
                U at = solver.dag.offsets[component];
                at < solver.dag.offsets[component + 1];
                ++at
            ) {
                reached[solver.dag.targets[at]] |= active;
            }
        }

        for (U component = 0; component < solver.component_count; ++component) {
            W bad_lanes = reached[component]
                & reached[solver.complement_component[component]];
            while (bad_lanes) {
                U lane = first_bit(bad_lanes);
                bad_lanes &= bad_lanes - 1;
                if (conflict[lane] == NIL) conflict[lane] = component;
            }
        }

        W sat_lanes = 0;
        for (U lane = 0; lane < lanes; ++lane) {
            if (empty[lane] == NIL && conflict[lane] == NIL)
                sat_lanes |= W(1) << lane;
        }
        if (sat_lanes) {
            for (U vertex = 0; vertex < packet.n; ++vertex) {
                W low = reached[solver.component[2 * vertex]];
                W high = reached[solver.component[2 * vertex + 1]];
                if (low & high & sat_lanes)
                    throw std::logic_error(
                        "direct tile retained a contradiction in a SAT lane"
                    );
                bool base_high = solver.high_colour[vertex] != NIL
                    && solver.base_labels[vertex] == solver.high_colour[vertex];
                high_choices[vertex] = (
                    high | (base_high ? ~low : W(0))
                ) & sat_lanes;
            }
        }

        for (U lane = 0; lane < lanes; ++lane) {
            if (empty[lane] != NIL) {
                std::vector<std::uint8_t> certificate = make_proof(
                    1, std::vector<U>{empty[lane]}, {}, cap
                );
                append_answer(output, {}, certificate, packet.n, cap, stats);
                continue;
            }
            if (conflict[lane] != NIL) {
                if (proof_generation == 0) {
                    std::fill(proof_marks.begin(), proof_marks.end(), 0);
                    proof_generation = 1;
                }
                std::vector<std::uint8_t> labels;
                std::vector<std::uint8_t> certificate;
                solver.solve_row(
                    first_row + lane,
                    labels,
                    certificate,
                    proof_touched,
                    proof_marks,
                    proof_effective,
                    proof_generation++
                );
                if (!labels.empty())
                    throw std::logic_error(
                        "direct tile contradiction did not replay"
                    );
                append_answer(output, labels, certificate, packet.n, cap, stats);
                continue;
            }

            if (output.size() >= cap)
                throw Resource("SCQ2 output exceeds max_bytes");
            output.push_back(1);
            W bit = W(1) << lane;
            for (U vertex = 0; vertex < packet.n; ++vertex) {
                std::uint8_t label = (high_choices[vertex] & bit)
                    ? static_cast<std::uint8_t>(solver.high_colour[vertex])
                    : static_cast<std::uint8_t>(solver.low_colour[vertex]);
                if (output.size() >= cap)
                    throw Resource("SCQ2 output exceeds max_bytes");
                output.push_back(label);
            }
            put32(output, 0, cap);
            ++stats.sat;
        }
    }

    if (!observer.verify(output))
        throw std::logic_error(
            "independent observer rejected direct 2-SAT tile output"
        );
    stats.output_bytes = output.size();
    return output;
}

using Runner = std::vector<std::uint8_t> (*)(
    const std::uint8_t*, std::size_t, W, StatsInternal&
);

int run_export(
    Runner runner,
    const char* unknown_message,
    const std::uint8_t* packet,
    std::size_t packet_size,
    std::uint64_t max_bytes,
    std::uint8_t** output,
    std::size_t* output_size,
    Direct2SATStats* public_stats
) {
    if (output == nullptr || output_size == nullptr || public_stats == nullptr)
        return 2;
    *output = nullptr;
    *output_size = 0;
    std::memset(public_stats, 0, sizeof(*public_stats));
    try {
        StatsInternal stats;
        std::vector<std::uint8_t> result =
            runner(packet, packet_size, max_bytes, stats);
        void* memory = std::malloc(result.empty() ? 1 : result.size());
        if (memory == nullptr) throw std::bad_alloc();
        if (!result.empty()) std::memcpy(memory, result.data(), result.size());
        *output = static_cast<std::uint8_t*>(memory);
        *output_size = result.size();
        public_stats->n = stats.n;
        public_stats->edges = stats.edges;
        public_stats->queries = stats.queries;
        public_stats->pins = stats.pins;
        public_stats->components = stats.components;
        public_stats->dag_edges = stats.dag_edges;
        public_stats->sat = stats.sat;
        public_stats->unsat = stats.unsat;
        public_stats->engine = stats.engine;
        public_stats->closure_bytes = stats.closure_bytes;
        public_stats->output_bytes = stats.output_bytes;
        public_stats->proof_bytes = stats.proof_bytes;
        LAST_ERROR.clear();
        return 0;
    } catch (const Invalid& error) {
        LAST_ERROR = error.what();
        return 1;
    } catch (const Resource& error) {
        LAST_ERROR = error.what();
        return 2;
    } catch (const std::exception& error) {
        LAST_ERROR = error.what();
        return 3;
    } catch (...) {
        LAST_ERROR = unknown_message;
        return 3;
    }
}

}  // namespace

extern "C" {

D2S_EXPORT int direct_twosat_solve_dag(
    const std::uint8_t* packet,
    std::size_t packet_size,
    std::uint64_t max_bytes,
    std::uint8_t** output,
    std::size_t* output_size,
    Direct2SATStats* public_stats
) {
    return run_export(
        solve_packet_dag,
        "unknown direct 2-SAT DAG failure",
        packet,
        packet_size,
        max_bytes,
        output,
        output_size,
        public_stats
    );
}

D2S_EXPORT int direct_twosat_solve_tile(
    const std::uint8_t* packet,
    std::size_t packet_size,
    std::uint64_t max_bytes,
    std::uint8_t** output,
    std::size_t* output_size,
    Direct2SATStats* public_stats
) {
    return run_export(
        solve_packet_tile,
        "unknown direct 2-SAT tile failure",
        packet,
        packet_size,
        max_bytes,
        output,
        output_size,
        public_stats
    );
}

}  // extern "C"
