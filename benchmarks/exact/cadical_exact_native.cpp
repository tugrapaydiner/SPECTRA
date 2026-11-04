// Proof-carrying SBC1 -> SCQ2 adapter around the official CaDiCaL solver core.
//
// CaDiCaL itself is linked from the exact external commit recorded by the build
// receipt. Packet admission, proof translation and the original-input observer
// are project-owned and deliberately charged inside every call.
#include "direct_twosat_native.cpp"
#include "cadical.hpp"

namespace {

struct CadicalProofIndex {
    const Packet& packet;
    std::vector<U> low_colour;
    std::vector<U> high_colour;
    std::vector<std::pair<U, U>> arcs;
    CSR graph;
    CSR reverse;
    std::vector<U> component;
    U component_count = 0;
    bool base_unsat = false;
    U base_literal = NIL;
    std::vector<U> base_left;
    std::vector<U> base_right;

    explicit CadicalProofIndex(const Packet& input) : packet(input) {
        low_colour.resize(packet.n);
        high_colour.resize(packet.n, NIL);
        for (U vertex = 0; vertex < packet.n; ++vertex) {
            W mask = packet.masks[vertex];
            low_colour[vertex] = first_bit(mask);
            mask &= mask - 1;
            if (mask) high_colour[vertex] = first_bit(mask);
        }

        arcs.reserve(2 * packet.edge_count + packet.n);
        for (U vertex = 0; vertex < packet.n; ++vertex) {
            if (high_colour[vertex] == NIL)
                arcs.emplace_back(2 * vertex + 1, 2 * vertex);
        }
        for (const auto& edge : packet.edges) {
            W shared = packet.masks[edge.first] & packet.masks[edge.second];
            while (shared) {
                const U colour = first_bit(shared);
                shared &= shared - 1;
                const U left = literal(edge.first, colour);
                const U right = literal(edge.second, colour);
                arcs.emplace_back(left, right ^ 1U);
                arcs.emplace_back(right, left ^ 1U);
            }
        }
        std::sort(arcs.begin(), arcs.end());
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());
        graph = CSR(2 * packet.n, arcs, false);
        reverse = CSR(2 * packet.n, arcs, true);
        component = strongly_connected(graph, reverse);
        for (const U id : component)
            component_count = std::max(component_count, id + 1);
        for (U vertex = 0; vertex < packet.n; ++vertex) {
            if (component[2 * vertex] == component[2 * vertex + 1]) {
                base_unsat = true;
                base_literal = 2 * vertex;
                base_left = shortest_path(graph, base_literal, base_literal ^ 1U);
                base_right = shortest_path(graph, base_literal ^ 1U, base_literal);
                break;
            }
        }
    }

    U literal(U vertex, U colour) const {
        if (low_colour[vertex] == colour) return 2 * vertex;
        if (high_colour[vertex] == colour) return 2 * vertex + 1;
        throw std::logic_error("colour is outside the admitted original domain");
    }

    int signed_choice(U literal_value) const {
        const int variable = static_cast<int>((literal_value >> 1) + 1);
        return (literal_value & 1U) ? variable : -variable;
    }

    std::vector<U> trace_parent(const std::vector<U>& parent, U target) const {
        std::vector<U> path;
        for (U current = target;; current = parent[current]) {
            if (current == NIL || path.size() > parent.size())
                throw std::logic_error("invalid proof predecessor chain");
            path.push_back(current);
            if (parent[current] == current) break;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    std::vector<std::uint8_t> base_proof(W cap) const {
        if (!base_unsat) throw std::logic_error("base proof requested for satisfiable formula");
        return make_proof(3, base_left, base_right, cap);
    }

    std::vector<std::uint8_t> query_proof(const std::vector<U>& roots, W cap) const {
        if (roots.empty())
            throw std::logic_error("query contradiction has no current forced root");
        std::vector<U> parent(2 * packet.n, NIL);
        std::vector<U> queue;
        queue.reserve(2 * packet.n);
        U conflict = NIL;
        for (const U root : roots) {
            if (root >= parent.size()) throw std::logic_error("proof root outside graph");
            if (parent[root] == NIL) {
                parent[root] = root;
                queue.push_back(root);
            }
            if (parent[root ^ 1U] != NIL) {
                conflict = root;
                break;
            }
        }
        for (std::size_t head = 0; conflict == NIL && head < queue.size(); ++head) {
            const U current = queue[head];
            for (U at = graph.offsets[current]; at < graph.offsets[current + 1]; ++at) {
                const U next = graph.targets[at];
                if (parent[next] == NIL) {
                    parent[next] = current;
                    queue.push_back(next);
                    if (parent[next ^ 1U] != NIL) {
                        conflict = next;
                        break;
                    }
                }
            }
        }
        if (conflict == NIL)
            throw std::logic_error("CaDiCaL UNSAT did not replay in the original implication graph");
        return make_proof(
            2,
            trace_parent(parent, conflict),
            trace_parent(parent, conflict ^ 1U),
            cap
        );
    }
};

struct CadicalExactStatsInternal {
    U n = 0;
    U edges = 0;
    U queries = 0;
    U pins = 0;
    U clauses = 0;
    U components = 0;
    U sat = 0;
    U unsat = 0;
    U solves = 0;
    U failed_assumptions = 0;
    W output_bytes = 0;
    W proof_bytes = 0;
};

void append_answer(std::vector<std::uint8_t>& output, const Packet& packet,
                   bool sat, const std::vector<std::uint8_t>& labels,
                   const std::vector<std::uint8_t>& proof_bytes, W cap,
                   CadicalExactStatsInternal& stats) {
    if (output.size() >= cap) throw Resource("SCQ2 output exceeds max_bytes");
    output.push_back(sat ? 1 : 0);
    if (sat) {
        if (labels.size() != packet.n || !proof_bytes.empty())
            throw std::logic_error("CaDiCaL SAT row geometry differs");
        if (W(labels.size()) > cap - output.size())
            throw Resource("SCQ2 SAT labels exceed max_bytes");
        output.insert(output.end(), labels.begin(), labels.end());
        put32(output, 0, cap);
        ++stats.sat;
    } else {
        if (!labels.empty()) throw std::logic_error("CaDiCaL UNSAT row supplied labels");
        if (W(packet.n) > cap - output.size())
            throw Resource("SCQ2 UNSAT padding exceeds max_bytes");
        output.insert(output.end(), packet.n, 0);
        if (proof_bytes.size() > std::numeric_limits<U>::max())
            throw Resource("SCQ2 proof exceeds uint32 length");
        put32(output, static_cast<U>(proof_bytes.size()), cap);
        put_bytes(output, proof_bytes, cap);
        ++stats.unsat;
        stats.proof_bytes += proof_bytes.size();
    }
}

std::vector<std::uint8_t> solve_with_cadical(
    const std::uint8_t* data,
    std::size_t size,
    W cap,
    CadicalExactStatsInternal& stats
) {
    Packet packet(data, size, cap);
    CadicalProofIndex proof_index(packet);
    Observer observer(data, size);

    CaDiCaL::Solver solver;
    if (!solver.set("quiet", 1))
        throw std::logic_error("official CaDiCaL rejected the quiet option");
    if (packet.n)
        solver.declare_more_variables(static_cast<int>(packet.n));

    for (U vertex = 0; vertex < packet.n; ++vertex) {
        if (proof_index.high_colour[vertex] == NIL) {
            solver.clause(-static_cast<int>(vertex + 1));
            ++stats.clauses;
        }
    }
    for (const auto& edge : packet.edges) {
        W shared = packet.masks[edge.first] & packet.masks[edge.second];
        while (shared) {
            const U colour = first_bit(shared);
            shared &= shared - 1;
            const U left = proof_index.literal(edge.first, colour);
            const U right = proof_index.literal(edge.second, colour);
            solver.clause(
                -proof_index.signed_choice(left),
                -proof_index.signed_choice(right)
            );
            ++stats.clauses;
        }
    }

    stats.n = packet.n;
    stats.edges = packet.edge_count;
    stats.queries = packet.query_count;
    stats.pins = packet.pin_count;
    stats.components = proof_index.component_count;

    if (proof_index.base_unsat) {
        ++stats.solves;
        if (solver.solve() != CaDiCaL::UNSATISFIABLE)
            throw std::logic_error("official CaDiCaL disagreed with the static contradiction proof");
    }

    std::vector<std::uint8_t> output;
    const W minimum = 16 + W(packet.query_count) * (1 + W(packet.n) + 4);
    if (minimum > cap) throw Resource("minimum SCQ2 output exceeds max_bytes");
    output.reserve(static_cast<std::size_t>(std::min<W>(minimum + 4096, cap)));
    output.insert(output.end(), {'S', 'C', 'Q', '2'});
    put32(output, packet.n, cap);
    put64(output, packet.query_count, cap);

    std::vector<U> marks(packet.n, 0);
    std::vector<W> effective(packet.n, 0);
    std::vector<U> touched;
    std::vector<U> roots;
    std::vector<std::pair<int, U>> assumptions;
    U generation = 1;

    for (U row = 0; row < packet.query_count; ++row, ++generation) {
        if (generation == 0) {
            std::fill(marks.begin(), marks.end(), 0);
            generation = 1;
        }
        touched.clear();
        roots.clear();
        assumptions.clear();
        U empty_vertex = NIL;

        for (U pin = packet.row_begin(row); pin < packet.row_end(row); ++pin) {
            const U vertex = packet.pin_vertex(pin);
            if (marks[vertex] != generation) {
                marks[vertex] = generation;
                effective[vertex] = packet.masks[vertex];
                touched.push_back(vertex);
            }
            effective[vertex] &= packet.pin_allowed(pin);
        }
        std::sort(touched.begin(), touched.end());
        for (const U vertex : touched) {
            if (!effective[vertex]) {
                empty_vertex = vertex;
                break;
            }
        }
        if (empty_vertex != NIL) {
            append_answer(
                output,
                packet,
                false,
                {},
                make_proof(1, std::vector<U>{empty_vertex}, {}, cap),
                cap,
                stats
            );
            continue;
        }

        if (proof_index.base_unsat) {
            append_answer(
                output,
                packet,
                false,
                {},
                proof_index.base_proof(cap),
                cap,
                stats
            );
            continue;
        }

        for (const U vertex : touched) {
            const W domain = effective[vertex];
            if (domain != packet.masks[vertex] && bit_count(domain) == 1) {
                const U root = proof_index.literal(vertex, first_bit(domain));
                roots.push_back(root);
                assumptions.emplace_back(proof_index.signed_choice(root), root);
            }
        }
        for (const auto& assumption : assumptions)
            solver.assume(assumption.first);
        ++stats.solves;
        const int result = solver.solve();

        if (result == CaDiCaL::SATISFIABLE) {
            std::vector<std::uint8_t> labels(packet.n);
            for (U vertex = 0; vertex < packet.n; ++vertex) {
                const int value = solver.val(static_cast<int>(vertex + 1));
                if (!value)
                    throw std::logic_error("official CaDiCaL returned an incomplete model");
                if (value > 0) {
                    if (proof_index.high_colour[vertex] == NIL)
                        throw std::logic_error("official CaDiCaL selected a virtual singleton alternative");
                    labels[vertex] = static_cast<std::uint8_t>(proof_index.high_colour[vertex]);
                } else {
                    labels[vertex] = static_cast<std::uint8_t>(proof_index.low_colour[vertex]);
                }
            }
            append_answer(output, packet, true, labels, {}, cap, stats);
        } else if (result == CaDiCaL::UNSATISFIABLE) {
            std::vector<U> failed_roots;
            for (const auto& assumption : assumptions) {
                if (solver.failed(assumption.first)) {
                    failed_roots.push_back(assumption.second);
                    ++stats.failed_assumptions;
                }
            }
            std::vector<std::uint8_t> certificate;
            try {
                certificate = proof_index.query_proof(failed_roots, cap);
            } catch (const std::logic_error&) {
                certificate = proof_index.query_proof(roots, cap);
            }
            append_answer(output, packet, false, {}, certificate, cap, stats);
        } else {
            throw std::logic_error("official CaDiCaL returned UNKNOWN");
        }
    }

    if (!observer.verify(output))
        throw std::logic_error("independent original-input observer rejected CaDiCaL output");
    stats.output_bytes = output.size();
    return output;
}

}  // namespace

extern "C" {

struct CadicalExactStats {
    std::uint32_t n;
    std::uint32_t edges;
    std::uint32_t queries;
    std::uint32_t pins;
    std::uint32_t clauses;
    std::uint32_t components;
    std::uint32_t sat;
    std::uint32_t unsat;
    std::uint32_t solves;
    std::uint32_t failed_assumptions;
    std::uint64_t output_bytes;
    std::uint64_t proof_bytes;
};

D2S_EXPORT const char* cadical_exact_signature() {
    return CaDiCaL::Solver::signature();
}

D2S_EXPORT const char* cadical_exact_error() {
    return LAST_ERROR.c_str();
}

D2S_EXPORT void cadical_exact_free(void* pointer) {
    std::free(pointer);
}

D2S_EXPORT int cadical_exact_solve(
    const std::uint8_t* packet,
    std::size_t packet_size,
    std::uint64_t max_bytes,
    std::uint8_t** output,
    std::size_t* output_size,
    CadicalExactStats* public_stats
) {
    if (output == nullptr || output_size == nullptr || public_stats == nullptr)
        return 2;
    *output = nullptr;
    *output_size = 0;
    std::memset(public_stats, 0, sizeof(*public_stats));
    try {
        CadicalExactStatsInternal stats;
        std::vector<std::uint8_t> result =
            solve_with_cadical(packet, packet_size, max_bytes, stats);
        void* memory = std::malloc(result.empty() ? 1 : result.size());
        if (memory == nullptr) throw std::bad_alloc();
        if (!result.empty()) std::memcpy(memory, result.data(), result.size());
        *output = static_cast<std::uint8_t*>(memory);
        *output_size = result.size();
        public_stats->n = stats.n;
        public_stats->edges = stats.edges;
        public_stats->queries = stats.queries;
        public_stats->pins = stats.pins;
        public_stats->clauses = stats.clauses;
        public_stats->components = stats.components;
        public_stats->sat = stats.sat;
        public_stats->unsat = stats.unsat;
        public_stats->solves = stats.solves;
        public_stats->failed_assumptions = stats.failed_assumptions;
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
    } catch (const std::bad_alloc&) {
        LAST_ERROR = "CaDiCaL adapter allocation failed";
        return 2;
    } catch (const std::exception& error) {
        LAST_ERROR = error.what();
        return 3;
    } catch (...) {
        LAST_ERROR = "unknown CaDiCaL adapter failure";
        return 3;
    }
}

}  // extern "C"
