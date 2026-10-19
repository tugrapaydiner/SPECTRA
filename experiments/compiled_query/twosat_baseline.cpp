// Independent direct 2-SAT challenger for SPECTRA's SBC1 -> SCQ2 contract.
// MIT: repository LICENSE. Uses established SCC, DAG closure and implication paths.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define SPECTRA_EXPORT __declspec(dllexport)
#else
#define SPECTRA_EXPORT __attribute__((visibility("default")))
#endif

namespace {
using U = std::uint32_t;
using W = std::uint64_t;
constexpr U NIL = std::numeric_limits<U>::max();
constexpr U MAX_N = 100000;
constexpr U MAX_E = 2000000;
constexpr U MAX_P = 1000000;
constexpr W MIN_BYTES = W(1) << 20;
constexpr U ABI = 1;

struct Invalid final : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource final : std::runtime_error { using std::runtime_error::runtime_error; };

U load32(const std::uint8_t* data, std::size_t at) {
    U value = 0;
    for (U i = 0; i < 4; ++i) value |= U(data[at + i]) << (8U * i);
    return value;
}

W load64(const std::uint8_t* data, std::size_t at) {
    W value = 0;
    for (U i = 0; i < 8; ++i) value |= W(data[at + i]) << (8U * i);
    return value;
}

void put32(std::vector<std::uint8_t>& out, U value) {
    for (U i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8U * i)));
}

void put64(std::vector<std::uint8_t>& out, W value) {
    for (U i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8U * i)));
}

U bit_count(W value) {
    U count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

U first_bit(W value) {
    if (value == 0) throw Invalid("zero mask has no colour");
    U index = 0;
    while ((value & 1U) == 0U) {
        value >>= 1U;
        ++index;
    }
    return index;
}

void require_capacity(const std::vector<std::uint8_t>& out, W add, W cap) {
    if (add > cap || W(out.size()) > cap - add) {
        throw Resource("SCQ2 output exceeds max_bytes");
    }
}

struct Packet {
    const std::uint8_t* data;
    std::size_t size;
    U n = 0;
    U palette = 0;
    U edge_count = 0;
    U query_count = 0;
    U pin_count = 0;
    std::size_t masks_at = 32;
    std::size_t edges_at = 0;
    std::size_t offsets_at = 0;
    std::size_t vertices_at = 0;
    std::size_t allowed_at = 0;
    std::vector<W> masks;
    std::vector<std::pair<U, U>> edges;

    Packet(const std::uint8_t* bytes, std::size_t length, W cap)
        : data(bytes), size(length) {
        if (data == nullptr || size < 36 || std::memcmp(data, "SBC1", 4) != 0) {
            throw Invalid("invalid SBC1 packet");
        }
        n = load32(data, 4);
        palette = load32(data, 8);
        edge_count = load32(data, 12);
        query_count = load32(data, 16);
        pin_count = load32(data, 20);
        if (n > MAX_N || palette == 0 || palette > 64 || edge_count > MAX_E ||
            query_count > MAX_P || pin_count > MAX_P || load32(data, 24) != 0 ||
            load32(data, 28) != 0) {
            throw Invalid("SBC1 header is outside supported bounds");
        }
        const W expected = 36 + 8 * W(n) + 8 * W(edge_count) +
                           4 * W(query_count) + 12 * W(pin_count);
        if (expected != W(size)) throw Invalid("SBC1 packet length differs");
        if (W(size) > cap) throw Resource("SBC1 packet exceeds max_bytes");

        edges_at = masks_at + 8 * W(n);
        offsets_at = edges_at + 8 * W(edge_count);
        vertices_at = offsets_at + 4 * W(query_count + 1);
        allowed_at = vertices_at + 4 * W(pin_count);

        const W palette_mask = palette == 64 ? std::numeric_limits<W>::max()
                                             : ((W(1) << palette) - 1);
        masks.reserve(n);
        for (U vertex = 0; vertex < n; ++vertex) {
            const W mask = load64(data, masks_at + 8 * W(vertex));
            if (mask == 0 || (mask & ~palette_mask) != 0 || bit_count(mask) > 2) {
                throw Invalid("SBC1 domains require one or two palette colours");
            }
            masks.push_back(mask);
        }

        edges.reserve(edge_count);
        std::pair<U, U> previous{0, 0};
        bool have_previous = false;
        for (U index = 0; index < edge_count; ++index) {
            const U left = load32(data, edges_at + 8 * W(index));
            const U right = load32(data, edges_at + 8 * W(index) + 4);
            const std::pair<U, U> edge{left, right};
            if (left >= right || right >= n || (have_previous && !(previous < edge))) {
                throw Invalid("SBC1 edges must be sorted unique canonical pairs");
            }
            edges.push_back(edge);
            previous = edge;
            have_previous = true;
        }

        U prior = load32(data, offsets_at);
        if (prior != 0) throw Invalid("SBC1 offsets must start at zero");
        for (U index = 1; index <= query_count; ++index) {
            const U next = load32(data, offsets_at + 4 * W(index));
            if (next < prior || next > pin_count) throw Invalid("SBC1 offsets differ");
            prior = next;
        }
        if (prior != pin_count) throw Invalid("SBC1 offsets do not consume pins");
        for (U index = 0; index < pin_count; ++index) {
            if (load32(data, vertices_at + 4 * W(index)) >= n) {
                throw Invalid("SBC1 pin vertex lies outside graph");
            }
        }
    }

    U row_begin(U row) const { return load32(data, offsets_at + 4 * W(row)); }
    U row_end(U row) const { return load32(data, offsets_at + 4 * W(row + 1)); }
    U pin_vertex(U index) const { return load32(data, vertices_at + 4 * W(index)); }
    W pin_allowed(U index) const { return load64(data, allowed_at + 8 * W(index)); }
};

struct CSR {
    std::vector<U> off;
    std::vector<U> to;

    CSR() = default;
    CSR(U vertices, const std::vector<std::pair<U, U>>& arcs, bool reverse = false)
        : off(std::size_t(vertices) + 1, 0), to(arcs.size()) {
        for (const auto& arc : arcs) {
            const U source = reverse ? arc.second : arc.first;
            ++off[std::size_t(source) + 1];
        }
        for (U vertex = 0; vertex < vertices; ++vertex) {
            off[std::size_t(vertex) + 1] += off[vertex];
        }
        std::vector<U> cursor = off;
        for (const auto& arc : arcs) {
            const U source = reverse ? arc.second : arc.first;
            const U target = reverse ? arc.first : arc.second;
            to[cursor[source]++] = target;
        }
    }
};

std::vector<U> components(const CSR& graph, const CSR& reverse) {
    const U vertices = static_cast<U>(graph.off.size() - 1);
    std::vector<std::uint8_t> seen(vertices, 0);
    std::vector<U> order;
    order.reserve(vertices);
    std::vector<std::pair<U, U>> stack;
    stack.reserve(vertices);
    for (U root = 0; root < vertices; ++root) {
        if (seen[root] != 0) continue;
        seen[root] = 1;
        stack.emplace_back(root, graph.off[root]);
        while (!stack.empty()) {
            auto& top = stack.back();
            if (top.second == graph.off[std::size_t(top.first) + 1]) {
                order.push_back(top.first);
                stack.pop_back();
                continue;
            }
            const U next = graph.to[top.second++];
            if (seen[next] == 0) {
                seen[next] = 1;
                stack.emplace_back(next, graph.off[next]);
            }
        }
    }

    std::vector<U> comp(vertices, NIL);
    std::vector<U> todo;
    todo.reserve(vertices);
    U identifier = 0;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (comp[*it] != NIL) continue;
        comp[*it] = identifier;
        todo.push_back(*it);
        while (!todo.empty()) {
            const U vertex = todo.back();
            todo.pop_back();
            for (U at = reverse.off[vertex]; at < reverse.off[std::size_t(vertex) + 1]; ++at) {
                const U next = reverse.to[at];
                if (comp[next] == NIL) {
                    comp[next] = identifier;
                    todo.push_back(next);
                }
            }
        }
        ++identifier;
    }
    return comp;
}

std::vector<U> path_between(const CSR& graph, U source, U target,
                            W& visited, W& arcs_scanned) {
    const U vertices = static_cast<U>(graph.off.size() - 1);
    if (source >= vertices || target >= vertices) throw Invalid("literal path endpoint differs");
    std::vector<U> parent(vertices, NIL);
    std::deque<U> queue;
    parent[source] = source;
    queue.push_back(source);
    while (!queue.empty() && parent[target] == NIL) {
        const U vertex = queue.front();
        queue.pop_front();
        ++visited;
        for (U at = graph.off[vertex]; at < graph.off[std::size_t(vertex) + 1]; ++at) {
            ++arcs_scanned;
            const U next = graph.to[at];
            if (parent[next] == NIL) {
                parent[next] = vertex;
                queue.push_back(next);
            }
        }
    }
    if (parent[target] == NIL) throw std::logic_error("required implication path is missing");
    std::vector<U> path;
    for (U vertex = target;; vertex = parent[vertex]) {
        path.push_back(vertex);
        if (vertex == source) break;
        if (parent[vertex] == NIL || path.size() > parent.size()) {
            throw std::logic_error("implication parent chain differs");
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<U> trace_path(const std::vector<U>& parent, U target) {
    std::vector<U> path;
    U vertex = target;
    while (true) {
        path.push_back(vertex);
        if (parent[vertex] == vertex) break;
        if (parent[vertex] == NIL || path.size() > parent.size()) {
            throw std::logic_error("query implication path differs");
        }
        vertex = parent[vertex];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::pair<std::vector<U>, std::vector<U>> contradiction_paths(
    const CSR& graph, const std::vector<U>& roots, W& visited, W& arcs_scanned) {
    const U vertices = static_cast<U>(graph.off.size() - 1);
    std::vector<U> parent(vertices, NIL);
    std::deque<U> queue;
    for (const U root : roots) {
        if (root >= vertices) throw Invalid("query root literal differs");
        if (parent[root] == NIL) {
            parent[root] = root;
            queue.push_back(root);
        }
    }
    for (const U root : roots) {
        if (parent[root ^ 1U] != NIL) {
            return {trace_path(parent, root), trace_path(parent, root ^ 1U)};
        }
    }
    while (!queue.empty()) {
        const U vertex = queue.front();
        queue.pop_front();
        ++visited;
        for (U at = graph.off[vertex]; at < graph.off[std::size_t(vertex) + 1]; ++at) {
            ++arcs_scanned;
            const U next = graph.to[at];
            if (parent[next] == NIL) {
                parent[next] = vertex;
                if (parent[next ^ 1U] != NIL) {
                    return {trace_path(parent, next), trace_path(parent, next ^ 1U)};
                }
                queue.push_back(next);
            }
        }
    }
    throw std::logic_error("forced closure contradiction lacks literal paths");
}

std::vector<std::uint8_t> make_proof(
    U kind, const std::vector<U>& left, const std::vector<U>& right) {
    std::vector<std::uint8_t> out;
    out.reserve(16 + 4 * (left.size() + right.size()));
    out.insert(out.end(), {'S', 'B', 'P', '1'});
    put32(out, kind);
    put32(out, static_cast<U>(left.size()));
    put32(out, static_cast<U>(right.size()));
    for (const U value : left) put32(out, value);
    for (const U value : right) put32(out, value);
    return out;
}

struct DirectTwoSat {
    const Packet& packet;
    U literals;
    std::vector<std::array<U, 2>> colours;
    std::vector<std::uint8_t> widths;
    std::vector<std::pair<U, U>> arcs;
    CSR graph;
    CSR reverse;
    std::vector<U> comp;
    U component_count;
    std::vector<U> opposite;
    std::vector<std::pair<U, U>> dag_arcs;
    CSR dag;
    std::vector<U> topo;
    std::vector<W> closure;
    U closure_words = 0;
    bool dense = false;
    std::vector<U> static_roots;
    bool base_unsat = false;
    std::vector<U> base_left;
    std::vector<U> base_right;
    W path_visited = 0;
    W path_arcs = 0;

    DirectTwoSat(const Packet& source, W cap)
        : packet(source), literals(2 * source.n), colours(source.n), widths(source.n, 0),
          component_count(0) {
        if (W(literals) > std::numeric_limits<U>::max()) {
            throw Resource("literal geometry exceeds uint32");
        }
        static_roots.reserve(source.n);
        for (U vertex = 0; vertex < source.n; ++vertex) {
            const W mask = source.masks[vertex];
            const U first = first_bit(mask);
            colours[vertex][0] = first;
            const W remainder = mask & ~(W(1) << first);
            if (remainder == 0) {
                colours[vertex][1] = first;
                widths[vertex] = 1;
                static_roots.push_back(2 * vertex);
            } else {
                colours[vertex][1] = first_bit(remainder);
                widths[vertex] = 2;
            }
        }

        const W reserve_hint = W(source.n) + 8 * W(source.edge_count);
        if (reserve_hint > std::numeric_limits<std::size_t>::max()) {
            throw Resource("implication reserve exceeds address space");
        }
        arcs.reserve(static_cast<std::size_t>(reserve_hint));
        for (U vertex = 0; vertex < source.n; ++vertex) {
            if (widths[vertex] == 1) arcs.emplace_back(2 * vertex + 1, 2 * vertex);
        }
        for (const auto& edge : source.edges) {
            const U left = edge.first;
            const U right = edge.second;
            for (U left_side = 0; left_side < widths[left]; ++left_side) {
                for (U right_side = 0; right_side < widths[right]; ++right_side) {
                    if (colours[left][left_side] != colours[right][right_side]) continue;
                    const U left_literal = 2 * left + left_side;
                    const U right_literal = 2 * right + right_side;
                    arcs.emplace_back(left_literal, right_literal ^ 1U);
                    arcs.emplace_back(right_literal, left_literal ^ 1U);
                }
            }
        }
        std::sort(arcs.begin(), arcs.end());
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());
        const W graph_bytes = 8 * W(arcs.size()) + 16 * W(literals + 1);
        if (graph_bytes > cap) throw Resource("direct 2-SAT implication graph exceeds max_bytes");
        graph = CSR(literals, arcs, false);
        reverse = CSR(literals, arcs, true);
        comp = components(graph, reverse);
        component_count = comp.empty() ? 0 : (*std::max_element(comp.begin(), comp.end()) + 1);

        for (U vertex = 0; vertex < source.n; ++vertex) {
            if (comp[2 * vertex] == comp[2 * vertex + 1]) {
                base_unsat = true;
                base_left = path_between(graph, 2 * vertex, 2 * vertex + 1,
                                         path_visited, path_arcs);
                base_right = path_between(graph, 2 * vertex + 1, 2 * vertex,
                                          path_visited, path_arcs);
                break;
            }
        }

        opposite.assign(component_count, NIL);
        for (U literal = 0; literal < literals; ++literal) {
            const U current = comp[literal];
            const U other = comp[literal ^ 1U];
            if (opposite[current] == NIL) opposite[current] = other;
            if (opposite[current] != other) {
                throw std::logic_error("component complement mapping differs");
            }
        }

        dag_arcs.reserve(arcs.size());
        for (const auto& arc : arcs) {
            const U left = comp[arc.first];
            const U right = comp[arc.second];
            if (left != right) dag_arcs.emplace_back(left, right);
        }
        std::sort(dag_arcs.begin(), dag_arcs.end());
        dag_arcs.erase(std::unique(dag_arcs.begin(), dag_arcs.end()), dag_arcs.end());
        dag = CSR(component_count, dag_arcs, false);

        std::vector<U> indegree(component_count, 0);
        for (const auto& arc : dag_arcs) ++indegree[arc.second];
        std::deque<U> queue;
        for (U component = 0; component < component_count; ++component) {
            if (indegree[component] == 0) queue.push_back(component);
        }
        topo.reserve(component_count);
        while (!queue.empty()) {
            const U component = queue.front();
            queue.pop_front();
            topo.push_back(component);
            for (U at = dag.off[component]; at < dag.off[std::size_t(component) + 1]; ++at) {
                const U next = dag.to[at];
                if (--indegree[next] == 0) queue.push_back(next);
            }
        }
        if (topo.size() != component_count) {
            throw std::logic_error("component graph is not acyclic");
        }

        closure_words = (component_count + 63U) / 64U;
        const W closure_bytes = 8 * W(component_count) * W(closure_words);
        const W fixed_bytes = graph_bytes + 16 * W(component_count + dag_arcs.size() + 1);
        dense = component_count != 0 && component_count <= 8192 &&
                closure_bytes <= cap && fixed_bytes <= cap - closure_bytes;
        if (dense) {
            closure.assign(std::size_t(component_count) * closure_words, 0);
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                const U component = *it;
                W* row = closure.data() + std::size_t(component) * closure_words;
                row[component >> 6U] |= W(1) << (component & 63U);
                for (U at = dag.off[component]; at < dag.off[std::size_t(component) + 1]; ++at) {
                    const U next = dag.to[at];
                    const W* child = closure.data() + std::size_t(next) * closure_words;
                    for (U word = 0; word < closure_words; ++word) row[word] |= child[word];
                }
            }
        }
    }

    U literal_for(U vertex, W domain) const {
        const U colour = first_bit(domain);
        if (colour == colours[vertex][0]) return 2 * vertex;
        if (widths[vertex] == 2 && colour == colours[vertex][1]) return 2 * vertex + 1;
        throw std::logic_error("effective singleton is not an original choice");
    }

    bool closure_for(const std::vector<U>& roots, std::vector<W>& forced_bits,
                     std::vector<std::uint8_t>& forced_sparse,
                     W& visited, W& arcs_scanned) const {
        if (dense) {
            std::fill(forced_bits.begin(), forced_bits.end(), 0);
            for (const U root : roots) {
                const U component = comp[root];
                const W* row = closure.data() + std::size_t(component) * closure_words;
                for (U word = 0; word < closure_words; ++word) forced_bits[word] |= row[word];
            }
            for (U component = 0; component < component_count; ++component) {
                const bool selected =
                    (forced_bits[component >> 6U] >> (component & 63U)) & W(1);
                const U other = opposite[component];
                const bool selected_other =
                    (forced_bits[other >> 6U] >> (other & 63U)) & W(1);
                if (selected && selected_other) return true;
            }
            return false;
        }

        std::fill(forced_sparse.begin(), forced_sparse.end(), 0);
        std::vector<U> stack;
        stack.reserve(component_count);
        for (const U root : roots) {
            const U component = comp[root];
            if (forced_sparse[component] == 0) {
                forced_sparse[component] = 1;
                stack.push_back(component);
            }
        }
        while (!stack.empty()) {
            const U component = stack.back();
            stack.pop_back();
            ++visited;
            for (U at = dag.off[component]; at < dag.off[std::size_t(component) + 1]; ++at) {
                ++arcs_scanned;
                const U next = dag.to[at];
                if (forced_sparse[next] == 0) {
                    forced_sparse[next] = 1;
                    stack.push_back(next);
                }
            }
        }
        for (U component = 0; component < component_count; ++component) {
            if (forced_sparse[component] != 0 && forced_sparse[opposite[component]] != 0) {
                return true;
            }
        }
        return false;
    }

    bool is_forced(U component, const std::vector<W>& forced_bits,
                   const std::vector<std::uint8_t>& forced_sparse) const {
        if (dense) return ((forced_bits[component >> 6U] >> (component & 63U)) & W(1)) != 0;
        return forced_sparse[component] != 0;
    }

    std::vector<std::uint8_t> solve(W cap, std::array<W, 12>& stats) {
        std::vector<std::uint8_t> output;
        require_capacity(output, 16, cap);
        output.insert(output.end(), {'S', 'C', 'Q', '2'});
        put32(output, packet.n);
        put64(output, packet.query_count);

        std::vector<W> effective(packet.n, 0);
        std::vector<U> vertex_stamp(packet.n, 0);
        std::vector<U> root_stamp(literals, 0);
        U stamp = 1;
        std::vector<U> touched;
        std::vector<U> roots;
        std::vector<W> forced_bits(closure_words, 0);
        std::vector<std::uint8_t> forced_sparse(component_count, 0);
        std::vector<std::int8_t> state(component_count, 0);
        W sat = 0;
        W unsat = 0;
        W proof_bytes = 0;
        W visited = path_visited;
        W arcs_scanned = path_arcs;
        W assignment_fallbacks = 0;

        for (U row = 0; row < packet.query_count; ++row) {
            if (++stamp == 0) {
                std::fill(vertex_stamp.begin(), vertex_stamp.end(), 0);
                std::fill(root_stamp.begin(), root_stamp.end(), 0);
                stamp = 1;
            }
            touched.clear();
            roots.clear();
            for (const U root : static_roots) {
                if (root_stamp[root] != stamp) {
                    root_stamp[root] = stamp;
                    roots.push_back(root);
                }
            }

            U empty_vertex = NIL;
            for (U pin = packet.row_begin(row); pin < packet.row_end(row); ++pin) {
                const U vertex = packet.pin_vertex(pin);
                if (vertex_stamp[vertex] != stamp) {
                    vertex_stamp[vertex] = stamp;
                    effective[vertex] = packet.masks[vertex];
                    touched.push_back(vertex);
                }
                effective[vertex] &= packet.pin_allowed(pin);
            }
            for (const U vertex : touched) {
                if (effective[vertex] == 0) {
                    if (empty_vertex == NIL) empty_vertex = vertex;
                    continue;
                }
                if (bit_count(effective[vertex]) == 1) {
                    const U root = literal_for(vertex, effective[vertex]);
                    if (root_stamp[root] != stamp) {
                        root_stamp[root] = stamp;
                        roots.push_back(root);
                    }
                }
            }

            std::vector<std::uint8_t> proof;
            std::vector<std::uint8_t> labels(packet.n, 0);
            U code = 0;
            if (empty_vertex != NIL) {
                proof = make_proof(1, std::vector<U>{empty_vertex}, {});
                ++unsat;
            } else if (base_unsat) {
                proof = make_proof(3, base_left, base_right);
                ++unsat;
            } else {
                const bool contradiction = closure_for(
                    roots, forced_bits, forced_sparse, visited, arcs_scanned);
                if (contradiction) {
                    const auto paths = contradiction_paths(graph, roots, visited, arcs_scanned);
                    proof = make_proof(2, paths.first, paths.second);
                    ++unsat;
                } else {
                    std::fill(state.begin(), state.end(), std::int8_t{0});
                    for (U component = 0; component < component_count; ++component) {
                        if (!is_forced(component, forced_bits, forced_sparse)) continue;
                        state[component] = 1;
                        state[opposite[component]] = -1;
                    }
                    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                        const U component = *it;
                        if (state[component] == 0) {
                            state[component] = 1;
                            state[opposite[component]] = -1;
                        }
                    }
                    for (U vertex = 0; vertex < packet.n; ++vertex) {
                        const bool low = state[comp[2 * vertex]] == 1;
                        if (!low && widths[vertex] == 1) {
                            ++assignment_fallbacks;
                            throw std::logic_error("direct 2-SAT assignment rejected singleton");
                        }
                        labels[vertex] = static_cast<std::uint8_t>(
                            low ? colours[vertex][0] : colours[vertex][1]);
                        const W domain = vertex_stamp[vertex] == stamp
                                             ? effective[vertex]
                                             : packet.masks[vertex];
                        if (((domain >> labels[vertex]) & W(1)) == 0) {
                            ++assignment_fallbacks;
                            throw std::logic_error("direct 2-SAT assignment violates domain");
                        }
                    }
                    for (const auto& edge : packet.edges) {
                        if (labels[edge.first] == labels[edge.second]) {
                            ++assignment_fallbacks;
                            throw std::logic_error("direct 2-SAT assignment violates edge");
                        }
                    }
                    code = 1;
                    ++sat;
                }
            }

            require_capacity(output, W(packet.n) + 5 + proof.size(), cap);
            output.push_back(static_cast<std::uint8_t>(code));
            output.insert(output.end(), labels.begin(), labels.end());
            put32(output, static_cast<U>(proof.size()));
            output.insert(output.end(), proof.begin(), proof.end());
            proof_bytes += proof.size();
        }

        stats = {
            packet.query_count,
            sat,
            unsat,
            proof_bytes,
            literals,
            arcs.size(),
            component_count,
            8 * W(closure.size()),
            dense ? 1U : 0U,
            assignment_fallbacks,
            visited,
            arcs_scanned,
        };
        return output;
    }
};

void write_error(char* buffer, std::size_t size, const std::string& message) {
    if (buffer == nullptr || size == 0) return;
    const std::size_t count = std::min(size - 1, message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

}  // namespace

extern "C" {

SPECTRA_EXPORT U spectra_twosat_abi() { return ABI; }

SPECTRA_EXPORT int spectra_twosat_solve(
    const std::uint8_t* packet,
    W packet_bytes,
    W max_bytes,
    std::uint8_t** output,
    W* output_bytes,
    W* stats,
    U stats_count,
    char* error,
    W error_bytes) {
    if (output != nullptr) *output = nullptr;
    if (output_bytes != nullptr) *output_bytes = 0;
    try {
        if (output == nullptr || output_bytes == nullptr || stats == nullptr || stats_count < 12) {
            throw Invalid("output and twelve stats slots are required");
        }
        if (max_bytes < MIN_BYTES) throw Invalid("max_bytes must be at least one MiB");
        if (packet_bytes > std::numeric_limits<std::size_t>::max() ||
            error_bytes > std::numeric_limits<std::size_t>::max()) {
            throw Invalid("buffer length exceeds address space");
        }
        const Packet decoded(packet, static_cast<std::size_t>(packet_bytes), max_bytes);
        DirectTwoSat solver(decoded, max_bytes);
        std::array<W, 12> values{};
        std::vector<std::uint8_t> result = solver.solve(max_bytes, values);
        void* memory = std::malloc(result.size() == 0 ? 1 : result.size());
        if (memory == nullptr) throw Resource("unable to allocate SCQ2 result");
        if (!result.empty()) std::memcpy(memory, result.data(), result.size());
        *output = static_cast<std::uint8_t*>(memory);
        *output_bytes = result.size();
        for (U index = 0; index < 12; ++index) stats[index] = values[index];
        write_error(error, static_cast<std::size_t>(error_bytes), "");
        return 0;
    } catch (const Invalid& exception) {
        write_error(error, static_cast<std::size_t>(error_bytes), exception.what());
        return 1;
    } catch (const Resource& exception) {
        write_error(error, static_cast<std::size_t>(error_bytes), exception.what());
        return 2;
    } catch (const std::exception& exception) {
        write_error(error, static_cast<std::size_t>(error_bytes), exception.what());
        return 3;
    } catch (...) {
        write_error(error, static_cast<std::size_t>(error_bytes), "unknown direct 2-SAT failure");
        return 3;
    }
}

SPECTRA_EXPORT void spectra_twosat_free(void* memory) { std::free(memory); }

}  // extern "C"
