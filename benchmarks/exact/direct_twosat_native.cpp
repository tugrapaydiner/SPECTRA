#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define D2S_EXPORT __declspec(dllexport)
#else
#define D2S_EXPORT __attribute__((visibility("default")))
#endif

namespace {
using U = std::uint32_t;
using W = std::uint64_t;
constexpr U NIL = std::numeric_limits<U>::max();
constexpr U MAX_N = 100000;
constexpr U MAX_E = 2000000;
constexpr U MAX_Q = 1000000;
constexpr W MAX_CLOSURE_BYTES = W(128) << 20;

struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };
thread_local std::string LAST_ERROR;

U load32(const std::uint8_t* data, std::size_t at) {
    U value = 0;
    for (U i = 0; i < 4; ++i) value |= U(data[at + i]) << (8 * i);
    return value;
}
W load64(const std::uint8_t* data, std::size_t at) {
    W value = 0;
    for (U i = 0; i < 8; ++i) value |= W(data[at + i]) << (8 * i);
    return value;
}
void put32(std::vector<std::uint8_t>& out, U value, W cap) {
    if (out.size() > cap || cap - out.size() < 4) throw Resource("output exceeds max_bytes");
    for (U i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void put64(std::vector<std::uint8_t>& out, W value, W cap) {
    if (out.size() > cap || cap - out.size() < 8) throw Resource("output exceeds max_bytes");
    for (U i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void put_bytes(std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& bytes, W cap) {
    if (out.size() > cap || W(bytes.size()) > cap - out.size()) throw Resource("output exceeds max_bytes");
    out.insert(out.end(), bytes.begin(), bytes.end());
}
U bit_count(W value) {
#if defined(_MSC_VER)
    U count = 0;
    while (value) { value &= value - 1; ++count; }
    return count;
#else
    return static_cast<U>(__builtin_popcountll(value));
#endif
}
U first_bit(W value) {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<U>(index);
#else
    return static_cast<U>(__builtin_ctzll(value));
#endif
}
W palette_mask(U palette) {
    return palette == 64 ? std::numeric_limits<W>::max() : ((W(1) << palette) - 1);
}

struct Packet {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    U n = 0, palette = 0, edge_count = 0, query_count = 0, pin_count = 0;
    std::size_t masks_at = 32, edges_at = 0, offsets_at = 0, vertices_at = 0, allowed_at = 0;
    std::vector<W> masks;
    std::vector<std::pair<U, U>> edges;

    Packet(const std::uint8_t* bytes, std::size_t length, W cap) : data(bytes), size(length) {
        if (data == nullptr || size < 36 || std::memcmp(data, "SBC1", 4) != 0)
            throw Invalid("invalid SBC1 packet");
        n = load32(data, 4);
        palette = load32(data, 8);
        edge_count = load32(data, 12);
        query_count = load32(data, 16);
        pin_count = load32(data, 20);
        if (n > MAX_N || palette == 0 || palette > 64 || edge_count > MAX_E ||
            query_count > MAX_Q || pin_count > MAX_Q || load32(data, 24) != 0 ||
            load32(data, 28) != 0) {
            throw Invalid("SBC1 header outside supported bounds");
        }
        W expected = 36 + 8 * W(n) + 8 * W(edge_count) + 4 * W(query_count) + 12 * W(pin_count);
        if (expected != size) throw Invalid("SBC1 packet length differs");
        if (W(size) > cap) throw Resource("SBC1 packet exceeds max_bytes");
        edges_at = masks_at + 8 * W(n);
        offsets_at = edges_at + 8 * W(edge_count);
        vertices_at = offsets_at + 4 * W(query_count + 1);
        allowed_at = vertices_at + 4 * W(pin_count);

        const W allowed_palette = palette_mask(palette);
        masks.reserve(n);
        for (U v = 0; v < n; ++v) {
            W mask = load64(data, masks_at + 8 * W(v));
            if (mask == 0 || (mask & ~allowed_palette) != 0 || bit_count(mask) > 2)
                throw Invalid("original domains require one or two palette colours");
            masks.push_back(mask);
        }
        edges.reserve(edge_count);
        std::pair<U, U> previous{0, 0};
        bool have_previous = false;
        for (U i = 0; i < edge_count; ++i) {
            U left = load32(data, edges_at + 8 * W(i));
            U right = load32(data, edges_at + 8 * W(i) + 4);
            if (left >= right || right >= n) throw Invalid("SBC1 edges must be sorted in-range pairs");
            std::pair<U, U> edge{left, right};
            if (have_previous && !(previous < edge)) throw Invalid("SBC1 edges must be sorted unique");
            previous = edge;
            have_previous = true;
            edges.push_back(edge);
        }
        U previous_offset = load32(data, offsets_at);
        if (previous_offset != 0) throw Invalid("SBC1 offsets must begin at zero");
        for (U i = 1; i <= query_count; ++i) {
            U next = load32(data, offsets_at + 4 * W(i));
            if (next < previous_offset || next > pin_count) throw Invalid("SBC1 offsets differ");
            previous_offset = next;
        }
        if (previous_offset != pin_count) throw Invalid("SBC1 offsets do not consume pins");
        for (U i = 0; i < pin_count; ++i) {
            if (load32(data, vertices_at + 4 * W(i)) >= n)
                throw Invalid("SBC1 pin vertex outside graph");
        }
    }

    U row_begin(U row) const { return load32(data, offsets_at + 4 * W(row)); }
    U row_end(U row) const { return load32(data, offsets_at + 4 * W(row + 1)); }
    U pin_vertex(U pin) const { return load32(data, vertices_at + 4 * W(pin)); }
    W pin_allowed(U pin) const { return load64(data, allowed_at + 8 * W(pin)); }
};

struct CSR {
    std::vector<U> offsets;
    std::vector<U> targets;
    CSR() = default;
    CSR(U count, const std::vector<std::pair<U, U>>& arcs, bool reverse = false)
        : offsets(count + 1, 0), targets(arcs.size()) {
        for (const auto& arc : arcs) ++offsets[(reverse ? arc.second : arc.first) + 1];
        for (U i = 0; i < count; ++i) offsets[i + 1] += offsets[i];
        std::vector<U> cursor = offsets;
        for (const auto& arc : arcs) {
            U from = reverse ? arc.second : arc.first;
            U to = reverse ? arc.first : arc.second;
            targets[cursor[from]++] = to;
        }
    }
};

std::vector<U> strongly_connected(const CSR& graph, const CSR& reverse) {
    const U count = static_cast<U>(graph.offsets.size() - 1);
    std::vector<std::uint8_t> seen(count, 0);
    std::vector<U> finish;
    finish.reserve(count);
    std::vector<std::pair<U, U>> stack;
    stack.reserve(count);
    for (U root = 0; root < count; ++root) {
        if (seen[root]) continue;
        seen[root] = 1;
        stack.emplace_back(root, graph.offsets[root]);
        while (!stack.empty()) {
            auto& frame = stack.back();
            if (frame.second == graph.offsets[frame.first + 1]) {
                finish.push_back(frame.first);
                stack.pop_back();
                continue;
            }
            U next = graph.targets[frame.second++];
            if (!seen[next]) {
                seen[next] = 1;
                stack.emplace_back(next, graph.offsets[next]);
            }
        }
    }
    std::vector<U> component(count, NIL);
    std::vector<U> pending;
    pending.reserve(count);
    U id = 0;
    for (auto it = finish.rbegin(); it != finish.rend(); ++it) {
        if (component[*it] != NIL) continue;
        component[*it] = id;
        pending.push_back(*it);
        while (!pending.empty()) {
            U current = pending.back();
            pending.pop_back();
            for (U at = reverse.offsets[current]; at < reverse.offsets[current + 1]; ++at) {
                U next = reverse.targets[at];
                if (component[next] == NIL) {
                    component[next] = id;
                    pending.push_back(next);
                }
            }
        }
        ++id;
    }
    return component;
}

std::vector<U> shortest_path(const CSR& graph, U source, U target) {
    const U count = static_cast<U>(graph.offsets.size() - 1);
    if (source >= count || target >= count) throw std::logic_error("path endpoint outside graph");
    std::vector<U> parent(count, NIL);
    std::deque<U> queue;
    parent[source] = source;
    queue.push_back(source);
    while (!queue.empty() && parent[target] == NIL) {
        U current = queue.front();
        queue.pop_front();
        for (U at = graph.offsets[current]; at < graph.offsets[current + 1]; ++at) {
            U next = graph.targets[at];
            if (parent[next] == NIL) {
                parent[next] = current;
                queue.push_back(next);
            }
        }
    }
    if (parent[target] == NIL) throw std::logic_error("required implication path is missing");
    std::vector<U> path;
    for (U current = target;; current = parent[current]) {
        path.push_back(current);
        if (current == source) break;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<std::uint8_t> make_proof(U kind, const std::vector<U>& left,
                                     const std::vector<U>& right, W cap) {
    W bytes = 16 + 4 * W(left.size() + right.size());
    if (bytes > cap) throw Resource("proof exceeds max_bytes");
    std::vector<std::uint8_t> out;
    out.reserve(static_cast<std::size_t>(bytes));
    out.insert(out.end(), {'S', 'B', 'P', '1'});
    put32(out, kind, cap);
    put32(out, static_cast<U>(left.size()), cap);
    put32(out, static_cast<U>(right.size()), cap);
    for (U value : left) put32(out, value, cap);
    for (U value : right) put32(out, value, cap);
    return out;
}

struct Solver {
    const Packet& packet;
    U literal_count = 0;
    std::vector<U> low_colour;
    std::vector<U> high_colour;
    std::vector<std::pair<U, U>> arcs;
    CSR graph;
    CSR reverse;
    std::vector<U> component;
    U component_count = 0;
    std::vector<U> complement_component;
    std::vector<std::pair<U, U>> dag_arcs;
    CSR dag;
    std::vector<U> topological;
    std::vector<std::uint8_t> base_labels;
    bool base_unsat = false;
    U base_conflict_literal = NIL;
    std::vector<U> base_left_path;
    std::vector<U> base_right_path;
    U closure_words = 0;
    std::vector<W> closure;
    bool closure_enabled = false;

    explicit Solver(const Packet& input, W cap) : packet(input) {
        literal_count = 2 * packet.n;
        low_colour.resize(packet.n);
        high_colour.resize(packet.n, NIL);
        for (U v = 0; v < packet.n; ++v) {
            W mask = packet.masks[v];
            low_colour[v] = first_bit(mask);
            mask &= mask - 1;
            if (mask) high_colour[v] = first_bit(mask);
        }
        arcs.reserve(2 * packet.edge_count + packet.n);
        for (U v = 0; v < packet.n; ++v) {
            if (high_colour[v] == NIL) arcs.emplace_back(2 * v + 1, 2 * v);
        }
        for (const auto& edge : packet.edges) {
            U left = edge.first;
            U right = edge.second;
            W shared = packet.masks[left] & packet.masks[right];
            while (shared) {
                U colour = first_bit(shared);
                shared &= shared - 1;
                U left_lit = 2 * left + (high_colour[left] == colour ? 1 : 0);
                U right_lit = 2 * right + (high_colour[right] == colour ? 1 : 0);
                arcs.emplace_back(left_lit, right_lit ^ 1U);
                arcs.emplace_back(right_lit, left_lit ^ 1U);
            }
        }
        std::sort(arcs.begin(), arcs.end());
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());
        graph = CSR(literal_count, arcs, false);
        reverse = CSR(literal_count, arcs, true);
        component = strongly_connected(graph, reverse);
        for (U id : component) component_count = std::max(component_count, id + 1);

        for (U v = 0; v < packet.n; ++v) {
            if (component[2 * v] == component[2 * v + 1]) {
                base_unsat = true;
                base_conflict_literal = 2 * v;
                break;
            }
        }
        if (base_unsat) {
            base_left_path = shortest_path(graph, base_conflict_literal, base_conflict_literal ^ 1U);
            base_right_path = shortest_path(graph, base_conflict_literal ^ 1U, base_conflict_literal);
            return;
        }

        complement_component.assign(component_count, NIL);
        for (U literal = 0; literal < literal_count; ++literal) {
            U here = component[literal];
            U opposite = component[literal ^ 1U];
            if (complement_component[here] == NIL) complement_component[here] = opposite;
            else if (complement_component[here] != opposite)
                throw std::logic_error("component complements are inconsistent");
        }

        dag_arcs.reserve(arcs.size());
        for (const auto& arc : arcs) {
            U from = component[arc.first];
            U to = component[arc.second];
            if (from != to) dag_arcs.emplace_back(from, to);
        }
        std::sort(dag_arcs.begin(), dag_arcs.end());
        dag_arcs.erase(std::unique(dag_arcs.begin(), dag_arcs.end()), dag_arcs.end());
        dag = CSR(component_count, dag_arcs, false);

        std::vector<U> indegree(component_count, 0);
        for (const auto& arc : dag_arcs) ++indegree[arc.second];
        std::deque<U> queue;
        for (U id = 0; id < component_count; ++id) if (indegree[id] == 0) queue.push_back(id);
        while (!queue.empty()) {
            U current = queue.front();
            queue.pop_front();
            topological.push_back(current);
            for (U at = dag.offsets[current]; at < dag.offsets[current + 1]; ++at) {
                U next = dag.targets[at];
                if (--indegree[next] == 0) queue.push_back(next);
            }
        }
        if (topological.size() != component_count) throw std::logic_error("condensation graph is cyclic");

        std::vector<U> rank(component_count, 0);
        for (U i = 0; i < component_count; ++i) rank[topological[i]] = i;
        base_labels.resize(packet.n);
        for (U v = 0; v < packet.n; ++v) {
            bool lower = rank[component[2 * v]] > rank[component[2 * v + 1]];
            U colour = lower ? low_colour[v] : high_colour[v];
            if (colour == NIL) colour = low_colour[v];
            base_labels[v] = static_cast<std::uint8_t>(colour);
        }

        closure_words = (component_count + 63U) / 64U;
        W closure_bytes = 8 * W(component_count) * W(closure_words);
        W allowed = std::min<W>(MAX_CLOSURE_BYTES, cap / 3);
        if (component_count != 0 && closure_bytes <= allowed) {
            closure.resize(static_cast<std::size_t>(W(component_count) * closure_words), 0);
            for (auto it = topological.rbegin(); it != topological.rend(); ++it) {
                U current = *it;
                W* row = closure.data() + W(current) * closure_words;
                row[current >> 6] |= W(1) << (current & 63U);
                for (U at = dag.offsets[current]; at < dag.offsets[current + 1]; ++at) {
                    U next = dag.targets[at];
                    const W* source = closure.data() + W(next) * closure_words;
                    for (U word = 0; word < closure_words; ++word) row[word] |= source[word];
                }
            }
            closure_enabled = true;
        }
    }

    bool reached(const std::vector<W>& bits, U component_id) const {
        return (bits[component_id >> 6] >> (component_id & 63U)) & 1U;
    }

    std::vector<U> seeds_for_row(U row, U& empty_vertex,
                                 std::vector<U>& touched,
                                 std::vector<U>& marks,
                                 std::vector<W>& effective,
                                 U generation) const {
        touched.clear();
        empty_vertex = NIL;
        for (U pin = packet.row_begin(row); pin < packet.row_end(row); ++pin) {
            U vertex = packet.pin_vertex(pin);
            if (marks[vertex] != generation) {
                marks[vertex] = generation;
                effective[vertex] = packet.masks[vertex];
                touched.push_back(vertex);
            }
            effective[vertex] &= packet.pin_allowed(pin);
        }
        std::vector<U> seeds;
        seeds.reserve(touched.size());
        for (U vertex : touched) {
            W mask = effective[vertex];
            if (mask == 0) {
                empty_vertex = vertex;
                return seeds;
            }
            if (bit_count(packet.masks[vertex]) == 2 && bit_count(mask) == 1) {
                U colour = first_bit(mask);
                U literal = 2 * vertex + (high_colour[vertex] == colour ? 1 : 0);
                seeds.push_back(literal);
            }
        }
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
        return seeds;
    }

    std::vector<U> find_path_from_seed(const std::vector<U>& seeds, U target) const {
        if (closure_enabled) {
            U target_component = component[target];
            for (U seed : seeds) {
                const W* row = closure.data() + W(component[seed]) * closure_words;
                if ((row[target_component >> 6] >> (target_component & 63U)) & 1U)
                    return shortest_path(graph, seed, target);
            }
        } else {
            for (U seed : seeds) {
                try {
                    return shortest_path(graph, seed, target);
                } catch (const std::logic_error&) {
                }
            }
        }
        throw std::logic_error("no forced root reaches contradiction target");
    }

    void solve_row(U row, std::vector<std::uint8_t>& labels,
                   std::vector<std::uint8_t>& certificate,
                   std::vector<U>& touched, std::vector<U>& marks,
                   std::vector<W>& effective, U generation) const {
        U empty_vertex = NIL;
        std::vector<U> seeds = seeds_for_row(row, empty_vertex, touched, marks, effective, generation);
        if (empty_vertex != NIL) {
            certificate = make_proof(1, std::vector<U>{empty_vertex}, {}, std::numeric_limits<W>::max());
            labels.clear();
            return;
        }
        if (base_unsat) {
            certificate = make_proof(3, base_left_path, base_right_path, std::numeric_limits<W>::max());
            labels.clear();
            return;
        }

        U conflict_literal = NIL;
        std::vector<W> reached_bits(closure_words, 0);
        std::vector<U> reached_components;
        std::vector<U> root_literal(component_count, NIL);
        if (closure_enabled) {
            for (U seed : seeds) {
                const W* source = closure.data() + W(component[seed]) * closure_words;
                for (U word = 0; word < closure_words; ++word) reached_bits[word] |= source[word];
            }
            for (U v = 0; v < packet.n; ++v) {
                if (reached(reached_bits, component[2 * v]) &&
                    reached(reached_bits, component[2 * v + 1])) {
                    conflict_literal = 2 * v;
                    break;
                }
            }
        } else {
            std::vector<std::uint8_t> seen(component_count, 0);
            std::deque<U> queue;
            for (U seed : seeds) {
                U id = component[seed];
                if (!seen[id]) {
                    seen[id] = 1;
                    root_literal[id] = seed;
                    queue.push_back(id);
                    reached_components.push_back(id);
                }
            }
            while (!queue.empty()) {
                U current = queue.front();
                queue.pop_front();
                U opposite = complement_component[current];
                if (seen[opposite]) {
                    for (U v = 0; v < packet.n; ++v) {
                        U a = component[2 * v], b = component[2 * v + 1];
                        if ((a == current && b == opposite) || (b == current && a == opposite)) {
                            conflict_literal = 2 * v;
                            break;
                        }
                    }
                    if (conflict_literal != NIL) break;
                }
                for (U at = dag.offsets[current]; at < dag.offsets[current + 1]; ++at) {
                    U next = dag.targets[at];
                    if (!seen[next]) {
                        seen[next] = 1;
                        root_literal[next] = root_literal[current];
                        queue.push_back(next);
                        reached_components.push_back(next);
                    }
                }
            }
            reached_bits.assign(closure_words, 0);
            for (U id : reached_components) reached_bits[id >> 6] |= W(1) << (id & 63U);
        }

        if (conflict_literal != NIL) {
            std::vector<U> left = find_path_from_seed(seeds, conflict_literal);
            std::vector<U> right = find_path_from_seed(seeds, conflict_literal ^ 1U);
            certificate = make_proof(2, left, right, std::numeric_limits<W>::max());
            labels.clear();
            return;
        }

        labels.resize(packet.n);
        for (U v = 0; v < packet.n; ++v) {
            if (reached(reached_bits, component[2 * v])) labels[v] = static_cast<std::uint8_t>(low_colour[v]);
            else if (reached(reached_bits, component[2 * v + 1])) {
                if (high_colour[v] == NIL) throw std::logic_error("virtual singleton alternative became true");
                labels[v] = static_cast<std::uint8_t>(high_colour[v]);
            } else labels[v] = base_labels[v];
        }
        certificate.clear();
    }
};

struct Observer {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    U n = 0, palette = 0, edge_count = 0, query_count = 0, pin_count = 0;
    std::size_t masks_at = 32, edges_at = 0, offsets_at = 0, vertices_at = 0, allowed_at = 0;
    std::vector<W> masks;
    std::vector<std::pair<U, U>> edges;
    std::vector<U> low_colour;
    std::vector<U> high_colour;

    Observer(const std::uint8_t* bytes, std::size_t length) : data(bytes), size(length) {
        if (data == nullptr || size < 36 || data[0] != 'S' || data[1] != 'B' ||
            data[2] != 'C' || data[3] != '1') throw Invalid("observer rejected SBC1 magic");
        n = load32(data, 4); palette = load32(data, 8); edge_count = load32(data, 12);
        query_count = load32(data, 16); pin_count = load32(data, 20);
        if (n > MAX_N || palette == 0 || palette > 64 || edge_count > MAX_E ||
            query_count > MAX_Q || pin_count > MAX_Q || load32(data, 24) != 0 ||
            load32(data, 28) != 0) throw Invalid("observer rejected SBC1 header");
        W expected = 32 + 8 * W(n) + 8 * W(edge_count) + 4 * W(query_count + 1) +
                     4 * W(pin_count) + 8 * W(pin_count);
        if (expected != size) throw Invalid("observer rejected SBC1 length");
        edges_at = masks_at + 8 * W(n);
        offsets_at = edges_at + 8 * W(edge_count);
        vertices_at = offsets_at + 4 * W(query_count + 1);
        allowed_at = vertices_at + 4 * W(pin_count);
        W allowed_palette = palette_mask(palette);
        masks.reserve(n); low_colour.resize(n); high_colour.resize(n, NIL);
        for (U v = 0; v < n; ++v) {
            W mask = load64(data, masks_at + 8 * W(v));
            if (mask == 0 || (mask & ~allowed_palette) != 0 || bit_count(mask) > 2)
                throw Invalid("observer rejected original mask");
            masks.push_back(mask);
            low_colour[v] = first_bit(mask);
            mask &= mask - 1;
            if (mask) high_colour[v] = first_bit(mask);
        }
        edges.reserve(edge_count);
        std::pair<U, U> previous{0, 0}; bool have = false;
        for (U i = 0; i < edge_count; ++i) {
            U left = load32(data, edges_at + 8 * W(i));
            U right = load32(data, edges_at + 8 * W(i) + 4);
            std::pair<U, U> edge{left, right};
            if (left >= right || right >= n || (have && !(previous < edge)))
                throw Invalid("observer rejected edge bank");
            previous = edge; have = true; edges.push_back(edge);
        }
        U prior = 0;
        for (U i = 0; i <= query_count; ++i) {
            U next = load32(data, offsets_at + 4 * W(i));
            if ((i == 0 && next != 0) || next < prior || next > pin_count)
                throw Invalid("observer rejected offsets");
            prior = next;
        }
        if (prior != pin_count) throw Invalid("observer offsets do not consume pins");
        for (U i = 0; i < pin_count; ++i) if (load32(data, vertices_at + 4 * W(i)) >= n)
            throw Invalid("observer rejected pin vertex");
    }

    U row_begin(U row) const { return load32(data, offsets_at + 4 * W(row)); }
    U row_end(U row) const { return load32(data, offsets_at + 4 * W(row + 1)); }
    U pin_vertex(U pin) const { return load32(data, vertices_at + 4 * W(pin)); }
    W pin_allowed(U pin) const { return load64(data, allowed_at + 8 * W(pin)); }

    U colour(U literal) const {
        U vertex = literal >> 1;
        return (literal & 1U) == 0 ? low_colour[vertex] : high_colour[vertex];
    }

    bool edge_exists(U left, U right) const {
        if (left > right) std::swap(left, right);
        return std::binary_search(edges.begin(), edges.end(), std::make_pair(left, right));
    }

    bool verify_path(const std::vector<U>& path) const {
        if (path.empty() || path.size() > 2 * W(n)) return false;
        std::vector<U> sorted = path;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return false;
        for (U literal : path) if (literal >= 2 * n) return false;
        for (std::size_t i = 1; i < path.size(); ++i) {
            U before = path[i - 1], after = path[i];
            U u = before >> 1, v = after >> 1;
            if (u == v) {
                if (before != 2 * u + 1 || after != 2 * u || high_colour[u] != NIL) return false;
            } else {
                U before_colour = colour(before);
                U excluded_colour = colour(after ^ 1U);
                if (before_colour == NIL || excluded_colour == NIL ||
                    before_colour != excluded_colour || !edge_exists(u, v)) return false;
            }
        }
        return true;
    }

    bool verify(const std::vector<std::uint8_t>& output) const {
        if (output.size() < 16 || std::memcmp(output.data(), "SCQ2", 4) != 0 ||
            load32(output.data(), 4) != n || load64(output.data(), 8) != query_count)
            return false;
        std::size_t at = 16;
        std::vector<U> marks(n, 0);
        std::vector<W> effective(n, 0);
        U generation = 1;
        for (U row = 0; row < query_count; ++row, ++generation) {
            if (generation == 0) { std::fill(marks.begin(), marks.end(), 0); generation = 1; }
            if (at >= output.size()) return false;
            std::uint8_t status = output[at++];
            if (output.size() - at < n + 4) return false;
            const std::uint8_t* labels = output.data() + at;
            at += n;
            U proof_bytes = load32(output.data(), at);
            at += 4;
            if (output.size() - at < proof_bytes) return false;
            const std::uint8_t* proof = output.data() + at;
            at += proof_bytes;

            for (U pin = row_begin(row); pin < row_end(row); ++pin) {
                U vertex = pin_vertex(pin);
                if (marks[vertex] != generation) {
                    marks[vertex] = generation;
                    effective[vertex] = masks[vertex];
                }
                effective[vertex] &= pin_allowed(pin);
            }
            auto allowed = [&](U vertex) -> W {
                return marks[vertex] == generation ? effective[vertex] : masks[vertex];
            };

            if (status == 1) {
                if (proof_bytes != 0) return false;
                for (U v = 0; v < n; ++v) {
                    U label = labels[v];
                    if (label >= palette || ((allowed(v) >> label) & 1U) == 0) return false;
                }
                for (const auto& edge : edges) if (labels[edge.first] == labels[edge.second]) return false;
            } else if (status == 0) {
                for (U v = 0; v < n; ++v) if (labels[v] != 0) return false;
                if (proof_bytes < 16 || std::memcmp(proof, "SBP1", 4) != 0) return false;
                U kind = load32(proof, 4), left_count = load32(proof, 8), right_count = load32(proof, 12);
                if (16 + 4 * W(left_count + right_count) != proof_bytes) return false;
                std::vector<U> left(left_count), right(right_count);
                for (U i = 0; i < left_count; ++i) left[i] = load32(proof, 16 + 4 * W(i));
                for (U i = 0; i < right_count; ++i)
                    right[i] = load32(proof, 16 + 4 * W(left_count + i));
                if (kind == 1) {
                    if (left_count != 1 || right_count != 0 || left[0] >= n || allowed(left[0]) != 0)
                        return false;
                } else if (kind == 2) {
                    if (!verify_path(left) || !verify_path(right) ||
                        left.back() != (right.back() ^ 1U)) return false;
                    U roots[2] = {left.front(), right.front()};
                    for (U root : roots) {
                        U root_colour = colour(root);
                        if (root_colour == NIL || allowed(root >> 1) != (W(1) << root_colour)) return false;
                    }
                } else if (kind == 3) {
                    if (!verify_path(left) || !verify_path(right) ||
                        left.front() != right.back() || left.back() != right.front() ||
                        left.back() != (left.front() ^ 1U)) return false;
                } else return false;
            } else return false;
        }
        return at == output.size();
    }
};

struct StatsInternal {
    U n = 0, edges = 0, queries = 0, pins = 0, components = 0, dag_edges = 0;
    U sat = 0, unsat = 0, engine = 0;
    W closure_bytes = 0, output_bytes = 0, proof_bytes = 0;
};

std::vector<std::uint8_t> solve_packet(const std::uint8_t* data, std::size_t size,
                                       W cap, StatsInternal& stats) {
    Packet packet(data, size, cap);
    Solver solver(packet, cap);
    Observer observer(data, size);
    stats.n = packet.n; stats.edges = packet.edge_count; stats.queries = packet.query_count;
    stats.pins = packet.pin_count; stats.components = solver.component_count;
    stats.dag_edges = static_cast<U>(solver.dag_arcs.size());
    stats.engine = solver.closure_enabled ? 1U : 2U;
    stats.closure_bytes = 8 * W(solver.closure.size());

    std::vector<std::uint8_t> output;
    W minimum = 16 + W(packet.query_count) * (1 + W(packet.n) + 4);
    if (minimum > cap) throw Resource("minimum SCQ2 output exceeds max_bytes");
    output.reserve(static_cast<std::size_t>(std::min<W>(minimum + 4096, cap)));
    output.insert(output.end(), {'S', 'C', 'Q', '2'});
    put32(output, packet.n, cap);
    put64(output, packet.query_count, cap);

    std::vector<U> touched;
    std::vector<U> marks(packet.n, 0);
    std::vector<W> effective(packet.n, 0);
    U generation = 1;
    for (U row = 0; row < packet.query_count; ++row, ++generation) {
        if (generation == 0) { std::fill(marks.begin(), marks.end(), 0); generation = 1; }
        std::vector<std::uint8_t> labels;
        std::vector<std::uint8_t> certificate;
        solver.solve_row(row, labels, certificate, touched, marks, effective, generation);
        bool sat = !labels.empty() || (packet.n == 0 && certificate.empty());
        output.push_back(sat ? 1 : 0);
        if (sat) {
            if (labels.size() != packet.n) throw std::logic_error("SAT label count differs");
            if (output.size() > cap || W(labels.size()) > cap - output.size())
                throw Resource("SCQ2 output exceeds max_bytes");
            output.insert(output.end(), labels.begin(), labels.end());
            put32(output, 0, cap);
            ++stats.sat;
        } else {
            if (output.size() > cap || W(packet.n) > cap - output.size())
                throw Resource("SCQ2 output exceeds max_bytes");
            output.insert(output.end(), packet.n, 0);
            put32(output, static_cast<U>(certificate.size()), cap);
            put_bytes(output, certificate, cap);
            ++stats.unsat;
            stats.proof_bytes += certificate.size();
        }
    }
    if (!observer.verify(output)) throw std::logic_error("independent observer rejected direct 2-SAT output");
    stats.output_bytes = output.size();
    return output;
}

} // namespace

extern "C" {

struct Direct2SATStats {
    std::uint32_t n;
    std::uint32_t edges;
    std::uint32_t queries;
    std::uint32_t pins;
    std::uint32_t components;
    std::uint32_t dag_edges;
    std::uint32_t sat;
    std::uint32_t unsat;
    std::uint32_t engine;
    std::uint64_t closure_bytes;
    std::uint64_t output_bytes;
    std::uint64_t proof_bytes;
};

D2S_EXPORT const char* direct_twosat_error() {
    return LAST_ERROR.c_str();
}

D2S_EXPORT void direct_twosat_free(void* pointer) {
    std::free(pointer);
}

D2S_EXPORT int direct_twosat_solve(const std::uint8_t* packet, std::size_t packet_size,
                                   std::uint64_t max_bytes, std::uint8_t** output,
                                   std::size_t* output_size, Direct2SATStats* public_stats) {
    if (output == nullptr || output_size == nullptr || public_stats == nullptr) return 2;
    *output = nullptr;
    *output_size = 0;
    std::memset(public_stats, 0, sizeof(*public_stats));
    try {
        StatsInternal stats;
        std::vector<std::uint8_t> result = solve_packet(packet, packet_size, max_bytes, stats);
        void* memory = std::malloc(result.size() == 0 ? 1 : result.size());
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
        LAST_ERROR = "unknown direct 2-SAT failure";
        return 3;
    }
}

} // extern "C"
