#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {
using U = std::uint32_t;
using W = std::uint64_t;
constexpr U NIL = std::numeric_limits<U>::max();
constexpr U MAX_N = 100000;
constexpr U MAX_E = 2000000;
constexpr U MAX_ROWS = 1000000;
constexpr U MAX_PINS = 1000000;
constexpr U ABI = 1;

struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };

#if defined(_MSC_VER)
U first_bit(W value) {
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<U>(index);
}
U popcount(W value) {
    U count = 0;
    while (value) { value &= value - 1; ++count; }
    return count;
}
#else
U first_bit(W value) { return static_cast<U>(__builtin_ctzll(value)); }
U popcount(W value) { return static_cast<U>(__builtin_popcountll(value)); }
#endif

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
void put32(std::vector<std::uint8_t>& out, U value) {
    for (U i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void put64(std::vector<std::uint8_t>& out, W value) {
    for (U i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

struct CSR {
    std::vector<U> offset;
    std::vector<U> target;

    CSR() = default;
    CSR(U vertices, const std::vector<std::pair<U, U>>& arcs, bool reverse = false)
        : offset(vertices + 1, 0), target(arcs.size()) {
        for (auto [from, to] : arcs) {
            ++offset[(reverse ? to : from) + 1];
        }
        for (U vertex = 0; vertex < vertices; ++vertex) {
            offset[vertex + 1] += offset[vertex];
        }
        auto cursor = offset;
        for (auto [from, to] : arcs) {
            U source = reverse ? to : from;
            U destination = reverse ? from : to;
            target[cursor[source]++] = destination;
        }
    }
};

struct Packet {
    const std::uint8_t* data;
    std::size_t size;
    U n = 0;
    U palette = 0;
    U edge_count = 0;
    U row_count = 0;
    U pin_count = 0;
    std::size_t masks_at = 32;
    std::size_t edges_at = 0;
    std::size_t offsets_at = 0;
    std::size_t vertices_at = 0;
    std::size_t allowed_at = 0;
    std::vector<W> masks;
    std::vector<std::pair<U, U>> edges;

    Packet(const std::uint8_t* bytes, std::size_t length, W max_bytes)
        : data(bytes), size(length) {
        if (size < 36 || std::memcmp(data, "SBC1", 4) != 0) {
            throw Invalid("invalid SBC1 packet");
        }
        n = load32(data, 4);
        palette = load32(data, 8);
        edge_count = load32(data, 12);
        row_count = load32(data, 16);
        pin_count = load32(data, 20);
        if (n > MAX_N || palette == 0 || palette > 64 || edge_count > MAX_E ||
            row_count > MAX_ROWS || pin_count > MAX_PINS || load32(data, 24) != 0 ||
            load32(data, 28) != 0) {
            throw Invalid("SBC1 header outside supported bounds");
        }
        W expected = 36 + 8 * W(n) + 8 * W(edge_count) + 4 * W(row_count) +
                     12 * W(pin_count);
        if (expected != size) throw Invalid("SBC1 packet length differs");
        if (W(size) > max_bytes) throw Resource("SBC1 packet exceeds memory cap");

        edges_at = masks_at + 8 * W(n);
        offsets_at = edges_at + 8 * W(edge_count);
        vertices_at = offsets_at + 4 * W(row_count + 1);
        allowed_at = vertices_at + 4 * W(pin_count);

        W palette_mask = palette == 64 ? std::numeric_limits<W>::max()
                                       : ((W(1) << palette) - 1);
        masks.reserve(n);
        for (U vertex = 0; vertex < n; ++vertex) {
            W mask = load64(data, masks_at + 8 * W(vertex));
            if (mask == 0 || (mask & ~palette_mask) != 0 || popcount(mask) > 2) {
                throw Invalid("SBC1 domains must contain one or two palette colours");
            }
            masks.push_back(mask);
        }

        edges.reserve(edge_count);
        U last_left = 0;
        U last_right = 0;
        bool have_edge = false;
        for (U index = 0; index < edge_count; ++index) {
            U left = load32(data, edges_at + 8 * W(index));
            U right = load32(data, edges_at + 8 * W(index) + 4);
            bool ordered = !have_edge || left > last_left ||
                           (left == last_left && right > last_right);
            if (left >= right || right >= n || !ordered) {
                throw Invalid("SBC1 edges must be sorted unique pairs");
            }
            edges.emplace_back(left, right);
            last_left = left;
            last_right = right;
            have_edge = true;
        }

        U previous = load32(data, offsets_at);
        if (previous != 0) throw Invalid("SBC1 offsets must start at zero");
        for (U row = 1; row <= row_count; ++row) {
            U next = load32(data, offsets_at + 4 * W(row));
            if (next < previous || next > pin_count) {
                throw Invalid("SBC1 offsets differ");
            }
            previous = next;
        }
        if (previous != pin_count) throw Invalid("SBC1 offsets do not consume pins");
        for (U pin = 0; pin < pin_count; ++pin) {
            if (load32(data, vertices_at + 4 * W(pin)) >= n) {
                throw Invalid("SBC1 query vertex outside graph");
            }
        }
    }

    U row_begin(U row) const { return load32(data, offsets_at + 4 * W(row)); }
    U row_end(U row) const { return load32(data, offsets_at + 4 * W(row + 1)); }
    U pin_vertex(U pin) const { return load32(data, vertices_at + 4 * W(pin)); }
    W pin_allowed(U pin) const { return load64(data, allowed_at + 8 * W(pin)); }
};

std::vector<U> strongly_connected(const CSR& graph, const CSR& reverse) {
    U vertices = static_cast<U>(graph.offset.size() - 1);
    std::vector<std::uint8_t> seen(vertices, 0);
    std::vector<U> order;
    order.reserve(vertices);
    std::vector<std::pair<U, U>> stack;
    stack.reserve(vertices);

    for (U root = 0; root < vertices; ++root) {
        if (seen[root]) continue;
        seen[root] = 1;
        stack.emplace_back(root, graph.offset[root]);
        while (!stack.empty()) {
            auto& frame = stack.back();
            U vertex = frame.first;
            if (frame.second == graph.offset[vertex + 1]) {
                order.push_back(vertex);
                stack.pop_back();
                continue;
            }
            U next = graph.target[frame.second++];
            if (!seen[next]) {
                seen[next] = 1;
                stack.emplace_back(next, graph.offset[next]);
            }
        }
    }

    std::vector<U> component(vertices, NIL);
    std::vector<U> work;
    work.reserve(vertices);
    U id = 0;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (component[*it] != NIL) continue;
        component[*it] = id;
        work.push_back(*it);
        while (!work.empty()) {
            U vertex = work.back();
            work.pop_back();
            for (U edge = reverse.offset[vertex]; edge < reverse.offset[vertex + 1]; ++edge) {
                U next = reverse.target[edge];
                if (component[next] == NIL) {
                    component[next] = id;
                    work.push_back(next);
                }
            }
        }
        ++id;
    }
    return component;
}

std::vector<std::uint8_t> make_proof(
    U kind,
    const std::vector<U>& first,
    const std::vector<U>& second
) {
    std::vector<std::uint8_t> output;
    output.reserve(16 + 4 * (first.size() + second.size()));
    output.insert(output.end(), {'S', 'B', 'P', '1'});
    put32(output, kind);
    put32(output, static_cast<U>(first.size()));
    put32(output, static_cast<U>(second.size()));
    for (U literal : first) put32(output, literal);
    for (U literal : second) put32(output, literal);
    return output;
}

std::vector<U> trace_path(const std::vector<U>& parent, U literal) {
    std::vector<U> path;
    while (true) {
        path.push_back(literal);
        if (parent[literal] == literal) break;
        if (parent[literal] == NIL || path.size() > parent.size()) {
            throw std::logic_error("invalid implication parent chain");
        }
        literal = parent[literal];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

struct Counters {
    W visits = 0;
    W arcs = 0;
};

struct QueryState {
    std::vector<U> domain_epoch;
    std::vector<W> effective;
    std::vector<U> touched;
    U epoch = 0;

    explicit QueryState(U n) : domain_epoch(n, 0), effective(n, 0) {
        touched.reserve(std::min<U>(n, 256));
    }

    U next_epoch() {
        if (++epoch == 0) {
            std::fill(domain_epoch.begin(), domain_epoch.end(), 0);
            epoch = 1;
        }
        touched.clear();
        return epoch;
    }
};

struct Baseline {
    U n;
    U palette;
    U components = 0;
    U bad_vertex = NIL;
    W max_bytes;
    W arc_count = 0;
    std::vector<W> masks;
    std::vector<std::uint8_t> low;
    std::vector<std::uint8_t> high;
    std::vector<std::uint8_t> base_side;
    CSR graph;
    CSR reverse;
    std::vector<U> component;
    std::vector<U> opposite;
    CSR dag;
    QueryState query_state;
    std::vector<U> reached_epoch;
    std::vector<U> parent_literal;
    std::vector<U> queue_literal;
    U literal_epoch = 0;
    std::vector<U> component_epoch;
    std::vector<U> queue_component;
    U comp_epoch = 0;
    std::vector<std::uint8_t> base_proof;

    explicit Baseline(const Packet& packet, W cap)
        : n(packet.n), palette(packet.palette), max_bytes(cap), masks(packet.masks),
          low(n), high(n, 255), base_side(n, 0), query_state(n),
          reached_epoch(2 * n, 0), parent_literal(2 * n, NIL),
          component_epoch(), queue_component() {
        std::vector<std::pair<U, U>> arcs;
        arcs.reserve(4 * packet.edges.size() + n);
        for (U vertex = 0; vertex < n; ++vertex) {
            low[vertex] = static_cast<std::uint8_t>(first_bit(masks[vertex]));
            W remaining = masks[vertex] & (masks[vertex] - 1);
            if (remaining) {
                high[vertex] = static_cast<std::uint8_t>(first_bit(remaining));
            } else {
                arcs.emplace_back(2 * vertex + 1, 2 * vertex);
            }
        }
        for (auto [left, right] : packet.edges) {
            W shared = masks[left] & masks[right];
            while (shared) {
                U colour = first_bit(shared);
                shared &= shared - 1;
                U a = 2 * left + (colour != low[left]);
                U b = 2 * right + (colour != low[right]);
                arcs.emplace_back(a, b ^ 1U);
                arcs.emplace_back(b, a ^ 1U);
            }
        }
        std::sort(arcs.begin(), arcs.end());
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());
        arc_count = arcs.size();
        graph = CSR(2 * n, arcs, false);
        reverse = CSR(2 * n, arcs, true);
        component = strongly_connected(graph, reverse);
        for (U id : component) components = std::max(components, id + 1);
        for (U vertex = 0; vertex < n; ++vertex) {
            if (component[2 * vertex] == component[2 * vertex + 1] && bad_vertex == NIL) {
                bad_vertex = vertex;
            }
            base_side[vertex] = static_cast<std::uint8_t>(
                component[2 * vertex] < component[2 * vertex + 1]
            );
        }
        opposite.assign(components, NIL);
        for (U literal = 0; literal < 2 * n; ++literal) {
            U here = component[literal];
            U other = component[literal ^ 1U];
            if (opposite[here] != NIL && opposite[here] != other) {
                throw std::logic_error("component complement mapping differs");
            }
            opposite[here] = other;
        }
        std::vector<std::pair<U, U>> links;
        links.reserve(arcs.size());
        for (auto [from, to] : arcs) {
            U left = component[from];
            U right = component[to];
            if (left != right) links.emplace_back(left, right);
        }
        std::sort(links.begin(), links.end());
        links.erase(std::unique(links.begin(), links.end()), links.end());
        dag = CSR(components, links, false);
        component_epoch.assign(components, 0);
        queue_literal.reserve(2 * n);
        queue_component.reserve(components);

        W retained = 8 * W(masks.capacity()) + low.capacity() + high.capacity() +
                     base_side.capacity() + 4 * W(component.capacity() + opposite.capacity() +
                     reached_epoch.capacity() + parent_literal.capacity() + component_epoch.capacity()) +
                     4 * W(graph.offset.capacity() + graph.target.capacity() +
                           reverse.offset.capacity() + reverse.target.capacity() +
                           dag.offset.capacity() + dag.target.capacity());
        if (retained > max_bytes) throw Resource("2-SAT baseline index exceeds memory cap");
        if (bad_vertex != NIL) base_proof = contradiction_path(2 * bad_vertex);
    }

    U next_literal_epoch() {
        if (++literal_epoch == 0) {
            std::fill(reached_epoch.begin(), reached_epoch.end(), 0);
            literal_epoch = 1;
        }
        queue_literal.clear();
        return literal_epoch;
    }

    U next_component_epoch() {
        if (++comp_epoch == 0) {
            std::fill(component_epoch.begin(), component_epoch.end(), 0);
            comp_epoch = 1;
        }
        queue_component.clear();
        return comp_epoch;
    }

    std::vector<U> bfs_path(U source, U target, Counters& counters) const {
        std::vector<U> parent(2 * n, NIL);
        std::vector<U> queue;
        queue.reserve(2 * n);
        parent[source] = source;
        queue.push_back(source);
        for (std::size_t head = 0; head < queue.size() && parent[target] == NIL; ++head) {
            U vertex = queue[head];
            ++counters.visits;
            for (U edge = graph.offset[vertex]; edge < graph.offset[vertex + 1]; ++edge) {
                ++counters.arcs;
                U next = graph.target[edge];
                if (parent[next] == NIL) {
                    parent[next] = vertex;
                    queue.push_back(next);
                }
            }
        }
        if (parent[target] == NIL) throw std::logic_error("implication path missing");
        return trace_path(parent, target);
    }

    std::vector<std::uint8_t> contradiction_path(U literal) const {
        Counters ignored;
        auto forward = bfs_path(literal, literal ^ 1U, ignored);
        auto backward = bfs_path(literal ^ 1U, literal, ignored);
        return make_proof(3, forward, backward);
    }

    bool prepare_seeds(
        const Packet& packet,
        U row,
        std::vector<U>& seeds,
        U& empty_vertex
    ) {
        U epoch = query_state.next_epoch();
        seeds.clear();
        empty_vertex = NIL;
        for (U pin = packet.row_begin(row); pin < packet.row_end(row); ++pin) {
            U vertex = packet.pin_vertex(pin);
            if (query_state.domain_epoch[vertex] != epoch) {
                query_state.domain_epoch[vertex] = epoch;
                query_state.effective[vertex] = masks[vertex];
                query_state.touched.push_back(vertex);
            }
            query_state.effective[vertex] &= packet.pin_allowed(pin);
        }
        std::sort(query_state.touched.begin(), query_state.touched.end());
        for (U vertex : query_state.touched) {
            W domain = query_state.effective[vertex];
            if (domain == 0) {
                empty_vertex = vertex;
                return false;
            }
            if (domain != masks[vertex]) {
                U side = first_bit(domain) != low[vertex];
                seeds.push_back(2 * vertex + side);
            }
        }
        return true;
    }

    std::vector<std::uint8_t> conflict_from_seeds(
        const std::vector<U>& seeds,
        Counters& counters
    ) {
        U epoch = next_literal_epoch();
        U conflict = NIL;
        for (U seed : seeds) {
            if (reached_epoch[seed] == epoch) continue;
            reached_epoch[seed] = epoch;
            parent_literal[seed] = seed;
            queue_literal.push_back(seed);
            if (reached_epoch[seed ^ 1U] == epoch) {
                conflict = seed;
                break;
            }
        }
        for (std::size_t head = 0; conflict == NIL && head < queue_literal.size(); ++head) {
            U vertex = queue_literal[head];
            ++counters.visits;
            for (U edge = graph.offset[vertex]; edge < graph.offset[vertex + 1]; ++edge) {
                ++counters.arcs;
                U next = graph.target[edge];
                if (reached_epoch[next] != epoch) {
                    reached_epoch[next] = epoch;
                    parent_literal[next] = vertex;
                    queue_literal.push_back(next);
                    if (reached_epoch[next ^ 1U] == epoch) {
                        conflict = next;
                        break;
                    }
                }
            }
        }
        if (conflict == NIL) return {};
        auto first = trace_path(parent_literal, conflict);
        auto second = trace_path(parent_literal, conflict ^ 1U);
        return make_proof(2, first, second);
    }

    bool raw_reach(const std::vector<U>& seeds, Counters& counters, U& epoch) {
        epoch = next_literal_epoch();
        for (U seed : seeds) {
            if (reached_epoch[seed] == epoch) continue;
            reached_epoch[seed] = epoch;
            parent_literal[seed] = seed;
            queue_literal.push_back(seed);
            if (reached_epoch[seed ^ 1U] == epoch) return false;
        }
        for (std::size_t head = 0; head < queue_literal.size(); ++head) {
            U vertex = queue_literal[head];
            ++counters.visits;
            for (U edge = graph.offset[vertex]; edge < graph.offset[vertex + 1]; ++edge) {
                ++counters.arcs;
                U next = graph.target[edge];
                if (reached_epoch[next] == epoch) continue;
                reached_epoch[next] = epoch;
                parent_literal[next] = vertex;
                queue_literal.push_back(next);
                if (reached_epoch[next ^ 1U] == epoch) return false;
            }
        }
        return true;
    }

    bool component_reach(const std::vector<U>& seeds, Counters& counters, U& epoch) {
        epoch = next_component_epoch();
        for (U seed : seeds) {
            U here = component[seed];
            if (component_epoch[here] == epoch) continue;
            component_epoch[here] = epoch;
            queue_component.push_back(here);
            if (component_epoch[opposite[here]] == epoch) return false;
        }
        for (std::size_t head = 0; head < queue_component.size(); ++head) {
            U here = queue_component[head];
            ++counters.visits;
            for (U edge = dag.offset[here]; edge < dag.offset[here + 1]; ++edge) {
                ++counters.arcs;
                U next = dag.target[edge];
                if (component_epoch[next] == epoch) continue;
                component_epoch[next] = epoch;
                queue_component.push_back(next);
                if (component_epoch[opposite[next]] == epoch) return false;
            }
        }
        return true;
    }

    std::vector<std::uint8_t> labels_from_raw(U epoch) const {
        std::vector<std::uint8_t> labels(n);
        for (U vertex = 0; vertex < n; ++vertex) {
            bool low_forced = reached_epoch[2 * vertex] == epoch;
            bool high_forced = reached_epoch[2 * vertex + 1] == epoch;
            U side = low_forced ? 0U : (high_forced ? 1U : base_side[vertex]);
            labels[vertex] = side ? high[vertex] : low[vertex];
        }
        return labels;
    }

    std::vector<std::uint8_t> labels_from_components(U epoch) const {
        std::vector<std::uint8_t> labels(n);
        for (U vertex = 0; vertex < n; ++vertex) {
            bool low_forced = component_epoch[component[2 * vertex]] == epoch;
            bool high_forced = component_epoch[component[2 * vertex + 1]] == epoch;
            U side = low_forced ? 0U : (high_forced ? 1U : base_side[vertex]);
            labels[vertex] = side ? high[vertex] : low[vertex];
        }
        return labels;
    }
};

struct RowAnswer {
    bool sat = false;
    std::vector<std::uint8_t> labels;
    std::vector<std::uint8_t> proof;
};

struct BankResult {
    std::vector<std::uint8_t> output;
    U sat = 0;
    U unsat = 0;
    W proof_bytes = 0;
    Counters counters;
    U tiles = 0;
    bool tiled = false;
    std::string selected_mode;
};

void append_answer(std::vector<std::uint8_t>& output, U n, const RowAnswer& answer) {
    output.push_back(answer.sat ? 1 : 0);
    if (answer.sat) {
        if (answer.labels.size() != n || !answer.proof.empty()) {
            throw std::logic_error("SAT row geometry differs");
        }
        output.insert(output.end(), answer.labels.begin(), answer.labels.end());
    } else {
        if (!answer.labels.empty()) throw std::logic_error("UNSAT labels must be empty");
        output.resize(output.size() + n, 0);
    }
    if (answer.proof.size() > std::numeric_limits<U>::max()) {
        throw Resource("proof exceeds SCQ2 bound");
    }
    put32(output, static_cast<U>(answer.proof.size()));
    output.insert(output.end(), answer.proof.begin(), answer.proof.end());
}

RowAnswer solve_scalar(
    Baseline& solver,
    const Packet& packet,
    U row,
    bool condensed,
    Counters& counters
) {
    if (solver.bad_vertex != NIL) {
        return RowAnswer{false, {}, solver.base_proof};
    }
    std::vector<U> seeds;
    seeds.reserve(packet.row_end(row) - packet.row_begin(row));
    U empty = NIL;
    if (!solver.prepare_seeds(packet, row, seeds, empty)) {
        return RowAnswer{false, {}, make_proof(1, {empty}, {})};
    }
    U epoch = 0;
    bool sat = condensed ? solver.component_reach(seeds, counters, epoch)
                         : solver.raw_reach(seeds, counters, epoch);
    if (!sat) {
        auto proof = solver.conflict_from_seeds(seeds, counters);
        if (proof.empty()) throw std::logic_error("query contradiction proof missing");
        return RowAnswer{false, {}, std::move(proof)};
    }
    return RowAnswer{
        true,
        condensed ? solver.labels_from_components(epoch) : solver.labels_from_raw(epoch),
        {},
    };
}

BankResult solve_serial(Baseline& solver, const Packet& packet, bool condensed) {
    BankResult result;
    result.selected_mode = condensed ? "scc" : "raw";
    W reserve = 16 + W(packet.row_count) * (W(packet.n) + 21);
    if (reserve > solver.max_bytes || reserve > std::numeric_limits<std::size_t>::max()) {
        throw Resource("SCQ2 output reserve exceeds memory cap");
    }
    result.output.reserve(static_cast<std::size_t>(reserve));
    result.output.insert(result.output.end(), {'S', 'C', 'Q', '2'});
    put32(result.output, packet.n);
    put64(result.output, packet.row_count);
    for (U row = 0; row < packet.row_count; ++row) {
        RowAnswer answer = solve_scalar(solver, packet, row, condensed, result.counters);
        if (answer.sat) ++result.sat;
        else { ++result.unsat; result.proof_bytes += answer.proof.size(); }
        append_answer(result.output, packet.n, answer);
        if (W(result.output.size()) > solver.max_bytes) {
            throw Resource("SCQ2 output exceeds memory cap");
        }
    }
    return result;
}

BankResult solve_tiled(Baseline& solver, const Packet& packet) {
    if (solver.bad_vertex != NIL || packet.row_count < 64) {
        return solve_serial(solver, packet, true);
    }
    BankResult result;
    result.selected_mode = "scc";
    result.tiled = true;
    W reserve = 16 + W(packet.row_count) * (W(packet.n) + 21);
    if (reserve > solver.max_bytes || reserve > std::numeric_limits<std::size_t>::max()) {
        throw Resource("SCQ2 output reserve exceeds memory cap");
    }
    result.output.reserve(static_cast<std::size_t>(reserve));
    result.output.insert(result.output.end(), {'S', 'C', 'Q', '2'});
    put32(result.output, packet.n);
    put64(result.output, packet.row_count);

    std::vector<W> reached(solver.components, 0);
    std::vector<std::vector<U>> seeds(64);
    std::vector<U> empty(64, NIL);
    for (auto& row : seeds) row.reserve(16);

    for (U first_row = 0; first_row < packet.row_count; first_row += 64) {
        ++result.tiles;
        U lanes = std::min<U>(64, packet.row_count - first_row);
        std::fill(reached.begin(), reached.end(), 0);
        std::fill(empty.begin(), empty.end(), NIL);
        for (U lane = 0; lane < lanes; ++lane) {
            seeds[lane].clear();
            if (!solver.prepare_seeds(packet, first_row + lane, seeds[lane], empty[lane])) {
                continue;
            }
            W bit = W(1) << lane;
            for (U literal : seeds[lane]) reached[solver.component[literal]] |= bit;
        }
        for (U component = 0; component < solver.components; ++component) {
            W active = reached[component];
            if (!active) continue;
            ++result.counters.visits;
            for (U edge = solver.dag.offset[component]; edge < solver.dag.offset[component + 1]; ++edge) {
                ++result.counters.arcs;
                reached[solver.dag.target[edge]] |= active;
            }
        }
        W conflicts = 0;
        for (U component = 0; component < solver.components; ++component) {
            conflicts |= reached[component] & reached[solver.opposite[component]];
        }
        for (U lane = 0; lane < lanes; ++lane) {
            RowAnswer answer;
            if (empty[lane] != NIL) {
                answer = RowAnswer{false, {}, make_proof(1, {empty[lane]}, {})};
            } else if ((conflicts >> lane) & 1U) {
                auto proof = solver.conflict_from_seeds(seeds[lane], result.counters);
                if (proof.empty()) throw std::logic_error("tile contradiction proof missing");
                answer = RowAnswer{false, {}, std::move(proof)};
            } else {
                answer.sat = true;
                answer.labels.resize(packet.n);
                W bit = W(1) << lane;
                for (U vertex = 0; vertex < packet.n; ++vertex) {
                    bool low_forced = (reached[solver.component[2 * vertex]] & bit) != 0;
                    bool high_forced = (reached[solver.component[2 * vertex + 1]] & bit) != 0;
                    U side = low_forced ? 0U : (high_forced ? 1U : solver.base_side[vertex]);
                    answer.labels[vertex] = side ? solver.high[vertex] : solver.low[vertex];
                }
            }
            if (answer.sat) ++result.sat;
            else { ++result.unsat; result.proof_bytes += answer.proof.size(); }
            append_answer(result.output, packet.n, answer);
            if (W(result.output.size()) > solver.max_bytes) {
                throw Resource("SCQ2 output exceeds memory cap");
            }
        }
    }
    return result;
}

struct Released {
    PyThreadState* state;
    Released() : state(PyEval_SaveThread()) {}
    ~Released() { PyEval_RestoreThread(state); }
};

PyObject* translate_exception() {
    try {
        throw;
    } catch (const Invalid& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const Resource& error) {
        PyErr_SetString(PyExc_MemoryError, error.what());
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown native 2-SAT failure");
    }
    return nullptr;
}

W exact_unsigned(PyObject* value, const char* message, W maximum) {
    if (!PyLong_CheckExact(value)) throw Invalid(message);
    W number = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw Invalid(message);
    }
    if (number > maximum) throw Invalid(message);
    return number;
}

PyObject* solve_session_packet(PyObject*, PyObject* args) {
    PyObject* packet_object = nullptr;
    PyObject* mode_object = nullptr;
    PyObject* cap_object = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &packet_object, &mode_object, &cap_object)) return nullptr;
    try {
        if (!PyBytes_CheckExact(packet_object)) {
            throw Invalid("session packet must be immutable exact bytes");
        }
        W mode = exact_unsigned(mode_object, "mode must be 0, 1 or 2", 2);
        W cap = exact_unsigned(cap_object, "invalid memory cap", std::numeric_limits<W>::max());
        if (cap < (W(1) << 20)) throw Invalid("memory cap must be at least 1 MiB");
        const auto* data = reinterpret_cast<const std::uint8_t*>(PyBytes_AS_STRING(packet_object));
        std::size_t size = static_cast<std::size_t>(PyBytes_GET_SIZE(packet_object));
        BankResult result;
        U vertices = 0;
        U components = 0;
        W original_arcs = 0;
        W condensation_arcs = 0;
        bool base_satisfiable = false;
        U row_count = 0;
        U pin_count = 0;
        {
            Released release;
            Packet packet(data, size, cap);
            Baseline solver(packet, cap);
            bool condensed = mode == 1;
            if (mode == 2) {
                condensed = packet.row_count >= 32 ||
                            (packet.row_count >= 4 && W(solver.components) * 4 <= W(packet.n) * 3);
            }
            result = condensed ? solve_tiled(solver, packet) : solve_serial(solver, packet, false);
            vertices = solver.n;
            components = solver.components;
            original_arcs = solver.arc_count;
            condensation_arcs = solver.dag.target.size();
            base_satisfiable = solver.bad_vertex == NIL;
            row_count = packet.row_count;
            pin_count = packet.pin_count;
        }
        PyObject* output = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(result.output.data()),
            static_cast<Py_ssize_t>(result.output.size())
        );
        if (!output) return nullptr;
        PyObject* metadata = Py_BuildValue(
            "{sI,sI,sK,sK,sO,ss,ss,sI,sI,sO}",
            "vertices", vertices,
            "components", components,
            "original_arcs", static_cast<unsigned long long>(original_arcs),
            "condensation_arcs", static_cast<unsigned long long>(condensation_arcs),
            "base_satisfiable", base_satisfiable ? Py_True : Py_False,
            "selected_mode", result.selected_mode.c_str(),
            "algorithm", "independent-static-scc-unit-closure-v1",
            "bank_tiles", result.tiles,
            "bank_query_count", row_count,
            "independent_implementation", Py_True
        );
        if (!metadata) {
            Py_DECREF(output);
            return nullptr;
        }
        PyObject* kernel = PyUnicode_FromString(result.tiled ? "tile64" : "serial");
        PyObject* pins = PyLong_FromUnsignedLong(pin_count);
        if (!kernel || !pins || PyDict_SetItemString(metadata, "bank_kernel", kernel) < 0 ||
            PyDict_SetItemString(metadata, "bank_pin_count", pins) < 0) {
            Py_XDECREF(kernel);
            Py_XDECREF(pins);
            Py_DECREF(output);
            Py_DECREF(metadata);
            return nullptr;
        }
        Py_DECREF(kernel);
        Py_DECREF(pins);
        return Py_BuildValue(
            "NKKKKKKN",
            output,
            static_cast<unsigned long long>(result.sat + result.unsat),
            static_cast<unsigned long long>(result.sat),
            static_cast<unsigned long long>(result.unsat),
            static_cast<unsigned long long>(result.proof_bytes),
            static_cast<unsigned long long>(result.counters.visits),
            static_cast<unsigned long long>(result.counters.arcs),
            metadata
        );
    } catch (...) {
        return translate_exception();
    }
}

PyObject* abi(PyObject*, PyObject*) { return PyLong_FromUnsignedLong(ABI); }

PyMethodDef methods[] = {
    {"solve_session_packet", solve_session_packet, METH_VARARGS, nullptr},
    {"abi", abi, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_spectra_twosat_baseline",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
}  // namespace

PyMODINIT_FUNC PyInit__spectra_twosat_baseline() { return PyModule_Create(&module); }
