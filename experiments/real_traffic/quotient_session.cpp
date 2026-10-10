// Exact compiled support sessions for binary quotient relations.
// Classical 2-SAT SCC evaluation; MIT, repository LICENSE.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U MAX_N = 100000;
constexpr W MAX_QUERIES = 1000000;
constexpr W MAX_RESTRICTIONS = 16000000;
struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };

W integer(PyObject* object, const char* message, W cap = UINT64_MAX) {
    if (!PyLong_CheckExact(object)) throw Invalid(message);
    W value = PyLong_AsUnsignedLongLong(object);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw Invalid(message);
    }
    if (value > cap) throw Invalid(message);
    return value;
}

struct Index {
    U n = 0;
    U qn = 0;
    U atoms = 0;
    W payload = 0;
    U words = 0;
    std::vector<U> node;
    std::vector<uint8_t> colour0;
    std::vector<uint8_t> colour1;
    std::vector<uint8_t> base_assumed;
    std::vector<U> forward_offsets;
    std::vector<U> forward_targets;
    std::vector<U> reverse_offsets;
    std::vector<U> reverse_targets;
    std::vector<U> topological;
    std::vector<W> closure;
    std::vector<W> base_forced;

    Index(PyObject* node_object, PyObject* colour0_object, PyObject* colour1_object,
          PyObject* palettes_object, PyObject* initial_object,
          PyObject* offsets_object, PyObject* arcs_object, W max_bytes) {
        if (!PyTuple_CheckExact(node_object) || !PyTuple_CheckExact(colour0_object)
                || !PyTuple_CheckExact(colour1_object)
                || !PyTuple_CheckExact(palettes_object)
                || !PyTuple_CheckExact(initial_object)
                || !PyTuple_CheckExact(offsets_object)
                || !PyTuple_CheckExact(arcs_object)) {
            throw Invalid("compiled support inputs must be exact tuples");
        }
        n = static_cast<U>(PyTuple_GET_SIZE(node_object));
        if (n > MAX_N) throw Resource("too many original vertices");
        if (PyTuple_GET_SIZE(colour0_object) != n || PyTuple_GET_SIZE(colour1_object) != n)
            throw Invalid("lift-map geometry differs");
        qn = static_cast<U>(PyTuple_GET_SIZE(palettes_object));
        if (!qn || qn > MAX_N || PyTuple_GET_SIZE(initial_object) != qn)
            throw Invalid("quotient geometry differs");
        atoms = 2 * qn;
        words = (atoms + 63) / 64;
        if (PyTuple_GET_SIZE(offsets_object) != static_cast<Py_ssize_t>(atoms + 1))
            throw Invalid("quotient offsets differ");
        const W arcs_count = static_cast<W>(PyTuple_GET_SIZE(arcs_object));
        const W required = 4 * W(n) + 2 * W(n) + W(atoms)
            + 8 * W(atoms + 1) + 8 * arcs_count
            + 8 * W(atoms) * words + 8 * words + 4096;
        if (required > max_bytes) throw Resource("compiled support index exceeds payload cap");

        node.resize(n);
        colour0.resize(n);
        colour1.resize(n);
        for (U v = 0; v < n; ++v) {
            node[v] = static_cast<U>(integer(PyTuple_GET_ITEM(node_object, v),
                                             "bad quotient owner", qn - 1));
            colour0[v] = static_cast<uint8_t>(integer(PyTuple_GET_ITEM(colour0_object, v),
                                                       "bad low colour", 255));
            colour1[v] = static_cast<uint8_t>(integer(PyTuple_GET_ITEM(colour1_object, v),
                                                       "bad high colour", 255));
            if (colour0[v] == colour1[v] || (colour0[v] == 255 && colour1[v] == 255))
                throw Invalid("noninvertible lift map");
        }
        base_assumed.assign(atoms, 0);
        for (U q = 0; q < qn; ++q) {
            W palette = integer(PyTuple_GET_ITEM(palettes_object, q), "bad palette");
            W initial = integer(PyTuple_GET_ITEM(initial_object, q), "bad initial domain");
            if (__builtin_popcountll(palette) != 2 || (initial & ~palette)
                    || __builtin_popcountll(initial) < 1) {
                throw Invalid("compiled support requires nonempty binary quotient domains");
            }
            // Public certificate palettes are canonical side masks 0b11 for binary nodes.
            if (palette != 3 || (initial != 1 && initial != 2 && initial != 3))
                throw Invalid("compiled support requires canonical binary quotient masks");
            if (initial == 1) base_assumed[2 * q] = 1;
            if (initial == 2) base_assumed[2 * q + 1] = 1;
        }

        std::vector<std::pair<U,U>> edges;
        edges.reserve(arcs_count);
        U previous = 0;
        for (U source = 0; source < atoms; ++source) {
            U begin = static_cast<U>(integer(PyTuple_GET_ITEM(offsets_object, source),
                                             "bad arc offset", arcs_count));
            U end = static_cast<U>(integer(PyTuple_GET_ITEM(offsets_object, source + 1),
                                           "bad arc offset", arcs_count));
            if (begin != previous || begin > end) throw Invalid("noncanonical arc offsets");
            previous = end;
            for (U index = begin; index < end; ++index) {
                PyObject* row = PyTuple_GET_ITEM(arcs_object, index);
                if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2)
                    throw Invalid("arc must be an exact target/mask pair");
                U target = static_cast<U>(integer(PyTuple_GET_ITEM(row, 0),
                                                  "bad arc target", qn - 1));
                W forbidden = integer(PyTuple_GET_ITEM(row, 1), "bad forbidden mask", 3);
                if (forbidden != 1 && forbidden != 2)
                    throw Invalid("compiled support requires one forbidden target side");
                U implied_side = forbidden == 1 ? 1 : 0;
                edges.emplace_back(source, 2 * target + implied_side);
            }
        }
        if (previous != arcs_count) throw Invalid("arc offsets do not consume arc bank");
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        // Exact binary implications must be closed under contraposition.
        for (auto edge : edges) {
            if (!std::binary_search(edges.begin(), edges.end(),
                                    std::make_pair(edge.second ^ 1U, edge.first ^ 1U)))
                throw Invalid("quotient implication bank lacks contraposition");
        }
        forward_offsets.assign(atoms + 1, 0);
        reverse_offsets.assign(atoms + 1, 0);
        for (auto edge : edges) {
            ++forward_offsets[edge.first + 1];
            ++reverse_offsets[edge.second + 1];
        }
        for (U i = 0; i < atoms; ++i) {
            forward_offsets[i + 1] += forward_offsets[i];
            reverse_offsets[i + 1] += reverse_offsets[i];
        }
        forward_targets.resize(edges.size());
        reverse_targets.resize(edges.size());
        std::vector<U> fc(forward_offsets), rc(reverse_offsets);
        for (auto edge : edges) {
            forward_targets[fc[edge.first]++] = edge.second;
            reverse_targets[rc[edge.second]++] = edge.first;
        }

        // The public quotient certificate already contracts implication SCCs.
        // Recheck that the residual relation is a DAG before compiling closure.
        std::vector<U> indegree(atoms, 0);
        for (auto edge : edges) ++indegree[edge.second];
        std::priority_queue<U, std::vector<U>, std::greater<U>> ready;
        for (U atom = 0; atom < atoms; ++atom) if (!indegree[atom]) ready.push(atom);
        topological.reserve(atoms);
        while (!ready.empty()) {
            U atom = ready.top();
            ready.pop();
            topological.push_back(atom);
            for (U i = forward_offsets[atom]; i < forward_offsets[atom + 1]; ++i)
                if (--indegree[forward_targets[i]] == 0) ready.push(forward_targets[i]);
        }
        if (topological.size() != atoms)
            throw Invalid("quotient certificate retains a residual implication cycle");
        closure.assign(W(atoms) * words, 0);
        for (auto it = topological.rbegin(); it != topological.rend(); ++it) {
            U atom = *it;
            W* row = closure.data() + W(atom) * words;
            row[atom / 64] |= W(1) << (atom % 64);
            for (U i = forward_offsets[atom]; i < forward_offsets[atom + 1]; ++i) {
                const W* child = closure.data() + W(forward_targets[i]) * words;
                for (U word = 0; word < words; ++word) row[word] |= child[word];
            }
        }
        base_forced.assign(words, 0);
        for (U atom = 0; atom < atoms; ++atom) if (base_assumed[atom]) {
            const W* row = closure.data() + W(atom) * words;
            for (U word = 0; word < words; ++word) base_forced[word] |= row[word];
        }
        constexpr W even_bits = UINT64_C(0x5555555555555555);
        for (W value : base_forced)
            if ((value & (value >> 1) & even_bits) != 0)
                throw Invalid("base quotient units are contradictory");

        payload = 4 * W(node.capacity() + forward_offsets.capacity()
                        + forward_targets.capacity() + reverse_offsets.capacity()
                        + reverse_targets.capacity() + topological.capacity())
            + 8 * W(closure.capacity() + base_forced.capacity())
            + colour0.capacity() + colour1.capacity() + base_assumed.capacity();
        if (payload > max_bytes) throw Resource("compiled support allocated payload exceeds cap");
    }
};

struct ParsedQueries {
    std::vector<U> offsets;
    std::vector<U> vertices;
    std::vector<W> masks;
};

ParsedQueries parse_queries(PyObject* queries, U n, W max_bytes) {
    if (!PyTuple_CheckExact(queries)) throw Invalid("queries must be an exact tuple");
    W count = static_cast<W>(PyTuple_GET_SIZE(queries));
    if (count > MAX_QUERIES) throw Resource("too many queries");
    ParsedQueries result;
    result.offsets.reserve(count + 1);
    result.offsets.push_back(0);
    W restrictions = 0;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(queries); ++i) {
        PyObject* query = PyTuple_GET_ITEM(queries, i);
        if (!PyTuple_CheckExact(query)) throw Invalid("each query must be an exact tuple");
        restrictions += static_cast<W>(PyTuple_GET_SIZE(query));
        if (restrictions > MAX_RESTRICTIONS) throw Resource("too many restrictions");
        result.offsets.push_back(static_cast<U>(restrictions));
    }
    W required = 4 * W(result.offsets.capacity()) + 12 * restrictions;
    if (required > max_bytes) throw Resource("query input exceeds payload cap");
    result.vertices.reserve(restrictions);
    result.masks.reserve(restrictions);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(queries); ++i) {
        PyObject* query = PyTuple_GET_ITEM(queries, i);
        U prior = UINT32_MAX;
        for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(query); ++j) {
            PyObject* row = PyTuple_GET_ITEM(query, j);
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2)
                throw Invalid("restriction must be an exact vertex/mask pair");
            U vertex = static_cast<U>(integer(PyTuple_GET_ITEM(row, 0),
                                              "restriction vertex outside graph", n - 1));
            W mask = integer(PyTuple_GET_ITEM(row, 1), "invalid restriction mask");
            if (j && vertex <= prior) throw Invalid("restriction vertices must be strictly increasing");
            prior = vertex;
            result.vertices.push_back(vertex);
            result.masks.push_back(mask);
        }
    }
    return result;
}

struct Worker {
    const Index& index;
    std::vector<uint8_t> assumed;
    std::vector<uint8_t> seen;
    std::vector<U> order;
    std::vector<U> component;
    std::vector<std::pair<U,U>> dfs;
    std::vector<U> stack;

    explicit Worker(const Index& x) : index(x), assumed(x.atoms), seen(x.atoms),
        order(), component(x.atoms), dfs(), stack() {
        order.reserve(x.atoms);
        dfs.reserve(x.atoms);
        stack.reserve(x.atoms);
    }

    bool next_forward(U vertex, U& cursor, U& child) const {
        const U begin = index.forward_offsets[vertex];
        const U end = index.forward_offsets[vertex + 1];
        const U base_count = end - begin;
        if (cursor < base_count) {
            child = index.forward_targets[begin + cursor++];
            return true;
        }
        if (cursor == base_count && assumed[vertex ^ 1U]) {
            ++cursor;
            child = vertex ^ 1U;
            return true;
        }
        return false;
    }

    bool next_reverse(U vertex, U& cursor, U& child) const {
        const U begin = index.reverse_offsets[vertex];
        const U end = index.reverse_offsets[vertex + 1];
        const U base_count = end - begin;
        if (cursor < base_count) {
            child = index.reverse_targets[begin + cursor++];
            return true;
        }
        if (cursor == base_count && assumed[vertex]) {
            ++cursor;
            child = vertex ^ 1U;
            return true;
        }
        return false;
    }

    bool solve(U begin, U end, const ParsedQueries& queries, char* labels,
               W& traversed_edges) {
        std::copy(index.base_assumed.begin(), index.base_assumed.end(), assumed.begin());
        bool immediate_conflict = false;
        for (U i = begin; i < end; ++i) {
            U v = queries.vertices[i];
            W mask = queries.masks[i];
            bool allow0 = index.colour0[v] != 255
                && (mask & (W(1) << index.colour0[v])) != 0;
            bool allow1 = index.colour1[v] != 255
                && (mask & (W(1) << index.colour1[v])) != 0;
            if (!allow0 && !allow1) {
                immediate_conflict = true;
                continue;
            }
            U q = index.node[v];
            if (allow0 != allow1) assumed[2 * q + (allow1 ? 1U : 0U)] = 1;
        }
        for (U q = 0; q < index.qn; ++q)
            if (assumed[2 * q] && assumed[2 * q + 1]) immediate_conflict = true;
        if (immediate_conflict) return false;

        std::fill(seen.begin(), seen.end(), 0);
        order.clear();
        for (U root = 0; root < index.atoms; ++root) {
            if (seen[root]) continue;
            seen[root] = 1;
            dfs.clear();
            dfs.emplace_back(root, 0);
            while (!dfs.empty()) {
                U vertex = dfs.back().first;
                U& cursor = dfs.back().second;
                U child = 0;
                if (!next_forward(vertex, cursor, child)) {
                    order.push_back(vertex);
                    dfs.pop_back();
                    continue;
                }
                ++traversed_edges;
                if (!seen[child]) {
                    seen[child] = 1;
                    dfs.emplace_back(child, 0);
                }
            }
        }

        std::fill(component.begin(), component.end(), UINT32_MAX);
        U next_component = 0;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            U root = *it;
            if (component[root] != UINT32_MAX) continue;
            component[root] = next_component;
            stack.clear();
            stack.push_back(root);
            while (!stack.empty()) {
                U vertex = stack.back();
                stack.pop_back();
                U cursor = 0, child = 0;
                while (next_reverse(vertex, cursor, child)) {
                    ++traversed_edges;
                    if (component[child] == UINT32_MAX) {
                        component[child] = next_component;
                        stack.push_back(child);
                    }
                }
            }
            ++next_component;
        }
        for (U q = 0; q < index.qn; ++q)
            if (component[2 * q] == component[2 * q + 1]) return false;
        for (U v = 0; v < index.n; ++v) {
            U q = index.node[v];
            bool side1 = component[2 * q + 1] > component[2 * q];
            U colour = side1 ? index.colour1[v] : index.colour0[v];
            if (colour == 255) throw std::logic_error("compiled support selected absent lift side");
            labels[v] = static_cast<char>(colour);
        }
        return true;
    }
};


struct ClosureWorker {
    const Index& index;
    std::vector<W> forced;

    explicit ClosureWorker(const Index& x) : index(x), forced(x.words) {}

    bool conflict_with(U atom, W& word_operations) const {
        constexpr W even_bits = UINT64_C(0x5555555555555555);
        const W* row = index.closure.data() + W(atom) * index.words;
        for (U word = 0; word < index.words; ++word) {
            W value = forced[word] | row[word];
            ++word_operations;
            if ((value & (value >> 1) & even_bits) != 0) return true;
        }
        return false;
    }

    void add(U atom, W& word_operations) {
        const W* row = index.closure.data() + W(atom) * index.words;
        for (U word = 0; word < index.words; ++word) {
            forced[word] |= row[word];
            ++word_operations;
        }
    }

    bool has_conflict() const {
        constexpr W even_bits = UINT64_C(0x5555555555555555);
        for (W value : forced)
            if ((value & (value >> 1) & even_bits) != 0) return true;
        return false;
    }

    bool solve(U begin, U end, const ParsedQueries& queries, char* labels,
               W& word_operations) {
        forced = index.base_forced;
        bool immediate_conflict = false;
        for (U i = begin; i < end; ++i) {
            U vertex = queries.vertices[i];
            W mask = queries.masks[i];
            bool allow0 = index.colour0[vertex] != 255
                && (mask & (W(1) << index.colour0[vertex])) != 0;
            bool allow1 = index.colour1[vertex] != 255
                && (mask & (W(1) << index.colour1[vertex])) != 0;
            if (!allow0 && !allow1) {
                immediate_conflict = true;
                continue;
            }
            if (allow0 != allow1)
                add(2 * index.node[vertex] + (allow1 ? 1U : 0U), word_operations);
        }
        if (immediate_conflict || has_conflict()) return false;

        // Complete the model one quotient choice at a time.  In 2-SAT, if one
        // literal's implication closure contradicts the current consistent unit
        // closure, its complement is forced and remains consistent.
        for (U q = 0; q < index.qn; ++q) {
            U atom0 = 2 * q;
            W pair = (forced[atom0 / 64] >> (atom0 % 64)) & 3;
            if (pair == 3) return false;
            if (pair == 0) {
                U selected = atom0;
                if (conflict_with(selected, word_operations)) selected ^= 1U;
                if (conflict_with(selected, word_operations))
                    throw std::logic_error("both binary choices contradict a consistent closure");
                add(selected, word_operations);
            }
        }
        if (has_conflict()) throw std::logic_error("compiled model completion is inconsistent");
        for (U vertex = 0; vertex < index.n; ++vertex) {
            U q = index.node[vertex];
            U atom0 = 2 * q;
            W pair = (forced[atom0 / 64] >> (atom0 % 64)) & 3;
            if (pair != 1 && pair != 2)
                throw std::logic_error("compiled support did not choose exactly one side");
            U colour = pair == 2 ? index.colour1[vertex] : index.colour0[vertex];
            if (colour == 255) throw std::logic_error("compiled support selected absent lift side");
            labels[vertex] = static_cast<char>(colour);
        }
        return true;
    }
};

constexpr const char* TAG = "spectra.quotient.session.v1";
void destroy(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, TAG);
    if (pointer) delete static_cast<Index*>(pointer);
    else PyErr_Clear();
}
PyObject* failure() {
    try { throw; }
    catch (const Invalid& error) { PyErr_SetString(PyExc_ValueError, error.what()); }
    catch (const Resource& error) { PyErr_SetString(PyExc_MemoryError, error.what()); }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown quotient session error"); }
    return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
    PyObject *node, *colour0, *colour1, *palettes, *initial, *offsets, *arcs, *cap;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &node, &colour0, &colour1, &palettes,
                          &initial, &offsets, &arcs, &cap)) return nullptr;
    try {
        auto index = std::make_unique<Index>(node, colour0, colour1, palettes, initial,
                                             offsets, arcs, integer(cap, "invalid payload cap"));
        PyObject* capsule = PyCapsule_New(index.get(), TAG, destroy);
        if (capsule) index.release();
        return capsule;
    } catch (...) { return failure(); }
}

PyObject* info(PyObject*, PyObject* capsule) {
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    return Py_BuildValue("(IIKK)", index->n, index->qn,
        static_cast<unsigned long long>(index->forward_targets.size()),
        static_cast<unsigned long long>(index->payload));
}

PyObject* solve_batch(PyObject*, PyObject* args) {
    PyObject *capsule, *queries, *cap;
    if (!PyArg_ParseTuple(args, "OOO", &capsule, &queries, &cap)) return nullptr;
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    try {
        W max_bytes = integer(cap, "invalid batch payload cap");
        ParsedQueries parsed = parse_queries(queries, index->n, max_bytes);
        W query_count = parsed.offsets.size() - 1;
        W output_bytes = query_count + query_count * W(index->n) + 8 * query_count;
        W scratch_bytes = W(index->atoms) * (3 + 4) + 16 * W(index->atoms);
        if (output_bytes + scratch_bytes > max_bytes)
            throw Resource("batch output and scratch exceed payload cap");
        PyObject* statuses = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(query_count));
        PyObject* labels = PyBytes_FromStringAndSize(nullptr,
            static_cast<Py_ssize_t>(query_count * W(index->n)));
        if (!statuses || !labels) {
            Py_XDECREF(statuses); Py_XDECREF(labels); return nullptr;
        }
        auto* status_data = PyBytes_AS_STRING(statuses);
        auto* label_data = PyBytes_AS_STRING(labels);
        std::vector<W> times(query_count, 0);
        std::vector<W> traversed(query_count, 0);
        {
            PyThreadState* state = PyEval_SaveThread();
            try {
                Worker worker(*index);
                for (W query = 0; query < query_count; ++query) {
                    auto started = std::chrono::steady_clock::now();
                    char* output = label_data + query * W(index->n);
                    std::memset(output, 0, index->n);
                    W edges = 0;
                    bool sat = worker.solve(parsed.offsets[query], parsed.offsets[query + 1],
                                            parsed, output, edges);
                    status_data[query] = sat ? 0 : 1;
                    traversed[query] = edges;
                    times[query] = static_cast<W>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count());
                }
            } catch (...) {
                PyEval_RestoreThread(state);
                Py_DECREF(statuses); Py_DECREF(labels);
                throw;
            }
            PyEval_RestoreThread(state);
        }
        PyObject* time_tuple = PyTuple_New(static_cast<Py_ssize_t>(query_count));
        PyObject* edge_tuple = PyTuple_New(static_cast<Py_ssize_t>(query_count));
        if (!time_tuple || !edge_tuple) {
            Py_DECREF(statuses); Py_DECREF(labels);
            Py_XDECREF(time_tuple); Py_XDECREF(edge_tuple); return nullptr;
        }
        for (W i = 0; i < query_count; ++i) {
            PyTuple_SET_ITEM(time_tuple, static_cast<Py_ssize_t>(i),
                             PyLong_FromUnsignedLongLong(times[i]));
            PyTuple_SET_ITEM(edge_tuple, static_cast<Py_ssize_t>(i),
                             PyLong_FromUnsignedLongLong(traversed[i]));
        }
        return Py_BuildValue("(NNNN)", statuses, labels, time_tuple, edge_tuple);
    } catch (...) { return failure(); }
}

PyObject* solve_batch_closure(PyObject*, PyObject* args) {
    PyObject *capsule, *queries, *cap;
    if (!PyArg_ParseTuple(args, "OOO", &capsule, &queries, &cap)) return nullptr;
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    try {
        W max_bytes = integer(cap, "invalid batch payload cap");
        ParsedQueries parsed = parse_queries(queries, index->n, max_bytes);
        W query_count = parsed.offsets.size() - 1;
        W output_bytes = query_count + query_count * W(index->n) + 8 * query_count;
        W scratch_bytes = W(index->atoms) * (3 + 4) + 16 * W(index->atoms);
        if (output_bytes + scratch_bytes > max_bytes)
            throw Resource("batch output and scratch exceed payload cap");
        PyObject* statuses = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(query_count));
        PyObject* labels = PyBytes_FromStringAndSize(nullptr,
            static_cast<Py_ssize_t>(query_count * W(index->n)));
        if (!statuses || !labels) {
            Py_XDECREF(statuses); Py_XDECREF(labels); return nullptr;
        }
        auto* status_data = PyBytes_AS_STRING(statuses);
        auto* label_data = PyBytes_AS_STRING(labels);
        std::vector<W> times(query_count, 0);
        std::vector<W> traversed(query_count, 0);
        {
            PyThreadState* state = PyEval_SaveThread();
            try {
                ClosureWorker worker(*index);
                for (W query = 0; query < query_count; ++query) {
                    auto started = std::chrono::steady_clock::now();
                    char* output = label_data + query * W(index->n);
                    std::memset(output, 0, index->n);
                    W edges = 0;
                    bool sat = worker.solve(parsed.offsets[query], parsed.offsets[query + 1],
                                            parsed, output, edges);
                    status_data[query] = sat ? 0 : 1;
                    traversed[query] = edges;
                    times[query] = static_cast<W>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count());
                }
            } catch (...) {
                PyEval_RestoreThread(state);
                Py_DECREF(statuses); Py_DECREF(labels);
                throw;
            }
            PyEval_RestoreThread(state);
        }
        PyObject* time_tuple = PyTuple_New(static_cast<Py_ssize_t>(query_count));
        PyObject* edge_tuple = PyTuple_New(static_cast<Py_ssize_t>(query_count));
        if (!time_tuple || !edge_tuple) {
            Py_DECREF(statuses); Py_DECREF(labels);
            Py_XDECREF(time_tuple); Py_XDECREF(edge_tuple); return nullptr;
        }
        for (W i = 0; i < query_count; ++i) {
            PyTuple_SET_ITEM(time_tuple, static_cast<Py_ssize_t>(i),
                             PyLong_FromUnsignedLongLong(times[i]));
            PyTuple_SET_ITEM(edge_tuple, static_cast<Py_ssize_t>(i),
                             PyLong_FromUnsignedLongLong(traversed[i]));
        }
        return Py_BuildValue("(NNNN)", statuses, labels, time_tuple, edge_tuple);
    } catch (...) { return failure(); }
}

PyObject* abi(PyObject*, PyObject*) { return PyLong_FromLong(1); }
PyMethodDef methods[] = {
    {"abi", abi, METH_NOARGS, "Return ABI version."},
    {"create", create, METH_VARARGS, "Compile a binary quotient session."},
    {"info", info, METH_O, "Return compiled geometry and payload."},
    {"solve_batch", solve_batch, METH_VARARGS, "Solve and lift a batch with per-query SCC."},
    {"solve_batch_closure", solve_batch_closure, METH_VARARGS, "Solve and lift a batch with compiled closures."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_spectra_quotient_session", nullptr, -1,
                      methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__spectra_quotient_session() { return PyModule_Create(&module); }
