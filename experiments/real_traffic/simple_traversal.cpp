// Exact implication traversal for binary-list support queries.
// Classical 2-SAT propagation; this is a cheap control, not a novelty claim.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U MAX_N = 100000;
constexpr W MAX_EDGES = 2000000;

struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };

W exact_uint(PyObject* object, const char* message, W maximum = UINT64_MAX) {
    if (!PyLong_CheckExact(object)) throw Invalid(message);
    W value = PyLong_AsUnsignedLongLong(object);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw Invalid(message);
    }
    if (value > maximum) throw Invalid(message);
    return value;
}
U first_bit(W value) { return static_cast<U>(__builtin_ctzll(value)); }
U bit_count(W value) { return static_cast<U>(__builtin_popcountll(value)); }

struct Index {
    U n = 0;
    std::vector<W> masks;
    std::vector<uint8_t> low;
    std::vector<uint8_t> high;
    std::vector<U> offsets;
    std::vector<U> targets;
    std::vector<U> units;
    std::vector<uint8_t> base_labels;
    std::vector<U> seen;
    std::vector<U> queue;
    U stamp = 0;
    W payload_bytes = 0;

    Index(PyObject* edge_bank, PyObject* mask_bank, W max_bytes) {
        if (!PyTuple_CheckExact(edge_bank) || !PyTuple_CheckExact(mask_bank))
            throw Invalid("edges and masks must be exact tuples");
        W n64 = static_cast<W>(PyTuple_GET_SIZE(mask_bank));
        W edge_count = static_cast<W>(PyTuple_GET_SIZE(edge_bank));
        if (n64 > MAX_N || edge_count > MAX_EDGES)
            throw Resource("input exceeds declared geometry");
        n = static_cast<U>(n64);
        masks.resize(n);
        low.resize(n);
        high.resize(n);
        units.reserve(n);

        for (U v = 0; v < n; ++v) {
            W mask = exact_uint(PyTuple_GET_ITEM(mask_bank, v), "invalid mask");
            U count = bit_count(mask);
            if (!mask || count > 2)
                throw Invalid("every mask must contain one or two colours");
            masks[v] = mask;
            low[v] = static_cast<uint8_t>(first_bit(mask));
            W remainder = mask & (mask - 1);
            high[v] = static_cast<uint8_t>(remainder ? first_bit(remainder) : low[v]);
            if (count == 1) units.push_back(2 * v);
        }

        std::vector<std::pair<U, U>> implications;
        implications.reserve(4 * edge_count + units.size());
        for (U node : units) implications.emplace_back(node ^ 1U, node);

        auto literal_for_colour = [&](U vertex, U colour) -> U {
            if (!(masks[vertex] & (W(1) << colour)))
                throw std::logic_error("colour is outside original mask");
            return 2 * vertex
                + (high[vertex] != low[vertex] && colour == high[vertex] ? 1U : 0U);
        };

        std::vector<std::pair<U, U>> original_edges;
        original_edges.reserve(edge_count);
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(edge_bank); ++i) {
            PyObject* item = PyTuple_GET_ITEM(edge_bank, i);
            if (!PyTuple_CheckExact(item) || PyTuple_GET_SIZE(item) != 2)
                throw Invalid("every edge must be an exact pair tuple");
            U left = static_cast<U>(exact_uint(
                PyTuple_GET_ITEM(item, 0), "invalid edge endpoint", MAX_N));
            U right = static_cast<U>(exact_uint(
                PyTuple_GET_ITEM(item, 1), "invalid edge endpoint", MAX_N));
            if (left >= n || right >= n || left == right)
                throw Invalid("edge endpoint outside graph or self-loop");
            original_edges.emplace_back(left, right);
            W shared = masks[left] & masks[right];
            while (shared) {
                W bit = shared & (~shared + 1);
                shared ^= bit;
                U colour = first_bit(bit);
                U a = literal_for_colour(left, colour);
                U b = literal_for_colour(right, colour);
                implications.emplace_back(a, b ^ 1U);
                implications.emplace_back(b, a ^ 1U);
            }
        }
        std::sort(implications.begin(), implications.end());
        implications.erase(
            std::unique(implications.begin(), implications.end()),
            implications.end());

        U nodes = 2 * n;
        offsets.assign(static_cast<size_t>(nodes) + 1, 0);
        for (auto edge : implications) ++offsets[edge.first + 1];
        for (U node = 0; node < nodes; ++node)
            offsets[node + 1] += offsets[node];
        targets.resize(implications.size());
        std::vector<U> cursor(offsets);
        for (auto edge : implications)
            targets[cursor[edge.first]++] = edge.second;

        std::vector<U> reverse_offsets(static_cast<size_t>(nodes) + 1, 0);
        for (auto edge : implications) ++reverse_offsets[edge.second + 1];
        for (U node = 0; node < nodes; ++node)
            reverse_offsets[node + 1] += reverse_offsets[node];
        std::vector<U> reverse_targets(implications.size());
        cursor = reverse_offsets;
        for (auto edge : implications)
            reverse_targets[cursor[edge.second]++] = edge.first;

        std::vector<uint8_t> visited(nodes, 0);
        std::vector<U> order;
        order.reserve(nodes);
        std::vector<std::pair<U, U>> stack;
        stack.reserve(nodes);
        for (U root = 0; root < nodes; ++root) {
            if (visited[root]) continue;
            visited[root] = 1;
            stack.emplace_back(root, offsets[root]);
            while (!stack.empty()) {
                auto& frame = stack.back();
                if (frame.second == offsets[frame.first + 1]) {
                    order.push_back(frame.first);
                    stack.pop_back();
                    continue;
                }
                U child = targets[frame.second++];
                if (!visited[child]) {
                    visited[child] = 1;
                    stack.emplace_back(child, offsets[child]);
                }
            }
        }

        std::vector<U> component(nodes, UINT32_MAX);
        U next_component = 0;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            U root = *it;
            if (component[root] != UINT32_MAX) continue;
            component[root] = next_component;
            std::vector<U> todo{root};
            while (!todo.empty()) {
                U vertex = todo.back();
                todo.pop_back();
                for (U at = reverse_offsets[vertex];
                     at < reverse_offsets[vertex + 1]; ++at) {
                    U child = reverse_targets[at];
                    if (component[child] == UINT32_MAX) {
                        component[child] = next_component;
                        todo.push_back(child);
                    }
                }
            }
            ++next_component;
        }
        for (U v = 0; v < n; ++v)
            if (component[2 * v] == component[2 * v + 1])
                throw Invalid("base formula is contradictory");

        auto construct = [&](bool greater) {
            std::vector<uint8_t> labels(n);
            for (U v = 0; v < n; ++v) {
                bool truth = greater
                    ? component[2 * v + 1] > component[2 * v]
                    : component[2 * v + 1] < component[2 * v];
                labels[v] = truth && high[v] != low[v] ? high[v] : low[v];
            }
            return labels;
        };
        base_labels = construct(true);
        auto valid = [&](const std::vector<uint8_t>& labels) {
            for (auto edge : original_edges)
                if (labels[edge.first] == labels[edge.second]) return false;
            return true;
        };
        if (!valid(base_labels)) base_labels = construct(false);
        if (!valid(base_labels))
            throw std::logic_error("could not reconstruct a base model");

        seen.assign(nodes, 0);
        queue.reserve(nodes);
        payload_bytes =
            sizeof(W) * masks.capacity()
            + sizeof(uint8_t)
                * (low.capacity() + high.capacity() + base_labels.capacity())
            + sizeof(U)
                * (offsets.capacity() + targets.capacity() + units.capacity()
                   + seen.capacity() + queue.capacity());
        if (payload_bytes > max_bytes)
            throw Resource("prepared traversal exceeds payload cap");
    }

    PyObject* solve(PyObject* restrictions) {
        if (!PyTuple_CheckExact(restrictions))
            throw Invalid("restrictions must be an exact tuple");
        if (static_cast<W>(PyTuple_GET_SIZE(restrictions))
                > 2 * static_cast<W>(n) + 1)
            throw Invalid("too many restrictions");
        if (++stamp == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            stamp = 1;
        }
        queue.clear();
        std::vector<uint8_t> restricted(n, 0);

        auto add = [&](U node) -> bool {
            if (seen[node] == stamp) return true;
            if (seen[node ^ 1U] == stamp) return false;
            seen[node] = stamp;
            queue.push_back(node);
            return true;
        };
        for (U node : units) {
            if (!add(node)) {
                PyObject* empty = PyBytes_FromStringAndSize("", 0);
                return Py_BuildValue("(NIKK)", empty, 1U, 0ULL, 0ULL);
            }
        }

        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(restrictions); ++i) {
            PyObject* pair = PyTuple_GET_ITEM(restrictions, i);
            if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2)
                throw Invalid("restriction must be an exact pair tuple");
            U vertex = static_cast<U>(exact_uint(
                PyTuple_GET_ITEM(pair, 0), "invalid restriction vertex", MAX_N));
            W allowed = exact_uint(
                PyTuple_GET_ITEM(pair, 1), "invalid restriction mask");
            if (vertex >= n || restricted[vertex])
                throw Invalid("restriction vertex outside graph or repeated");
            W choice = allowed & masks[vertex];
            if (bit_count(choice) != 1)
                throw Invalid("restriction must select one available colour");
            restricted[vertex] = 1;
            U colour = first_bit(choice);
            U node = 2 * vertex
                + (high[vertex] != low[vertex] && colour == high[vertex] ? 1U : 0U);
            if (!add(node)) {
                PyObject* empty = PyBytes_FromStringAndSize("", 0);
                return Py_BuildValue(
                    "(NIKK)", empty, 1U, 0ULL,
                    static_cast<unsigned long long>(queue.size()));
            }
        }

        W traversed = 0;
        for (size_t head = 0; head < queue.size(); ++head) {
            U source = queue[head];
            for (U at = offsets[source]; at < offsets[source + 1]; ++at) {
                ++traversed;
                if (!add(targets[at])) {
                    PyObject* empty = PyBytes_FromStringAndSize("", 0);
                    return Py_BuildValue(
                        "(NIKK)", empty, 1U,
                        static_cast<unsigned long long>(traversed),
                        static_cast<unsigned long long>(queue.size()));
                }
            }
        }

        PyObject* output = PyBytes_FromStringAndSize(nullptr, n);
        if (!output) return nullptr;
        auto* labels = reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(output));
        std::copy(base_labels.begin(), base_labels.end(), labels);
        for (U node : queue) {
            U vertex = node / 2;
            labels[vertex] = (node & 1U) && high[vertex] != low[vertex]
                ? high[vertex] : low[vertex];
        }
        return Py_BuildValue(
            "(NIKK)", output, 0U,
            static_cast<unsigned long long>(traversed),
            static_cast<unsigned long long>(queue.size()));
    }
};

constexpr const char* TAG = "spectra.simple.traversal.v1";
void destroy(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, TAG);
    if (pointer) delete static_cast<Index*>(pointer);
    else PyErr_Clear();
}
PyObject* fail() {
    try { throw; }
    catch (const Invalid& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    }
    catch (const Resource& error) {
        PyErr_SetString(PyExc_MemoryError, error.what());
    }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown traversal error"); }
    return nullptr;
}
PyObject* create(PyObject*, PyObject* args) {
    PyObject *edges, *masks, *limit;
    if (!PyArg_ParseTuple(args, "OOO", &edges, &masks, &limit)) return nullptr;
    try {
        auto index = std::make_unique<Index>(
            edges, masks, exact_uint(limit, "invalid payload cap"));
        PyObject* capsule = PyCapsule_New(index.get(), TAG, destroy);
        if (capsule) index.release();
        return capsule;
    } catch (...) { return fail(); }
}
PyObject* solve(PyObject*, PyObject* args) {
    PyObject *capsule, *restrictions;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &restrictions)) return nullptr;
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    try { return index->solve(restrictions); }
    catch (...) { return fail(); }
}
PyObject* info(PyObject*, PyObject* capsule) {
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    return Py_BuildValue(
        "(IKK)", index->n,
        static_cast<unsigned long long>(index->targets.size()),
        static_cast<unsigned long long>(index->payload_bytes));
}
PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Prepare a binary implication graph."},
    {"solve", solve, METH_VARARGS, "Propagate one exact restriction set."},
    {"info", info, METH_O, "Return prepared geometry."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_spectra_simple_traversal", nullptr, -1,
    methods, nullptr, nullptr, nullptr, nullptr
};
}
PyMODINIT_FUNC PyInit__spectra_simple_traversal() {
    return PyModule_Create(&module);
}
