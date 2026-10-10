// Exact compiled support queries for binary-list graph colouring.
// MIT; see repository LICENSE. Classical SCC and reachability compilation.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U MAX_VARIABLES = 100000;
constexpr W MAX_EDGES = 2000000;
constexpr U NIL = UINT32_MAX;

struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };

W integer(PyObject* object, const char* message, W maximum = UINT64_MAX) {
    if (!PyLong_CheckExact(object)) throw Invalid(message);
    W value = PyLong_AsUnsignedLongLong(object);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw Invalid(message);
    }
    if (value > maximum) throw Invalid(message);
    return value;
}

unsigned first(W value) {
    return static_cast<unsigned>(__builtin_ctzll(value));
}

U literal_node(U variable, U color, const std::vector<uint8_t>& low,
               const std::vector<uint8_t>& high) {
    if (low[variable] == high[variable]) {
        if (color != low[variable]) throw Invalid("color outside singleton list");
        return 2 * variable + 1;
    }
    if (color == low[variable]) return 2 * variable;
    if (color == high[variable]) return 2 * variable + 1;
    throw Invalid("color outside binary list");
}

struct Index {
    U variables = 0;
    U colors = 0;
    U components = 0;
    size_t words = 0;
    bool impossible = false;
    W payload_bytes = 0;
    W build_bound = 0;
    std::vector<uint8_t> low;
    std::vector<uint8_t> high;
    std::vector<U> component;
    std::vector<U> opposite;
    std::vector<U> topological;
    std::vector<W> closure;

    Index(PyObject* nv, PyObject* kv, PyObject* edge_bank,
          PyObject* mask_bank, W maximum_bytes) {
        variables = static_cast<U>(integer(
            nv, "variables outside supported range", MAX_VARIABLES));
        colors = static_cast<U>(integer(kv, "colors outside supported range", 64));
        if (!PyTuple_CheckExact(edge_bank) || !PyTuple_CheckExact(mask_bank)) {
            throw Invalid("edges and masks must be exact tuples");
        }
        const W edge_count = static_cast<W>(PyTuple_GET_SIZE(edge_bank));
        if (edge_count > MAX_EDGES) throw Resource("too many input edges");
        if (static_cast<W>(PyTuple_GET_SIZE(mask_bank)) != variables) {
            throw Invalid("mask count differs from variables");
        }
        low.resize(variables);
        high.resize(variables);
        for (U variable = 0; variable < variables; ++variable) {
            W mask = integer(PyTuple_GET_ITEM(mask_bank, variable), "invalid binary mask");
            if (colors < 64 && (mask >> colors)) throw Invalid("mask uses a color outside the palette");
            const unsigned choices = static_cast<unsigned>(__builtin_popcountll(mask));
            if (!mask || choices > 2) throw Invalid("compiled support requires one or two colors per variable");
            const unsigned first_color = first(mask);
            W remaining = mask & (mask - 1);
            const unsigned second_color = remaining ? first(remaining) : first_color;
            low[variable] = static_cast<uint8_t>(first_color);
            high[variable] = static_cast<uint8_t>(second_color);
        }

        const U nodes = 2 * variables;
        std::vector<std::pair<U, U>> arcs;
        arcs.reserve(static_cast<size_t>(2 * variables + 4 * edge_count));
        for (U variable = 0; variable < variables; ++variable) {
            if (low[variable] == high[variable]) {
                arcs.emplace_back(2 * variable, 2 * variable + 1);
            }
        }
        std::vector<std::pair<U, U>> seen_edges;
        seen_edges.reserve(static_cast<size_t>(edge_count));
        for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(edge_bank); ++index) {
            PyObject* row = PyTuple_GET_ITEM(edge_bank, index);
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) {
                throw Invalid("edge must be an exact pair tuple");
            }
            U left = static_cast<U>(integer(
                PyTuple_GET_ITEM(row, 0), "invalid edge endpoint", MAX_VARIABLES));
            U right = static_cast<U>(integer(
                PyTuple_GET_ITEM(row, 1), "invalid edge endpoint", MAX_VARIABLES));
            if (left >= variables || right >= variables || left >= right) {
                throw Invalid("edges must be canonical nonloops");
            }
            seen_edges.emplace_back(left, right);
            W left_mask = (W(1) << low[left]) | (W(1) << high[left]);
            W right_mask = (W(1) << low[right]) | (W(1) << high[right]);
            W shared = left_mask & right_mask;
            while (shared) {
                W bit = shared & (~shared + 1);
                shared &= shared - 1;
                U color = static_cast<U>(first(bit));
                U left_literal = literal_node(left, color, low, high);
                U right_literal = literal_node(right, color, low, high);
                arcs.emplace_back(left_literal, right_literal ^ 1U);
                arcs.emplace_back(right_literal, left_literal ^ 1U);
            }
        }
        if (!std::is_sorted(seen_edges.begin(), seen_edges.end())
                || std::adjacent_find(seen_edges.begin(), seen_edges.end()) != seen_edges.end()) {
            throw Invalid("edges must be distinct and sorted");
        }
        std::sort(arcs.begin(), arcs.end());
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());

        build_bound = 64 * W(nodes + 1) + 32 * W(arcs.size() + 1) + 4096;
        if (build_bound > maximum_bytes) throw Resource("implication construction exceeds payload cap");

        std::vector<U> forward_offset(static_cast<size_t>(nodes) + 1, 0);
        std::vector<U> reverse_offset(static_cast<size_t>(nodes) + 1, 0);
        for (auto arc : arcs) {
            ++forward_offset[arc.first + 1];
            ++reverse_offset[arc.second + 1];
        }
        for (U node = 0; node < nodes; ++node) {
            forward_offset[node + 1] += forward_offset[node];
            reverse_offset[node + 1] += reverse_offset[node];
        }
        std::vector<U> forward_cursor(forward_offset);
        std::vector<U> reverse_cursor(reverse_offset);
        std::vector<U> forward(arcs.size());
        std::vector<U> reverse(arcs.size());
        for (auto arc : arcs) {
            forward[forward_cursor[arc.first]++] = arc.second;
            reverse[reverse_cursor[arc.second]++] = arc.first;
        }

        std::vector<uint8_t> seen(nodes, 0);
        std::vector<U> order;
        order.reserve(nodes);
        std::vector<std::pair<U, U>> stack;
        stack.reserve(nodes);
        for (U root = 0; root < nodes; ++root) {
            if (seen[root]) continue;
            seen[root] = 1;
            stack.emplace_back(root, forward_offset[root]);
            while (!stack.empty()) {
                auto& frame = stack.back();
                if (frame.second == forward_offset[frame.first + 1]) {
                    order.push_back(frame.first);
                    stack.pop_back();
                    continue;
                }
                U child = forward[frame.second++];
                if (!seen[child]) {
                    seen[child] = 1;
                    stack.emplace_back(child, forward_offset[child]);
                }
            }
        }

        component.assign(nodes, NIL);
        components = 0;
        for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
            U root = *iterator;
            if (component[root] != NIL) continue;
            component[root] = components;
            stack.emplace_back(root, 0);
            while (!stack.empty()) {
                U node = stack.back().first;
                stack.pop_back();
                for (U offset = reverse_offset[node]; offset < reverse_offset[node + 1]; ++offset) {
                    U child = reverse[offset];
                    if (component[child] == NIL) {
                        component[child] = components;
                        stack.emplace_back(child, 0);
                    }
                }
            }
            ++components;
        }
        for (U variable = 0; variable < variables; ++variable) {
            if (component[2 * variable] == component[2 * variable + 1]) {
                impossible = true;
            }
        }

        opposite.assign(components, NIL);
        for (U node = 0; node < nodes; ++node) {
            U source = component[node];
            U target = component[node ^ 1U];
            if (opposite[source] != NIL && opposite[source] != target) {
                throw std::logic_error("component complement is not well-defined");
            }
            opposite[source] = target;
        }
        for (U source = 0; source < components; ++source) {
            if (opposite[source] == NIL || opposite[opposite[source]] != source
                    || opposite[source] == source) {
                throw std::logic_error("invalid component complement");
            }
        }

        std::vector<std::pair<U, U>> dag_edges;
        dag_edges.reserve(arcs.size());
        for (auto arc : arcs) {
            U source = component[arc.first];
            U target = component[arc.second];
            if (source != target) dag_edges.emplace_back(source, target);
        }
        std::sort(dag_edges.begin(), dag_edges.end());
        dag_edges.erase(std::unique(dag_edges.begin(), dag_edges.end()), dag_edges.end());
        std::vector<U> dag_offset(static_cast<size_t>(components) + 1, 0);
        std::vector<U> indegree(components, 0);
        for (auto edge : dag_edges) {
            ++dag_offset[edge.first + 1];
            ++indegree[edge.second];
        }
        for (U source = 0; source < components; ++source) {
            dag_offset[source + 1] += dag_offset[source];
        }
        std::vector<U> dag_cursor(dag_offset);
        std::vector<U> dag(dag_edges.size());
        for (auto edge : dag_edges) dag[dag_cursor[edge.first]++] = edge.second;

        std::priority_queue<U, std::vector<U>, std::greater<U>> ready;
        for (U component_id = 0; component_id < components; ++component_id) {
            if (!indegree[component_id]) ready.push(component_id);
        }
        topological.reserve(components);
        while (!ready.empty()) {
            U source = ready.top();
            ready.pop();
            topological.push_back(source);
            for (U offset = dag_offset[source]; offset < dag_offset[source + 1]; ++offset) {
                U target = dag[offset];
                if (--indegree[target] == 0) ready.push(target);
            }
        }
        if (topological.size() != components) throw std::logic_error("component graph is cyclic");

        words = (static_cast<size_t>(components) + 63) / 64;
        const W closure_bytes = 8 * W(components) * W(words);
        const W persistent = 2 * W(low.capacity())
            + 4 * W(component.capacity() + opposite.capacity() + topological.capacity())
            + closure_bytes;
        if (persistent > maximum_bytes) throw Resource("compiled closure exceeds payload cap");
        closure.assign(static_cast<size_t>(components) * words, 0);
        for (auto iterator = topological.rbegin(); iterator != topological.rend(); ++iterator) {
            U source = *iterator;
            W* row = &closure[static_cast<size_t>(source) * words];
            row[source / 64] |= W(1) << (source % 64);
            for (U offset = dag_offset[source]; offset < dag_offset[source + 1]; ++offset) {
                U target = dag[offset];
                const W* child = &closure[static_cast<size_t>(target) * words];
                for (size_t word = 0; word < words; ++word) row[word] |= child[word];
            }
        }
        payload_bytes = persistent;
    }

    bool contains(const std::vector<W>& bits, U component_id) const {
        return bits[component_id / 64] & (W(1) << (component_id % 64));
    }

    void add_closure(std::vector<W>& bits, U component_id) const {
        const W* row = &closure[static_cast<size_t>(component_id) * words];
        for (size_t word = 0; word < words; ++word) bits[word] |= row[word];
    }

    bool contradictory(const std::vector<W>& bits) const {
        for (U component_id = 0; component_id < components; ++component_id) {
            if (contains(bits, component_id) && contains(bits, opposite[component_id])) return true;
        }
        return false;
    }

    std::pair<bool, std::vector<char>> solve(const std::vector<U>& assumptions) const {
        if (impossible) return {false, {}};
        std::vector<W> truth(words, 0);
        for (U component_id : assumptions) add_closure(truth, component_id);
        if (contradictory(truth)) return {false, {}};
        for (auto iterator = topological.rbegin(); iterator != topological.rend(); ++iterator) {
            U component_id = *iterator;
            U complement = opposite[component_id];
            if (contains(truth, component_id) || contains(truth, complement)) continue;
            std::vector<W> trial(truth);
            add_closure(trial, component_id);
            if (contradictory(trial)) {
                add_closure(truth, complement);
                if (contradictory(truth)) throw std::logic_error("satisfiable closure lost both choices");
            } else {
                truth.swap(trial);
            }
        }
        std::vector<char> labels(variables);
        for (U variable = 0; variable < variables; ++variable) {
            const bool positive = contains(truth, component[2 * variable + 1]);
            const bool negative = contains(truth, component[2 * variable]);
            if (positive == negative) throw std::logic_error("compiled model leaves a variable undecided");
            labels[variable] = static_cast<char>(positive ? high[variable] : low[variable]);
        }
        return {true, std::move(labels)};
    }
};

constexpr const char* TAG = "spectra.real_traffic.compiled_support.v1";

void destroy(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, TAG);
    if (pointer) delete static_cast<Index*>(pointer);
    else PyErr_Clear();
}

PyObject* failure() {
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
        PyErr_SetString(PyExc_RuntimeError, "unknown compiled support error");
    }
    return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
    PyObject* n;
    PyObject* k;
    PyObject* edges;
    PyObject* masks;
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "OOOOO", &n, &k, &edges, &masks, &cap)) return nullptr;
    try {
        auto index = std::make_unique<Index>(
            n, k, edges, masks, integer(cap, "invalid compiled support payload cap"));
        PyObject* capsule = PyCapsule_New(index.get(), TAG, destroy);
        if (capsule) index.release();
        return capsule;
    } catch (...) {
        return failure();
    }
}

PyObject* solve(PyObject*, PyObject* args) {
    PyObject* capsule;
    PyObject* query;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &query)) return nullptr;
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    try {
        if (!PyTuple_CheckExact(query)) throw Invalid("query restrictions must be an exact tuple");
        if (static_cast<W>(PyTuple_GET_SIZE(query)) > 2 * W(index->variables) + 1) {
            throw Invalid("too many query restrictions");
        }
        std::vector<U> assumptions;
        assumptions.reserve(static_cast<size_t>(PyTuple_GET_SIZE(query)));
        std::vector<U> seen_variables;
        seen_variables.reserve(assumptions.capacity());
        U previous = NIL;
        for (Py_ssize_t offset = 0; offset < PyTuple_GET_SIZE(query); ++offset) {
            PyObject* row = PyTuple_GET_ITEM(query, offset);
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) {
                throw Invalid("restriction must be an exact vertex/mask pair");
            }
            U variable = static_cast<U>(integer(
                PyTuple_GET_ITEM(row, 0), "invalid restriction vertex", MAX_VARIABLES));
            W allowed = integer(PyTuple_GET_ITEM(row, 1), "invalid restriction mask");
            if (variable >= index->variables || (previous != NIL && variable <= previous)) {
                throw Invalid("restriction vertices must be distinct and increasing");
            }
            previous = variable;
            W original = (W(1) << index->low[variable]) | (W(1) << index->high[variable]);
            W selected = allowed & original;
            if (__builtin_popcountll(selected) != 1) {
                throw Invalid("restriction must select one available color");
            }
            U color = static_cast<U>(first(selected));
            U node = literal_node(variable, color, index->low, index->high);
            assumptions.push_back(index->component[node]);
        }
        std::pair<bool, std::vector<char>> result;
        PyThreadState* state = PyEval_SaveThread();
        try {
            result = index->solve(assumptions);
        } catch (...) {
            PyEval_RestoreThread(state);
            throw;
        }
        PyEval_RestoreThread(state);
        PyObject* labels = PyBytes_FromStringAndSize(
            result.second.data(), static_cast<Py_ssize_t>(result.second.size()));
        if (!labels) return nullptr;
        return Py_BuildValue("(NO)", labels, result.first ? Py_True : Py_False);
    } catch (...) {
        return failure();
    }
}

PyObject* info(PyObject*, PyObject* capsule) {
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    return Py_BuildValue(
        "(IIKKK)", index->variables, index->components,
        static_cast<unsigned long long>(index->payload_bytes),
        static_cast<unsigned long long>(index->build_bound),
        static_cast<unsigned long long>(index->words));
}

PyObject* abi(PyObject*, PyObject*) {
    return PyLong_FromLong(1);
}

PyMethodDef methods[] = {
    {"abi", abi, METH_NOARGS, "Return the compiled support ABI version."},
    {"create", create, METH_VARARGS, "Compile the binary implication relation."},
    {"solve", solve, METH_VARARGS, "Answer one exact support query."},
    {"info", info, METH_O, "Return compiled relation geometry."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_spectra_compiled_support",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
}  // namespace

PyMODINIT_FUNC PyInit__spectra_compiled_support() {
    return PyModule_Create(&module);
}
