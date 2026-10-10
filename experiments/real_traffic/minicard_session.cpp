// Direct persistent MiniCard session adapter for the real-traffic study.
// MiniCard is compiled from its pinned upstream source and remains unmodified.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "minicard/Solver.h"

namespace {
using U = uint32_t;
using W = uint64_t;
using Minisat::Lit;
using Minisat::Solver;
using Minisat::Var;
using Minisat::lbool;
using Minisat::mkLit;

constexpr U MAX_N = 100000;
constexpr W MAX_EDGES = 2000000;
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

U first_bit(W mask) {
    if (!mask) throw Invalid("empty colour mask");
    return static_cast<U>(__builtin_ctzll(mask));
}

struct Index {
    U n = 0;
    U palette = 0;
    W input_edges = 0;
    W clauses = 0;
    W payload = 0;
    bool base_unsat = false;
    std::unique_ptr<Solver> solver;
    std::vector<uint8_t> low;
    std::vector<uint8_t> high;
    std::vector<W> masks;

    Lit choice_literal(U vertex, U colour) const {
        if (low[vertex] == high[vertex] && colour == low[vertex])
            return mkLit(static_cast<Var>(vertex), false);
        if (colour == low[vertex]) return mkLit(static_cast<Var>(vertex), true);
        if (colour == high[vertex]) return mkLit(static_cast<Var>(vertex), false);
        throw Invalid("colour outside variable domain");
    }

    Index(PyObject* edges_object, PyObject* masks_object, W max_bytes) {
        if (!PyTuple_CheckExact(edges_object) || !PyTuple_CheckExact(masks_object))
            throw Invalid("edges and masks must be exact tuples");
        n = static_cast<U>(PyTuple_GET_SIZE(masks_object));
        if (!n || n > MAX_N) throw Resource("variable inventory outside supported range");
        input_edges = static_cast<W>(PyTuple_GET_SIZE(edges_object));
        if (input_edges > MAX_EDGES) throw Resource("too many edges");
        W required = 18 * W(n) + 16 * input_edges + 4096;
        if (required > max_bytes) throw Resource("MiniCard session exceeds payload cap");
        low.resize(n);
        high.resize(n);
        masks.resize(n);
        palette = 0;
        solver = std::make_unique<Solver>();
        solver->verbosity = 0;
        for (U v = 0; v < n; ++v) {
            W mask = integer(PyTuple_GET_ITEM(masks_object, v), "invalid binary colour mask");
            U count = static_cast<U>(__builtin_popcountll(mask));
            if (count < 1 || count > 2) throw Invalid("MiniCard session requires one or two colours");
            masks[v] = mask;
            low[v] = static_cast<uint8_t>(first_bit(mask));
            W remainder = mask & (mask - 1);
            high[v] = static_cast<uint8_t>(remainder ? first_bit(remainder) : low[v]);
            palette = std::max<U>(palette, static_cast<U>(high[v]) + 1);
            solver->newVar();
            if (count == 1) {
                if (!base_unsat && !solver->addClause(mkLit(static_cast<Var>(v), false)))
                    base_unsat = true;
                ++clauses;
            }
        }
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(edges_object); ++i) {
            PyObject* row = PyTuple_GET_ITEM(edges_object, i);
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2)
                throw Invalid("edge must be an exact pair");
            U left = static_cast<U>(integer(PyTuple_GET_ITEM(row, 0), "bad edge endpoint", n - 1));
            U right = static_cast<U>(integer(PyTuple_GET_ITEM(row, 1), "bad edge endpoint", n - 1));
            if (left >= right) throw Invalid("edges must be canonical nonloops");
            W shared = masks[left] & masks[right];
            while (shared) {
                W bit = shared & (~shared + 1);
                shared &= shared - 1;
                U colour = first_bit(bit);
                Lit a = choice_literal(left, colour);
                Lit b = choice_literal(right, colour);
                if (!base_unsat && !solver->addClause(~a, ~b)) base_unsat = true;
                ++clauses;
            }
        }
        payload = masks.capacity() * 8 + low.capacity() + high.capacity();
        if (payload > max_bytes) throw Resource("MiniCard adapter payload exceeds cap");
    }
};

struct ParsedQueries {
    std::vector<U> offsets;
    std::vector<Lit> assumptions;
    std::vector<uint8_t> immediate_unsat;
};

ParsedQueries parse_queries(PyObject* queries, const Index& index, W max_bytes) {
    if (!PyTuple_CheckExact(queries)) throw Invalid("queries must be an exact tuple");
    W count = static_cast<W>(PyTuple_GET_SIZE(queries));
    if (count > MAX_QUERIES) throw Resource("too many queries");
    ParsedQueries result;
    result.offsets.reserve(count + 1);
    result.immediate_unsat.assign(count, 0);
    result.offsets.push_back(0);
    W observed_restrictions = 0;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(queries); ++i) {
        PyObject* query = PyTuple_GET_ITEM(queries, i);
        if (!PyTuple_CheckExact(query)) throw Invalid("each query must be an exact tuple");
        observed_restrictions += static_cast<W>(PyTuple_GET_SIZE(query));
        if (observed_restrictions > MAX_RESTRICTIONS) throw Resource("too many restrictions");
        U prior = UINT32_MAX;
        for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(query); ++j) {
            PyObject* row = PyTuple_GET_ITEM(query, j);
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2)
                throw Invalid("restriction must be an exact vertex/mask pair");
            U vertex = static_cast<U>(integer(PyTuple_GET_ITEM(row, 0),
                                              "restriction vertex outside graph", index.n - 1));
            W allowed = integer(PyTuple_GET_ITEM(row, 1), "invalid restriction mask");
            if (j && vertex <= prior) throw Invalid("restriction vertices must be strictly increasing");
            prior = vertex;
            W available = allowed & index.masks[vertex];
            U count_available = static_cast<U>(__builtin_popcountll(available));
            if (!count_available) result.immediate_unsat[i] = 1;
            else if (count_available == 1)
                result.assumptions.push_back(
                    index.choice_literal(vertex, first_bit(available)));
        }
        result.offsets.push_back(static_cast<U>(result.assumptions.size()));
    }
    W required = 4 * W(result.offsets.capacity())
        + sizeof(Lit) * W(result.assumptions.capacity()) + count;
    if (required > max_bytes) throw Resource("query assumptions exceed payload cap");
    return result;
}

constexpr const char* TAG = "spectra.minicard.session.v1";
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
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown MiniCard session error"); }
    return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
    PyObject *edges, *masks, *cap;
    if (!PyArg_ParseTuple(args, "OOO", &edges, &masks, &cap)) return nullptr;
    try {
        auto index = std::make_unique<Index>(edges, masks, integer(cap, "invalid payload cap"));
        PyObject* capsule = PyCapsule_New(index.get(), TAG, destroy);
        if (capsule) index.release();
        return capsule;
    } catch (...) { return failure(); }
}

PyObject* info(PyObject*, PyObject* capsule) {
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    return Py_BuildValue("(IIKK)", index->n, index->palette,
        static_cast<unsigned long long>(index->clauses),
        static_cast<unsigned long long>(index->payload));
}

PyObject* solve_batch(PyObject*, PyObject* args) {
    PyObject *capsule, *queries, *cap;
    if (!PyArg_ParseTuple(args, "OOO", &capsule, &queries, &cap)) return nullptr;
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    try {
        W max_bytes = integer(cap, "invalid batch payload cap");
        ParsedQueries parsed = parse_queries(queries, *index, max_bytes);
        W query_count = parsed.offsets.size() - 1;
        W output_bytes = query_count + query_count * W(index->n) + 24 * query_count;
        if (output_bytes > max_bytes) throw Resource("MiniCard output exceeds payload cap");
        PyObject* statuses = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(query_count));
        PyObject* labels = PyBytes_FromStringAndSize(nullptr,
            static_cast<Py_ssize_t>(query_count * W(index->n)));
        if (!statuses || !labels) {
            Py_XDECREF(statuses); Py_XDECREF(labels); return nullptr;
        }
        char* status_data = PyBytes_AS_STRING(statuses);
        char* label_data = PyBytes_AS_STRING(labels);
        std::vector<W> times(query_count, 0), decisions(query_count, 0), conflicts(query_count, 0);
        {
            PyThreadState* thread = PyEval_SaveThread();
            try {
                Minisat::vec<Lit> assumptions;
                for (W query = 0; query < query_count; ++query) {
                    auto started = std::chrono::steady_clock::now();
                    char* output = label_data + query * W(index->n);
                    std::memset(output, 0, index->n);
                    bool sat = false;
                    W prior_decisions = index->solver->decisions;
                    W prior_conflicts = index->solver->conflicts;
                    if (!parsed.immediate_unsat[query] && !index->base_unsat) {
                        assumptions.clear();
                        for (U i = parsed.offsets[query]; i < parsed.offsets[query + 1]; ++i)
                            assumptions.push(parsed.assumptions[i]);
                        sat = index->solver->solve(assumptions);
                    }
                    if (sat) {
                        for (U v = 0; v < index->n; ++v) {
                            lbool value = index->solver->modelValue(static_cast<Var>(v));
                            bool high = value == l_True;
                            if (value != l_True && value != l_False && value != l_Undef)
                                throw std::logic_error("unexpected MiniCard model value");
                            output[v] = static_cast<char>(high ? index->high[v] : index->low[v]);
                        }
                        status_data[query] = 0;
                    } else {
                        status_data[query] = 1;
                    }
                    decisions[query] = index->solver->decisions - prior_decisions;
                    conflicts[query] = index->solver->conflicts - prior_conflicts;
                    times[query] = static_cast<W>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count());
                }
            } catch (...) {
                PyEval_RestoreThread(thread);
                Py_DECREF(statuses); Py_DECREF(labels);
                throw;
            }
            PyEval_RestoreThread(thread);
        }
        auto make_tuple = [query_count](const std::vector<W>& values) -> PyObject* {
            PyObject* result = PyTuple_New(static_cast<Py_ssize_t>(query_count));
            if (!result) return nullptr;
            for (W i = 0; i < query_count; ++i)
                PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(i),
                    PyLong_FromUnsignedLongLong(values[i]));
            return result;
        };
        PyObject* time_tuple = make_tuple(times);
        PyObject* decision_tuple = make_tuple(decisions);
        PyObject* conflict_tuple = make_tuple(conflicts);
        if (!time_tuple || !decision_tuple || !conflict_tuple) {
            Py_DECREF(statuses); Py_DECREF(labels);
            Py_XDECREF(time_tuple); Py_XDECREF(decision_tuple); Py_XDECREF(conflict_tuple);
            return nullptr;
        }
        return Py_BuildValue("(NNNNN)", statuses, labels, time_tuple,
                             decision_tuple, conflict_tuple);
    } catch (...) { return failure(); }
}

PyObject* abi(PyObject*, PyObject*) { return PyLong_FromLong(1); }
PyMethodDef methods[] = {
    {"abi", abi, METH_NOARGS, "Return ABI version."},
    {"create", create, METH_VARARGS, "Prepare an unmodified persistent MiniCard solver."},
    {"info", info, METH_O, "Return adapter geometry."},
    {"solve_batch", solve_batch, METH_VARARGS, "Solve a batch under assumptions."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_spectra_minicard_session", nullptr, -1,
                      methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__spectra_minicard_session() { return PyModule_Create(&module); }
