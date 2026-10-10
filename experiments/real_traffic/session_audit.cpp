// Independent native verifier for complete real-traffic solver sessions.
// It rebuilds only the original graph/list relation and retained contradiction paths.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U MAX_N = 100000;
constexpr W MAX_EDGES = 2000000;
constexpr W MAX_QUERIES = 1000000;
constexpr W MAX_PATH_LITERALS = 64000000;
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

int64_t signed_integer(PyObject* object, const char* message, int64_t cap) {
    if (!PyLong_CheckExact(object)) throw Invalid(message);
    long long value = PyLong_AsLongLong(object);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw Invalid(message);
    }
    if (!value || value > cap || value < -cap) throw Invalid(message);
    return static_cast<int64_t>(value);
}

U first_bit(W mask) {
    if (!mask) throw Invalid("empty colour mask");
    return static_cast<U>(__builtin_ctzll(mask));
}

struct Index {
    U n = 0;
    U literals = 0;
    U words = 0;
    U vertex_words = 0;
    W payload = 0;
    bool dense_edges = false;
    std::vector<W> masks;
    std::vector<std::pair<U,U>> edges;
    std::vector<W> adjacency_bits;
    std::vector<W> base_implications;
    std::vector<U> query_offsets;
    std::vector<U> query_vertices;
    std::vector<W> query_allowed;
    std::vector<uint8_t> expected;
    std::vector<U> proof_offsets;
    std::vector<int32_t> proof_literals;

    U node(int32_t literal) const {
        U variable = static_cast<U>(literal > 0 ? literal - 1 : -int64_t(literal) - 1);
        return 2 * variable + (literal > 0 ? 1U : 0U);
    }

    int32_t choice_literal(U vertex, U colour) const {
        W mask = masks[vertex];
        U low = first_bit(mask);
        W rest = mask & (mask - 1);
        if (!rest) {
            if (colour != low) throw Invalid("colour outside singleton domain");
            return static_cast<int32_t>(vertex + 1);
        }
        U high = first_bit(rest);
        if (colour == low) return -static_cast<int32_t>(vertex + 1);
        if (colour == high) return static_cast<int32_t>(vertex + 1);
        throw Invalid("colour outside binary domain");
    }

    void add_implication(U source, U target) {
        base_implications[W(source) * words + target / 64] |= W(1) << (target % 64);
    }

    bool has_base(U source, U target) const {
        return (base_implications[W(source) * words + target / 64] >> (target % 64)) & 1;
    }

    Index(PyObject* edges_object, PyObject* masks_object, PyObject* queries_object,
          PyObject* statuses_object, PyObject* contradictions_object, W max_bytes) {
        if (!PyTuple_CheckExact(edges_object) || !PyTuple_CheckExact(masks_object)
                || !PyTuple_CheckExact(queries_object) || !PyTuple_CheckExact(statuses_object)
                || !PyTuple_CheckExact(contradictions_object))
            throw Invalid("audit inputs must be exact tuples");
        n = static_cast<U>(PyTuple_GET_SIZE(masks_object));
        if (!n || n > MAX_N) throw Resource("audit variable inventory outside contract");
        literals = 2 * n;
        words = (literals + 63) / 64;
        vertex_words = (n + 63) / 64;
        W edge_count = static_cast<W>(PyTuple_GET_SIZE(edges_object));
        W query_count = static_cast<W>(PyTuple_GET_SIZE(queries_object));
        if (edge_count > MAX_EDGES || query_count > MAX_QUERIES)
            throw Resource("audit inventory exceeds geometry limits");
        if (PyTuple_GET_SIZE(statuses_object) != static_cast<Py_ssize_t>(query_count)
                || PyTuple_GET_SIZE(contradictions_object) != static_cast<Py_ssize_t>(query_count))
            throw Invalid("audit query records differ");
        W required = 8 * W(n) + 8 * edge_count + 8 * W(literals) * words
            + 24 * query_count + 4096;
        if (required > max_bytes) throw Resource("audit index exceeds payload cap");
        W dense_bytes = 8 * W(n) * vertex_words;
        dense_edges = W(n) * vertex_words <= 2 * edge_count
            && dense_bytes <= max_bytes - required;

        masks.resize(n);
        if (dense_edges) adjacency_bits.assign(W(n) * vertex_words, 0);
        base_implications.assign(W(literals) * words, 0);
        for (U vertex = 0; vertex < n; ++vertex) {
            W mask = integer(PyTuple_GET_ITEM(masks_object, vertex), "invalid audit mask");
            U count = static_cast<U>(__builtin_popcountll(mask));
            if (count < 1 || count > 2) throw Invalid("audit requires one or two colours");
            masks[vertex] = mask;
            if (count == 1) {
                U positive = 2 * vertex + 1;
                add_implication(positive ^ 1U, positive);
            }
        }

        edges.reserve(edge_count);
        U previous_left = 0, previous_right = 0;
        bool have_previous = false;
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(edges_object); ++i) {
            PyObject* row = PyTuple_GET_ITEM(edges_object, i);
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2)
                throw Invalid("audit edge must be an exact pair");
            U left = static_cast<U>(integer(PyTuple_GET_ITEM(row, 0), "bad edge endpoint", n - 1));
            U right = static_cast<U>(integer(PyTuple_GET_ITEM(row, 1), "bad edge endpoint", n - 1));
            if (left >= right || (have_previous && std::make_pair(left, right)
                    <= std::make_pair(previous_left, previous_right)))
                throw Invalid("audit edges must be strictly sorted canonical nonloops");
            previous_left = left; previous_right = right; have_previous = true;
            edges.emplace_back(left, right);
            if (dense_edges) {
                adjacency_bits[W(left) * vertex_words + right / 64]
                    |= W(1) << (right % 64);
                adjacency_bits[W(right) * vertex_words + left / 64]
                    |= W(1) << (left % 64);
            }
            W shared = masks[left] & masks[right];
            while (shared) {
                W bit = shared & (~shared + 1);
                shared &= shared - 1;
                U colour = first_bit(bit);
                U a = node(choice_literal(left, colour));
                U b = node(choice_literal(right, colour));
                add_implication(a, b ^ 1U);
                add_implication(b, a ^ 1U);
            }
        }

        query_offsets.reserve(query_count + 1);
        query_offsets.push_back(0);
        expected.resize(query_count);
        proof_offsets.reserve(2 * query_count + 1);
        proof_offsets.push_back(0);
        W total_paths = 0;
        for (U query_index = 0; query_index < query_count; ++query_index) {
            PyObject* query = PyTuple_GET_ITEM(queries_object, query_index);
            if (!PyTuple_CheckExact(query)) throw Invalid("audit query must be an exact tuple");
            U prior = UINT32_MAX;
            for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(query); ++j) {
                PyObject* row = PyTuple_GET_ITEM(query, j);
                if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2)
                    throw Invalid("audit restriction must be an exact pair");
                U vertex = static_cast<U>(integer(PyTuple_GET_ITEM(row, 0),
                                                  "bad restriction vertex", n - 1));
                W allowed = integer(PyTuple_GET_ITEM(row, 1), "bad restriction mask");
                if (j && vertex <= prior) throw Invalid("audit restrictions are not canonical");
                prior = vertex;
                W available = allowed & masks[vertex];
                if (__builtin_popcountll(available) != 1)
                    throw Invalid("audit restriction must choose one available colour");
                query_vertices.push_back(vertex);
                query_allowed.push_back(allowed);
            }
            query_offsets.push_back(static_cast<U>(query_vertices.size()));

            PyObject* status = PyTuple_GET_ITEM(statuses_object, query_index);
            if (!PyUnicode_CheckExact(status)) throw Invalid("audit status must be exact text");
            const char* text = PyUnicode_AsUTF8(status);
            if (!text) throw Invalid("audit status is not UTF-8");
            if (std::strcmp(text, "SAT") == 0) expected[query_index] = 0;
            else if (std::strcmp(text, "UNSAT") == 0) expected[query_index] = 1;
            else throw Invalid("audit status must be SAT or UNSAT");

            PyObject* certificate = PyTuple_GET_ITEM(contradictions_object, query_index);
            if (!expected[query_index]) {
                if (certificate != Py_None) throw Invalid("SAT query carries a contradiction");
                proof_offsets.push_back(static_cast<U>(proof_literals.size()));
                proof_offsets.push_back(static_cast<U>(proof_literals.size()));
                continue;
            }
            if (!PyDict_CheckExact(certificate)) throw Invalid("UNSAT query lacks a contradiction dict");
            PyObject* schema = PyDict_GetItemString(certificate, "schema");
            PyObject* variable_object = PyDict_GetItemString(certificate, "variable");
            PyObject* first_path = PyDict_GetItemString(certificate, "positive_to_negative");
            PyObject* second_path = PyDict_GetItemString(certificate, "negative_to_positive");
            if (!schema || !variable_object || !first_path || !second_path
                    || PyDict_Size(certificate) != 4 || !PyUnicode_CheckExact(schema)
                    || std::strcmp(PyUnicode_AsUTF8(schema),
                                   "spectra.real_traffic.contradiction.v1") != 0)
                throw Invalid("invalid contradiction schema");
            U variable = static_cast<U>(integer(variable_object, "bad contradiction variable", n - 1));
            PyObject* paths[2] = {first_path, second_path};
            int32_t starts[2] = {static_cast<int32_t>(variable + 1),
                                 -static_cast<int32_t>(variable + 1)};
            int32_t targets[2] = {-starts[0], -starts[1]};
            for (int which = 0; which < 2; ++which) {
                PyObject* path = paths[which];
                if (!PyList_CheckExact(path) || PyList_GET_SIZE(path) < 2)
                    throw Invalid("contradiction path is too short");
                total_paths += static_cast<W>(PyList_GET_SIZE(path));
                if (total_paths > MAX_PATH_LITERALS) throw Resource("too many proof literals");
                size_t begin = proof_literals.size();
                for (Py_ssize_t j = 0; j < PyList_GET_SIZE(path); ++j)
                    proof_literals.push_back(static_cast<int32_t>(signed_integer(
                        PyList_GET_ITEM(path, j), "bad contradiction literal", n)));
                if (proof_literals[begin] != starts[which]
                        || proof_literals.back() != targets[which])
                    throw Invalid("contradiction endpoints differ");
                proof_offsets.push_back(static_cast<U>(proof_literals.size()));
            }
        }
        payload = 8 * W(masks.capacity() + adjacency_bits.capacity()
                        + base_implications.capacity() + query_allowed.capacity())
            + 8 * W(edges.capacity())
            + 4 * W(query_offsets.capacity() + query_vertices.capacity()
                    + proof_offsets.capacity() + proof_literals.capacity())
            + expected.capacity();
        if (payload > max_bytes) throw Resource("audit allocated payload exceeds cap");
    }

    bool query_has(U query_index, U source, U target) const {
        for (U i = query_offsets[query_index]; i < query_offsets[query_index + 1]; ++i) {
            U vertex = query_vertices[i];
            U colour = first_bit(query_allowed[i] & masks[vertex]);
            U literal = node(choice_literal(vertex, colour));
            if ((literal ^ 1U) == source && literal == target) return true;
        }
        return false;
    }
};

constexpr const char* TAG = "spectra.real.traffic.audit.v1";
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
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown session audit error"); }
    return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
    PyObject *edges, *masks, *queries, *statuses, *contradictions, *cap;
    if (!PyArg_ParseTuple(args, "OOOOOO", &edges, &masks, &queries, &statuses,
                          &contradictions, &cap)) return nullptr;
    try {
        auto index = std::make_unique<Index>(edges, masks, queries, statuses,
                                             contradictions, integer(cap, "invalid audit cap"));
        PyObject* capsule = PyCapsule_New(index.get(), TAG, destroy);
        if (capsule) index.release();
        return capsule;
    } catch (...) { return failure(); }
}

PyObject* info(PyObject*, PyObject* capsule) {
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    return Py_BuildValue("(IIKKK)", index->n,
        static_cast<unsigned>(index->expected.size()),
        static_cast<unsigned long long>(index->edges.size()),
        static_cast<unsigned long long>(index->proof_literals.size()),
        static_cast<unsigned long long>(index->payload));
}

PyObject* verify_batch(PyObject*, PyObject* args) {
    PyObject *capsule, *statuses_object, *labels_object;
    if (!PyArg_ParseTuple(args, "OOO", &capsule, &statuses_object, &labels_object))
        return nullptr;
    auto* index = static_cast<Index*>(PyCapsule_GetPointer(capsule, TAG));
    if (!index) return nullptr;
    try {
        if (!PyBytes_CheckExact(statuses_object) || !PyBytes_CheckExact(labels_object))
            throw Invalid("session outputs must be exact bytes");
        W queries = index->expected.size();
        if (PyBytes_GET_SIZE(statuses_object) != static_cast<Py_ssize_t>(queries)
                || PyBytes_GET_SIZE(labels_object) != static_cast<Py_ssize_t>(queries * W(index->n)))
            throw Invalid("session output geometry differs");
        const uint8_t* statuses = reinterpret_cast<const uint8_t*>(
            PyBytes_AS_STRING(statuses_object));
        const uint8_t* labels = reinterpret_cast<const uint8_t*>(
            PyBytes_AS_STRING(labels_object));
        W sat = 0, unsat = 0, path_edges = 0;
        W check_ns = 0, proof_ns = 0;
        std::vector<W> query_times(queries, 0);
        std::vector<W> seen_by_colour;
        if (index->dense_edges) seen_by_colour.resize(64 * W(index->vertex_words));
        {
            PyThreadState* thread = PyEval_SaveThread();
            try {
                for (U query = 0; query < queries; ++query) {
                    if (statuses[query] > 1 || statuses[query] != index->expected[query])
                        throw std::logic_error("solver status differs from audited outcome");
                    if (!statuses[query]) {
                        auto started = std::chrono::steady_clock::now();
                        const uint8_t* answer = labels + W(query) * index->n;
                        if (index->dense_edges)
                            std::fill(seen_by_colour.begin(), seen_by_colour.end(), 0);
                        for (U vertex = 0; vertex < index->n; ++vertex) {
                            U colour = answer[vertex];
                            if (colour >= 64
                                    || !(index->masks[vertex] & (W(1) << colour)))
                                throw std::logic_error("SAT answer violates an original list");
                            if (index->dense_edges) {
                                W* seen = seen_by_colour.data() + W(colour) * index->vertex_words;
                                const W* neighbors = index->adjacency_bits.data()
                                    + W(vertex) * index->vertex_words;
                                for (U word = 0; word < index->vertex_words; ++word)
                                    if (seen[word] & neighbors[word])
                                        throw std::logic_error("SAT answer violates an original conflict edge");
                                seen[vertex / 64] |= W(1) << (vertex % 64);
                            }
                        }
                        if (!index->dense_edges)
                            for (auto edge : index->edges)
                                if (answer[edge.first] == answer[edge.second])
                                    throw std::logic_error("SAT answer violates an original conflict edge");
                        for (U i = index->query_offsets[query];
                             i < index->query_offsets[query + 1]; ++i)
                            if (!(index->query_allowed[i] & (W(1) << answer[index->query_vertices[i]])))
                                throw std::logic_error("SAT answer violates its request");
                        W elapsed = static_cast<W>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started).count());
                        check_ns += elapsed;
                        query_times[query] = elapsed;
                        ++sat;
                    } else {
                        auto started = std::chrono::steady_clock::now();
                        U paths_begin = 2 * query;
                        for (U path = paths_begin; path < paths_begin + 2; ++path) {
                            U begin = index->proof_offsets[path];
                            U end = index->proof_offsets[path + 1];
                            if (end <= begin + 1) throw std::logic_error("empty retained contradiction path");
                            for (U i = begin; i + 1 < end; ++i) {
                                U source = index->node(index->proof_literals[i]);
                                U target = index->node(index->proof_literals[i + 1]);
                                if (!index->has_base(source, target)
                                        && !index->query_has(query, source, target))
                                    throw std::logic_error("retained contradiction uses a missing implication");
                                ++path_edges;
                            }
                        }
                        W elapsed = static_cast<W>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started).count());
                        proof_ns += elapsed;
                        query_times[query] = elapsed;
                        ++unsat;
                    }
                }
            } catch (...) {
                PyEval_RestoreThread(thread);
                throw;
            }
            PyEval_RestoreThread(thread);
        }
        PyObject* timing = PyTuple_New(static_cast<Py_ssize_t>(queries));
        if (!timing) return nullptr;
        for (W query = 0; query < queries; ++query)
            PyTuple_SET_ITEM(timing, static_cast<Py_ssize_t>(query),
                PyLong_FromUnsignedLongLong(query_times[query]));
        return Py_BuildValue("(KKKKKN)", static_cast<unsigned long long>(sat),
            static_cast<unsigned long long>(unsat),
            static_cast<unsigned long long>(check_ns),
            static_cast<unsigned long long>(proof_ns),
            static_cast<unsigned long long>(path_edges), timing);
    } catch (...) { return failure(); }
}

PyObject* abi(PyObject*, PyObject*) { return PyLong_FromLong(1); }
PyMethodDef methods[] = {
    {"abi", abi, METH_NOARGS, "Return ABI version."},
    {"create", create, METH_VARARGS, "Prepare an independent original-input session auditor."},
    {"info", info, METH_O, "Return audit geometry and payload."},
    {"verify_batch", verify_batch, METH_VARARGS, "Verify all SAT witnesses and UNSAT paths."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_spectra_session_audit", nullptr, -1,
                      methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__spectra_session_audit() { return PyModule_Create(&module); }
