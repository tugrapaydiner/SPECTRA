// Native MiniCard model materialisation for the real-traffic control arm.
// This does not change MiniCard search; it removes a Python per-variable loop.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U MAX_VARIABLES = 100000;

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

struct Decoder {
    U variables;
    std::vector<uint8_t> low;
    std::vector<uint8_t> high;
    W payload_bytes = 0;

    Decoder(PyObject* masks, W maximum_bytes) {
        if (!PyTuple_CheckExact(masks)) throw Invalid("masks must be an exact tuple");
        const Py_ssize_t count = PyTuple_GET_SIZE(masks);
        if (count < 0 || static_cast<W>(count) > MAX_VARIABLES) {
            throw Resource("too many model variables");
        }
        variables = static_cast<U>(count);
        const W required = 2 * static_cast<W>(variables);
        if (required > maximum_bytes) throw Resource("decoder payload exceeds cap");
        low.resize(variables);
        high.resize(variables);
        for (U variable = 0; variable < variables; ++variable) {
            W mask = integer(PyTuple_GET_ITEM(masks, variable), "invalid binary mask");
            const unsigned choices = static_cast<unsigned>(__builtin_popcountll(mask));
            if (!mask || choices > 2) throw Invalid("decoder requires one or two colors per variable");
            const unsigned first_color = first(mask);
            W remaining = mask & (mask - 1);
            const unsigned second_color = remaining ? first(remaining) : first_color;
            if (first_color > 255 || second_color > 255) throw Invalid("color does not fit a byte");
            low[variable] = static_cast<uint8_t>(first_color);
            high[variable] = static_cast<uint8_t>(second_color);
        }
        payload_bytes = low.capacity() + high.capacity();
        if (payload_bytes > maximum_bytes) throw std::logic_error("decoder payload bound underestimated");
    }
};

constexpr const char* TAG = "spectra.real_traffic.model_decoder.v1";

void destroy(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, TAG);
    if (pointer) delete static_cast<Decoder*>(pointer);
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
        PyErr_SetString(PyExc_RuntimeError, "unknown model decoder error");
    }
    return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
    PyObject* masks;
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "OO", &masks, &cap)) return nullptr;
    try {
        auto decoder = std::make_unique<Decoder>(
            masks, integer(cap, "invalid decoder payload cap"));
        PyObject* capsule = PyCapsule_New(decoder.get(), TAG, destroy);
        if (capsule) decoder.release();
        return capsule;
    } catch (...) {
        return failure();
    }
}

PyObject* decode(PyObject*, PyObject* args) {
    PyObject* capsule;
    PyObject* model;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &model)) return nullptr;
    auto* decoder = static_cast<Decoder*>(PyCapsule_GetPointer(capsule, TAG));
    if (!decoder) return nullptr;
    try {
        if (!PyList_CheckExact(model) && !PyTuple_CheckExact(model)) {
            throw Invalid("model must be an exact list or tuple");
        }
        const Py_ssize_t count = PySequence_Size(model);
        if (count < 0) throw Invalid("invalid model length");
        std::vector<char> output(decoder->variables);
        std::vector<uint8_t> seen(decoder->variables, 0);
        for (U variable = 0; variable < decoder->variables; ++variable) {
            output[variable] = static_cast<char>(decoder->low[variable]);
        }
        for (Py_ssize_t index = 0; index < count; ++index) {
            PyObject* item = PyList_CheckExact(model)
                ? PyList_GET_ITEM(model, index)
                : PyTuple_GET_ITEM(model, index);
            if (!PyLong_CheckExact(item)) throw Invalid("model literal must be an exact integer");
            long long literal = PyLong_AsLongLong(item);
            if (PyErr_Occurred()) {
                PyErr_Clear();
                throw Invalid("model literal exceeds signed range");
            }
            if (!literal) throw Invalid("model contains literal zero");
            const unsigned long long magnitude = literal < 0
                ? static_cast<unsigned long long>(-(literal + 1)) + 1
                : static_cast<unsigned long long>(literal);
            if (magnitude > decoder->variables) {
                throw Invalid("model literal exceeds the declared variables");
            }
            const U variable = static_cast<U>(magnitude - 1);
            if (seen[variable]) throw Invalid("model assigns a variable more than once");
            seen[variable] = 1;
            if (literal > 0) output[variable] = static_cast<char>(decoder->high[variable]);
        }
        return PyBytes_FromStringAndSize(
            output.data(), static_cast<Py_ssize_t>(output.size()));
    } catch (...) {
        return failure();
    }
}

PyObject* info(PyObject*, PyObject* capsule) {
    auto* decoder = static_cast<Decoder*>(PyCapsule_GetPointer(capsule, TAG));
    if (!decoder) return nullptr;
    return Py_BuildValue(
        "(IK)", decoder->variables,
        static_cast<unsigned long long>(decoder->payload_bytes));
}

PyObject* abi(PyObject*, PyObject*) {
    return PyLong_FromLong(1);
}

PyMethodDef methods[] = {
    {"abi", abi, METH_NOARGS, "Return the decoder ABI version."},
    {"create", create, METH_VARARGS, "Prepare immutable binary-list decoding."},
    {"decode", decode, METH_VARARGS, "Materialize one complete byte witness."},
    {"info", info, METH_O, "Return decoder geometry and payload."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_spectra_model_decoder",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
}  // namespace

PyMODINIT_FUNC PyInit__spectra_model_decoder() {
    return PyModule_Create(&module);
}
