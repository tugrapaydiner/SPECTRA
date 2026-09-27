// Optional CPython preprocessing accelerator. No SVM arithmetic lives here.
// Build explicitly for the running CPython ABI with strict FP compiler flags.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cfenv>
#include <limits>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "IEEE binary64 required");

namespace {
constexpr const char* capsule_name = "spectra.preprocessing.plan.v1";
constexpr Py_ssize_t max_elements = 8000000;
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* p=nullptr): p(p) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    PyObject* release() { auto* v=p; p=nullptr; return v; }
};
struct Numeric { Py_ssize_t col,out; double fill,mean,scale; };
struct Category { Py_ssize_t col,offset,count; PyObject* lookup; bool strict; };
struct Plan {
    Py_ssize_t columns,features,cap;
    std::vector<Numeric> numeric;
    std::vector<Category> category;
    ~Plan() { for (auto& c:category) Py_DECREF(c.lookup); }
};
void require(bool ok,const char* msg) { if(!ok) throw std::invalid_argument(msg); }
Py_ssize_t integer(PyObject* p) {
    require(PyLong_CheckExact(p),"expected exact integer");
    auto v=PyLong_AsSsize_t(p);
    if(PyErr_Occurred()) throw std::invalid_argument("integer outside range");
    return v;
}
double constant(PyObject* p) {
    require(PyFloat_CheckExact(p),"expected exact float constant");
    double v=PyFloat_AS_DOUBLE(p);
    require(std::isfinite(v),"nonfinite plan constant");
    return v;
}
void destroy(PyObject* cap) {
    auto* plan=static_cast<Plan*>(PyCapsule_GetPointer(cap,capsule_name));
    if(plan) delete plan;
    else PyErr_Clear();
}
// No exception crosses the Python C boundary, including allocation failures.
template<class F> PyObject* protect(F f) {
    try { return f(); }
    catch(const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch(const std::exception& e) {
        if(!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,e.what());
        return nullptr;
    }
}
PyObject* prepare(PyObject*,PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *pc,*pf,*pr,*ns,*cs;
        if(!PyArg_ParseTuple(args,"OOOOO",&pc,&pf,&pr,&ns,&cs)) return nullptr;
        auto p=std::make_unique<Plan>();
        p->columns=integer(pc);p->features=integer(pf);p->cap=integer(pr);
        require(p->columns>=1&&p->columns<=4096&&p->features>=1&&p->features<=4096,
                "invalid plan dimensions");
        require(p->cap>=1&&p->cap<=65536&&p->cap*std::max(p->columns,p->features)<=max_elements,
                "invalid plan row cap");
        require(PyTuple_CheckExact(ns)&&PyTuple_CheckExact(cs),"plan operations must be tuples");
        auto nn=PyTuple_GET_SIZE(ns),nc=PyTuple_GET_SIZE(cs);
        require(nn+nc>=1&&nn+nc<=4096,"invalid operation count");
        std::vector<bool> used(size_t(p->features),false);
        for(Py_ssize_t i=0;i<nn;++i) {
            auto* op=PyTuple_GET_ITEM(ns,i);
            require(PyTuple_CheckExact(op)&&PyTuple_GET_SIZE(op)==5,"invalid numeric tuple");
            Numeric n{integer(PyTuple_GET_ITEM(op,0)),integer(PyTuple_GET_ITEM(op,1)),
                constant(PyTuple_GET_ITEM(op,2)),constant(PyTuple_GET_ITEM(op,3)),constant(PyTuple_GET_ITEM(op,4))};
            require(n.col>=0&&n.col<p->columns&&n.out>=0&&n.out<p->features&&n.scale>0,
                    "invalid numeric operation");
            require(!used[size_t(n.out)],"overlapping outputs");used[size_t(n.out)]=true;
            p->numeric.push_back(n);
        }
        for(Py_ssize_t i=0;i<nc;++i) {
            auto* op=PyTuple_GET_ITEM(cs,i);
            require(PyTuple_CheckExact(op)&&PyTuple_GET_SIZE(op)==4,"invalid categorical tuple");
            auto col=integer(PyTuple_GET_ITEM(op,0)),offset=integer(PyTuple_GET_ITEM(op,1));
            auto* values=PyTuple_GET_ITEM(op,2);auto* strict=PyTuple_GET_ITEM(op,3);
            require(PyTuple_CheckExact(values)&&PyBool_Check(strict),"invalid categories or policy");
            auto count=PyTuple_GET_SIZE(values);
            require(col>=0&&col<p->columns&&offset>=0&&count>=1&&count<=4096&&offset+count<=p->features,
                    "invalid categorical geometry");
            Ref dict(PyDict_New()); if(!dict.p) return nullptr;
            for(Py_ssize_t j=0;j<count;++j) {
                auto* key=PyTuple_GET_ITEM(values,j);
                require(PyUnicode_CheckExact(key),"category must be exact string");
                Py_ssize_t len=0;
                if(!PyUnicode_AsUTF8AndSize(key,&len)) return nullptr;
                require(len<=4096,"category exceeds byte limit");
                require(!used[size_t(offset+j)],"overlapping outputs");used[size_t(offset+j)]=true;
                int contains=PyDict_Contains(dict.p,key);if(contains<0) return nullptr;
                require(!contains,"duplicate category");
                Ref index(PyLong_FromSsize_t(offset+j));if(!index.p) return nullptr;
                if(PyDict_SetItem(dict.p,key,index.p)<0) return nullptr;
            }
            p->category.push_back(Category{col,offset,count,dict.p,strict==Py_True});
            dict.release();
        }
        for(bool v:used) require(v,"missing output feature");
        Ref capsule(PyCapsule_New(p.get(),capsule_name,destroy));
        if(!capsule.p) return nullptr;
        p.release();return capsule.release();
    });
}
PyObject* transform_impl(PyObject* args, bool raw_input) {
    return protect([&]() -> PyObject* {
        PyObject *cap,*rows;
        if(!PyArg_ParseTuple(args,"OO",&cap,&rows)) return nullptr;
        auto* p=static_cast<Plan*>(PyCapsule_GetPointer(cap,capsule_name));
        if(!p) return nullptr;
        require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
        if(raw_input && !PyList_CheckExact(rows) && !PyTuple_CheckExact(rows)) {
            Py_INCREF(Py_NotImplemented);return Py_NotImplemented;
        }
        require(raw_input || PyList_CheckExact(rows),"expected checked row list");
        auto n=PySequence_Fast_GET_SIZE(rows);
        require(n<=p->cap,"batch exceeds row/element cap");
        // Validate structural bounds before allocating; no Python callbacks occur
        // during the fast path. The wrapper owns the bounded materialization.
        for(Py_ssize_t r=0;r<n;++r) {
            auto* row=PySequence_Fast_GET_ITEM(rows,r);
            if(raw_input && !PyTuple_CheckExact(row) && !PyList_CheckExact(row)) {
                Py_INCREF(Py_NotImplemented);return Py_NotImplemented;
            }
            require(raw_input || PyTuple_CheckExact(row),"invalid raw row");
            require(PySequence_Fast_GET_SIZE(row)==p->columns,"invalid raw row");
        }
        Py_ssize_t size=n*p->features*Py_ssize_t(sizeof(double));
        Ref result(PyByteArray_FromStringAndSize(nullptr,size));if(!result.p) return nullptr;
        char* raw=PyByteArray_AS_STRING(result.p);
        std::memset(raw,0,size_t(size));
        auto* out=reinterpret_cast<double*>(raw);
        for(Py_ssize_t r=0;r<n;++r) {
            auto* row=PySequence_Fast_GET_ITEM(rows,r);auto base=r*p->features;
            for(const auto& op:p->numeric) {
                auto* v=PySequence_Fast_GET_ITEM(row,op.col);double x;
                if(v==Py_None) x=op.fill;
                else if(PyFloat_CheckExact(v)) x=PyFloat_AS_DOUBLE(v);
                else if(PyLong_CheckExact(v)) {
                    x=PyLong_AsDouble(v);
                    if(PyErr_Occurred()) { PyErr_Clear();throw std::invalid_argument("invalid numeric feature"); }
                } else {
                    // Preserve numbers.Real and subclass semantics in Python.
                    // No transformed output is exposed on this fallback path.
                    Py_INCREF(Py_NotImplemented);return Py_NotImplemented;
                }
                if(std::isnan(x)) x=op.fill;
                require(std::isfinite(x),"infinite raw feature");
                const double centered=x-op.mean;
                const double value=centered/op.scale;
                require(std::isfinite(value),"nonfinite transformed feature");
                out[base+op.out]=value;
            }
            for(const auto& op:p->category) {
                auto* v=PySequence_Fast_GET_ITEM(row,op.col);
                if(!PyUnicode_CheckExact(v)) { Py_INCREF(Py_NotImplemented);return Py_NotImplemented; }
                Py_ssize_t len=0;if(!PyUnicode_AsUTF8AndSize(v,&len)) return nullptr;
                require(len<=4096,"categorical value exceeds byte limit");
                auto* index=PyDict_GetItemWithError(op.lookup,v);
                if(!index) {
                    if(PyErr_Occurred()) return nullptr;
                    require(!op.strict,"unknown categorical value");
                } else out[base+PyLong_AsSsize_t(index)]=1.;
            }
        }
        return result.release();
    });
}
PyObject* transform(PyObject*,PyObject* args) { return transform_impl(args,false); }
PyObject* transform_raw(PyObject*,PyObject* args) { return transform_impl(args,true); }
PyObject* abi(PyObject*,PyObject*) { return PyLong_FromLong(2); }
PyMethodDef methods[]={
    {"prepare",prepare,METH_VARARGS,"Create an owned, checked preprocessing plan."},
    {"transform",transform,METH_VARARGS,"Return fresh binary64 bytes or NotImplemented for reference fallback."},
    {"transform_raw",transform_raw,METH_VARARGS,"Traverse exact built-in containers under the GIL; otherwise request reference fallback."},
    {"abi",abi,METH_NOARGS,"Return the preprocessing ABI version."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef definition={PyModuleDef_HEAD_INIT,"_spectra_preprocess",
    "Optional strict-arithmetic preprocessing. GIL held; not a sandbox.",0,methods};
}
PyMODINIT_FUNC PyInit__spectra_preprocess() { return PyModule_Create(&definition); }
