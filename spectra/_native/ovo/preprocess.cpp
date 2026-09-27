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
#include <cstdint>

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

// The private binding accepts addresses only from the already-loaded, ABI-checked
// SVM library. It is not a public safe deserialization surface for raw pointers.
using RunWorker = int (*)(void*,const double*,int,int,int,int,int*,int,uint64_t*,int,int);
using LastError = const char* (*)();
constexpr const char* worker_capsule = "spectra.preprocessing.worker.v1";
struct BoundWorker {
    void* handle;
    RunWorker run;
    LastError error;
    PyObject* labels;
    PyObject* owner;  // Holds the Python worker and thus its CDLL alive.
    BoundWorker(void* h,RunWorker r,LastError e,PyObject* l,PyObject* o):
        handle(h),run(r),error(e),labels(l),owner(o) {Py_INCREF(labels);Py_INCREF(owner);}
    BoundWorker(const BoundWorker&)=delete;
    BoundWorker& operator=(const BoundWorker&)=delete;
    ~BoundWorker() { Py_DECREF(labels); Py_DECREF(owner); }
};
void destroy_worker(PyObject* cap) {
    auto* p=static_cast<BoundWorker*>(PyCapsule_GetPointer(cap,worker_capsule));
    if(p) delete p; else PyErr_Clear();
}
void* address(PyObject* p) {
    require(PyLong_CheckExact(p),"native address must be an integer");
    auto* value=PyLong_AsVoidPtr(p);
    require(!PyErr_Occurred() && value,"null or invalid native address");
    return value;
}
PyObject* bind_worker(PyObject*,PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *handle,*run,*error,*labels,*owner;
        if(!PyArg_ParseTuple(args,"OOOOO",&handle,&run,&error,&labels,&owner)) return nullptr;
        auto* h=address(handle);auto* f=address(run);auto* e=address(error);
        require(PyTuple_CheckExact(labels),"labels must be an immutable tuple");
        auto n=PyTuple_GET_SIZE(labels);require(n>=2&&n<=128,"invalid label count");
        // All pointers are trusted; the Python caller must hold the worker lock
        // and check its live handle before every invocation, not just binding.
        auto p=std::make_unique<BoundWorker>(h,reinterpret_cast<RunWorker>(f),
                                              reinterpret_cast<LastError>(e),labels,owner);
        Ref cap(PyCapsule_New(p.get(),worker_capsule,destroy_worker));
        if(!cap.p) return nullptr;
        p.release();return cap.release();
    });
}

struct Detached {
    PyThreadState* state;
    Detached():state(PyEval_SaveThread()) {}
    ~Detached() { PyEval_RestoreThread(state); }
    Detached(const Detached&)=delete;
    Detached& operator=(const Detached&)=delete;
};

// Return -1 for unsupported custom objects, otherwise the size. No user callback
// is invoked, so fallback neither consumes iterators nor converts a scalar twice.
Py_ssize_t eligible_rows(const Plan& p,PyObject* rows) {
    if(!PyList_CheckExact(rows)&&!PyTuple_CheckExact(rows)) return -1;
    auto n=PySequence_Fast_GET_SIZE(rows);
    require(n<=p.cap,"batch exceeds row/element cap");
    for(Py_ssize_t r=0;r<n;++r) {
        auto* row=PySequence_Fast_GET_ITEM(rows,r);
        if(!PyList_CheckExact(row)&&!PyTuple_CheckExact(row)) return -1;
        require(PySequence_Fast_GET_SIZE(row)==p.columns,"invalid raw row");
        for(const auto& op:p.numeric) {
            auto* x=PySequence_Fast_GET_ITEM(row,op.col);
            if(x!=Py_None&&!PyFloat_CheckExact(x)&&!PyLong_CheckExact(x)) return -1;
        }
        for(const auto& op:p.category)
            if(!PyUnicode_CheckExact(PySequence_Fast_GET_ITEM(row,op.col))) return -1;
    }
    return n;
}

// This traversal never runs with a detached thread state. Between tiles another
// thread may have run; check containers before using unchecked indexing macros.
// Mutating caller input is unsupported, but never an excuse for unsafe indexing.
bool fill_tile(const Plan& p,PyObject* rows,Py_ssize_t total,Py_ssize_t first,
               Py_ssize_t count,double* out) {
    require(PySequence_Fast_GET_SIZE(rows)==total,"input batch mutated during inference");
    std::fill(out,out+count*p.features,0.);
    for(Py_ssize_t r=0;r<count;++r) {
        auto* row=PySequence_Fast_GET_ITEM(rows,first+r);
        require((PyList_CheckExact(row)||PyTuple_CheckExact(row))&&
                PySequence_Fast_GET_SIZE(row)==p.columns,"input row mutated during inference");
        auto base=r*p.features;
        for(const auto& op:p.numeric) {
            auto* v=PySequence_Fast_GET_ITEM(row,op.col);double x;
            if(v==Py_None) x=op.fill;
            else if(PyFloat_CheckExact(v)) x=PyFloat_AS_DOUBLE(v);
            else if(PyLong_CheckExact(v)) {
                x=PyLong_AsDouble(v);
                if(PyErr_Occurred()) {PyErr_Clear();throw std::invalid_argument("invalid numeric feature");}
            } else throw std::invalid_argument("input scalar mutated during inference");
            if(std::isnan(x)) x=op.fill;
            require(std::isfinite(x),"infinite raw feature");
            const double centered=x-op.mean;
            const double value=centered/op.scale;
            require(std::isfinite(value),"nonfinite transformed feature");
            out[base+op.out]=value;
        }
        for(const auto& op:p.category) {
            auto* v=PySequence_Fast_GET_ITEM(row,op.col);
            require(PyUnicode_CheckExact(v),"input category mutated during inference");
            Py_ssize_t length=0;
            if(!PyUnicode_AsUTF8AndSize(v,&length)) return false;
            require(length<=4096,"categorical value exceeds byte limit");
            auto* value=PyDict_GetItemWithError(op.lookup,v);
            if(!value) {
                if(PyErr_Occurred()) return false;
                require(!op.strict,"unknown categorical value");
            } else out[base+PyLong_AsSsize_t(value)]=1.;
        }
    }
    return true;
}

PyObject* predict_fused(PyObject*,PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *pc,*wc,*rows,*mode_object,*hint_object,*tile_object;
        if(!PyArg_ParseTuple(args,"OOOOOO",&pc,&wc,&rows,&mode_object,&hint_object,&tile_object)) return nullptr;
        auto* p=static_cast<Plan*>(PyCapsule_GetPointer(pc,capsule_name));if(!p) return nullptr;
        auto* w=static_cast<BoundWorker*>(PyCapsule_GetPointer(wc,worker_capsule));if(!w) return nullptr;
        auto mode=integer(mode_object),hint=integer(hint_object),requested=integer(tile_object);
        const auto classes=PyTuple_GET_SIZE(w->labels);
        require(mode>=0&&mode<=6&&hint>=-1&&hint<classes,"invalid worker schedule/hint");
        require(requested>=1&&requested<=128,"tile_rows must be between 1 and 128");
        require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
        auto n=eligible_rows(*p,rows);
        if(n<0) {Py_INCREF(Py_NotImplemented);return Py_NotImplemented;}
        auto tile=std::min({requested,Py_ssize_t(16384)/p->features,std::max(n,Py_ssize_t(1))});
        std::vector<double> values(size_t(tile*p->features));
        std::vector<int> indices(static_cast<size_t>(tile),0);
        Ref output(PyList_New(n));if(!output.p) return nullptr;
        uint64_t stats[5]={};
        // An empty batch still invalidates a preceding native certificate.
        if(n==0) {
            int status=w->run(w->handle,nullptr,0,int(p->features),int(mode),int(hint),nullptr,0,stats,5,0);
            if(status) throw std::invalid_argument(w->error());
        }
        for(Py_ssize_t first=0;first<n;first+=tile) {
            auto count=std::min(tile,n-first);
            if(!fill_tile(*p,rows,n,first,count,values.data())) return nullptr;
            int status;
            {
                Detached detached; // Only private plain C++ buffers are touched.
                status=w->run(w->handle,values.data(),int(count),int(p->features),int(mode),int(hint),
                              indices.data(),int(count),stats,5,0);
            }
            if(status) throw std::invalid_argument(w->error());
            for(Py_ssize_t r=0;r<count;++r) {
                auto index=indices[size_t(r)];
                require(index>=0&&index<classes,"native class index out of range");
                auto* label=PyTuple_GET_ITEM(w->labels,index);Py_INCREF(label);
                PyList_SET_ITEM(output.p,first+r,label);
            }
            if(PyErr_CheckSignals()<0) return nullptr;
        }
        return output.release();
    });
}
PyObject* abi(PyObject*,PyObject*) { return PyLong_FromLong(3); }
PyMethodDef methods[]={
    {"prepare",prepare,METH_VARARGS,"Create an owned, checked preprocessing plan."},
    {"transform",transform,METH_VARARGS,"Return fresh binary64 bytes or NotImplemented for reference fallback."},
    {"transform_raw",transform_raw,METH_VARARGS,"Traverse exact built-in containers under the GIL; otherwise request reference fallback."},
    {"_bind_worker",bind_worker,METH_VARARGS,"PRIVATE trusted native-address binding; caller must protect worker lifetime and lock."},
    {"predict_fused",predict_fused,METH_VARARGS,"Optional bounded-tile pipeline; requires live locked private worker binding."},
    {"abi",abi,METH_NOARGS,"Return the preprocessing ABI version."},
    {nullptr,nullptr,0,nullptr}};
PyModuleDef definition={PyModuleDef_HEAD_INIT,"_spectra_preprocess",
    "Optional strict-arithmetic preprocessing. GIL held; not a sandbox.",0,methods};
}
PyMODINIT_FUNC PyInit__spectra_preprocess() { return PyModule_Create(&definition); }
