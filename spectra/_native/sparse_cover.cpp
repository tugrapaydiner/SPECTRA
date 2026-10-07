// SPECTRA sparse covering/exclusion execution. MIT; repository LICENSE.
// Sparse incidence, exact counters, reversible assignments. No novelty claim.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using U = uint32_t;
constexpr U NONE = UINT32_MAX;
constexpr uint64_t MAX_VARS=1000000, MAX_GROUPS=2000000, MAX_ENTRIES=16000000;
struct Resource:std::runtime_error {using std::runtime_error::runtime_error;};
struct Invalid:std::runtime_error {using std::runtime_error::runtime_error;};
uint64_t integer(PyObject* obj,const char* label,uint64_t cap=UINT64_MAX) {
    if (!PyLong_CheckExact(obj)) throw Invalid(std::string(label)+" must be an exact nonnegative integer");
    uint64_t value=PyLong_AsUnsignedLongLong(obj);
    if (PyErr_Occurred()) {PyErr_Clear();throw Invalid(std::string(label)+" is outside unsigned range");}
    if (value>cap) throw Invalid(std::string(label)+" exceeds its supported bound");
    return value;
}
struct Index {
    U n,p,g;
    std::vector<U> offsets, vars, voff, groups;
    std::vector<uint64_t> weight;
    uint64_t bytes=0;
    Index(PyObject* nv,PyObject* positive,PyObject* exclusive,uint64_t budget) {
        n=static_cast<U>(integer(nv,"nvars",MAX_VARS));
        if (!PyTuple_CheckExact(positive)||!PyTuple_CheckExact(exclusive)) throw Invalid("constraint banks must be exact tuples");
        const uint64_t pc=PyTuple_GET_SIZE(positive),gc=pc+PyTuple_GET_SIZE(exclusive);
        if (gc>MAX_GROUPS) throw Resource("too many groups");
        p=static_cast<U>(pc);g=static_cast<U>(gc);
        uint64_t count=0;
        for (U i=0;i<g;++i) {
            PyObject* row=PyTuple_GET_ITEM(i<p?positive:exclusive,i<p?i:i-p);
            if (!PyTuple_CheckExact(row)) throw Invalid("groups must be exact tuples");
            count+=static_cast<uint64_t>(PyTuple_GET_SIZE(row));
            if (count>MAX_ENTRIES) throw Resource("too many incidences");
        }
        // Persistent compressed adjacency plus one temporary last-seen/cursor array.
        bytes=4*(uint64_t(g)+1+uint64_t(n)+1+2*count)+8*uint64_t(p);
        if (bytes+4*uint64_t(n)>budget) throw Resource("index plus temporary input validation exceeds build payload cap");
        offsets.resize(size_t(g)+1);voff.assign(size_t(n)+1,0);vars.resize(count);groups.resize(count);
        std::vector<U> seen(n,NONE);
        U at=0;
        for (U i=0;i<g;++i) {
            offsets[i]=at;
            PyObject* row=PyTuple_GET_ITEM(i<p?positive:exclusive,i<p?i:i-p);
            const auto sz=PyTuple_GET_SIZE(row);
            for (Py_ssize_t j=0;j<sz;++j) {
                const uint64_t x=integer(PyTuple_GET_ITEM(row,j),"group variable",n);
                if (!x) throw Invalid("group variables are one-based");
                const U v=static_cast<U>(x-1);
                if (seen[v]==i) throw Invalid("a group contains a repeated variable");
                seen[v]=i;vars[at++]=v;++voff[size_t(v)+1];
            }
        }
        offsets[g]=at;
        for (U v=0;v<n;++v) voff[v+1]+=voff[v];
        std::copy(voff.begin(),voff.begin()+n,seen.begin());
        for (U i=0;i<g;++i) for (U j=offsets[i];j<offsets[i+1];++j) groups[seen[vars[j]]++]=i;
        // Reuse the temporary bank for static exclusion incidence degree.
        // Each variable's degree is at most the total input incidence inventory.
        std::fill(seen.begin(),seen.end(),0);weight.assign(p,0);
        for(U i=p;i<g;++i) {
            const U width=offsets[i+1]-offsets[i];
            if(width>1) for(U j=offsets[i];j<offsets[i+1];++j) seen[vars[j]]+=width-1;
        }
        for(U i=0;i<p;++i) for(U j=offsets[i];j<offsets[i+1];++j) weight[i]+=seen[vars[j]];
    }
};
struct Frame {size_t mark;U v;bool second;};
struct Answer {
    std::vector<int8_t> value;
    U reason=2;
    uint64_t work=0,nodes=0,forced=0,backtracks=0,index_bytes=0,state_bytes=0,trace=14695981039346656037ULL;
};
// Every mutable bank is allocated once from a closed-form linear payload bound.
struct Search {
    const Index& x; bool heap_mode,active_mode,xor_mode,degree_mode,lcv_mode; uint64_t limit;
    Answer result;
    std::vector<U> left,covered,heap,pos,trail,xor_left;
    U active_count=0;
    std::vector<int32_t> queue;
    std::vector<int8_t> pending;
    std::vector<Frame> frames;
    size_t head=0;
    U unsatisfied;
    bool conflict=false;
    uint64_t work=0,nodes=0,forced=0,backtracks=0,trace=14695981039346656037ULL;
    Search(const Index& index,U flags,uint64_t cap,uint64_t memory):x(index),heap_mode(flags&1),active_mode(flags&2),xor_mode(flags&4),degree_mode(flags&8),lcv_mode(flags&16),limit(cap),unsatisfied(x.p) {
        result.index_bytes=x.bytes;
        result.state_bytes=uint64_t(x.n)*(2+4+4+sizeof(Frame))+uint64_t(x.p)*(8+((heap_mode||active_mode)?8:0)+(xor_mode?4:0));
        if (result.state_bytes>memory) throw Resource("linear search state exceeds payload cap");
        result.value.assign(x.n,0);pending.assign(x.n,0);
        left.resize(x.p);covered.assign(x.p,0);
        if (heap_mode||active_mode) {heap.resize(x.p);pos.resize(x.p);active_count=x.p;}
        if (xor_mode) xor_left.assign(x.p,0);
        trail.reserve(x.n);queue.reserve(x.n);frames.reserve(x.n);
        for (U g=0;g<x.p;++g) {
            left[g]=x.offsets[g+1]-x.offsets[g];
            if (heap_mode||active_mode) heap[g]=pos[g]=g;
            if (xor_mode) for(U j=x.offsets[g];j<x.offsets[g+1];++j) xor_left[g]^=x.vars[j];
        }
        if (heap_mode) for (size_t i=heap.size()/2;i>0;--i) down(static_cast<U>(i-1));
    }
    U key(U g) const {return covered[g]?NONE:left[g];}
    bool before(U a,U b)const {
        if(key(a)!=key(b))return key(a)<key(b);
        if(degree_mode&&x.weight[a]!=x.weight[b])return x.weight[a]>x.weight[b];
        return a<b;
    }
    void swapheap(U a,U b) {std::swap(heap[a],heap[b]);pos[heap[a]]=a;pos[heap[b]]=b;}
    void up(U i) {while(i) {U par=(i-1)/2;if(!before(heap[i],heap[par]))break;swapheap(i,par);i=par;}}
    void down(U i) {while(2*uint64_t(i)+1<heap.size()) {U j=2*i+1;if(j+1<heap.size()&&before(heap[j+1],heap[j]))++j;if(!before(heap[j],heap[i]))break;swapheap(i,j);i=j;}}
    void changed(U g) {
        if(heap_mode) {up(pos[g]);down(pos[g]);}
        else if(active_mode) {
            // Sparse-set membership only; MRV still breaks ties by original group id.
            const bool present=pos[g]<active_count;
            if(covered[g]&&present) {--active_count;swapheap(pos[g],active_count);}
            else if(!covered[g]&&!present) {swapheap(pos[g],active_count);++active_count;}
        }
    }
    void enqueue(U v,int8_t value) {
        if (result.value[v]) {conflict|=result.value[v]!=value;return;}
        if (pending[v]) {conflict|=pending[v]!=value;return;}
        pending[v]=value;queue.push_back(value>0?static_cast<int32_t>(v+1):-static_cast<int32_t>(v+1));
    }
    void force_unit(U g) {
        if(covered[g])return;
        if(!left[g]) {conflict=true;return;}
        if(left[g]==1) {
            if(xor_mode) {enqueue(xor_left[g],1);return;}
            for(U i=x.offsets[g];i<x.offsets[g+1];++i) if(!result.value[x.vars[i]]) {enqueue(x.vars[i],1);return;}
        }
    }
    void clear_queue() {
        for(size_t i=head;i<queue.size();++i) {const int64_t lit=queue[i];pending[static_cast<U>((lit<0?-lit:lit)-1)]=0;}
        queue.clear();head=0;
    }
    // Return 0 stable, 1 conflict, 2 work cap. A partial variable update never escapes.
    U propagate() {
        while(head<queue.size()&&!conflict) {
            if(work==limit)return 2;
            const int64_t lit=queue[head++];const U v=static_cast<U>((lit<0?-lit:lit)-1);const int8_t val=lit>0?1:-1;
            pending[v]=0;
            if(result.value[v]) {conflict|=result.value[v]!=val;continue;}
            result.value[v]=val;trail.push_back(v);++work;++forced;
            trace=(trace^static_cast<uint64_t>(static_cast<int64_t>(lit)))*1099511628211ULL;
            for(U j=x.voff[v];j<x.voff[v+1];++j) {
                const U g=x.groups[j];if(g>=x.p)continue;
                --left[g];if(xor_mode)xor_left[g]^=v;
                if(val>0 && covered[g]++==0)--unsatisfied;
                changed(g);
            }
            // Counters have been changed completely and can always be rolled back.
            for(U j=x.voff[v];j<x.voff[v+1];++j) {
                const U g=x.groups[j];
                if(g<x.p) force_unit(g);
                else if(val>0) for(U i=x.offsets[g];i<x.offsets[g+1];++i) if(x.vars[i]!=v) enqueue(x.vars[i],-1);
            }
        }
        if(conflict)return 1;
        clear_queue();return 0;
    }
    void rollback(size_t mark) {
        clear_queue();conflict=false;
        while(trail.size()>mark) {
            const U v=trail.back();trail.pop_back();const int8_t val=result.value[v];result.value[v]=0;
            for(U j=x.voff[v];j<x.voff[v+1];++j) {
                const U g=x.groups[j];if(g>=x.p)continue;
                ++left[g];if(xor_mode)xor_left[g]^=v;
                if(val>0 && --covered[g]==0)++unsatisfied;
                changed(g);
            }
        }
        trace=(trace^UINT64_MAX)*1099511628211ULL;
    }
    U choose()const {
        if(heap_mode)return heap.empty()?NONE:heap[0];
        U best=NONE;
        if(active_mode) {
            for(U i=0;i<active_count;++i) {
                const U g=heap[i];
                if(best==NONE||before(g,best))best=g;
            }
        } else for(U g=0;g<x.p;++g)if(!covered[g]&&(best==NONE||before(g,best)))best=g;
        return best;
    }
    U choose_variable(U g)const {
        U first=NONE,best=NONE,best_impact=NONE,visits=0;
        for(U i=x.offsets[g];i<x.offsets[g+1];++i) {
            const U v=x.vars[i];if(result.value[v])continue;
            if(first==NONE)first=v;
            if(!lcv_mode||left[g]>8)return first;
            U impact=0;
            for(U j=x.voff[v];j<x.voff[v+1];++j) {
                const U e=x.groups[j];if(e<x.p)continue;
                for(U k=x.offsets[e];k<x.offsets[e+1];++k) {
                    if(++visits>4096)return first; // explicit per-decision heuristic scan cap
                    const U u=x.vars[k];impact+=u!=v&&!result.value[u];
                }
            }
            if(best==NONE||impact<best_impact) {best=v;best_impact=impact;}
        }
        return best;
    }
    Answer run(const std::vector<int32_t>& assumptions) {
        for(const int32_t a:assumptions) {const int64_t lit=a;enqueue(static_cast<U>((lit<0?-lit:lit)-1),lit>0?1:-1);}
        for(U g=0;g<x.p;++g) force_unit(g);
        while(true) {
            U status=propagate();
            if(status==2) {result.reason=1;break;}
            if(status==1) {
                bool found=false;
                while(!frames.empty()) {
                    Frame& f=frames.back();rollback(f.mark);++backtracks;
                    if(!f.second) {f.second=true;enqueue(f.v,-1);found=true;break;}
                    frames.pop_back();
                }
                if(!found){result.reason=2;break;}
                continue;
            }
            if(!unsatisfied) {result.reason=0;break;}
            if(work==limit) {result.reason=1;break;}
            const U g=choose();
            if(g==NONE||!left[g]) {conflict=true;continue;}
            const U v=choose_variable(g);
            if(v==NONE)throw std::logic_error("remaining count disagrees with incidence");
            ++work;++nodes;frames.push_back({trail.size(),v,false});enqueue(v,1);
        }
        result.work=work;result.nodes=nodes;result.forced=forced;result.backtracks=backtracks;result.trace=trace;
        return std::move(result);
    }
};
constexpr const char* TAG="spectra.sparse.index.v1";
void destroy(PyObject* capsule) {void* p=PyCapsule_GetPointer(capsule,TAG);if(p)delete static_cast<Index*>(p);else PyErr_Clear();}
PyObject* error() {
    try {throw;} catch(const Resource& e) {PyErr_SetString(PyExc_MemoryError,e.what());}
    catch(const std::bad_alloc&) {PyErr_NoMemory();}
    catch(const Invalid& e) {PyErr_SetString(PyExc_ValueError,e.what());}
    catch(const std::exception& e) {PyErr_SetString(PyExc_RuntimeError,e.what());}
    catch(...) {PyErr_SetString(PyExc_RuntimeError,"unknown sparse native exception");}
    return nullptr;
}
PyObject* create(PyObject*,PyObject* args) {
    PyObject *n,*p,*e,*b;
    if(!PyArg_ParseTuple(args,"OOOO",&n,&p,&e,&b))return nullptr;
    try {
        const uint64_t budget=integer(b,"max_build_bytes");
        auto index=std::make_unique<Index>(n,p,e,budget);
        PyObject* c=PyCapsule_New(index.get(),TAG,destroy);
        if(c) index.release();
        return c;
    }catch(...){return error();}
}
struct GilRelease {PyThreadState* s;GilRelease():s(PyEval_SaveThread()){}~GilRelease(){PyEval_RestoreThread(s);}};
PyObject* solve(PyObject*,PyObject* args) {
    PyObject *capsule,*w,*memory,*mode,*assumptions;
    if(!PyArg_ParseTuple(args,"OOOOO",&capsule,&w,&memory,&mode,&assumptions))return nullptr;
    Index* index=static_cast<Index*>(PyCapsule_GetPointer(capsule,TAG));if(!index)return nullptr;
    try {
        const auto limit=integer(w,"max_work"),budget=integer(memory,"max_state_bytes");
        const U flags=static_cast<U>(integer(mode,"search flags",31));
        if((flags&3)==3)throw Invalid("heap and active strategies cannot be combined");
        if(!PyTuple_CheckExact(assumptions))throw Invalid("assumptions must be an exact tuple");
        const Py_ssize_t ac=PyTuple_GET_SIZE(assumptions);
        if(static_cast<uint64_t>(ac)>2*uint64_t(index->n))throw Invalid("too many assumptions");
        const uint64_t assumption_bytes=4*static_cast<uint64_t>(ac);
        if(assumption_bytes>budget)throw Resource("assumption conversion exceeds state payload cap");
        std::vector<int32_t> signed_vars; signed_vars.reserve(ac);
        for(Py_ssize_t i=0;i<ac;++i) {
            PyObject* a=PyTuple_GET_ITEM(assumptions,i);
            if(!PyLong_CheckExact(a))throw Invalid("assumption must be an exact integer");
            const long long value=PyLong_AsLongLong(a);
            if(PyErr_Occurred()) {PyErr_Clear();throw Invalid("assumption exceeds signed bounds");}
            if(!value||value>index->n||value<-int64_t(index->n))throw Invalid("assumption outside declared variables");
            signed_vars.push_back(static_cast<int32_t>(value));
        }
        Answer r;
        {GilRelease release; Search search(*index,flags,limit,budget-assumption_bytes);r=search.run(signed_vars);r.state_bytes+=assumption_bytes;}
        PyObject* witness=PyTuple_New(index->n);if(!witness)return nullptr;
        for(U v=0;v<index->n;++v) {PyObject* bit=r.value[v]>0?Py_True:Py_False;Py_INCREF(bit);PyTuple_SET_ITEM(witness,v,bit);}
        return Py_BuildValue("(NIKKKKKKK)",witness,static_cast<unsigned int>(r.reason),
            static_cast<unsigned long long>(r.work),static_cast<unsigned long long>(r.nodes),
            static_cast<unsigned long long>(r.forced),static_cast<unsigned long long>(r.backtracks),
            static_cast<unsigned long long>(r.index_bytes),static_cast<unsigned long long>(r.state_bytes),
            static_cast<unsigned long long>(r.trace));
    }catch(...){return error();}
}
PyObject* abi(PyObject*,PyObject*) {return PyLong_FromLong(3);}
PyMethodDef methods[]={{"abi",abi,METH_NOARGS,"Native API version."},{"create",create,METH_VARARGS,"Create immutable sparse constraint index."},{"solve",solve,METH_VARARGS,"Solve with an independent reversible state."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_sparse",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_sparse(){return PyModule_Create(&module);}
