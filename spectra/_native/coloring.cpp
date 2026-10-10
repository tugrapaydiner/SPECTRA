// SPECTRA finite-domain inequality execution. MIT; see repository LICENSE.
// Established CSP/DSATUR/2-SAT techniques; no neural or novelty claim.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using U=uint32_t; using M=uint64_t;
constexpr U NONE=UINT32_MAX;
constexpr uint64_t MAX_N=100000, MAX_M=2000000;
struct Invalid:std::runtime_error {using std::runtime_error::runtime_error;};
struct Resource:std::runtime_error {using std::runtime_error::runtime_error;};
uint64_t integer(PyObject* o,const char* name,uint64_t cap=UINT64_MAX) {
    if(!PyLong_CheckExact(o))throw Invalid(name);
    auto v=PyLong_AsUnsignedLongLong(o);
    if(PyErr_Occurred()){PyErr_Clear();throw Invalid(name);}
    if(v>cap)throw Invalid(name);
    return v;
}
unsigned bits(M x){return static_cast<unsigned>(__builtin_popcountll(x));}
unsigned first(M x){return static_cast<unsigned>(__builtin_ctzll(x));}
bool unit(M x){return x && !(x&(x-1));}
M full(U k){return k==64?~M(0):((M(1)<<k)-1);}
struct Graph {
    U n,k; uint64_t m,bytes=0; bool loops=false,unrestricted=true;
    std::vector<U> off,to,rank,vertex; std::vector<M> initial;
    Graph(PyObject* nv,PyObject* kv,PyObject* edges,PyObject* masks,uint64_t cap) {
        n=static_cast<U>(integer(nv,"n must be an exact integer in [0,100000]",MAX_N));
        k=static_cast<U>(integer(kv,"k must be an exact integer in [0,64]",64));
        if(!PyTuple_CheckExact(edges)||!PyTuple_CheckExact(masks))throw Invalid("edges and masks must be exact tuples");
        m=static_cast<uint64_t>(PyTuple_GET_SIZE(edges));
        if(m>MAX_M)throw Resource("too many input edges");
        if(PyTuple_GET_SIZE(masks)!=0&&static_cast<uint64_t>(PyTuple_GET_SIZE(masks))!=n)throw Invalid("mask count differs from n");
        bytes=4*(uint64_t(n)+1+2*m)+16*uint64_t(n);
        if(bytes+4*uint64_t(n)>cap)throw Resource("graph plus construction scratch exceeds payload cap");
        off.assign(size_t(n)+1,0);to.resize(2*m);initial.assign(n,full(k));
        if(PyTuple_GET_SIZE(masks))for(U v=0;v<n;++v) {
            initial[v]=integer(PyTuple_GET_ITEM(masks,v),"masks must be exact nonnegative uint64 integers");
            if(initial[v]&~full(k))throw Invalid("mask uses a colour outside k");
            unrestricted&=initial[v]==full(k);
        }
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i) {
            PyObject* e=PyTuple_GET_ITEM(edges,i);
            if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)throw Invalid("each edge must be a pair tuple");
            const auto a=integer(PyTuple_GET_ITEM(e,0),"invalid edge endpoint"),b=integer(PyTuple_GET_ITEM(e,1),"invalid edge endpoint");
            if(a>=n||b>=n)throw Invalid("edge endpoint outside n");
            loops|=a==b;++off[a+1];++off[b+1];
        }
        for(U v=0;v<n;++v)off[v+1]+=off[v];
        std::vector<U> cursor(off.begin(),off.end()-1);
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i) {
            PyObject* e=PyTuple_GET_ITEM(edges,i);
            const U a=static_cast<U>(PyLong_AsUnsignedLong(PyTuple_GET_ITEM(e,0)));
            const U b=static_cast<U>(PyLong_AsUnsignedLong(PyTuple_GET_ITEM(e,1)));
            to[cursor[a]++]=b;to[cursor[b]++]=a;
        }
        // Static rank preserves the scan's degree/id tie rule exactly. The
        // bounded-domain priority buckets below change scheduling cost, not policy.
        vertex.resize(n);rank.resize(n);std::iota(vertex.begin(),vertex.end(),0);
        std::sort(vertex.begin(),vertex.end(),[&](U a,U b){
            const U da=off[a+1]-off[a],db=off[b+1]-off[b];
            return da!=db?da>db:a<b;
        });
        for(U i=0;i<n;++i)rank[vertex[i]]=i;
    }
};
struct Change {U v;M old;};
struct Frame {size_t mark;U v;M choices;};
struct Result {
    std::vector<M> domains;U reason=2;
    uint64_t work=0,branches=0,backtracks=0,removals=0,binary_calls=0,binary_edges=0,binary_sat=0;
    uint64_t index_bytes=0,state_bytes=0,trail_peak=0,trace=14695981039346656037ULL;
};
struct Search {
    const Graph& g;uint64_t cap,budget;bool symmetry,binary,buckets;
    Result r;std::vector<Change> trail;std::vector<Frame> frames;std::vector<U> queue;
    size_t qhead=0;uint64_t trail_bound;
    std::vector<M> members,summaries,heads;std::vector<U> used_counts;
    size_t words=0,summary_words=0;M active_sizes=0,used_mask=0;U wide=0;
    Search(const Graph& graph,uint64_t work_cap,uint64_t state_cap,bool sym,bool bin,bool bucket):g(graph),cap(work_cap),budget(state_cap),symmetry(sym&&g.unrestricted),binary(bin),buckets(bucket) {
        // On an active path, a vertex becomes singleton at most once; each directed
        // edge can remove a colour at most once. Each decision fixes one vertex.
        // Also at most 64 strict domain reductions per vertex.
        trail_bound=std::min(64*uint64_t(g.n),2*g.m+g.n);
        r.index_bytes=g.bytes;
        r.state_bytes=uint64_t(g.n)*(sizeof(M)+sizeof(Frame)+sizeof(U))+trail_bound*sizeof(Change);
        if(buckets) {
            words=(size_t(g.n)+63)/64;summary_words=(words+63)/64;
            if(summary_words>64)throw std::logic_error("priority hierarchy exceeds admitted geometry");
            r.state_bytes+=8*(uint64_t(g.k)+1)*(words+summary_words+1)+256;
        }
        if(r.state_bytes>budget)throw Resource("bounded reversible state exceeds payload cap");
        r.domains=g.initial;trail.reserve(trail_bound);frames.reserve(g.n);queue.reserve(g.n);
        if(buckets) {
            members.assign((size_t(g.k)+1)*words,0);summaries.assign((size_t(g.k)+1)*summary_words,0);
            heads.assign(size_t(g.k)+1,0);used_counts.assign(64,0);
            for(U v=0;v<g.n;++v){priority_add(v,r.domains[v]);colour_add(r.domains[v]);}
        }
    }
    // At most 65 domain-size rows, with a three-level bit hierarchy over static
    // degree/id ranks. A domain update and minimum selection use constant work
    // within the explicit n<=100000,k<=64 contract; no unbounded heap walk.
    void priority_add(U v,M d) {
        unsigned size=bits(d);if(size<2)return;
        const U rnk=g.rank[v],word=rnk/64,block=word/64;
        members[size*words+word]|=M(1)<<(rnk%64);
        summaries[size*summary_words+block]|=M(1)<<(word%64);
        heads[size]|=M(1)<<block;active_sizes|=M(1)<<(size-1);
    }
    void priority_remove(U v,M d) {
        unsigned size=bits(d);if(size<2)return;
        const U rnk=g.rank[v],word=rnk/64,block=word/64;
        M& entry=members[size*words+word];entry&=~(M(1)<<(rnk%64));
        if(!entry){M& summary=summaries[size*summary_words+block];summary&=~(M(1)<<(word%64));
            if(!summary){heads[size]&=~(M(1)<<block);if(!heads[size])active_sizes&=~(M(1)<<(size-1));}}
    }
    void colour_add(M d) {
        if(unit(d)){if(used_counts[first(d)]++==0)used_mask|=d;}
        else if(bits(d)>2)++wide;
    }
    void colour_remove(M d) {
        if(unit(d)){if(--used_counts[first(d)]==0)used_mask&=~d;}
        else if(bits(d)>2)--wide;
    }
    void priority_change(U v,M before,M after) {
        if(!buckets)return;
        priority_remove(v,before);colour_remove(before);priority_add(v,after);colour_add(after);
    }
    void trace(M event){r.trace=(r.trace^event)*1099511628211ULL;}
    bool reduce(U v,M d) {
        const M old=r.domains[v];if(d==old)return true;
        if(d&~old)throw std::logic_error("domain enlargement outside rollback");
        if(trail.size()>=trail_bound)throw std::logic_error("active trail bound violated");
        trail.push_back({v,old});priority_change(v,old,d);r.domains[v]=d;++r.removals;
        r.trail_peak=std::max(r.trail_peak,uint64_t(trail.size()));trace((uint64_t(v)<<32)^d);
        if(!d)return false;
        if(unit(d))queue.push_back(v);
        return true;
    }
    U propagate() {
        while(qhead<queue.size()) {
            const U v=queue[qhead++];const M colour=r.domains[v];
            if(!unit(colour))return 1;
            for(U j=g.off[v];j<g.off[v+1];++j) {
                if(r.work==cap)return 2;
                ++r.work;const U u=g.to[j];
                if(!reduce(u,r.domains[u]&~colour))return 1;
            }
        }
        queue.clear();qhead=0;return 0;
    }
    void restore(size_t mark) {
        queue.clear();qhead=0;
        while(trail.size()>mark){const auto c=trail.back();trail.pop_back();priority_change(c.v,r.domains[c.v],c.old);r.domains[c.v]=c.old;}
        trace(UINT64_MAX);
    }
    U choose(M& used,bool& all_binary)const {
        if(buckets){
            used=used_mask;all_binary=wide==0;if(!active_sizes)return NONE;
            const unsigned size=first(active_sizes)+1,block=first(heads[size]);
            const size_t word=64*block+first(summaries[size*summary_words+block]);
            const size_t rnk=64*word+first(members[size*words+word]);
            return g.vertex[rnk];
        }
        U best=NONE;unsigned count=65;U degree=0;used=0;all_binary=true;
        for(U v=0;v<g.n;++v) {
            const M d=r.domains[v];if(!d)return v;
            const unsigned size=bits(d);
            if(size==1){used|=d;continue;}
            all_binary&=size<=2;
            const U deg=g.off[v+1]-g.off[v];
            if(size<count||(size==count&&deg>degree)){best=v;count=size;degree=deg;}
        }
        return best;
    }
    // Literal 2*v means choosing the lower available colour for vertex v;
    // 2*v+1 means its upper colour. A singleton has a true unit literal.
    // Return 0=resource-ineligible,1=SAT,2=conflict,3=work-cap.
    U finish_binary() {
        uint64_t arcs=0;const U nv=2*g.n;
        for(U v=0;v<g.n;++v) {
            if(!r.domains[v])return 2;
            if(bits(r.domains[v])>2)return 0;
            if(unit(r.domains[v]))++arcs;
            for(U j=g.off[v];j<g.off[v+1];++j)if(v<g.to[j])arcs+=2*bits(r.domains[v]&r.domains[g.to[j]]);
        }
        // Edge list; forward/reverse CSR and fill cursors; DFS stack, order,
        // components, visited flags. Singletons remain explicit in this simple
        // closure, avoiding any invalid inference from omitted fixed variables.
        const uint64_t scratch=33*uint64_t(nv)+16*arcs+16;
        if(scratch>budget-r.state_bytes)return 0;
        if(uint64_t(nv)+arcs+1>cap-r.work)return 3;
        r.work+=uint64_t(nv)+arcs+1;++r.binary_calls;r.binary_edges+=arcs;
        r.state_bytes+=scratch;
        std::vector<std::pair<U,U>> edges;edges.reserve(arcs);
        auto clause=[&](U a,U b){edges.emplace_back(a^1U,b);if(a!=b)edges.emplace_back(b^1U,a);};
        for(U v=0;v<g.n;++v) {
            const M d=r.domains[v];if(unit(d))clause(2*v,2*v);
            for(U j=g.off[v];j<g.off[v+1];++j) {
                const U u=g.to[j];if(v>=u)continue;
                M shared=d&r.domains[u];
                while(shared) {
                    M colour=shared&(~shared+1);shared&=shared-1;
                    const U a=2*v+(colour==(d&(~d+1))?0:1);
                    const M ud=r.domains[u];const U b=2*u+(colour==(ud&(~ud+1))?0:1);
                    clause(a^1U,b^1U);
                }
            }
        }
        if(edges.size()!=arcs)throw std::logic_error("binary edge accounting differs");
        std::vector<U> off(size_t(nv)+1,0),rev(size_t(nv)+1,0);
        for(auto e:edges){++off[e.first+1];++rev[e.second+1];}
        for(U v=0;v<nv;++v){off[v+1]+=off[v];rev[v+1]+=rev[v];}
        std::vector<U> next(off),rn(rev),to(arcs),from(arcs);
        for(auto e:edges){to[next[e.first]++]=e.second;from[rn[e.second]++]=e.first;}
        std::vector<uint8_t> visited(nv,0);std::vector<U> order,comp(nv,NONE);order.reserve(nv);
        std::vector<std::pair<U,U>> stack;stack.reserve(nv);
        for(U root=0;root<nv;++root)if(!visited[root]) {
            visited[root]=1;stack.emplace_back(root,off[root]);
            while(!stack.empty()) {
                auto& f=stack.back();if(f.second==off[f.first+1]){order.push_back(f.first);stack.pop_back();continue;}
                U u=to[f.second++];if(!visited[u]){visited[u]=1;stack.emplace_back(u,off[u]);}
            }
        }
        U c=0;
        for(auto it=order.rbegin();it!=order.rend();++it)if(comp[*it]==NONE) {
            comp[*it]=c;stack.emplace_back(*it,0);
            while(!stack.empty()) {
                U v=stack.back().first;stack.pop_back();
                for(U j=rev[v];j<rev[v+1];++j){U u=from[j];if(comp[u]==NONE){comp[u]=c;stack.emplace_back(u,0);}}
            }++c;
        }
        for(U v=0;v<g.n;++v)if(comp[2*v]==comp[2*v+1])return 2;
        for(U v=0;v<g.n;++v) {
            M d=r.domains[v],lo=d&(~d+1),hi=d^lo;
            r.domains[v]=comp[2*v]>comp[2*v+1]?lo:hi;
            if(!unit(r.domains[v]))throw std::logic_error("binary closure violates singleton clause");
            trace((uint64_t(v)<<32)^r.domains[v]);
        }
        ++r.binary_sat;return 1;
    }
    Result run() {
        if(g.loops){r.reason=2;return std::move(r);}
        for(U v=0;v<g.n;++v){if(!r.domains[v]){r.reason=2;return std::move(r);}if(unit(r.domains[v]))queue.push_back(v);}
        uint64_t peak=r.state_bytes;bool conflict=false;
        while(true) {
            U state=conflict?1:propagate();
            if(state==2){r.reason=1;break;}
            if(state==1) {
                bool resumed=false;
                while(!frames.empty()) {
                    auto& f=frames.back();restore(f.mark);++r.backtracks;
                    if(f.choices) {
                        if(r.work==cap){r.reason=1;r.state_bytes=peak;return std::move(r);}
                        M c=f.choices&(~f.choices+1);f.choices&=f.choices-1;++r.work;++r.branches;
                        conflict=!reduce(f.v,c);resumed=true;break;
                    }
                    frames.pop_back();
                }
                if(!resumed){r.reason=2;break;}
                continue;
            }
            M used=0;bool all_binary=false;const U v=choose(used,all_binary);
            if(v==NONE){r.reason=0;break;}
            if(!r.domains[v]){conflict=true;continue;}
            if(binary&&all_binary) {
                const auto live=r.state_bytes;U b=finish_binary();peak=std::max(peak,r.state_bytes);r.state_bytes=live;
                if(b==1){r.reason=0;break;}if(b==2){conflict=true;continue;}if(b==3){r.reason=1;break;}
            }
            if(r.work==cap){r.reason=1;break;}
            M options=r.domains[v];
            if(symmetry) {
                M unused=options&~used;
                // On unrestricted problems unused colour names are exchangeable;
                // never use this reduction under arbitrary vertex lists/precolours.
                options=(options&used)|(unused?(unused&(~unused+1)):0);
            }
            if(!options){conflict=true;continue;}
            M c=options&(~options+1);frames.push_back({trail.size(),v,options^c});
            ++r.work;++r.branches;conflict=!reduce(v,c);
        }
        r.state_bytes=peak;return std::move(r);
    }
};
constexpr const char* TAG="spectra.colouring.v1";
void destroy(PyObject* c){void* p=PyCapsule_GetPointer(c,TAG);if(p)delete static_cast<Graph*>(p);else PyErr_Clear();}
PyObject* error(){
    try{throw;}catch(const Resource& e){PyErr_SetString(PyExc_MemoryError,e.what());}
    catch(const Invalid& e){PyErr_SetString(PyExc_ValueError,e.what());}
    catch(const std::bad_alloc&){PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());}
    catch(...){PyErr_SetString(PyExc_RuntimeError,"unknown native exception");}return nullptr;
}
PyObject* create(PyObject*,PyObject* args){
    PyObject *n,*k,*edges,*masks,*cap;if(!PyArg_ParseTuple(args,"OOOOO",&n,&k,&edges,&masks,&cap))return nullptr;
    try{auto p=std::make_unique<Graph>(n,k,edges,masks,integer(cap,"invalid build cap"));auto c=PyCapsule_New(p.get(),TAG,destroy);if(c)p.release();return c;}catch(...){return error();}
}
struct Release{PyThreadState* t;Release():t(PyEval_SaveThread()){}~Release(){PyEval_RestoreThread(t);}};
PyObject* solve(PyObject*,PyObject* args){
    PyObject *capsule,*work,*memory,*sym,*bin,*bucket;if(!PyArg_ParseTuple(args,"OOOOOO",&capsule,&work,&memory,&sym,&bin,&bucket))return nullptr;
    auto* g=static_cast<Graph*>(PyCapsule_GetPointer(capsule,TAG));if(!g)return nullptr;
    try{
        auto w=integer(work,"invalid work cap"),m=integer(memory,"invalid state cap");
        if(!PyBool_Check(sym)||!PyBool_Check(bin)||!PyBool_Check(bucket))throw Invalid("search switches must be bool");
        Result r;{Release release;Search s(*g,w,m,sym==Py_True,bin==Py_True,bucket==Py_True);r=s.run();}
        PyObject* labels=PyTuple_New(g->n);if(!labels)return nullptr;
        for(U v=0;v<g->n;++v){PyObject* c=PyLong_FromLong(unit(r.domains[v])?static_cast<long>(first(r.domains[v])):-1);if(!c){Py_DECREF(labels);return nullptr;}PyTuple_SET_ITEM(labels,v,c);}
        return Py_BuildValue("(NIKKKKKKKKKKK)",labels,static_cast<unsigned>(r.reason),
            static_cast<unsigned long long>(r.work),static_cast<unsigned long long>(r.branches),
            static_cast<unsigned long long>(r.backtracks),static_cast<unsigned long long>(r.removals),
            static_cast<unsigned long long>(r.binary_calls),static_cast<unsigned long long>(r.binary_edges),
            static_cast<unsigned long long>(r.binary_sat),static_cast<unsigned long long>(r.index_bytes),
            static_cast<unsigned long long>(r.state_bytes),static_cast<unsigned long long>(r.trail_peak),static_cast<unsigned long long>(r.trace));
    }catch(...){return error();}
}
// Separate original-input observer: no adjacency, domain state, trail or SCC reuse.
PyObject* check(PyObject*,PyObject* args){
    PyObject *nv,*kv,*edges,*masks,*labels;if(!PyArg_ParseTuple(args,"OOOOO",&nv,&kv,&edges,&masks,&labels))return nullptr;
    try{
        uint64_t n=integer(nv,"invalid n",MAX_N),k=integer(kv,"invalid k",64);
        if(!PyTuple_CheckExact(edges)||!PyTuple_CheckExact(masks)||!PyTuple_CheckExact(labels))Py_RETURN_FALSE;
        if(static_cast<uint64_t>(PyTuple_GET_SIZE(labels))!=n)Py_RETURN_FALSE;
        if(PyTuple_GET_SIZE(masks)&&static_cast<uint64_t>(PyTuple_GET_SIZE(masks))!=n)Py_RETURN_FALSE;
        if(static_cast<uint64_t>(PyTuple_GET_SIZE(edges))>MAX_M)Py_RETURN_FALSE;
        for(uint64_t i=0;i<n;++i){uint64_t c=integer(PyTuple_GET_ITEM(labels,i),"invalid colour");if(c>=k)Py_RETURN_FALSE;
            if(PyTuple_GET_SIZE(masks)){M d=integer(PyTuple_GET_ITEM(masks,i),"invalid mask");if(d&~full(k))Py_RETURN_FALSE;if(!(d&(M(1)<<c)))Py_RETURN_FALSE;}}
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i){PyObject* e=PyTuple_GET_ITEM(edges,i);if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)Py_RETURN_FALSE;
            uint64_t a=integer(PyTuple_GET_ITEM(e,0),"invalid edge"),b=integer(PyTuple_GET_ITEM(e,1),"invalid edge");if(a>=n||b>=n)Py_RETURN_FALSE;
            if(PyLong_AsUnsignedLong(PyTuple_GET_ITEM(labels,a))==PyLong_AsUnsignedLong(PyTuple_GET_ITEM(labels,b)))Py_RETURN_FALSE;}
        Py_RETURN_TRUE;
    }catch(const Invalid&){Py_RETURN_FALSE;}catch(...){return error();}
}
PyObject* abi(PyObject*,PyObject*){return PyLong_FromLong(2);}
PyMethodDef methods[]={{"create",create,METH_VARARGS,"Prepare an immutable graph."},{"solve",solve,METH_VARARGS,"Bounded exact finite-domain search."},{"check",check,METH_VARARGS,"Check original edge/list constraints."},{"abi",abi,METH_NOARGS,"Native ABI."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_coloring",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_coloring(){return PyModule_Create(&module);}
