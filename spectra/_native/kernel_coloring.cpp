// SPECTRA two-choice implication kernel with a multi-choice search core.
// MIT; repository LICENSE. Established 2-SAT/backdoor techniques, not a novelty claim.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
namespace {
using U=uint32_t;using M=uint64_t;
constexpr U NONE=UINT32_MAX;
constexpr U MAX_N=100000,MAX_E=2000000;
struct Invalid:std::runtime_error {using std::runtime_error::runtime_error;};
struct Resource:std::runtime_error {using std::runtime_error::runtime_error;};
uint64_t number(PyObject* o,const char* message,uint64_t cap=UINT64_MAX){
    if(!PyLong_CheckExact(o))throw Invalid(message);
    auto value=PyLong_AsUnsignedLongLong(o);
    if(PyErr_Occurred()){PyErr_Clear();throw Invalid(message);}
    if(value>cap)throw Invalid(message);
    return value;
}
U count(M d){return static_cast<U>(__builtin_popcountll(d));}
U first(M d){return static_cast<U>(__builtin_ctzll(d));}
M low(M d){return d&(~d+1);}
bool single(M d){return d&&!(d&(d-1));}
M full(U k){return k==64?~M(0):((M(1)<<k)-1);}
struct Index {
    U n,k,nb=0,nw=0,nc=0;uint64_t m,arcs=0,index_bytes=0,build_bound=0;
    bool loop=false,empty=false,kernel_conflict=false,unrestricted=true;
    std::vector<M> initial;
    std::vector<U> off,to,bid,wid,bvertices,wvertices;
    std::vector<U> component,opposite,coff,members,doff,dto;
    Index(PyObject* nv,PyObject* kv,PyObject* edges,PyObject* masks,uint64_t cap){
        n=static_cast<U>(number(nv,"invalid n",MAX_N));k=static_cast<U>(number(kv,"invalid k",64));
        if(!PyTuple_CheckExact(edges)||!PyTuple_CheckExact(masks))throw Invalid("exact edge/mask tuples required");
        m=static_cast<uint64_t>(PyTuple_GET_SIZE(edges));if(m>MAX_E)throw Resource("too many edges");
        if(PyTuple_GET_SIZE(masks)&&static_cast<uint64_t>(PyTuple_GET_SIZE(masks))!=n)throw Invalid("mask count differs");
        if(28*uint64_t(n)+8*m+64>cap)throw Resource("input index exceeds build payload cap");
        initial.assign(n,full(k));bid.assign(n,NONE);wid.assign(n,NONE);off.assign(size_t(n)+1,0);
        for(U v=0;v<n;++v){
            if(PyTuple_GET_SIZE(masks))initial[v]=number(PyTuple_GET_ITEM(masks,v),"invalid mask");
            if(initial[v]&~full(k))throw Invalid("mask outside colour range");
            unrestricted&=initial[v]==full(k);const U size=count(initial[v]);empty|=size==0;
            if(size&&size<=2)bid[v]=nb++;else if(size>2)wid[v]=nw++;
        }
        bvertices.resize(nb);wvertices.resize(nw);
        for(U v=0;v<n;++v){if(bid[v]!=NONE)bvertices[bid[v]]=v;if(wid[v]!=NONE)wvertices[wid[v]]=v;}
        for(U b=0;b<nb;++b)arcs+=single(initial[bvertices[b]]);
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i){
            PyObject* e=PyTuple_GET_ITEM(edges,i);
            if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)throw Invalid("invalid edge pair");
            uint64_t a=number(PyTuple_GET_ITEM(e,0),"invalid endpoint"),b=number(PyTuple_GET_ITEM(e,1),"invalid endpoint");
            if(a>=n||b>=n)throw Invalid("endpoint outside n");
            loop|=a==b;++off[a+1];++off[b+1];
            if(a!=b&&bid[a]!=NONE&&bid[b]!=NONE)arcs+=2*count(initial[a]&initial[b]);
        }
        // Conservative closed-form bound on simultaneously live native array
        // payload during construction, including SCC scratch and condensation.
        // Python input objects, allocator headers and native stack are excluded.
        build_bound=28*uint64_t(n)+8*m+64*uint64_t(2*nb)+24*arcs+64;
        if(build_bound>cap)throw Resource("implication construction exceeds build payload cap");
        to.resize(2*m);
        for(U v=0;v<n;++v)off[v+1]+=off[v];
        {
            std::vector<U> cursor(off.begin(),off.end()-1);
            for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i){PyObject* e=PyTuple_GET_ITEM(edges,i);
                U a=static_cast<U>(PyLong_AsUnsignedLong(PyTuple_GET_ITEM(e,0))),b=static_cast<U>(PyLong_AsUnsignedLong(PyTuple_GET_ITEM(e,1)));
                to[cursor[a]++]=b;to[cursor[b]++]=a;
            }
        }
        if(!loop&&!empty)compile();
        index_bytes=8*initial.capacity()+4*(off.capacity()+to.capacity()+bid.capacity()+wid.capacity()+bvertices.capacity()+wvertices.capacity()+component.capacity()+opposite.capacity()+coff.capacity()+members.capacity()+doff.capacity()+dto.capacity());
        if(index_bytes>build_bound)throw std::logic_error("index payload bound violated");
    }
    U literal(U v,M colour)const{
        if(bid[v]==NONE||!(initial[v]&colour)||!single(colour))throw std::logic_error("invalid binary literal");
        return 2*bid[v]+(colour==low(initial[v])?0:1);
    }
    void compile(){
        const U nv=2*nb;
        std::vector<std::pair<U,U>> links;links.reserve(arcs);
        for(U b=0;b<nb;++b){const U v=bvertices[b];
            if(single(initial[v]))links.emplace_back(2*b+1,2*b);
            for(U j=off[v];j<off[v+1];++j){const U u=to[j];if(v>=u||bid[u]==NONE)continue;
                M common=initial[v]&initial[u];
                while(common){const M colour=low(common);common&=common-1;
                    U a=literal(v,colour),c=literal(u,colour);links.emplace_back(a,c^1U);links.emplace_back(c,a^1U);
                }
            }
        }
        if(links.size()!=arcs)throw std::logic_error("binary arc inventory differs");
        std::vector<U> foff(size_t(nv)+1,0),roff(size_t(nv)+1,0);
        for(auto e:links){++foff[e.first+1];++roff[e.second+1];}
        for(U v=0;v<nv;++v){foff[v+1]+=foff[v];roff[v+1]+=roff[v];}
        std::vector<U> fcur(foff),rcur(roff),fto(arcs),rto(arcs);
        for(auto e:links){fto[fcur[e.first]++]=e.second;rto[rcur[e.second]++]=e.first;}
        std::vector<uint8_t> seen(nv,0);std::vector<U> order;order.reserve(nv);
        std::vector<std::pair<U,U>> stack;stack.reserve(nv);
        for(U root=0;root<nv;++root)if(!seen[root]){
            seen[root]=1;stack.emplace_back(root,foff[root]);
            while(!stack.empty()){
                auto& item=stack.back();
                if(item.second==foff[item.first+1]){order.push_back(item.first);stack.pop_back();continue;}
                U u=fto[item.second++];if(!seen[u]){seen[u]=1;stack.emplace_back(u,foff[u]);}
            }
        }
        component.assign(nv,NONE);
        for(auto it=order.rbegin();it!=order.rend();++it)if(component[*it]==NONE){
            component[*it]=nc;stack.emplace_back(*it,0);
            while(!stack.empty()){
                U v=stack.back().first;stack.pop_back();
                for(U j=roff[v];j<roff[v+1];++j){U u=rto[j];if(component[u]==NONE){component[u]=nc;stack.emplace_back(u,0);}}
            }
            ++nc;
        }
        for(U b=0;b<nb;++b)if(component[2*b]==component[2*b+1]){kernel_conflict=true;return;}
        opposite.assign(nc,NONE);coff.assign(size_t(nc)+1,0);doff.assign(size_t(nc)+1,0);
        for(U lit=0;lit<nv;++lit){U c=component[lit],other=component[lit^1U];
            if(opposite[c]!=NONE&&opposite[c]!=other)throw std::logic_error("SCC complement is not well-defined");
            opposite[c]=other;++coff[c+1];
        }
        for(auto e:links)if(component[e.first]!=component[e.second]){
            if(component[e.first]>=component[e.second])throw std::logic_error("SCC order is not topological");
            ++doff[component[e.first]+1];
        }
        for(U c=0;c<nc;++c){coff[c+1]+=coff[c];doff[c+1]+=doff[c];}
        members.resize(nv);dto.resize(doff[nc]);
        std::vector<U> cc(coff),dc(doff);
        for(U lit=0;lit<nv;++lit)members[cc[component[lit]]++]=lit;
        for(auto e:links)if(component[e.first]!=component[e.second])dto[dc[component[e.first]]++]=component[e.second];
    }
};
struct Change {U id;bool is_component;M old;};
struct Frame {size_t mark;U core;M options;};
struct Event {U id;bool is_component;};
struct Result {
    std::vector<int> labels;U reason=2;
    uint64_t work=0,branches=0,backtracks=0,domain_changes=0,component_changes=0,events=0,trail_peak=0,state_bytes=0,trace=14695981039346656037ULL;
};
struct Search {
    const Index& g;uint64_t cap,trail_bound;bool symmetry,conflict=false;
    std::vector<M> domains;std::vector<int8_t> truth;
    std::vector<Change> trail;std::vector<Frame> frames;std::vector<Event> queue;size_t qhead=0;
    Result r;
    Search(const Index& index,uint64_t work,uint64_t memory,bool sym):g(index),cap(work),symmetry(sym&&g.unrestricted){
        const uint64_t domain_bound=std::min(64*uint64_t(g.nw),2*g.m+g.nw);
        trail_bound=domain_bound+g.nb;
        r.state_bytes=8*uint64_t(g.nw)+2*uint64_t(g.nc)+4*uint64_t(g.n)+trail_bound*sizeof(Change)+uint64_t(g.nw)*sizeof(Frame)+uint64_t(g.nw+g.nc)*sizeof(Event);
        if(r.state_bytes>memory)throw Resource("kernel search state exceeds payload cap");
        domains.resize(g.nw);for(U w=0;w<g.nw;++w)domains[w]=g.initial[g.wvertices[w]];
        truth.assign(g.nc,0);trail.reserve(trail_bound);frames.reserve(g.nw);queue.reserve(g.nw+g.nc);
    }
    void trace(M value){r.trace=(r.trace^value)*1099511628211ULL;}
    void changed(){r.trail_peak=std::max(r.trail_peak,uint64_t(trail.size()));if(trail.size()>trail_bound)throw std::logic_error("active trail exceeds proven bound");}
    void activate(U c){
        if(truth[c]){conflict|=truth[c]<0;return;}
        const U other=g.opposite[c];
        if(truth[other])throw std::logic_error("asymmetric component truth state");
        truth[c]=1;truth[other]=-1;trail.push_back({c,true,0});changed();
        ++r.component_changes;trace((M(1)<<63)|c);queue.push_back({c,true});
    }
    void reduce(U w,M d){
        const M before=domains[w];if(before==d)return;
        if(d&~before)throw std::logic_error("domain enlargement outside restore");
        trail.push_back({w,false,before});changed();domains[w]=d;++r.domain_changes;trace((uint64_t(w)<<32)^d);
        if(!d)conflict=true;else if(single(d))queue.push_back({w,false});
    }
    bool consume(){if(r.work==cap)return false;++r.work;return true;}
    U propagate(){
        while(qhead<queue.size()&&!conflict){
            if(!consume())return 2;
            const Event e=queue[qhead++];++r.events;
            if(e.is_component){
                for(U j=g.doff[e.id];j<g.doff[e.id+1]&&!conflict;++j){if(!consume())return 2;activate(g.dto[j]);}
                for(U j=g.coff[e.id];j<g.coff[e.id+1]&&!conflict;++j){
                    const U lit=g.members[j],v=g.bvertices[lit/2];const M d=g.initial[v],colour=lit%2?(d^low(d)):low(d);
                    if(!single(colour))throw std::logic_error("impossible singleton complement activated");
                    for(U a=g.off[v];a<g.off[v+1]&&!conflict;++a){const U u=g.to[a];if(g.wid[u]==NONE)continue;
                        if(!consume())return 2;
                        reduce(g.wid[u],domains[g.wid[u]]&~colour);
                    }
                }
            }else{
                const U v=g.wvertices[e.id];const M colour=domains[e.id];if(!single(colour))throw std::logic_error("non-singleton core event");
                for(U j=g.off[v];j<g.off[v+1]&&!conflict;++j){if(!consume())return 2;const U u=g.to[j];
                    if(g.wid[u]!=NONE)reduce(g.wid[u],domains[g.wid[u]]&~colour);
                    else if(g.initial[u]&colour)activate(g.component[g.literal(u,colour)^1U]);
                }
            }
        }
        if(conflict)return 1;
        queue.clear();qhead=0;return 0;
    }
    void restore(size_t mark){
        queue.clear();qhead=0;conflict=false;
        while(trail.size()>mark){const auto c=trail.back();trail.pop_back();
            if(c.is_component){truth[c.id]=0;truth[g.opposite[c.id]]=0;}else domains[c.id]=c.old;
        }
        trace(UINT64_MAX);
    }
    U choose(M& used)const{
        U best=NONE,size=65,degree=0;used=0;
        for(U w=0;w<g.nw;++w){const M d=domains[w];if(!d)return w;
            if(single(d)){used|=d;continue;}
            const U s=count(d),v=g.wvertices[w],dg=g.off[v+1]-g.off[v];
            if(s<size||(s==size&&dg>degree)){best=w;size=s;degree=dg;}
        }
        return best;
    }
    void finish(){
        // Existing true components are implication-closed. In reverse topological
        // order an unassigned component can be set true and its complement false:
        // a false successor would already force this component false by contraposition.
        std::vector<int8_t> completed(truth);
        for(U c=g.nc;c>0;--c)if(!completed[c-1]){completed[c-1]=1;completed[g.opposite[c-1]]=-1;}
        r.labels.assign(g.n,-1);
        for(U w=0;w<g.nw;++w){if(!single(domains[w]))throw std::logic_error("unassigned search core");r.labels[g.wvertices[w]]=static_cast<int>(first(domains[w]));}
        for(U b=0;b<g.nb;++b){const U v=g.bvertices[b];const M d=g.initial[v],colour=completed[g.component[2*b]]>0?low(d):(d^low(d));
            if(!single(colour))throw std::logic_error("invalid completed binary choice");
            r.labels[v]=static_cast<int>(first(colour));
        }
    }
    Result run(){
        if(g.loop){r.reason=4;return std::move(r);}if(g.empty){r.reason=5;return std::move(r);}
        if(g.kernel_conflict){r.reason=3;return std::move(r);}
        for(U b=0;b<g.nb;++b)if(single(g.initial[g.bvertices[b]]))activate(g.component[2*b]);
        while(true){
            U status=propagate();if(status==2){r.reason=1;break;}
            if(status==1){bool resumed=false;
                while(!frames.empty()){
                    auto& f=frames.back();restore(f.mark);++r.backtracks;
                    if(f.options){if(!consume()){r.reason=1;return std::move(r);}M c=low(f.options);f.options&=f.options-1;++r.branches;reduce(f.core,c);resumed=true;break;}
                    frames.pop_back();
                }
                if(!resumed){r.reason=2;break;}
                continue;
            }
            M used=0;const U w=choose(used);
            if(w==NONE){finish();r.reason=0;break;}
            if(!domains[w]){conflict=true;continue;}
            if(!consume()){r.reason=1;break;}
            M options=domains[w];if(symmetry){M unused=options&~used;options=(options&used)|(unused?low(unused):0);}
            if(!options){conflict=true;continue;}
            const M c=low(options);frames.push_back({trail.size(),w,options^c});++r.branches;reduce(w,c);
        }
        return std::move(r);
    }
};
constexpr const char* TAG="spectra.twochoice.kernel.v1";
void destroy(PyObject* capsule){void* p=PyCapsule_GetPointer(capsule,TAG);if(p)delete static_cast<Index*>(p);else PyErr_Clear();}
PyObject* failure(){
    try{throw;}catch(const Invalid& e){PyErr_SetString(PyExc_ValueError,e.what());}
    catch(const Resource& e){PyErr_SetString(PyExc_MemoryError,e.what());}
    catch(const std::bad_alloc&){PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());}
    catch(...){PyErr_SetString(PyExc_RuntimeError,"unknown kernel exception");}return nullptr;
}
PyObject* create(PyObject*,PyObject* args){
    PyObject *n,*k,*e,*m,*cap;if(!PyArg_ParseTuple(args,"OOOOO",&n,&k,&e,&m,&cap))return nullptr;
    try{auto ptr=std::make_unique<Index>(n,k,e,m,number(cap,"invalid build cap"));auto obj=PyCapsule_New(ptr.get(),TAG,destroy);if(obj)ptr.release();return obj;}catch(...){return failure();}
}
struct Release{PyThreadState* t;Release():t(PyEval_SaveThread()){}~Release(){PyEval_RestoreThread(t);}};
PyObject* solve(PyObject*,PyObject* args){
    PyObject *capsule,*work,*memory,*sym;if(!PyArg_ParseTuple(args,"OOOO",&capsule,&work,&memory,&sym))return nullptr;
    auto* g=static_cast<Index*>(PyCapsule_GetPointer(capsule,TAG));if(!g)return nullptr;
    try{
        auto w=number(work,"invalid search work cap"),m=number(memory,"invalid state cap");if(!PyBool_Check(sym))throw Invalid("symmetry must be bool");
        Result r;{Release release;Search search(*g,w,m,sym==Py_True);r=search.run();}
        PyObject* labels=PyTuple_New(r.labels.size());if(!labels)return nullptr;
        for(size_t i=0;i<r.labels.size();++i){auto obj=PyLong_FromLong(r.labels[i]);if(!obj){Py_DECREF(labels);return nullptr;}PyTuple_SET_ITEM(labels,i,obj);}
        return Py_BuildValue("(NIKKKKKKKKKKKKKKKK)",labels,static_cast<unsigned>(r.reason),
            static_cast<unsigned long long>(r.work),static_cast<unsigned long long>(r.branches),
            static_cast<unsigned long long>(r.backtracks),static_cast<unsigned long long>(r.domain_changes),
            static_cast<unsigned long long>(r.component_changes),static_cast<unsigned long long>(r.events),
            static_cast<unsigned long long>(g->nw),static_cast<unsigned long long>(g->nb),
            static_cast<unsigned long long>(g->nc),static_cast<unsigned long long>(g->arcs),
            static_cast<unsigned long long>(g->index_bytes),static_cast<unsigned long long>(g->build_bound),
            static_cast<unsigned long long>(r.state_bytes),static_cast<unsigned long long>(r.trail_peak),
            static_cast<unsigned long long>(r.trace),static_cast<unsigned long long>(g->dto.size()));
    }catch(...){return failure();}
}
// Independent original tuple observer, deliberately not consulting the compiled index.
PyObject* check(PyObject*,PyObject* args){
    PyObject *nv,*kv,*edges,*masks,*labels;if(!PyArg_ParseTuple(args,"OOOOO",&nv,&kv,&edges,&masks,&labels))return nullptr;
    try{
        const uint64_t n=number(nv,"invalid n",MAX_N),k=number(kv,"invalid k",64);
        if(!PyTuple_CheckExact(edges)||!PyTuple_CheckExact(masks)||!PyTuple_CheckExact(labels))Py_RETURN_FALSE;
        if(static_cast<uint64_t>(PyTuple_GET_SIZE(labels))!=n||(PyTuple_GET_SIZE(masks)&&static_cast<uint64_t>(PyTuple_GET_SIZE(masks))!=n)||PyTuple_GET_SIZE(edges)>MAX_E)Py_RETURN_FALSE;
        for(U v=0;v<n;++v){const auto c=number(PyTuple_GET_ITEM(labels,v),"invalid label");if(c>=k)Py_RETURN_FALSE;
            if(PyTuple_GET_SIZE(masks)){const M d=number(PyTuple_GET_ITEM(masks,v),"invalid mask");if((d&~full(k))||!(d&(M(1)<<c)))Py_RETURN_FALSE;}}
        for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i){PyObject* e=PyTuple_GET_ITEM(edges,i);if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)Py_RETURN_FALSE;
            auto a=number(PyTuple_GET_ITEM(e,0),"invalid endpoint"),b=number(PyTuple_GET_ITEM(e,1),"invalid endpoint");if(a>=n||b>=n)Py_RETURN_FALSE;
            if(PyLong_AsUnsignedLong(PyTuple_GET_ITEM(labels,a))==PyLong_AsUnsignedLong(PyTuple_GET_ITEM(labels,b)))Py_RETURN_FALSE;}
        Py_RETURN_TRUE;
    }catch(const Invalid&){Py_RETURN_FALSE;}catch(...){return failure();}
}
PyObject* abi(PyObject*,PyObject*){return PyLong_FromLong(1);}
PyMethodDef methods[]={{"create",create,METH_VARARGS,"Compile exact two-choice implication kernel."},{"solve",solve,METH_VARARGS,"Branch only on the original multi-choice core."},{"check",check,METH_VARARGS,"Check original edge/list constraints."},{"abi",abi,METH_NOARGS,"ABI version."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_kernel_coloring",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_kernel_coloring(){return PyModule_Create(&module);}
