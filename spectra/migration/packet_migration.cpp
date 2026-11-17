// MIT, repository LICENSE. Exact two-plan migration with streamed query bits.
// SCC condensation and word-parallel reachability are established algorithms.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U NIL = UINT32_MAX, MAX_N = 100000;
constexpr W MAX_Q = 1000000, MAX_PINS = 16000000, MAX_E = 2000000;
struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };
W number(PyObject* p, const char* error, W cap = UINT64_MAX) {
    if (!PyLong_CheckExact(p)) throw Invalid(error);
    W v = PyLong_AsUnsignedLongLong(p);
    if (PyErr_Occurred()) { PyErr_Clear(); throw Invalid(error); }
    if (v > cap) throw Invalid(error);
    return v;
}
struct CSR {
    std::vector<U> offsets, targets;
    void build(U n, const std::vector<std::pair<U,U>>& arcs, bool reverse = false) {
        offsets.assign(size_t(n) + 1, 0);
        targets.resize(arcs.size());
        for (auto e : arcs) ++offsets[(reverse ? e.second : e.first) + 1];
        for (U v = 0; v < n; ++v) offsets[v+1] += offsets[v];
        std::vector<U> next(offsets);
        for (auto e : arcs) targets[next[reverse ? e.second : e.first]++] = reverse ? e.first : e.second;
    }
    W bytes() const { return 4 * W(offsets.capacity() + targets.capacity()); }
};
struct Index {
    U n = 0, components = 0, changed = 0;
    W payload = 0, build_bound = 0, source_arcs = 0;
    std::vector<uint8_t> a, b;
    std::vector<U> owner, member_offsets, members;
    CSR original, dag;
    Index(PyObject* edges, PyObject* aa, PyObject* bb, W cap, bool sparse) {
        if (!PyTuple_CheckExact(edges) || !PyTuple_CheckExact(aa) || !PyTuple_CheckExact(bb))
            throw Invalid("edges and plans must be exact tuples");
        W nv = PyTuple_GET_SIZE(aa), m = PyTuple_GET_SIZE(edges);
        if (!nv || nv > MAX_N || m > MAX_E || PyTuple_GET_SIZE(bb) != static_cast<Py_ssize_t>(nv))
            throw Invalid("unsupported graph or plan geometry");
        // Includes simultaneous construction arrays, excluding Python inputs and allocator metadata.
        build_bound = 128 * nv + 128 * m + 8192;
        if (build_bound > cap) throw Resource("construction array-payload bound exceeds cap");
        n = static_cast<U>(nv); a.resize(n); b.resize(n); owner.assign(n, NIL);
        for (U v = 0; v < n; ++v) {
            a[v] = static_cast<uint8_t>(number(PyTuple_GET_ITEM(aa,v), "invalid reference colour",63));
            b[v] = static_cast<uint8_t>(number(PyTuple_GET_ITEM(bb,v), "invalid target colour",63));
            changed += a[v] != b[v];
        }
        std::vector<std::pair<U,U>> arcs; arcs.reserve(2*m);
        for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(edges); ++j) {
            auto e = PyTuple_GET_ITEM(edges,j);
            if (!PyTuple_CheckExact(e) || PyTuple_GET_SIZE(e) != 2) throw Invalid("edge must be an exact pair");
            U u = static_cast<U>(number(PyTuple_GET_ITEM(e,0),"edge endpoint out of range",n-1));
            U v = static_cast<U>(number(PyTuple_GET_ITEM(e,1),"edge endpoint out of range",n-1));
            if (a[u] == a[v] || b[u] == b[v]) throw Invalid("both original plans must be proper");
            if (b[u] == a[v]) arcs.emplace_back(u,v);
            if (b[v] == a[u]) arcs.emplace_back(v,u);
        }
        std::sort(arcs.begin(),arcs.end()); arcs.erase(std::unique(arcs.begin(),arcs.end()),arcs.end());
        source_arcs=arcs.size();
        original.build(n,arcs);
        CSR reverse; reverse.build(n,arcs,true);
        std::vector<uint8_t> seen(n,0);
        std::vector<U> order, stack; order.reserve(changed); stack.reserve(changed);
        std::vector<std::pair<U,U>> dfs; dfs.reserve(changed);
        for (U root = 0; root < n; ++root) if (a[root] != b[root] && !seen[root]) {
            seen[root]=1; dfs.emplace_back(root,original.offsets[root]);
            while (!dfs.empty()) {
                auto& f=dfs.back(); U v=f.first;
                if (f.second == original.offsets[v+1]) { order.push_back(v); dfs.pop_back(); continue; }
                U w=original.targets[f.second++];
                if (!seen[w]) { seen[w]=1; dfs.emplace_back(w,original.offsets[w]); }
            }
        }
        for (auto it=order.rbegin(); it!=order.rend(); ++it) if (owner[*it]==NIL) {
            owner[*it]=components; stack.push_back(*it);
            while (!stack.empty()) {
                U v=stack.back(); stack.pop_back();
                for (U j=reverse.offsets[v]; j<reverse.offsets[v+1]; ++j) {
                    U w=reverse.targets[j];
                    if (owner[w]==NIL) { owner[w]=components; stack.push_back(w); }
                }
            }
            ++components;
        }
        std::vector<std::pair<U,U>> condensed; condensed.reserve(arcs.size());
        for (auto e:arcs) {
            U u=owner[e.first], v=owner[e.second];
            if (u==NIL || v==NIL) throw std::logic_error("unchanged vertex has a migration dependency");
            if (u!=v) {
                if (u>v) throw std::logic_error("SCC numbering is not topological");
                condensed.emplace_back(u,v);
            }
        }
        std::sort(condensed.begin(),condensed.end());
        condensed.erase(std::unique(condensed.begin(),condensed.end()),condensed.end());
        dag.build(components,condensed);
        member_offsets.assign(size_t(components)+1,0); members.resize(changed);
        for (U v=0; v<n; ++v) if (owner[v]!=NIL) ++member_offsets[owner[v]+1];
        for (U c=0; c<components; ++c) member_offsets[c+1]+=member_offsets[c];
        std::vector<U> cursor(member_offsets);
        for (U v=0; v<n; ++v) if (owner[v]!=NIL) members[cursor[owner[v]]++]=v;
        if(sparse && !arcs.empty()) {
            // Retain two original-edge trees per SCC and one real edge per DAG arc.
            // Every retained edge existed originally; reachability is preserved both ways.
            std::vector<std::pair<U,U>> witness; witness.reserve(arcs.size());
            for(U c=0;c<components;++c) {
                U root=members[member_offsets[c]];
                stack.clear();stack.push_back(root);seen[root]=2;
                for(size_t at=0;at<stack.size();++at) {
                    U v=stack[at];
                    for(U j=original.offsets[v];j<original.offsets[v+1];++j) {
                        U w=original.targets[j];
                        if(owner[w]==c && seen[w]!=2) { seen[w]=2;stack.push_back(w);witness.emplace_back(v,w); }
                    }
                }
                stack.clear();stack.push_back(root);seen[root]=3;
                for(size_t at=0;at<stack.size();++at) {
                    U v=stack[at];
                    for(U j=reverse.offsets[v];j<reverse.offsets[v+1];++j) {
                        U w=reverse.targets[j];
                        if(owner[w]==c && seen[w]!=3) { seen[w]=3;stack.push_back(w);witness.emplace_back(w,v); }
                    }
                }
            }
            // Reuse sorted DAG-arc slots, so one representative is selected deterministically.
            std::vector<uint8_t> represented(condensed.size(),0);
            for(auto e:arcs) {
                U u=owner[e.first],v=owner[e.second];if(u==v)continue;
                auto it=std::lower_bound(condensed.begin(),condensed.end(),std::make_pair(u,v));
                size_t pos=static_cast<size_t>(it-condensed.begin());
                if(it==condensed.end() || *it!=std::make_pair(u,v))throw std::logic_error("missing quotient edge");
                if(!represented[pos]) { represented[pos]=1;witness.push_back(e); }
            }
            std::sort(witness.begin(),witness.end());witness.erase(std::unique(witness.begin(),witness.end()),witness.end());
            if(witness.size()>arcs.size())throw std::logic_error("witness graph expanded");
            // Replace, not resize: vector capacity must really release discarded edges.
            CSR smaller;smaller.build(n,witness);original=std::move(smaller);
        }
        payload = a.capacity()+b.capacity()+4*W(owner.capacity()+member_offsets.capacity()+members.capacity())+original.bytes()+dag.bytes();
        if (payload>build_bound) throw std::logic_error("construction payload bound underestimated");
    }
};
U read_u32(const unsigned char* p) {
    return U(p[0])|(U(p[1])<<8)|(U(p[2])<<16)|(U(p[3])<<24);
}
W read_u64(const unsigned char* p) {
    return W(read_u32(p))|(W(read_u32(p+4))<<32);
}
struct Queries {
    std::vector<U> offsets, vertices; std::vector<W> masks;
    const unsigned char *wire_offsets=nullptr,*wire_vertices=nullptr,*wire_masks=nullptr;
    W bound=0, borrowed=0;
    U rows=0, pins=0;
    U offset(U i) const {return wire_offsets?read_u32(wire_offsets+4*W(i)):offsets[i];}
    U vertex(U i) const {return wire_vertices?read_u32(wire_vertices+4*W(i)):vertices[i];}
    W mask(U i) const {return wire_masks?read_u64(wire_masks+8*W(i)):masks[i];}
    U count() const {return wire_offsets?rows:static_cast<U>(offsets.size()-1);}
};
Queries parse_packet(PyObject* qs,const Index& x,W cap) {
    if(!PyBytes_CheckExact(qs))throw Invalid("packet must be exact immutable bytes");
    W size=PyBytes_GET_SIZE(qs);
    if(size<20)throw Invalid("truncated query packet");
    const auto* p=reinterpret_cast<const unsigned char*>(PyBytes_AS_STRING(qs));
    if(std::memcmp(p,"SPQ1",4)||read_u32(p+12)!=0)throw Invalid("unknown query packet format");
    Queries q;q.rows=read_u32(p+4);q.pins=read_u32(p+8);
    if(q.rows>MAX_Q||q.pins>MAX_PINS)throw Resource("query packet exceeds count contract");
    W required=16+4*(W(q.rows)+1)+12*W(q.pins);
    if(required!=size)throw Invalid("query packet length differs");
    if(W(q.rows)*x.n>UINT32_MAX)throw Resource("output/proof geometry exceeds uint32");
    q.bound=20*W(q.rows)*x.n+size+80*W(q.rows)+64*W(x.n+x.components)+16384;
    if(q.bound>cap)throw Resource("batch array-payload bound exceeds cap");
    q.wire_offsets=p+16;
    q.wire_vertices=q.wire_offsets+4*(W(q.rows)+1);
    q.wire_masks=q.wire_vertices+4*W(q.pins);q.borrowed=size;
    if(q.offset(0)!=0||q.offset(q.rows)!=q.pins)throw Invalid("unconsumed query pins");
    for(U i=0;i<q.rows;++i)
        if(q.offset(i)>q.offset(i+1)||q.offset(i+1)>q.pins)throw Invalid("invalid query packet offsets");
    for(U j=0;j<q.pins;++j)if(q.vertex(j)>=x.n)throw Invalid("restriction vertex out of range");
    return q;
}
Queries parse(PyObject* qs, const Index& x, W cap) {
    if(PyBytes_CheckExact(qs))return parse_packet(qs,x,cap);
    if (!PyTuple_CheckExact(qs)) throw Invalid("query bank must be an exact tuple");
    W count=PyTuple_GET_SIZE(qs), pins=0;
    if (count>MAX_Q) throw Resource("too many queries");
    for (W i=0; i<count; ++i) {
        auto q=PyTuple_GET_ITEM(qs,i);
        if (!PyTuple_CheckExact(q)) throw Invalid("each query must be an exact tuple");
        pins += PyTuple_GET_SIZE(q); if (pins>MAX_PINS) throw Resource("too many restrictions");
    }
    if (count*W(x.n)>UINT32_MAX) throw Resource("output/proof geometry exceeds uint32");
    Queries p;
    // Worst-case admission is deliberately conservative; no Q*N proof reserve is allocated.
    p.bound=20*count*x.n+16*pins+80*count+64*W(x.n+x.components)+16384;
    if (p.bound>cap) throw Resource("batch array-payload bound exceeds cap");
    p.offsets.resize(count+1); p.vertices.reserve(pins); p.masks.reserve(pins);
    for (W i=0; i<count; ++i) {
        auto q=PyTuple_GET_ITEM(qs,i);
        for (Py_ssize_t j=0; j<PyTuple_GET_SIZE(q); ++j) {
            auto item=PyTuple_GET_ITEM(q,j);
            if (!PyTuple_CheckExact(item)||PyTuple_GET_SIZE(item)!=2) throw Invalid("restriction must be an exact pair");
            p.vertices.push_back(static_cast<U>(number(PyTuple_GET_ITEM(item,0),"restriction vertex out of range",x.n-1)));
            p.masks.push_back(number(PyTuple_GET_ITEM(item,1),"invalid restriction mask"));
        }
        p.offsets[i+1]=static_cast<U>(p.vertices.size());
    }
    return p;
}
struct PrivateAnswer {
    PyObject* object=nullptr;
    W length=0;
    PrivateAnswer(W n):length(n) {
        object=PyBytes_FromStringAndSize(nullptr,static_cast<Py_ssize_t>(n));
        if(!object)throw std::bad_alloc();
        if(n)std::memset(PyBytes_AS_STRING(object),0,n);
    }
    ~PrivateAnswer(){Py_XDECREF(object);}
    PrivateAnswer(const PrivateAnswer&)=delete;
    PrivateAnswer& operator=(const PrivateAnswer&)=delete;
    char* data(){return PyBytes_AS_STRING(object);}
    W size() const{return length;}
    PyObject* release(){auto p=object;object=nullptr;return p;}
};
struct Output {
    std::vector<char> status;
    PrivateAnswer labels;
    std::vector<char> kinds;
    std::vector<U> offsets, paths;
    W dag_visits=0, proof_visits=0, tiles=0, query_payload=0;
    explicit Output(W q,U n):status(q,0),labels(q*n),kinds(q,0),offsets(q+1,0) {}
};
struct Worker {
    const Index& x; const Queries& q; Output& out;
    std::vector<W> forced, forbidden;
    std::vector<U> epoch, queue, parents, original_queue, proof;
    std::vector<uint8_t> stays;
    U stamp=0;
    Worker(const Index& xx,const Queries& qq,Output& oo):x(xx),q(qq),out(oo),forced(x.components),forbidden(x.components),epoch(x.components,0),parents(x.n,NIL),stays(x.n) {
        queue.reserve(x.components); original_queue.reserve(x.changed); proof.reserve(x.changed ? x.changed : 1);
    }
    // Every path uses original edges. Deliberately no stored answer/proof bank.
    void contradiction(U i) {
        std::fill(parents.begin(),parents.end(),NIL); std::fill(stays.begin(),stays.end(),0);
        original_queue.clear();
        for(U j=q.offset(i);j<q.offset(i+1);++j) {
            U v=q.vertex(j); W mask=q.mask(j);
            if(x.a[v]==x.b[v]) continue;
            if (!(mask&(W(1)<<x.a[v])) && parents[v]==NIL) { parents[v]=v; original_queue.push_back(v); }
            if (!(mask&(W(1)<<x.b[v]))) stays[v]=1;
        }
        U conflict=NIL;
        for(size_t head=0;head<original_queue.size();++head) {
            U v=original_queue[head];
            if(stays[v]) { conflict=v; break; }
            for(U j=x.original.offsets[v];j<x.original.offsets[v+1];++j) {
                ++out.proof_visits; U w=x.original.targets[j];
                if(parents[w]==NIL) { parents[w]=v; original_queue.push_back(w); }
            }
        }
        if(conflict==NIL) throw std::logic_error("SCC contradiction has no original path");
        proof.clear(); U v=conflict;
        while(parents[v]!=v) {
            proof.push_back(v); v=parents[v];
            if(v==NIL || proof.size()>x.changed) throw std::logic_error("invalid predecessor chain");
        }
        proof.push_back(v); std::reverse(proof.begin(),proof.end());
        out.kinds[i]=1; out.status[i]=1; out.paths.insert(out.paths.end(),proof.begin(),proof.end());
    }
    U empty_vertex(U i) const {
        U empty=NIL;
        for(U j=q.offset(i);j<q.offset(i+1);++j) {
            U v=q.vertex(j); W mask=q.mask(j);
            if(!(mask & ((W(1)<<x.a[v])|(W(1)<<x.b[v])))) empty=v;
        }
        return empty;
    }
    void finish(U i,U empty,bool bad) {
        if(empty!=NIL) { out.status[i]=1;out.kinds[i]=2;out.paths.push_back(empty); }
        else if(bad) contradiction(i);
        out.offsets[i+1]=static_cast<U>(out.paths.size());
    }
    void single(U i) {
        U empty=empty_vertex(i);
        if(empty!=NIL) { finish(i,empty,true); return; }
        ++stamp; // Q<=1e6, cannot wrap.
        queue.clear(); bool bad=false;
        for(U j=q.offset(i);j<q.offset(i+1);++j) {
            U v=q.vertex(j), c=x.owner[v];
            if(c!=NIL && !(q.mask(j)&(W(1)<<x.a[v])) && epoch[c]!=stamp) {
                epoch[c]=stamp; queue.push_back(c);
            }
            if(c!=NIL && !(q.mask(j)&(W(1)<<x.b[v]))) forbidden[c]=stamp;
        }
        for(size_t head=0;head<queue.size();++head) {
            U c=queue[head];
            if(forbidden[c]==stamp) { bad=true; break; }
            for(U j=x.dag.offsets[c];j<x.dag.offsets[c+1];++j) {
                ++out.dag_visits; U w=x.dag.targets[j];
                if(epoch[w]!=stamp) { epoch[w]=stamp; queue.push_back(w); }
            }
        }
        if(!bad) {
            char* dest=out.labels.data()+W(i)*x.n; std::memcpy(dest,x.a.data(),x.n);
            for(U c:queue) for(U j=x.member_offsets[c];j<x.member_offsets[c+1];++j) {
                U v=x.members[j];dest[v]=static_cast<char>(x.b[v]);
            }
        }
        finish(i,NIL,bad);
    }
    void tile(U begin,U end) {
        ++out.tiles;
        std::fill(forced.begin(),forced.end(),0);std::fill(forbidden.begin(),forbidden.end(),0);
        U empty[64]; W invalid=0;
        for(U i=begin;i<end;++i) {
            W bit=W(1)<<(i-begin); empty[i-begin]=NIL;
            for(U j=q.offset(i);j<q.offset(i+1);++j) {
                U v=q.vertex(j),c=x.owner[v]; W mask=q.mask(j);
                bool aa=mask&(W(1)<<x.a[v]),bb=mask&(W(1)<<x.b[v]);
                if(!aa&&!bb) { empty[i-begin]=v;invalid|=bit;continue; }
                if(c!=NIL) { if(!aa)forced[c]|=bit;if(!bb)forbidden[c]|=bit; }
            }
        }
        W bad=invalid;
        // Kosaraju component IDs are checked topological during preparation.
        // Each graph edge is visited at most once per 64-request tile.
        for(U c=0;c<x.components;++c) {
            W bits=forced[c]&~bad;bad|=bits&forbidden[c];bits&=~bad;
            if(!bits)continue;
            for(U j=x.dag.offsets[c];j<x.dag.offsets[c+1];++j) {
                ++out.dag_visits; forced[x.dag.targets[j]]|=bits;
            }
        }
        for(U i=begin;i<end;++i) if(!(bad&(W(1)<<(i-begin))))
            std::memcpy(out.labels.data()+W(i)*x.n,x.a.data(),x.n);
        for(U c=0;c<x.components;++c) {
            W bits=forced[c]&~bad;
            while(bits) {
                U k=static_cast<U>(__builtin_ctzll(bits));bits&=bits-1;
                char* dest=out.labels.data()+W(begin+k)*x.n;
                for(U j=x.member_offsets[c];j<x.member_offsets[c+1];++j) {
                    U v=x.members[j];dest[v]=static_cast<char>(x.b[v]);
                }
            }
        }
        for(U i=begin;i<end;++i) finish(i,empty[i-begin],bad&(W(1)<<(i-begin)));
    }
};
constexpr const char* TAG="spectra.packet.migration.v1";
void destroy(PyObject* c) { void* p=PyCapsule_GetPointer(c,TAG);if(p)delete static_cast<Index*>(p);else PyErr_Clear(); }
PyObject* fail() {
    try { throw; }
    catch(const Invalid& e) { PyErr_SetString(PyExc_ValueError,e.what()); }
    catch(const Resource& e) { PyErr_SetString(PyExc_MemoryError,e.what()); }
    catch(const std::bad_alloc&) { PyErr_NoMemory(); }
    catch(const std::exception& e) { PyErr_SetString(PyExc_RuntimeError,e.what()); }
    catch(...) { PyErr_SetString(PyExc_RuntimeError,"compact migration failure"); }
    return nullptr;
}
PyObject* create(PyObject*,PyObject* args) {
    PyObject *edges,*a,*b,*cap,*sparse;
    if(!PyArg_ParseTuple(args,"OOOOO",&edges,&a,&b,&cap,&sparse))return nullptr;
    try {
        if(!PyBool_Check(sparse))throw Invalid("sparsify must be bool");
        auto x=std::make_unique<Index>(edges,a,b,number(cap,"invalid index cap"),sparse==Py_True);
        PyObject* c=PyCapsule_New(x.get(),TAG,destroy);if(c)x.release();return c;
    } catch(...) { return fail(); }
}
PyObject* info(PyObject*,PyObject* c) {
    auto x=static_cast<Index*>(PyCapsule_GetPointer(c,TAG));if(!x)return nullptr;
    return Py_BuildValue("(IIIKKKKK)",x->n,x->changed,x->components,
        static_cast<unsigned long long>(x->original.targets.size()),static_cast<unsigned long long>(x->dag.targets.size()),
        static_cast<unsigned long long>(x->payload),static_cast<unsigned long long>(x->build_bound),static_cast<unsigned long long>(x->source_arcs));
}
PyObject* solve(PyObject*,PyObject* args) {
    PyObject *c,*qs,*mode_object,*cap;
    if(!PyArg_ParseTuple(args,"OOOO",&c,&qs,&mode_object,&cap))return nullptr;
    auto x=static_cast<Index*>(PyCapsule_GetPointer(c,TAG));if(!x)return nullptr;
    try {
        U mode=static_cast<U>(number(mode_object,"mode must be 0 (SCC-BFS), 1 (tiled), or 2 (auto)",2));
        Queries q=parse(qs,*x,number(cap,"invalid batch cap")); U count=q.count();
        // Fixed engineering policy, not selected from outcomes/timings. Explicit modes remain available.
        if(mode==2)mode=count<64?0:1;
        Output out(count,x->n);
        PyThreadState* thread=PyEval_SaveThread();
        try {
            Worker worker(*x,q,out);
            if(mode==0)for(U i=0;i<count;++i)worker.single(i);
            else for(U i=0;i<count;i+=64)worker.tile(i,std::min<U>(count,i+64));
            out.query_payload=8*W(worker.forced.capacity()+worker.forbidden.capacity())
                +4*W(worker.epoch.capacity()+worker.queue.capacity()+worker.parents.capacity()+worker.original_queue.capacity()+worker.proof.capacity())
                +worker.stays.capacity();
        } catch(...) { PyEval_RestoreThread(thread);throw; }
        PyEval_RestoreThread(thread);
        std::vector<char> paths(4*out.paths.size());
        for(size_t i=0;i<out.paths.size();++i)for(U j=0;j<4;++j)paths[4*i+j]=static_cast<char>((out.paths[i]>>(8*j))&255);
        // The full answer was filled in a new private PyBytes buffer before publication.
        // Smaller status/proof buffers remain copied. No published object is mutated.
        PyObject* s=PyBytes_FromStringAndSize(out.status.data(),out.status.size());
        PyObject* l=out.labels.release();
        PyObject* k=PyBytes_FromStringAndSize(out.kinds.data(),out.kinds.size());
        PyObject* p=PyBytes_FromStringAndSize(paths.data(),paths.size());
        PyObject* offsets=PyTuple_New(out.offsets.size());
        auto cleanup=[&](){Py_XDECREF(s);Py_XDECREF(l);Py_XDECREF(k);Py_XDECREF(p);Py_XDECREF(offsets);};
        if(!s||!l||!k||!p||!offsets){cleanup();return nullptr;}
        for(size_t i=0;i<out.offsets.size();++i) {
            PyObject* v=PyLong_FromUnsignedLong(out.offsets[i]);if(!v){cleanup();return nullptr;}PyTuple_SET_ITEM(offsets,i,v);
        }
        return Py_BuildValue("(NNNNNIKKKKK)",s,l,k,offsets,p,mode,
            static_cast<unsigned long long>(out.dag_visits),static_cast<unsigned long long>(out.proof_visits),
            static_cast<unsigned long long>(out.tiles),static_cast<unsigned long long>(out.query_payload),
            static_cast<unsigned long long>(q.bound));
    } catch(...) { return fail(); }
}

void put_u32(unsigned char* p,U v){for(U j=0;j<4;++j)p[j]=static_cast<unsigned char>(v>>(8*j));}
void put_u64(unsigned char* p,W v){put_u32(p,static_cast<U>(v));put_u32(p+4,static_cast<U>(v>>32));}
PyObject* pack(PyObject*,PyObject* args) {
    PyObject *qs,*cap;
    if(!PyArg_ParseTuple(args,"OO",&qs,&cap))return nullptr;
    try {
        W budget=number(cap,"invalid packet cap");
        if(!PyTuple_CheckExact(qs))throw Invalid("queries must be an exact frozen tuple");
        W rows=PyTuple_GET_SIZE(qs),pins=0;
        if(rows>MAX_Q)throw Resource("too many queries");
        for(W i=0;i<rows;++i) {
            auto q=PyTuple_GET_ITEM(qs,i);
            if(!PyTuple_CheckExact(q))throw Invalid("each query must be exact tuple");
            pins+=PyTuple_GET_SIZE(q);
            if(pins>MAX_PINS)throw Resource("too many pins");
        }
        W size=16+4*(rows+1)+12*pins;
        if(size>budget)throw Resource("query packet exceeds cap");
        // New private storage only; never modify a published or interned bytes object.
        PyObject* obj=PyBytes_FromStringAndSize(nullptr,static_cast<Py_ssize_t>(size));
        if(!obj)return nullptr;
        try {
            auto* data=reinterpret_cast<unsigned char*>(PyBytes_AS_STRING(obj));
            std::memcpy(data,"SPQ1",4);put_u32(data+4,static_cast<U>(rows));
            put_u32(data+8,static_cast<U>(pins));put_u32(data+12,0);
            auto* off=data+16;auto* vertices=off+4*(rows+1);auto* masks=vertices+4*pins;
            U at=0;put_u32(off,0);
            for(U i=0;i<rows;++i) {
                auto q=PyTuple_GET_ITEM(qs,i);
                for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(q);++j) {
                    auto pin=PyTuple_GET_ITEM(q,j);
                    if(!PyTuple_CheckExact(pin)||PyTuple_GET_SIZE(pin)!=2)throw Invalid("restriction must be an exact pair");
                    U v=static_cast<U>(number(PyTuple_GET_ITEM(pin,0),"restriction vertex out of range",MAX_N-1));
                    W m=number(PyTuple_GET_ITEM(pin,1),"invalid restriction mask");
                    put_u32(vertices+4*W(at),v);put_u64(masks+8*W(at),m);++at;
                }
                put_u32(off+4*(W(i)+1),at);
            }
        } catch(...) {Py_DECREF(obj);throw;}
        return obj;
    } catch(...) {return fail();}
}

PyMethodDef methods[]={{"pack",pack,METH_VARARGS,"Pack immutable original-address pins; no formula or answer is used."},{"create",create,METH_VARARGS,"Prepare a sparse exact migration index."},
    {"solve",solve,METH_VARARGS,"Return full assignments or original dependency paths."},
    {"info",info,METH_O,"Actual retained array capacities and conservative construction bound."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_packet_migration",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_packet_migration(){return PyModule_Create(&module);}
