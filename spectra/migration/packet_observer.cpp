// MIT, repository LICENSE. Independent verification from ORIGINAL two-plan inputs.
// No solver SCCs, reachability tables, proof graph, or cached answers are accepted.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U NIL = UINT32_MAX, MAX_N = 100000;
struct Bad : std::runtime_error { using std::runtime_error::runtime_error; };
struct Large : std::runtime_error { using std::runtime_error::runtime_error; };
W integer(PyObject* p, const char* message, W cap = UINT64_MAX) {
    if (!PyLong_CheckExact(p)) throw Bad(message);
    W value = PyLong_AsUnsignedLongLong(p);
    if (PyErr_Occurred()) { PyErr_Clear(); throw Bad(message); }
    if (value > cap) throw Bad(message);
    return value;
}
struct Observer {
    U n = 0;
    W original_edges = 0, payload = 0, build_bound = 0;
    std::vector<uint8_t> a, b;
    std::vector<U> starts, following;
    Observer(PyObject* ee, PyObject* aa, PyObject* bb, W cap) {
        if (!PyTuple_CheckExact(ee) || !PyTuple_CheckExact(aa) || !PyTuple_CheckExact(bb))
            throw Bad("original inputs must be exact tuples");
        W nv = PyTuple_GET_SIZE(aa), m = PyTuple_GET_SIZE(ee);
        if (!nv || nv > MAX_N || m > 2000000 || PyTuple_GET_SIZE(bb) != static_cast<Py_ssize_t>(nv))
            throw Bad("invalid original geometry");
        build_bound = 64 * nv + 64 * m + 8192;
        if (build_bound > cap) throw Large("original observer construction exceeds cap");
        n = static_cast<U>(nv);
        a.resize(n); b.resize(n);
        for (U v = 0; v < n; ++v) {
            a[v] = static_cast<uint8_t>(integer(PyTuple_GET_ITEM(aa, v), "invalid original colour", 63));
            b[v] = static_cast<uint8_t>(integer(PyTuple_GET_ITEM(bb, v), "invalid target colour", 63));
        }
        std::vector<std::pair<U,U>> original;
        original.reserve(m);
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(ee); ++i) {
            auto edge = PyTuple_GET_ITEM(ee, i);
            if (!PyTuple_CheckExact(edge) || PyTuple_GET_SIZE(edge) != 2)
                throw Bad("edge must be an exact pair");
            U u = static_cast<U>(integer(PyTuple_GET_ITEM(edge, 0), "bad endpoint", n-1));
            U v = static_cast<U>(integer(PyTuple_GET_ITEM(edge, 1), "bad endpoint", n-1));
            // This checks every original edge, including edges later discarded.
            if (a[u] == a[v] || b[u] == b[v]) throw Bad("original plans are not proper");
            original.emplace_back(std::min(u,v), std::max(u,v));
        }
        std::sort(original.begin(), original.end());
        original.erase(std::unique(original.begin(), original.end()), original.end());
        original_edges = original.size();
        std::vector<std::pair<U,U>> deps;
        deps.reserve(2*original_edges);
        for (auto e : original) {
            if (b[e.first] == a[e.second]) deps.emplace_back(e.first, e.second);
            if (b[e.second] == a[e.first]) deps.emplace_back(e.second, e.first);
        }
        std::sort(deps.begin(), deps.end());
        starts.assign(n+1, 0);
        following.resize(deps.size());
        for (size_t i = 0; i < deps.size(); ++i) {
            ++starts[deps[i].first+1]; following[i] = deps[i].second;
        }
        for (U v = 0; v < n; ++v) starts[v+1] += starts[v];
        // After proper A/B validation, the only possible collisions are BA/AB.
        // Keeping these implications is equivalent to keeping ALL original edges.
        // No SCC contraction or transitive reduction is performed by this checker.
        payload = a.capacity()+b.capacity()+4*W(starts.capacity()+following.capacity());
        if (payload > build_bound) throw std::logic_error("observer admission underestimated");
    }
    bool original_dependency(U u, U v) const {
        return b[u] == a[v] && std::binary_search(
            following.begin()+starts[u], following.begin()+starts[u+1], v);
    }
};
W from_le(const uint8_t* data, unsigned bytes) {
    W value=0;
    for(unsigned k=0;k<bytes;++k)value|=W(data[k])<<(8*k);
    return value;
}
struct Bank {
    std::vector<U> starts, vertices;
    std::vector<W> masks;
    W admission=0,borrowed=0;
    const uint8_t* packet=nullptr;
    U rows=0,pins=0;
    U count() const{return packet?rows:static_cast<U>(starts.size()-1);}
    U start(U i) const{return packet?static_cast<U>(from_le(packet+16+4*W(i),4)):starts[i];}
    U vertex(U i) const{return packet?static_cast<U>(from_le(packet+16+4*(W(rows)+1)+4*W(i),4)):vertices[i];}
    W mask(U i) const{return packet?from_le(packet+16+4*(W(rows)+1)+4*W(pins)+8*W(i),8):masks[i];}
};
Bank read_packet(PyObject* object,U n,W cap) {
    // Independent transport reader: does not call the producer/solver parser.
    W bytes=PyBytes_GET_SIZE(object);
    if(bytes<20)throw Bad("short query packet");
    const auto* p=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(object));
    if(p[0]!='S'||p[1]!='P'||p[2]!='Q'||p[3]!='1'||from_le(p+12,4))
        throw Bad("invalid packet schema or flags");
    W rows=from_le(p+4,4),pins=from_le(p+8,4);
    if(rows>1000000||pins>16000000)throw Large("packet count exceeds observer contract");
    if(16+4*(rows+1)+12*pins!=bytes)throw Bad("packet byte inventory differs");
    Bank q;q.rows=static_cast<U>(rows);q.pins=static_cast<U>(pins);q.packet=p;q.borrowed=bytes;
    q.admission=bytes+8*(rows+1)+96*W(n)+8192;
    if(q.admission>cap)throw Large("original observer scratch exceeds cap");
    U previous=0;
    for(U i=0;i<=q.rows;++i) {
        U current=q.start(i);
        if((i==0&&current!=0)||current<previous||current>pins)throw Bad("noncanonical query spans");
        previous=current;
    }
    if(previous!=pins)throw Bad("unconsumed packet pins");
    for(U j=0;j<q.pins;++j)if(q.vertex(j)>=n)throw Bad("bad query vertex");
    return q;
}
Bank read_queries(PyObject* qs, U n, W cap) {
    if(PyBytes_CheckExact(qs))return read_packet(qs,n,cap);
    if (!PyTuple_CheckExact(qs)) throw Bad("queries must be an exact tuple");
    W count = PyTuple_GET_SIZE(qs), pins = 0;
    if (count > 1000000) throw Large("too many queries");
    for (W i = 0; i < count; ++i) {
        auto q = PyTuple_GET_ITEM(qs, i);
        if (!PyTuple_CheckExact(q)) throw Bad("query must be an exact tuple");
        pins += PyTuple_GET_SIZE(q);
        if (pins > 16000000) throw Large("too many query pins");
    }
    Bank bank;
    // Covers simultaneous bank, offsets, scalar-proof and tile work arrays.
    // Borrowed Python input/output bytes and Python allocator costs are excluded.
    bank.admission = 16*pins + 8*(count+1) + 96*W(n) + 8192;
    if (bank.admission > cap) throw Large("original observer scratch exceeds cap");
    bank.starts.resize(count+1, 0); bank.vertices.reserve(pins); bank.masks.reserve(pins);
    for (W i = 0; i < count; ++i) {
        auto q = PyTuple_GET_ITEM(qs, i);
        for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(q); ++j) {
            auto pin = PyTuple_GET_ITEM(q, j);
            if (!PyTuple_CheckExact(pin) || PyTuple_GET_SIZE(pin) != 2)
                throw Bad("pin must be an exact pair");
            bank.vertices.push_back(static_cast<U>(integer(PyTuple_GET_ITEM(pin,0), "bad query vertex", n-1)));
            bank.masks.push_back(integer(PyTuple_GET_ITEM(pin,1), "bad query mask"));
        }
        bank.starts[i+1] = static_cast<U>(bank.vertices.size());
    }
    return bank;
}
U decode(const uint8_t* p) { return U(p[0]) | (U(p[1])<<8) | (U(p[2])<<16) | (U(p[3])<<24); }
struct Counts {
    W sat = 0, unsat = 0, migrations = 0, proof_edges = 0, dependency_visits = 0, scratch = 0;
};
struct Worker {
    const Observer& x;
    const Bank& q;
    const uint8_t *statuses, *labels, *kinds, *paths;
    const std::vector<U>& offsets;
    Counts result;
    std::vector<U> moving, staying, visited, path_seen, queue;
    Worker(const Observer& xx, const Bank& qq, const uint8_t* ss, const uint8_t* ll,
           const uint8_t* kk, const std::vector<U>& oo, const uint8_t* pp, bool minimal, bool scalar)
        : x(xx), q(qq), statuses(ss), labels(ll), kinds(kk), paths(pp), offsets(oo),
          moving(x.n,0), staying(x.n,0), visited(minimal && scalar ? x.n:0,0), path_seen(x.n,0), queue(minimal ? x.n:0) {
        result.scratch = 4*W(moving.capacity()+staying.capacity()+visited.capacity()+path_seen.capacity()+queue.capacity());
    }
    U pins(U i) {
        U stamp=i+1, empty=NIL;
        for (U j=q.start(i); j<q.start(i+1); ++j) {
            U v=q.vertex(j); W mask=q.mask(j);
            bool aa=(mask>>x.a[v])&1, bb=(mask>>x.b[v])&1;
            if (!aa && !bb) empty=v;
            if (x.a[v]!=x.b[v]) {
                if (!aa) moving[v]=stamp;
                if (!bb) staying[v]=stamp;
            }
        }
        return empty;
    }
    void proof(U i) {
        U begin=offsets[i], end=offsets[i+1], stamp=i+1;
        const uint8_t* answer=labels+W(i)*x.n;
        uint8_t nonzero=0;
        for (U v=0; v<x.n; ++v) nonzero |= answer[v];
        if (nonzero) throw Bad("UNSAT output padding must be zero");
        if (!(end>begin && W(end)-begin<=x.n)) throw Bad("invalid proof path length");
        U first=decode(paths+W(begin)*4), last=decode(paths+W(end-1)*4);
        if (first>=x.n || last>=x.n) throw Bad("proof vertex outside graph");
        if (kinds[i]==2) {
            if (end-begin!=1) throw Bad("empty-domain proof must name one vertex");
            bool found=false;
            for (U j=q.start(i); j<q.start(i+1); ++j)
                if (q.vertex(j)==first && !(q.mask(j)&((W(1)<<x.a[first])|(W(1)<<x.b[first])))) found=true;
            if (!found) throw Bad("false empty-domain proof");
        } else if (kinds[i]==1) {
            if (moving[first]!=stamp || staying[last]!=stamp)
                throw Bad("contradiction endpoints are not pinned oppositely");
            U prev=NIL;
            for (U j=begin; j<end; ++j) {
                U v=decode(paths+W(j)*4);
                if (v>=x.n || path_seen[v]==stamp) throw Bad("invalid or repeated proof vertex");
                path_seen[v]=stamp;
                if (prev!=NIL) {
                    if (!x.original_dependency(prev,v)) throw Bad("proof step is not an original conflict implication");
                    ++result.proof_edges;
                }
                prev=v;
            }
        } else throw Bad("unknown contradiction kind");
        ++result.unsat;
    }
    void sat_header(U i, U empty) const {
        if (kinds[i]!=0 || offsets[i]!=offsets[i+1] || empty!=NIL)
            throw Bad("SAT output carries a contradiction");
    }
    void request_check(U i) const {
        const uint8_t* answer=labels+W(i)*x.n;
        // Membership must have been checked first, so every shift is in [0,63].
        for (U j=q.start(i); j<q.start(i+1); ++j)
            if (!(q.mask(j)>>answer[q.vertex(j)]&1)) throw Bad("answer violates a request");
    }
    void scalar(bool minimal) {
        U count=q.count();
        for (U i=0; i<count; ++i) {
            U stamp=i+1, empty=pins(i);
            if (statuses[i]==1) { proof(i); continue; }
            if (statuses[i]!=0) throw Bad("unknown result status");
            sat_header(i,empty);
            const uint8_t* answer=labels+W(i)*x.n;
            for (U v=0; v<x.n; ++v) {
                if (answer[v]!=x.a[v] && answer[v]!=x.b[v]) throw Bad("answer is outside original plans");
                result.migrations += answer[v]!=x.a[v];
            }
            request_check(i);
            // Exact original-constraint check, independent of minimality.
            for (U v=0; v<x.n; ++v) if (answer[v]!=x.a[v])
                for (U j=x.starts[v]; j<x.starts[v+1]; ++j)
                    if (answer[x.following[j]]==x.a[x.following[j]]) throw Bad("answer violates an original conflict");
            if (minimal) {
                U head=0, tail=0;
                for (U j=q.start(i); j<q.start(i+1); ++j) {
                    U v=q.vertex(j);
                    if (moving[v]==stamp && visited[v]!=stamp) { visited[v]=stamp; queue[tail++]=v; }
                }
                while (head<tail) {
                    U v=queue[head++];
                    for (U j=x.starts[v]; j<x.starts[v+1]; ++j) {
                        ++result.dependency_visits;
                        U w=x.following[j];
                        if (visited[w]!=stamp) { visited[w]=stamp; queue[tail++]=w; }
                    }
                }
                for (U v=0; v<x.n; ++v)
                    if ((answer[v]!=x.a[v])!=(visited[v]==stamp)) throw Bad("answer is not the minimum-change closure");
            }
            ++result.sat;
        }
    }
    void tiled(bool minimal) {
        U count=q.count();
        std::vector<W> actual(x.n), required(minimal ? x.n:0);
        std::vector<uint8_t> queued(minimal ? x.n:0);
        result.scratch += 8*W(actual.capacity()+required.capacity())+queued.capacity();
        for (U first=0; first<count; first+=64) {
            std::fill(actual.begin(),actual.end(),0);
            std::fill(required.begin(),required.end(),0);
            U end=std::min<U>(count,first+64);
            W valid=0;
            for (U i=first; i<end; ++i) {
                U empty=pins(i);
                if (statuses[i]==1) { proof(i); continue; }
                if (statuses[i]!=0) throw Bad("unknown result status");
                sat_header(i,empty);
                W bit=W(1)<<(i-first); valid|=bit;
                const uint8_t* answer=labels+W(i)*x.n;
                uint8_t invalid=0;
                for (U v=0; v<x.n; ++v) {
                    invalid |= uint8_t(answer[v]!=x.a[v] && answer[v]!=x.b[v]);
                    actual[v] |= W(answer[v]!=x.a[v])<<(i-first);
                }
                if (invalid) throw Bad("answer is outside original plans");
                request_check(i);
                if (minimal) for (U j=q.start(i); j<q.start(i+1); ++j) {
                    U v=q.vertex(j); if (moving[v]==i+1) required[v]|=bit;
                }
                ++result.sat;
            }
            if (!valid) continue;
            for (U v=0; v<x.n; ++v) {
                result.migrations += static_cast<W>(__builtin_popcountll(actual[v]));
                for (U j=x.starts[v]; j<x.starts[v+1]; ++j)
                    if (actual[v]&~actual[x.following[j]]) throw Bad("answer violates an original conflict");
            }
            if (!minimal) continue;
            // Least fixed point on ORIGINAL dependencies, including cycles.
            // A vertex is queued at most once concurrently; each query bit can
            // enter a vertex only once. No SCCs or closure table from the solver.
            std::fill(queued.begin(),queued.end(),0);
            U head=0, tail=0, active=0;
            auto enqueue=[&](U v) {
                if (queued[v]) return;
                if (active==x.n) throw std::logic_error("observer work queue overflow");
                queue[tail]=v; tail=(tail+1==x.n)?0:tail+1; ++active; queued[v]=1;
            };
            for (U v=0; v<x.n; ++v) if (required[v]) enqueue(v);
            while (active) {
                U v=queue[head]; head=(head+1==x.n)?0:head+1; --active; queued[v]=0;
                W delta=required[v];
                for (U j=x.starts[v]; j<x.starts[v+1]; ++j) {
                    ++result.dependency_visits;
                    U w=x.following[j]; W added=delta&~required[w];
                    if (added) { required[w]|=added; enqueue(w); }
                }
            }
            for (U v=0; v<x.n; ++v)
                if (actual[v]!=required[v]) throw Bad("answer is not the minimum-change closure");
        }
    }

    // Recompute the unique minimum from ORIGINAL implications, then compare
    // the whole submitted assignment directly. No answer-derived bit matrix.
    void direct(bool minimal) {
        if (!minimal) { tiled(false); return; }
        U count=q.count();
        std::vector<W> required(x.n,0);
        std::vector<uint8_t> queued(x.n,0), expected(x.n);
        std::vector<U> touched;touched.reserve(x.n);
        result.scratch += 8*W(required.capacity())+queued.capacity()+expected.capacity()+4*W(touched.capacity());
        for (U first=0; first<count; first+=64) {
            std::fill(required.begin(),required.end(),0);
            U end=std::min<U>(count,first+64);
            W valid=0;
            for (U i=first;i<end;++i) {
                if(statuses[i]==1) {pins(i);proof(i);continue;}
                if(statuses[i]!=0) throw Bad("unknown result status");
                sat_header(i,NIL);
                W bit=W(1)<<(i-first);valid|=bit;
                // Read each raw request once to seed the independently computed
                // closure. Proof-only endpoint stamps are not needed for SAT.
                for(U j=q.start(i);j<q.start(i+1);++j) {
                    U v=q.vertex(j);W mask=q.mask(j);
                    if(!(mask&(W(1)<<x.a[v]))) {
                        if(!(mask&(W(1)<<x.b[v])))throw Bad("SAT output carries an empty request");
                        required[v]|=bit;
                    }
                }
                ++result.sat;
            }
            if(!valid)continue;
            std::fill(queued.begin(),queued.end(),0);
            U head=0,tail=0,active=0;
            auto enqueue=[&](U v) {
                if(queued[v])return;
                if(active==x.n)throw std::logic_error("observer work queue overflow");
                queue[tail]=v;tail=(tail+1==x.n)?0:tail+1;++active;queued[v]=1;
            };
            for(U v=0;v<x.n;++v)if(required[v])enqueue(v);
            while(active) {
                U v=queue[head];head=(head+1==x.n)?0:head+1;--active;queued[v]=0;
                W bits=required[v];
                for(U j=x.starts[v];j<x.starts[v+1];++j) {
                    ++result.dependency_visits;
                    U w=x.following[j];W added=bits&~required[w];
                    if(added){required[w]|=added;enqueue(w);}
                }
            }
            touched.clear();
            W active_lanes=0;
            for(U v=0;v<x.n;++v)if(required[v]) {
                touched.push_back(v);active_lanes|=required[v];
                result.migrations+=static_cast<W>(__builtin_popcountll(required[v]));
            }
            for(U i=first;i<end;++i)if(valid&(W(1)<<(i-first))) {
                const uint8_t* answer=labels+W(i)*x.n;
                W bit=W(1)<<(i-first);
                const uint8_t* compare=x.a.data();
                if(active_lanes&bit) {
                    std::memcpy(expected.data(),x.a.data(),x.n);
                    for(U v:touched)if(required[v]&bit)expected[v]=x.b[v];
                    compare=expected.data();
                }
                if(std::memcmp(answer,compare,x.n))throw Bad("answer is not the original minimum-change closure");
                request_check(i);
            }
        }
    }


    void rows(bool minimal) {
        if(!minimal){scalar(false);return;}
        std::vector<uint8_t> expected(x.n);
        result.scratch+=expected.capacity();
        for(U i=0;i<q.count();++i) {
            if(statuses[i]==1){pins(i);proof(i);continue;}
            if(statuses[i]!=0)throw Bad("unknown result status");
            sat_header(i,NIL);
            std::memcpy(expected.data(),x.a.data(),x.n);
            U head=0,tail=0;
            for(U j=q.start(i);j<q.start(i+1);++j) {
                U v=q.vertex(j);
                W mask=q.mask(j);
                if(!(mask&(W(1)<<x.a[v]))) {
                    if(!(mask&(W(1)<<x.b[v])))throw Bad("SAT output carries an empty request");
                    if(expected[v]==x.a[v]){expected[v]=x.b[v];queue[tail++]=v;}
                }
            }
            while(head<tail) {
                U v=queue[head++];
                for(U j=x.starts[v];j<x.starts[v+1];++j) {
                    ++result.dependency_visits;U w=x.following[j];
                    if(expected[w]==x.a[w]) {
                        expected[w]=x.b[w];
                        if(tail>=x.n)throw std::logic_error("original queue overflow");
                        queue[tail++]=w;
                    }
                }
            }
            const uint8_t* answer=labels+W(i)*x.n;
            if(std::memcmp(answer,expected.data(),x.n))throw Bad("answer is not the original minimum-change closure");
            request_check(i);result.migrations+=tail;++result.sat;
        }
    }

};
constexpr const char* TAG="spectra.original.packet.observer.v1";
void destroy(PyObject* c) {
    void* p=PyCapsule_GetPointer(c,TAG);
    if (p) delete static_cast<Observer*>(p); else PyErr_Clear();
}
PyObject* failure() {
    try { throw; }
    catch (const Bad& e) { PyErr_SetString(PyExc_ValueError,e.what()); }
    catch (const Large& e) { PyErr_SetString(PyExc_MemoryError,e.what()); }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError,e.what()); }
    catch (...) { PyErr_SetString(PyExc_RuntimeError,"observer failed"); }
    return nullptr;
}
PyObject* create(PyObject*,PyObject* args) {
    PyObject *e,*a,*b,*cap;
    if (!PyArg_ParseTuple(args,"OOOO",&e,&a,&b,&cap)) return nullptr;
    try {
        auto x=std::make_unique<Observer>(e,a,b,integer(cap,"invalid observer cap"));
        auto c=PyCapsule_New(x.get(),TAG,destroy); if (c) x.release(); return c;
    } catch (...) { return failure(); }
}
PyObject* info(PyObject*,PyObject* c) {
    auto x=static_cast<Observer*>(PyCapsule_GetPointer(c,TAG)); if (!x) return nullptr;
    return Py_BuildValue("(IKKKK)",x->n,static_cast<unsigned long long>(x->original_edges),
        static_cast<unsigned long long>(x->following.size()),static_cast<unsigned long long>(x->payload),
        static_cast<unsigned long long>(x->build_bound));
}
PyObject* verify(PyObject*,PyObject* args) {
    PyObject *c,*qs,*ss,*ls,*ks,*os,*ps,*min,*mode,*cap;
    if (!PyArg_ParseTuple(args,"OOOOOOOOOO",&c,&qs,&ss,&ls,&ks,&os,&ps,&min,&mode,&cap)) return nullptr;
    auto x=static_cast<Observer*>(PyCapsule_GetPointer(c,TAG)); if (!x) return nullptr;
    try {
        if (!PyBool_Check(min)) throw Bad("minimal must be bool");
        U execution=static_cast<U>(integer(mode,"invalid checker mode",4));
        Bank q=read_queries(qs,x->n,integer(cap,"invalid scratch cap")); W count=q.count();
        if (execution==2) execution=count<64 ? 4:3;
        if (!PyBytes_CheckExact(ss)||!PyBytes_CheckExact(ls)||!PyBytes_CheckExact(ks)
            ||!PyBytes_CheckExact(ps)||!PyTuple_CheckExact(os)) throw Bad("invalid output types");
        W path_bytes=PyBytes_GET_SIZE(ps);
        if (W(PyBytes_GET_SIZE(ss))!=count||W(PyBytes_GET_SIZE(ks))!=count
            ||W(PyBytes_GET_SIZE(ls))!=count*x->n||W(PyTuple_GET_SIZE(os))!=count+1
            ||path_bytes%4||path_bytes/4>UINT32_MAX) throw Bad("output geometry differs");
        std::vector<U> offsets(count+1);
        for (W i=0;i<=count;++i) {
            offsets[i]=static_cast<U>(integer(PyTuple_GET_ITEM(os,i),"invalid proof offset",path_bytes/4));
            if (i&&offsets[i]<offsets[i-1]) throw Bad("nonmonotonic proof offsets");
        }
        if (offsets[0]!=0||W(offsets[count])*4!=path_bytes) throw Bad("unconsumed proof bytes");
        auto bytes=[](PyObject* p) {return reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(p));};
        Worker worker(*x,q,bytes(ss),bytes(ls),bytes(ks),offsets,bytes(ps),min==Py_True,execution==0);
        PyThreadState* thread=PyEval_SaveThread();
        try {
            if(execution==4)worker.rows(min==Py_True);
            else if(execution==3)worker.direct(min==Py_True);
            else if (execution) worker.tiled(min==Py_True); else worker.scalar(min==Py_True);
        } catch (...) {PyEval_RestoreThread(thread);throw;}
        PyEval_RestoreThread(thread);
        Counts r=worker.result;
        r.scratch+=4*W(offsets.capacity()+q.starts.capacity()+q.vertices.capacity())+8*W(q.masks.capacity());
        if (r.scratch>q.admission) throw std::logic_error("observer scratch admission underestimated");
        return Py_BuildValue("(KKKKIKKK)",static_cast<unsigned long long>(r.sat),
            static_cast<unsigned long long>(r.unsat),static_cast<unsigned long long>(r.migrations),
            static_cast<unsigned long long>(r.proof_edges),execution,static_cast<unsigned long long>(r.dependency_visits),
            static_cast<unsigned long long>(r.scratch),static_cast<unsigned long long>(q.admission));
    } catch (...) { return failure(); }
}
// Freeze only request data, not solver decisions. Every solver and observer still
// validates its own native input bank. Fully immutable exact tuples are reused.
struct PythonError {};
struct Decref { void operator()(PyObject* p) const {Py_XDECREF(p);} };
using Ref = std::unique_ptr<PyObject,Decref>;
Ref bounded_tuple(PyObject* source, W limit, const char* message) {
    if (PyTuple_CheckExact(source)) {
        if (W(PyTuple_GET_SIZE(source))>limit) throw Large(message);
        return Ref(Py_NewRef(source));
    }
    Ref iterator(PyObject_GetIter(source)); if (!iterator) throw PythonError{};
    Ref items(PyList_New(0)); if (!items) throw PythonError{};
    W size=0;
    while (true) {
        Ref item(PyIter_Next(iterator.get()));
        if (!item) {if (PyErr_Occurred()) throw PythonError{}; break;}
        if (size++==limit) throw Large(message);
        if (PyList_Append(items.get(),item.get())<0) throw PythonError{};
    }
    Ref result(PyList_AsTuple(items.get())); if (!result) throw PythonError{};
    return result;
}
void pin_shape(PyObject* pin) {
    if (!PyTuple_CheckExact(pin)||PyTuple_GET_SIZE(pin)!=2
        ||!PyLong_CheckExact(PyTuple_GET_ITEM(pin,0))||!PyLong_CheckExact(PyTuple_GET_ITEM(pin,1)))
        throw Bad("pin must contain two exact integers");
}
PyObject* freeze_queries(PyObject*,PyObject* args) {
    PyObject *source,*qcap,*pcap;
    if (!PyArg_ParseTuple(args,"OOO",&source,&qcap,&pcap)) return nullptr;
    try {
        W maxq=integer(qcap,"invalid query count cap",1000000),maxp=integer(pcap,"invalid pin count cap",16000000);
        if (PyTuple_CheckExact(source)) {
            W pins=0,count=PyTuple_GET_SIZE(source); bool exact=true;
            if (count>maxq) throw Large("too many queries");
            for (W i=0;i<count&&exact;++i) {
                auto row=PyTuple_GET_ITEM(source,i);
                if (!PyTuple_CheckExact(row)) {exact=false;break;}
                pins+=PyTuple_GET_SIZE(row);if(pins>maxp)throw Large("too many query pins");
                for (Py_ssize_t j=0;j<PyTuple_GET_SIZE(row);++j) {
                    auto pin=PyTuple_GET_ITEM(row,j);
                    if (!PyTuple_CheckExact(pin)) {exact=false;break;}
                    pin_shape(pin);
                }
            }
            if (exact) return Py_NewRef(source);
        }
        Ref outer=bounded_tuple(source,maxq,"too many queries");
        W count=PyTuple_GET_SIZE(outer.get()),pins=0;
        Ref output(PyTuple_New(count)); if (!output) throw PythonError{};
        for (W i=0;i<count;++i) {
            Ref row=bounded_tuple(PyTuple_GET_ITEM(outer.get(),i),maxp-pins,"too many query pins");
            W width=PyTuple_GET_SIZE(row.get());pins+=width;
            Ref frozen(PyTuple_New(width));if(!frozen)throw PythonError{};
            for (W j=0;j<width;++j) {
                Ref pin=bounded_tuple(PyTuple_GET_ITEM(row.get(),j),2,"pin has more than two fields");
                pin_shape(pin.get());PyTuple_SET_ITEM(frozen.get(),j,pin.release());
            }
            PyTuple_SET_ITEM(output.get(),i,frozen.release());
        }
        return output.release();
    } catch (const PythonError&) {return nullptr;} catch (...) {return failure();}
}

PyMethodDef methods[]={{"freeze_queries",freeze_queries,METH_VARARGS,"Freeze request iterables with bounded counts; reuse immutable exact tuples."},{"create",create,METH_VARARGS,"Validate and own original mixed-plan constraints."},
    {"verify",verify,METH_VARARGS,"Check original constraints, proofs and minimum necessary change."},
    {"info",info,METH_O,"Report original observer array payload."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_packet_observer",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_packet_observer() {return PyModule_Create(&module);}
