// Exact binary-list support with original-literal contradiction paths.
// MIT: repository LICENSE. Established 2-SAT/SCC/implication closure, not novelty.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U NIL = UINT32_MAX, MAX_N = 100000, MAX_E = 2000000, MAX_P = 1000000;
constexpr const char* SOLVER = "spectra.exact_binary.solver.v3";
constexpr const char* OBSERVER = "spectra.exact_binary.observer.v3";
struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };
W number(PyObject* value, const char* message, W cap = UINT64_MAX) {
    if (!PyLong_CheckExact(value)) throw Invalid(message);
    W n = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) { PyErr_Clear(); throw Invalid(message); }
    if (n > cap) throw Invalid(message);
    return n;
}
void tuple(PyObject* value, const char* message) {
    if (!PyTuple_CheckExact(value)) throw Invalid(message);
}
W palette(U k) { return k == 64 ? UINT64_MAX : ((W(1) << k) - 1); }
#if defined(_MSC_VER)
U first(W v) { unsigned long index=0;_BitScanForward64(&index,v);return static_cast<U>(index); }
// Do not require POPCNT on the Windows portable build.
U bits(W v) { U count=0;while(v) {v&=v-1;++count;}return count; }
#else
U first(W v) { return static_cast<U>(__builtin_ctzll(v)); }
U bits(W v) { return static_cast<U>(__builtin_popcountll(v)); }
#endif
struct Pin { U vertex; W allowed; };
std::vector<Pin> read_query(PyObject* q, U n, W cap) {
    tuple(q, "queries require exact tuples");
    W size = static_cast<W>(PyTuple_GET_SIZE(q));
    if (size > MAX_P || 32*size + 64*W(n+1) + 4096 > cap)
        throw Resource("query scratch/pin admission exceeds cap");
    std::vector<Pin> pins; pins.reserve(size);
    for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(q); ++i) {
        PyObject* row=PyTuple_GET_ITEM(q,i); tuple(row,"pin must be tuple");
        if (PyTuple_GET_SIZE(row)!=2) throw Invalid("pin must be a pair");
        W v=number(PyTuple_GET_ITEM(row,0),"invalid pin vertex");
        W allowed=number(PyTuple_GET_ITEM(row,1),"invalid uint64 mask");
        if (v>=n) throw Invalid("pin vertex outside graph");
        pins.push_back({static_cast<U>(v),allowed});
    }
    return pins; // every row is checked, including after an earlier contradiction
}
struct Input {
    U n,k; W cap;
    std::vector<W> masks;
    std::vector<std::pair<U,U>> edges;
    Input(U nv,U kv,std::vector<std::pair<U,U>> es,std::vector<W> ms,W limit)
        :n(nv),k(kv),cap(limit),masks(std::move(ms)),edges(std::move(es)) {}
    Input(PyObject* nv, PyObject* kv, PyObject* es, PyObject* ms, W limit):cap(limit) {
        n=static_cast<U>(number(nv,"n outside supported bound",MAX_N));
        k=static_cast<U>(number(kv,"palette outside [1,64]",64));
        if (!k) throw Invalid("palette outside [1,64]");
        tuple(es,"edges require exact tuples"); tuple(ms,"masks require exact tuples");
        if (PyTuple_GET_SIZE(ms)!=n) throw Invalid("mask count differs");
        W m=static_cast<W>(PyTuple_GET_SIZE(es));
        if (m>MAX_E || 768*W(n+1)+384*(m+1)+8192>cap)
            throw Resource("conservative construction payload admission exceeds cap");
        masks.reserve(n);
        for (U v=0;v<n;++v) {
            W d=number(PyTuple_GET_ITEM(ms,v),"invalid original mask");
            if (!d || (d & ~palette(k)) || bits(d)>2)
                throw Invalid("original domains require one or two palette colours");
            masks.push_back(d);
        }
        edges.reserve(m);
        for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(es);++i) {
            PyObject* row=PyTuple_GET_ITEM(es,i); tuple(row,"edge must be tuple");
            if (PyTuple_GET_SIZE(row)!=2) throw Invalid("edge must be pair");
            W u=number(PyTuple_GET_ITEM(row,0),"invalid edge endpoint");
            W v=number(PyTuple_GET_ITEM(row,1),"invalid edge endpoint");
            if (u>=n || v>=n || u==v) throw Invalid("edge needs distinct in-range vertices");
            edges.emplace_back(std::min(u,v),std::max(u,v));
        }
        std::sort(edges.begin(),edges.end());
        edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    }
};
struct CSR {
    std::vector<U> off, to;
    CSR()=default;
    CSR(U n, const std::vector<std::pair<U,U>>& arcs, bool reverse=false):off(n+1,0),to(arcs.size()) {
        for(auto e:arcs) ++off[(reverse?e.second:e.first)+1];
        for(U v=0;v<n;++v) off[v+1]+=off[v];
        auto cursor=off;
        for(auto e:arcs) to[cursor[reverse?e.second:e.first]++]=reverse?e.first:e.second;
    }
    W bytes() const {return 4*W(off.capacity()+to.capacity());}
};
std::vector<U> components(const CSR& g,const CSR& r) {
    U n=static_cast<U>(g.off.size()-1);
    std::vector<uint8_t> seen(n,0);
    std::vector<U> order; order.reserve(n);
    std::vector<std::pair<U,U>> stack; stack.reserve(n);
    for(U root=0;root<n;++root) if(!seen[root]) {
        seen[root]=1;stack.emplace_back(root,g.off[root]);
        while(!stack.empty()) {
            auto& top=stack.back();
            if(top.second==g.off[top.first+1]) {order.push_back(top.first);stack.pop_back();continue;}
            U u=g.to[top.second++];
            if(!seen[u]) {seen[u]=1;stack.emplace_back(u,g.off[u]);}
        }
    }
    std::vector<U> comp(n,NIL), todo; todo.reserve(n); U id=0;
    for(auto it=order.rbegin();it!=order.rend();++it) if(comp[*it]==NIL) {
        comp[*it]=id;todo.push_back(*it);
        while(!todo.empty()) {U v=todo.back();todo.pop_back();
            for(U j=r.off[v];j<r.off[v+1];++j) if(comp[r.to[j]]==NIL) {
                comp[r.to[j]]=id;todo.push_back(r.to[j]);
            }
        }
        ++id;
    }
    return comp; // topological component numbering for the original implication DAG
}
void word(std::vector<uint8_t>& out,U x) {for(U j=0;j<4;++j) out.push_back(static_cast<uint8_t>(x>>(8*j)));}
void qword(std::vector<uint8_t>& out,W x) {for(U j=0;j<8;++j) out.push_back(static_cast<uint8_t>(x>>(8*j)));}
U load32(const uint8_t* data,size_t at) {U x=0;for(U j=0;j<4;++j)x|=U(data[at+j])<<(8*j);return x;}
W load64(const uint8_t* data,size_t at) {W x=0;for(U j=0;j<8;++j)x|=W(data[at+j])<<(8*j);return x;}
std::vector<uint8_t> proof(U kind,const std::vector<U>& a,const std::vector<U>& b) {
    std::vector<uint8_t> out;out.reserve(16+4*(a.size()+b.size()));
    out.insert(out.end(),{'S','B','P','1'}); word(out,kind);word(out,static_cast<U>(a.size()));word(out,static_cast<U>(b.size()));
    for(U v:a) word(out,v);
    for(U v:b) word(out,v);
    return out;
}
std::vector<U> trace(const std::vector<U>& parent,U v) {
    std::vector<U> path;
    while(true) {path.push_back(v);if(parent[v]==v) break;
        if(parent[v]==NIL || path.size()>parent.size()) throw std::logic_error("invalid parent chain");
        v=parent[v];
    }
    std::reverse(path.begin(),path.end());return path;
}

// SBQ1 solver-side parser. It owns no Python objects and borrows immutable bytes
// only for one native call. The observer has a separately written parser below,
// so a shared offset bug cannot silently validate its own output.
struct SolverBank {
    const uint8_t* data;size_t size;U count,pins;size_t offsets_at=16,vertices_at,masks_at;
    SolverBank(const uint8_t* bytes,size_t length,U n):data(bytes),size(length),count(0),pins(0),vertices_at(0),masks_at(0) {
        if(size<20 || std::memcmp(data,"SBQ1",4)) throw Invalid("invalid SBQ1 solver packet");
        count=load32(data,4);pins=load32(data,8);
        if(load32(data,12)!=0 || count>MAX_P || pins>MAX_P) throw Invalid("invalid SBQ1 solver header");
        W expected=20+4*W(count)+12*W(pins);
        if(expected!=size) throw Invalid("SBQ1 solver length differs");
        vertices_at=20+4*W(count);masks_at=vertices_at+4*W(pins);
        U previous=load32(data,offsets_at);
        if(previous!=0) throw Invalid("SBQ1 solver offsets must start at zero");
        for(U i=1;i<=count;++i) {
            U next=load32(data,offsets_at+4*W(i));
            if(next<previous || next>pins) throw Invalid("SBQ1 solver offsets differ");
            previous=next;
        }
        if(previous!=pins) throw Invalid("SBQ1 solver offsets do not consume pins");
        for(U i=0;i<pins;++i) if(load32(data,vertices_at+4*W(i))>=n)
            throw Invalid("SBQ1 solver vertex outside graph");
    }
    U begin(U row) const {return load32(data,offsets_at+4*W(row));}
    U end(U row) const {return load32(data,offsets_at+4*W(row+1));}
    U query_count() const {return count;}
    U pin_count() const {return pins;}
    U row_begin(U index) const {return begin(index);}
    U row_end(U index) const {return end(index);}
    U pin_vertex(U index) const {return load32(data,vertices_at+4*W(index));}
    W pin_allowed(U index) const {return load64(data,masks_at+8*W(index));}
    std::vector<Pin> row(U index) const {
        U first_pin=begin(index),last_pin=end(index);std::vector<Pin> result;result.reserve(last_pin-first_pin);
        for(U i=first_pin;i<last_pin;++i)
            result.push_back({load32(data,vertices_at+4*W(i)),load64(data,masks_at+8*W(i))});
        return result;
    }
};

struct ObserverBank {
    const uint8_t* payload;size_t bytes;U rows,total;size_t table=16,vertices,masks;
    ObserverBank(const uint8_t* source,size_t length,U n):payload(source),bytes(length),rows(0),total(0),vertices(0),masks(0) {
        if(bytes<QUERY_MIN || payload[0]!='S' || payload[1]!='B' || payload[2]!='Q' || payload[3]!='1')
            throw Invalid("invalid SBQ1 observer packet");
        rows=load32(payload,4);total=load32(payload,8);
        if(load32(payload,12)!=0 || rows>MAX_P || total>MAX_P)
            throw Invalid("invalid SBQ1 observer header");
        W geometry=16+4*W(rows+1)+12*W(total);
        if(geometry!=bytes) throw Invalid("SBQ1 observer geometry differs");
        vertices=16+4*W(rows+1);masks=vertices+4*W(total);
        U at=0;
        for(U i=0;i<=rows;++i) {
            U next=load32(payload,table+4*W(i));
            if((i==0 && next!=0) || next<at || next>total)
                throw Invalid("SBQ1 observer offsets differ");
            at=next;
        }
        if(at!=total) throw Invalid("SBQ1 observer pin bank is not consumed");
        for(U i=0;i<total;++i) if(load32(payload,vertices+4*W(i))>=n)
            throw Invalid("SBQ1 observer vertex outside graph");
    }
    std::vector<Pin> row(U index) const {
        U begin=load32(payload,table+4*W(index));
        U finish=load32(payload,table+4*W(index+1));
        std::vector<Pin> result;result.reserve(finish-begin);
        for(U i=begin;i<finish;++i)
            result.push_back({load32(payload,vertices+4*W(i)),load64(payload,masks+8*W(i))});
        return result;
    }
    U query_count() const {return rows;}
    U pin_count() const {return total;}
private:
    static constexpr size_t QUERY_MIN=20;
};

// SBC1 contains the complete fixed problem and one immutable request bank. The
// solver and observer intentionally use separate parsers and separately owned
// graph/domain arrays. A ready packet is a deployment input format, not a claim
// that Python/JSON preparation disappeared.
struct SolverSessionPacket {
    const uint8_t* data;size_t size;U n,k,edge_count,count,pins;
    size_t masks_at=32,edges_at,offsets_at,vertices_at,allowed_at;
    std::vector<W> original_masks;
    std::vector<std::pair<U,U>> original_edges;
    SolverSessionPacket(const uint8_t* bytes,size_t length,W cap):data(bytes),size(length),
        n(0),k(0),edge_count(0),count(0),pins(0),edges_at(0),offsets_at(0),vertices_at(0),allowed_at(0) {
        if(size<36 || std::memcmp(data,"SBC1",4)) throw Invalid("invalid SBC1 solver packet");
        n=load32(data,4);k=load32(data,8);edge_count=load32(data,12);
        count=load32(data,16);pins=load32(data,20);
        if(n>MAX_N || !k || k>64 || edge_count>MAX_E || count>MAX_P || pins>MAX_P
                || load32(data,24)!=0 || load32(data,28)!=0)
            throw Invalid("invalid SBC1 solver header");
        W expected=36+8*W(n)+8*W(edge_count)+4*W(count)+12*W(pins);
        if(expected!=size) throw Invalid("SBC1 solver length differs");
        W construction=768*W(n+1)+384*W(edge_count+1)+8192;
        if(construction>cap || W(size)>cap-construction)
            throw Resource("SBC1 solver construction admission exceeds cap");
        edges_at=masks_at+8*W(n);offsets_at=edges_at+8*W(edge_count);
        vertices_at=offsets_at+4*W(count+1);allowed_at=vertices_at+4*W(pins);
        original_masks.reserve(n);
        for(U v=0;v<n;++v) {W domain=load64(data,masks_at+8*W(v));
            if(!domain || (domain&~palette(k)) || bits(domain)>2)
                throw Invalid("SBC1 solver original domain differs");
            original_masks.push_back(domain);
        }
        original_edges.reserve(edge_count);U previous_left=0,previous_right=0;bool have_previous=false;
        for(U i=0;i<edge_count;++i) {U left=load32(data,edges_at+8*W(i));U right=load32(data,edges_at+8*W(i)+4);
            if(left>=right || right>=n || (have_previous && (left<previous_left || (left==previous_left && right<=previous_right))))
                throw Invalid("SBC1 solver edges must be sorted unique pairs");
            original_edges.emplace_back(left,right);previous_left=left;previous_right=right;have_previous=true;
        }
        U previous=load32(data,offsets_at);
        if(previous!=0) throw Invalid("SBC1 solver offsets must start at zero");
        for(U i=1;i<=count;++i) {
            U next=load32(data,offsets_at+4*W(i));
            if(next<previous || next>pins) throw Invalid("SBC1 solver offsets differ");
            previous=next;
        }
        if(previous!=pins) throw Invalid("SBC1 solver offsets do not consume pins");
        for(U i=0;i<pins;++i) if(load32(data,vertices_at+4*W(i))>=n)
            throw Invalid("SBC1 solver query vertex outside graph");
    }
    U query_count() const {return count;}
    U pin_count() const {return pins;}
    U row_begin(U index) const {return load32(data,offsets_at+4*W(index));}
    U row_end(U index) const {return load32(data,offsets_at+4*W(index+1));}
    U pin_vertex(U index) const {return load32(data,vertices_at+4*W(index));}
    W pin_allowed(U index) const {return load64(data,allowed_at+8*W(index));}
    std::vector<Pin> row(U index) const {
        U begin=row_begin(index),end=row_end(index);
        std::vector<Pin> result;result.reserve(end-begin);
        for(U i=begin;i<end;++i)
            result.push_back({load32(data,vertices_at+4*W(i)),load64(data,allowed_at+8*W(i))});
        return result;
    }
    Input take_input(W cap) {return Input(n,k,std::move(original_edges),std::move(original_masks),cap);}
};

struct ObserverSessionPacket {
    const uint8_t* payload;size_t bytes;U vertices_count,palette_size,edges_count,rows,total;
    size_t domain_table=32,edge_table,offset_table,vertex_table,mask_table;
    std::vector<W> domains;
    std::vector<std::pair<U,U>> edges;
    ObserverSessionPacket(const uint8_t* source,size_t length,W cap):payload(source),bytes(length),
        vertices_count(0),palette_size(0),edges_count(0),rows(0),total(0),edge_table(0),offset_table(0),vertex_table(0),mask_table(0) {
        if(bytes<36 || payload[0]!='S' || payload[1]!='B' || payload[2]!='C' || payload[3]!='1')
            throw Invalid("invalid SBC1 observer packet");
        vertices_count=load32(payload,4);palette_size=load32(payload,8);edges_count=load32(payload,12);
        rows=load32(payload,16);total=load32(payload,20);
        if(vertices_count>MAX_N || palette_size<1 || palette_size>64 || edges_count>MAX_E
                || rows>MAX_P || total>MAX_P || load32(payload,24) || load32(payload,28))
            throw Invalid("invalid SBC1 observer header");
        W geometry=32+8*W(vertices_count)+8*W(edges_count)+4*W(rows+1)+12*W(total);
        if(geometry!=bytes) throw Invalid("SBC1 observer geometry differs");
        W construction=768*W(vertices_count+1)+384*W(edges_count+1)+8192;
        if(construction>cap || W(bytes)>cap-construction)
            throw Resource("SBC1 observer construction admission exceeds cap");
        edge_table=domain_table+8*W(vertices_count);offset_table=edge_table+8*W(edges_count);
        vertex_table=offset_table+4*W(rows+1);mask_table=vertex_table+4*W(total);
        domains.reserve(vertices_count);
        for(U v=0;v<vertices_count;++v) {W domain=load64(payload,domain_table+8*W(v));
            if(domain==0 || (domain&~palette(palette_size))!=0 || bits(domain)>2)
                throw Invalid("SBC1 observer original domain differs");
            domains.push_back(domain);
        }
        edges.reserve(edges_count);U last_left=0,last_right=0;bool have=false;
        for(U i=0;i<edges_count;++i) {U left=load32(payload,edge_table+8*W(i));U right=load32(payload,edge_table+8*W(i)+4);
            bool ordered=!have || left>last_left || (left==last_left && right>last_right);
            if(left>=right || right>=vertices_count || !ordered)
                throw Invalid("SBC1 observer edge bank differs");
            edges.emplace_back(left,right);last_left=left;last_right=right;have=true;
        }
        U consumed=0;
        for(U i=0;i<=rows;++i) {
            U next=load32(payload,offset_table+4*W(i));
            if((i==0 && next!=0) || next<consumed || next>total)
                throw Invalid("SBC1 observer offsets differ");
            consumed=next;
        }
        if(consumed!=total) throw Invalid("SBC1 observer pin bank is not consumed");
        for(U i=0;i<total;++i) if(load32(payload,vertex_table+4*W(i))>=vertices_count)
            throw Invalid("SBC1 observer query vertex outside graph");
    }
    U query_count() const {return rows;}
    U pin_count() const {return total;}
    std::vector<Pin> row(U index) const {
        U begin=load32(payload,offset_table+4*W(index)),end=load32(payload,offset_table+4*W(index+1));
        std::vector<Pin> result;result.reserve(end-begin);
        for(U i=begin;i<end;++i)
            result.push_back({load32(payload,vertex_table+4*W(i)),load64(payload,mask_table+8*W(i))});
        return result;
    }
    Input take_input(W cap) {return Input(vertices_count,palette_size,std::move(edges),std::move(domains),cap);}
};
struct Answer {bool sat=true;std::vector<uint8_t> labels,certificate;W visits=0,arcs=0;};
struct Index {
    U n,k,nc=0,bad=NIL;bool condensed,sparse,routed;W cap;W full_arcs=0;
    const char* auto_reason="explicit_mode";
    std::vector<W> masks;
    std::vector<uint8_t> lo,hi,base;
    std::vector<U> comp,opposite;
    std::vector<U> inward,outward,representative,cross_from,cross_to;
    CSR original,dag;
    Index(Input p,U mode,U bank_size=0,U bank_pins=0):n(p.n),k(p.k),condensed(mode!=0),sparse(mode==2),routed(mode==3),cap(p.cap),masks(std::move(p.masks)),lo(n),hi(n,255),base(n) {
        std::vector<std::pair<U,U>> arcs;arcs.reserve(4*p.edges.size()+n);
        for(U v=0;v<n;++v) {lo[v]=first(masks[v]);W h=masks[v] & (masks[v]-1);
            if(h) hi[v]=first(h); else arcs.emplace_back(2*v+1,2*v);
        }
        for(auto e:p.edges) {W shared=masks[e.first]&masks[e.second];
            while(shared) {U c=first(shared);shared&=shared-1;
                U a=2*e.first+(c!=lo[e.first]), b=2*e.second+(c!=lo[e.second]);
                arcs.emplace_back(a,b^1U);arcs.emplace_back(b,a^1U);
            }
        }
        // Input owns sorted unique (u,v), u<v edges. In that edge stream each
        // vertex sees its neighbours in increasing order, and a source/target
        // literal contributes at most one arc per neighbour. Singleton unit
        // arcs have otherwise empty source/target rows. Stable CSR scattering
        // therefore produces exactly the same two ordered adjacencies as a
        // global arc sort, with no duplicates. See LINEAR_CONSTRUCTION.md.
        full_arcs=arcs.size();original=CSR(2*n,arcs);CSR reverse(2*n,arcs,true);
#if defined(SPECTRA_EXACT_VALIDATE_LINEAR_CSR)
        // Diagnostic builds only: compare actual native arrays, not a model of
        // them, with the previous sorted construction on every admitted graph.
        auto reference_arcs=arcs;
        std::sort(reference_arcs.begin(),reference_arcs.end());
        reference_arcs.erase(std::unique(reference_arcs.begin(),reference_arcs.end()),reference_arcs.end());
        CSR reference_forward(2*n,reference_arcs),reference_reverse(2*n,reference_arcs,true);
        if(reference_arcs.size()!=arcs.size() || original.off!=reference_forward.off
                || original.to!=reference_forward.to || reverse.off!=reference_reverse.off
                || reverse.to!=reference_reverse.to)
            throw std::logic_error("linear implication construction differs from sorted reference");
#endif
        comp=components(original,reverse);
        for(U c:comp) nc=std::max(nc,c+1);
        for(U v=0;v<n;++v) {if(comp[2*v]==comp[2*v+1] && bad==NIL) bad=v;
            base[v]=static_cast<uint8_t>(comp[2*v]<comp[2*v+1]);
        }
        if(mode==4) {
            // Structural auto policy v2.  set9 development evidence showed that
            // large, heavily pinned repair banks can lose to raw traversal even
            // when tile64 is available.  The policy therefore uses only immutable
            // request geometry (never answers or measured timings): raw below one
            // tile, on a base contradiction, or at eight pins per row and above.
            // The threshold is frozen before the set10/set12 holdout transfer.
            if(bank_size<64) {
                condensed=false;routed=false;auto_reason="bank_below_tile64";
            } else if(bad!=NIL) {
                condensed=false;routed=false;auto_reason="base_infeasible";
            } else if(W(bank_pins)>=8*W(bank_size)) {
                condensed=false;routed=false;auto_reason="high_pin_density";
            } else {
                condensed=true;routed=nc<=n;
                auto_reason=routed?"route_component_compression":"scc_tile64";
            }
        }
        if(condensed && bad==NIL) {
            opposite.assign(nc,NIL);
            for(U v=0;v<2*n;++v) {
                U a=comp[v],b=comp[v^1U];
                if(a==b || (opposite[a]!=NIL && opposite[a]!=b)) throw std::logic_error("complement mapping differs");
                opposite[a]=b;
            }
            // Routed mode builds the same ordered DAG from its original-edge
            // bridges below. Sorting both the links and bridges repeats work.
            if(!routed) {
                std::vector<std::pair<U,U>> links;links.reserve(arcs.size());
                for(auto e:arcs) if(comp[e.first]!=comp[e.second]) links.emplace_back(comp[e.first],comp[e.second]);
                std::sort(links.begin(),links.end());links.erase(std::unique(links.begin(),links.end()),links.end());
                dag=CSR(nc,links);
            }
        } else if(!condensed) {std::vector<U>().swap(comp);}
        if(sparse || routed) {
            if(routed) {
                inward.assign(2*n,NIL);outward.assign(2*n,NIL);
                representative.assign(nc,NIL);
                for(U v=0;v<2*n;++v) if(representative[comp[v]]==NIL) representative[comp[v]]=v;
            }
            // Reachability-preserving original-edge subgraph: inward/outward
            // trees in each SCC plus one original arc per condensation edge.
            // Full original graph remains separately owned by the verifier.
            std::vector<std::pair<U,U>> keep;
            if(sparse) keep.reserve(std::min<W>(full_arcs,4*W(n)+dag.to.size()));
            std::vector<uint8_t> seen(2*n,0);
            std::vector<U> queue;queue.reserve(2*n);
            for(U direction=0;direction<2;++direction) {
                std::fill(seen.begin(),seen.end(),0);
                const CSR& g=direction?reverse:original;
                for(U root=0;root<2*n;++root) if(!seen[root]) {
                    queue.clear();queue.push_back(root);seen[root]=1;
                    if(routed) (direction?inward:outward)[root]=root;
                    for(size_t head=0;head<queue.size();++head) {U v=queue[head];
                        for(U j=g.off[v];j<g.off[v+1];++j) {U u=g.to[j];
                            if(comp[u]==comp[root] && !seen[u]) {seen[u]=1;queue.push_back(u);
                                if(sparse) keep.emplace_back(direction?u:v,direction?v:u);
                                if(routed) (direction?inward:outward)[u]=v;
                            }
                        }
                    }
                }
            }
            std::vector<std::tuple<U,U,U,U>> bridges;bridges.reserve(arcs.size());
            for(auto e:arcs) if(comp[e.first]!=comp[e.second])
                bridges.emplace_back(comp[e.first],comp[e.second],e.first,e.second);
            std::sort(bridges.begin(),bridges.end());
            std::vector<std::pair<U,U>> routed_links;
            if(routed && bad==NIL) routed_links.reserve(bridges.size());
            U previous_a=NIL,previous_b=NIL;
            for(auto e:bridges) {U a=std::get<0>(e),b=std::get<1>(e);
                if(a!=previous_a || b!=previous_b) {
                    if(sparse) keep.emplace_back(std::get<2>(e),std::get<3>(e));
                    if(routed && bad==NIL) {
                        routed_links.emplace_back(a,b);
                        cross_from.push_back(std::get<2>(e));cross_to.push_back(std::get<3>(e));
                    }
                    previous_a=a;previous_b=b;
                }
            }
            std::sort(keep.begin(),keep.end());keep.erase(std::unique(keep.begin(),keep.end()),keep.end());
            if(keep.size()>arcs.size()) throw std::logic_error("proof subgraph grew beyond original");
            if(routed) {
                if(bad==NIL) dag=CSR(nc,routed_links);
                if(cross_from.size()!=dag.to.size()) throw std::logic_error("routing bridge geometry differs");
                for(U c=0;c<nc && bad==NIL;++c) for(U j=dag.off[c];j<dag.off[c+1];++j)
                    if(comp[cross_from[j]]!=c || comp[cross_to[j]]!=dag.to[j])
                        throw std::logic_error("routing bridge identity differs");
                original=CSR(); // no original adjacency or BFS is needed for proof reconstruction
            } else original=CSR(2*n,keep); // release dense original solver adjacency
        }
    }
    W bytes() const {return 8*W(masks.capacity())+lo.capacity()+hi.capacity()+base.capacity()+
        4*W(comp.capacity()+opposite.capacity())+original.bytes()+dag.bytes()+routing_bytes();}
    W routing_bytes() const {return 4*W(inward.capacity()+outward.capacity()+representative.capacity()+cross_from.capacity()+cross_to.capacity());}
    std::vector<U> within(U from,U target,Answer& answer) const {
        if(from==target) return {from};
        if(comp[from]!=comp[target]) throw std::logic_error("intra-component routing crosses SCCs");
        std::vector<U> tail;
        for(U v=target;;v=outward[v]) {
            tail.push_back(v);
            if(outward[v]==v) break;
            if(tail.size()>2*W(n) || outward[v]==NIL) throw std::logic_error("invalid outward routing tree");
            ++answer.arcs;
        }
        // Join at the first inward vertex on the outward root-to-target path.
        // This removes tree-overlap cycles without a graph search or n-sized
        // scratch. Sorted membership avoids a hash-table worst-case assumption.
        auto membership=tail;std::sort(membership.begin(),membership.end());
        std::vector<U> result;
        U meeting=NIL;
        for(U v=from;;v=inward[v]) {
            result.push_back(v);
            if(std::binary_search(membership.begin(),membership.end(),v)) {meeting=v;break;}
            if(result.size()>2*W(n) || inward[v]==NIL || inward[v]==v)
                throw std::logic_error("routing trees do not meet");
            ++answer.arcs;
        }
        size_t at=static_cast<size_t>(std::find(tail.begin(),tail.end(),meeting)-tail.begin());
        while(at) result.push_back(tail[--at]);
        return result;
    }
    std::vector<U> lift(U target,const std::vector<U>& parent,const std::vector<U>& edge,
                        const std::vector<U>& seed,Answer& answer) const {
        U c=comp[target];std::vector<U> chain;
        while(parent[c]!=c) {
            if(parent[c]==NIL || edge[c]==NIL || chain.size()>=nc)
                throw std::logic_error("routing predecessor chain missing");
            chain.push_back(edge[c]);c=parent[c];
        }
        U current=seed[c];if(current==NIL) throw std::logic_error("current request root missing");
        std::vector<U> result{current};
        for(auto it=chain.rbegin();it!=chain.rend();++it) {
            auto segment=within(current,cross_from[*it],answer);
            result.insert(result.end(),segment.begin()+1,segment.end());
            current=cross_to[*it];result.push_back(current);++answer.arcs;
        }
        auto segment=within(current,target,answer);
        result.insert(result.end(),segment.begin()+1,segment.end());
        if(result.size()>2*W(n)) throw std::logic_error("lifted proof exceeds simple-path bound");
        return result;
    }
    std::vector<U> path(U from,U target,Answer& a) const {
        if(routed) return within(from,target,a);
        std::vector<U> parent(2*n,NIL),queue;queue.reserve(2*n);parent[from]=from;queue.push_back(from);
        for(size_t head=0;head<queue.size() && parent[target]==NIL;++head) {U v=queue[head];++a.visits;
            for(U j=original.off[v];j<original.off[v+1];++j) {++a.arcs;U u=original.to[j];
                if(parent[u]==NIL) {parent[u]=v;queue.push_back(u);}
            }
        }
        if(parent[target]==NIL) throw std::logic_error("base contradiction path missing");
        return trace(parent,target);
    }
    bool raw_closure(const std::vector<U>& seeds,std::vector<U>& parent,Answer& a) const {
        if(routed) throw std::logic_error("routed proofs must not invoke original-graph search");
        parent.assign(2*n,NIL);std::vector<U> queue;queue.reserve(2*n);U conflict=NIL;
        for(U v:seeds) if(parent[v]==NIL) {parent[v]=v;queue.push_back(v);
            if(parent[v^1U]!=NIL) {conflict=v;break;}}
        for(size_t head=0;conflict==NIL && head<queue.size();++head) {U v=queue[head];++a.visits;
            for(U j=original.off[v];j<original.off[v+1];++j) {++a.arcs;U u=original.to[j];
                if(parent[u]==NIL) {parent[u]=v;queue.push_back(u);
                    if(parent[u^1U]!=NIL) {conflict=u;break;}}
            }
        }
        if(conflict==NIL) return true;
        a.sat=false;a.certificate=proof(2,trace(parent,conflict),trace(parent,conflict^1U));return false;
    }
    Answer solve(const std::vector<Pin>& pins,
                 const std::vector<uint8_t>* known_base_proof=nullptr) const {
        Answer answer;
        // Give every exact mode the same sparse request preparation, including
        // the cheap raw control. All pins were validated before this function.
        auto restrictions=pins;
        std::sort(restrictions.begin(),restrictions.end(),[](Pin a,Pin b){return a.vertex<b.vertex;});
        std::vector<U> seeds;seeds.reserve(std::min<W>(pins.size(),n));
        for(size_t i=0;i<restrictions.size();) {
            U v=restrictions[i].vertex;W domain=masks[v];
            do {domain&=restrictions[i++].allowed;} while(i<restrictions.size() && restrictions[i].vertex==v);
            if(!domain) {answer.sat=false;answer.certificate=proof(1,{v},{});return answer;}
            if(domain!=masks[v]) seeds.push_back(2*v+(first(domain)!=lo[v]));
        }
        if(bad!=NIL) {answer.sat=false;
            // A contradiction in the fixed graph survives every restriction.
            // Current pins are still checked first, preserving kind-1 proofs.
            if(known_base_proof && !known_base_proof->empty()) {
                answer.certificate=*known_base_proof;return answer;
            }
            auto a=path(2*bad,2*bad+1,answer),b=path(2*bad+1,2*bad,answer);
            answer.certificate=proof(3,a,b);return answer;
        }
        return solve_roots(seeds);
    }
    // Roots belong to this request only. The caller has already intersected
    // every restriction and ruled out empty domains and a fixed contradiction.
    Answer solve_roots(const std::vector<U>& seeds) const {
        Answer answer;
        std::vector<U> parent;
        std::vector<uint8_t> reached;
        if(condensed) {
            reached.assign(nc,0);std::vector<U> queue;queue.reserve(nc);U conflict=NIL;
            std::vector<U> route_parent,route_edge,route_seed;
            if(routed) {route_parent.assign(nc,NIL);route_edge.assign(nc,NIL);route_seed.assign(nc,NIL);}
            for(U v:seeds) {U c=comp[v];if(!reached[c]) {
                reached[c]=1;queue.push_back(c);
                if(routed) {route_parent[c]=c;route_seed[c]=v;}
                if(reached[opposite[c]]) {conflict=c;break;}
            }}
            for(size_t head=0;conflict==NIL && head<queue.size();++head) {U c=queue[head];++answer.visits;
                for(U j=dag.off[c];j<dag.off[c+1];++j) {++answer.arcs;U d=dag.to[j];
                    if(!reached[d]) {
                        reached[d]=1;queue.push_back(d);
                        if(routed) {route_parent[d]=c;route_edge[d]=j;}
                        if(reached[opposite[d]]) {conflict=d;break;}
                    }
                }
            }
            if(conflict!=NIL) {
                if(routed) {
                    U target=representative[conflict];answer.sat=false;
                    auto left=lift(target,route_parent,route_edge,route_seed,answer);
                    auto right=lift(target^1U,route_parent,route_edge,route_seed,answer);
                    answer.certificate=proof(2,left,right);
                } else if(raw_closure(seeds,parent,answer)) throw std::logic_error("quotient contradiction did not replay");
                return answer;
            }
        } else if(!raw_closure(seeds,parent,answer)) return answer;
        answer.labels.resize(n);
        for(U v=0;v<n;++v) {bool l=condensed?reached[comp[2*v]]:parent[2*v]!=NIL;
            bool h=condensed?reached[comp[2*v+1]]:parent[2*v+1]!=NIL;
            U side=l?0:(h?1:base[v]);answer.labels[v]=side?hi[v]:lo[v];
        }
        return answer;
    }
};
// Separately owned observer. It never receives Index, its graph, SCC labels,
// base assignment, propagation flags, parents, or a previously checked answer.
struct ProofScratch {
    std::vector<U> seen;U epoch=0;
    U next(U literals) {
        if(seen.size()!=literals) {seen.assign(literals,0);epoch=0;}
        if(epoch==UINT32_MAX) {std::fill(seen.begin(),seen.end(),0);epoch=0;}
        return ++epoch;
    }
};
struct Observer {
    U n,k;W cap;
    std::vector<W> masks;
    std::vector<std::pair<U,U>> edges,conflicts;
    std::vector<uint8_t> colors;
    explicit Observer(Input p):n(p.n),k(p.k),cap(p.cap),masks(std::move(p.masks)),edges(std::move(p.edges)),colors(2*n,255) {
        for(U v=0;v<n;++v) {U side=0;for(U c=0;c<k;++c) if((masks[v]>>c)&1) colors[2*v+side++]=static_cast<uint8_t>(c);}
        // Independently derive exactly the original equal-colour conflicts. The
        // observer never receives the solver's implication graph or SCC index.
        conflicts.reserve(2*edges.size());
        for(auto edge:edges) {
            W shared=masks[edge.first]&masks[edge.second];
            while(shared) {
                U color=first(shared);shared&=shared-1;
                U left=2*edge.first+(colors[2*edge.first]!=color);
                U right=2*edge.second+(colors[2*edge.second]!=color);
                conflicts.emplace_back(left,right);
            }
        }
        // Each admitted original edge is unique; each shared colour creates
        // one distinct forbidden literal pair. Verification scans every pair,
        // so neither sorting nor duplicate elimination is needed here. Keep
        // the independently owned original edges sorted for proof-hop lookup.
    }
    W bytes() const {return 8*W(masks.capacity()+edges.capacity()+conflicts.capacity())+colors.capacity();}
    bool hop(U a,U b) const {
        U u=a/2,v=b/2;
        if(u==v) return (a%2)==1 && (b%2)==0 && bits(masks[u])==1;
        if(colors[a]==255 || colors[b^1U]==255 || colors[a]!=colors[b^1U]) return false;
        return std::binary_search(edges.begin(),edges.end(),std::make_pair(std::min(u,v),std::max(u,v)));
    }
    bool verify(const std::vector<Pin>& pins,U status,const uint8_t* labels,size_t len,const uint8_t* data,size_t size,
                ProofScratch* scratch=nullptr) const {
        if(status==1) {
            if(len!=n || size) return false;
            for(U v=0;v<n;++v) if(labels[v]>=k || !((masks[v]>>labels[v])&1)) return false;
            for(auto pair:conflicts)
                if(labels[pair.first/2]==colors[pair.first]
                        && labels[pair.second/2]==colors[pair.second]) return false;
            for(auto p:pins) if(!((p.allowed>>labels[p.vertex])&1)) return false;
            return true;
        }
        if(status!=0 || len || size<16 || std::memcmp(data,"SBP1",4)) return false;
        auto get=[data](size_t at)->U {U value=0;for(U j=0;j<4;++j) value|=U(data[at+j])<<(8*j);return value;};
        U kind=get(4), na=get(8),nb=get(12);
        if(W(16)+4*(W(na)+nb)!=size || na>2*W(n)+1 || nb>2*W(n)) return false;
        // Proof checks need only the cited original root domains, not an
        // n-element domain copy. Pin validation is still complete and unsampled.
        auto effective=[&](U vertex) {W d=masks[vertex];for(auto pin:pins) if(pin.vertex==vertex) d&=pin.allowed;return d;};
        if(kind==1) return na==1 && nb==0 && get(16)<n && effective(get(16))==0;
        if((kind!=2 && kind!=3) || !na || !nb || na>2*W(n)) return false;
        // Read bounded paths directly from the serialized certificate. A fresh
        // generation checks simplicity for each path; the backing array can be
        // reused within this call without retaining any validation result.
        ProofScratch local;auto& work=scratch?*scratch:local;
        auto path_ok=[&](size_t offset,U count) {
            U stamp=work.next(2*n),previous=NIL;
            for(U i=0;i<count;++i) {
                U literal=get(offset+4*W(i));
                if(literal>=2*n || work.seen[literal]==stamp) return false;
                work.seen[literal]=stamp;
                if(i && !hop(previous,literal)) return false;
                previous=literal;
            }
            return true;
        };
        if(!path_ok(16,na) || !path_ok(16+4*W(na),nb)) return false;
        U a_first=get(16),a_last=get(16+4*W(na-1));
        U b_first=get(16+4*W(na)),b_last=get(16+4*(W(na)+nb-1));
        if(kind==3) return a_first==b_last && a_last==b_first && a_last==(a_first^1U);
        auto forced=[&](U literal) {return colors[literal]!=255 && effective(literal/2)==(W(1)<<colors[literal]);};
        return a_last==(b_last^1U) && forced(a_first) && forced(b_first);
    }
    template<class Bank>
    bool verify_bank(const Bank& bank,const uint8_t* output,size_t size) const {
        U rows=bank.query_count();
        if(size<16 || std::memcmp(output,"SCQ2",4) || load32(output,4)!=n || load64(output,8)!=rows)
            return false;
        size_t at=16;
        std::vector<W> selected(n,0);
        ProofScratch proof_scratch; // call-local, also for concurrent observer use
        for(U first_row=0;first_row<rows;first_row+=64) {
            U lanes=std::min<U>(64,rows-first_row);W sat_mask=0;
            std::fill(selected.begin(),selected.end(),0);
            for(U lane=0;lane<lanes;++lane) {
                if(at+W(n)+5>size) return false;
                U status=output[at];const uint8_t* labels=output+at+1;
                U length=load32(output,at+1+W(n));at+=W(n)+5;
                if(W(length)>size-at) return false;
                auto pins=bank.row(first_row+lane);
                if(status==1) {
                    if(length) return false;
                    W bit=W(1)<<lane;sat_mask|=bit;
                    for(U v=0;v<n;++v) {
                        U label=labels[v];
                        if(label==colors[2*v]) continue;
                        if(label<k && label==colors[2*v+1]) selected[v]|=bit;
                        else return false;
                    }
                    for(auto pin:pins)
                        if(!((pin.allowed>>labels[pin.vertex])&1)) return false;
                } else if(status==0) {
                    for(U v=0;v<n;++v) if(labels[v]) return false;
                    if(!verify(pins,0,nullptr,0,output+at,length,&proof_scratch)) return false;
                } else return false;
                at+=length;
            }
            // Check all SAT lanes against independently reconstructed original
            // equal-colour conflicts with one word operation per conflict/tile.
            for(auto pair:conflicts) {
                W left=(pair.first%2)?selected[pair.first/2]:~selected[pair.first/2];
                W right=(pair.second%2)?selected[pair.second/2]:~selected[pair.second/2];
                if(left&right&sat_mask) return false;
            }
        }
        return at==size;
    }
};
struct Released {
    PyThreadState* state;
    Released():state(PyEval_SaveThread()){} ~Released(){PyEval_RestoreThread(state);}
};
struct BankResult {
    std::vector<uint8_t> output;
    W visits=0,arcs=0,proof_bytes=0;
    U sat=0,unsat=0,tiles=0;
    bool tiled=false;
};
template<class SolverRows,class ObserverRows>
BankResult run_rows(Index& solver,Observer& observer,const SolverRows& solver_rows,
                    const ObserverRows& observer_rows,size_t packet_size,W total_cap,
                    const char* schema) {
    if(solver.n!=observer.n || solver.k!=observer.k)
        throw Invalid("solver and observer geometry differs");
    U count=solver_rows.query_count(),pins=solver_rows.pin_count();
    if(count!=observer_rows.query_count() || pins!=observer_rows.pin_count())
        throw std::logic_error("independent request parsers disagree");
    W retained=solver.bytes()+observer.bytes();
    W per_row=25+17*W(solver.n);
    W output_bound=16+per_row*W(count);
    // Covers parsed request copies, up to 64 complete answers, bit-lane state,
    // proof replay scratch, vector metadata, and output serialization. Python
    // inputs and allocator metadata remain outside this conservative native cap.
    // Two output envelopes cover vector reallocation and the immutable Python
    // bytes handoff while the native vector is still alive. Not an RSS limit.
    W scratch=2048*W(solver.n+1)+64*W(pins)+64*W(count)+8192;
    if(retained>total_cap || W(packet_size)>total_cap-retained
            || output_bound>(total_cap-retained-W(packet_size))/2
            || scratch>total_cap-retained-W(packet_size)-2*output_bound)
        throw Resource("combined exact bank solve/verify/output admission exceeds cap");
    BankResult result;
    W initial=16+W(count)*(W(solver.n)+25);
    if(initial>std::numeric_limits<size_t>::max())
        throw Resource("exact bank output reserve exceeds platform size");
    result.output.reserve(static_cast<size_t>(initial));
    result.output.insert(result.output.end(),{'S','C','Q','2'});word(result.output,solver.n);qword(result.output,count);

    auto reserve_row=[&](size_t proof_size) {
        W row_bytes=5+W(solver.n)+proof_size;
        if(row_bytes>output_bound-result.output.size())
            throw Resource("SCQ2 output exceeds admitted bound");
        W required=result.output.size()+row_bytes;
        if(required>result.output.capacity()) {
            // Bound capacity explicitly instead of relying on vector growth.
            W grown=std::min(output_bound,std::max(required,2*W(result.output.capacity())));
            result.output.reserve(static_cast<size_t>(grown));
        }
    };
    auto append=[&](const Answer& answer) {
        reserve_row(answer.certificate.size());
        result.visits+=answer.visits;result.arcs+=answer.arcs;
        result.output.push_back(answer.sat?1:0);
        if(answer.sat) {
            if(answer.labels.size()!=solver.n || !answer.certificate.empty())
                throw std::logic_error("SAT bank row geometry differs");
            result.output.insert(result.output.end(),answer.labels.begin(),answer.labels.end());++result.sat;
        } else {
            if(!answer.labels.empty() || answer.certificate.size()>UINT32_MAX)
                throw std::logic_error("UNSAT bank row geometry differs");
            result.output.resize(result.output.size()+solver.n,0);++result.unsat;
        }
        word(result.output,static_cast<U>(answer.certificate.size()));
        result.output.insert(result.output.end(),answer.certificate.begin(),answer.certificate.end());
        result.proof_bytes+=answer.certificate.size();
        if(W(result.output.size())>output_bound)
            throw Resource("SCQ2 output exceeds admitted bound");
    };

    // The quotient component IDs are topological. For sufficiently large banks,
    // propagate 64 independent requests as bits through the same DAG traversal.
    // Contradiction certificates are still reconstructed by the ordinary exact
    // solver per UNSAT lane, so proof semantics and independent checking remain
    // unchanged. This is a batching optimization, not a new satisfiability rule.
    const bool use_tiles=solver.condensed && solver.bad==NIL && count>=64;
    if(use_tiles) {
        result.tiled=true;
        // Reuse one allocation set for the whole bank. Each request is reduced
        // directly from the immutable packet into sorted unique roots; no
        // temporary Pin vector or per-lane vector allocation is needed.
        std::vector<U> seen(solver.n,0),touched;
        std::vector<W> effective(solver.n),reached(solver.nc,0),high_choices(solver.n);
        std::vector<U> roots,root_offsets(65,0),empty(64,NIL),conflict(64,NIL);
        std::vector<U> replay_roots,parent;
        touched.reserve(std::min<U>(solver.n,64));
        roots.reserve(static_cast<size_t>(std::min<W>(pins,64*W(solver.n))));
        replay_roots.reserve(std::min<U>(solver.n,64));
        U epoch=0;
        for(U first_row=0;first_row<count;first_row+=64) {
            ++result.tiles;
            U lanes=std::min<U>(64,count-first_row);
            roots.clear();root_offsets[0]=0;
            std::fill(empty.begin(),empty.begin()+lanes,NIL);
            std::fill(conflict.begin(),conflict.begin()+lanes,NIL);
            std::fill(reached.begin(),reached.end(),0);
            for(U lane=0;lane<lanes;++lane) {
                if(++epoch==0) {std::fill(seen.begin(),seen.end(),0);epoch=1;}
                touched.clear();
                U row=first_row+lane;
                for(U pin=solver_rows.row_begin(row);pin<solver_rows.row_end(row);++pin) {
                    U v=solver_rows.pin_vertex(pin);
                    if(seen[v]!=epoch) {
                        seen[v]=epoch;effective[v]=solver.masks[v];touched.push_back(v);
                    }
                    effective[v]&=solver_rows.pin_allowed(pin);
                }
                // Match scalar preparation exactly: the first empty-domain
                // proof and seed order are by original vertex, not packet order.
                std::sort(touched.begin(),touched.end());
                for(U v:touched) if(!effective[v]) {empty[lane]=v;break;}
                if(empty[lane]==NIL) for(U v:touched) if(effective[v]!=solver.masks[v])
                    roots.push_back(2*v+(first(effective[v])!=solver.lo[v]));
                root_offsets[lane+1]=static_cast<U>(roots.size());
                if(empty[lane]==NIL) {
                    W bit=W(1)<<lane;
                    for(U at=root_offsets[lane];at<root_offsets[lane+1];++at)
                        reached[solver.comp[roots[at]]]|=bit;
                }
            }
            for(U component=0;component<solver.nc;++component) if(reached[component]) {
                ++result.visits;
                W active=reached[component];
                for(U j=solver.dag.off[component];j<solver.dag.off[component+1];++j) {
                    ++result.arcs;reached[solver.dag.to[j]]|=active;
                }
            }
            for(U component=0;component<solver.nc;++component) {
                W lanes_in_conflict=reached[component]&reached[solver.opposite[component]];
                while(lanes_in_conflict) {
                    U lane=first(lanes_in_conflict);lanes_in_conflict&=lanes_in_conflict-1;
                    if(conflict[lane]==NIL) conflict[lane]=component;
                }
            }
            W sat_lanes=0;
            for(U lane=0;lane<lanes;++lane)
                if(empty[lane]==NIL && conflict[lane]==NIL) sat_lanes|=W(1)<<lane;
            // Resolve component choices once per vertex for the whole tile.
            // The output still contains every original label for every lane.
            if(sat_lanes) {
                for(U v=0;v<solver.n;++v) {
                    W low=reached[solver.comp[2*v]],high=reached[solver.comp[2*v+1]];
                    if(low&high&sat_lanes) throw std::logic_error("SAT bit lane retains contradiction");
                    high_choices[v]=(high|(solver.base[v]?~low:0))&sat_lanes;
                }
            }
            for(U lane=0;lane<lanes;++lane) {
                Answer answer;
                if(empty[lane]!=NIL) {
                    answer.sat=false;answer.certificate=proof(1,{empty[lane]},{});
                } else if(conflict[lane]!=NIL) {
                    // The batch already established a contradiction. Reuse its
                    // prepared roots, without reparsing or sorting the request.
                    replay_roots.assign(roots.begin()+root_offsets[lane],
                                        roots.begin()+root_offsets[lane+1]);
                    if(solver.routed) answer=solver.solve_roots(replay_roots);
                    else solver.raw_closure(replay_roots,parent,answer);
                    if(answer.sat)
                        throw std::logic_error("bit-lane contradiction did not replay");
                } else {
                    // Write directly into the admitted SCQ2 row. Its four-byte
                    // proof length stays zero; no temporary label vector/copy.
                    reserve_row(0);
                    size_t at=result.output.size();
                    result.output.resize(at+W(solver.n)+5,0);
                    result.output[at]=1;++result.sat;
                    W bit=W(1)<<lane;
                    for(U v=0;v<solver.n;++v)
                        result.output[at+1+v]=(high_choices[v]&bit)?solver.hi[v]:solver.lo[v];
                    continue;
                }
                append(answer);
            }
        }
    } else {
        // Keep this immutable-problem proof local to this call. Do not cache
        // request-dependent contradictions or share mutable state across calls.
        // The observer independently checks every emitted copy below.
        std::vector<uint8_t> base_proof;
        for(U row=0;row<count;++row) {
            Answer answer=solver.solve(solver_rows.row(row),&base_proof);
            if(row+1<count && solver.bad!=NIL && base_proof.empty() && answer.certificate.size()>=16
                    && load32(answer.certificate.data(),4)==3)
                base_proof=answer.certificate;
            append(answer);
        }
    }
    if(!observer.verify_bank(observer_rows,result.output.data(),result.output.size()))
        throw std::logic_error(std::string("serialized ")+schema+" bank failed independent observer");
    return result;
}
BankResult run_bank(Index& solver,Observer& observer,const uint8_t* data,size_t size) {
    SolverBank solver_bank(data,size,solver.n);
    ObserverBank observer_bank(data,size,observer.n);
    W observer_bytes=observer.bytes();
    W total_cap=solver.cap>UINT64_MAX-observer_bytes?UINT64_MAX:solver.cap+observer_bytes;
    return run_rows(solver,observer,solver_bank,observer_bank,size,total_cap,"SBQ1");
}
PyObject* exception() {
    try {throw;} catch(const Invalid& e){PyErr_SetString(PyExc_ValueError,e.what());}
    catch(const Resource& e){PyErr_SetString(PyExc_MemoryError,e.what());}
    catch(const std::bad_alloc&){PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());}
    catch(...){PyErr_SetString(PyExc_RuntimeError,"unknown native failure");}
    return nullptr;
}
void destroy_index(PyObject* c){delete static_cast<Index*>(PyCapsule_GetPointer(c,SOLVER));}
void destroy_observer(PyObject* c){delete static_cast<Observer*>(PyCapsule_GetPointer(c,OBSERVER));}
template<class T> T* handle(PyObject* p,const char* tag) {auto* t=static_cast<T*>(PyCapsule_GetPointer(p,tag));if(!t){PyErr_Clear();throw Invalid("invalid exact-service handle");}return t;}
PyObject* create(PyObject*,PyObject* args) {
    PyObject *nv,*kv,*es,*ms,*mode,*cap;
    if(!PyArg_ParseTuple(args,"OOOOOO",&nv,&kv,&es,&ms,&mode,&cap)) return nullptr;
    try {W limit=number(cap,"invalid memory payload cap"),which=number(mode,"mode must be 0, 1, 2 or 3",3);
        Input input(nv,kv,es,ms,limit);std::unique_ptr<Index> result;
        {Released guard;result=std::make_unique<Index>(std::move(input),static_cast<U>(which));}
        PyObject* capsule=PyCapsule_New(result.get(),SOLVER,destroy_index);if(capsule) result.release();
        return capsule;
    }catch(...){return exception();}
}
PyObject* observer_create(PyObject*,PyObject* args) {
    PyObject *nv,*kv,*es,*ms,*cap;
    if(!PyArg_ParseTuple(args,"OOOOO",&nv,&kv,&es,&ms,&cap)) return nullptr;
    try {Input input(nv,kv,es,ms,number(cap,"invalid memory payload cap"));
        auto result=std::make_unique<Observer>(std::move(input));PyObject* capsule=PyCapsule_New(result.get(),OBSERVER,destroy_observer);
        if(capsule) result.release();
        return capsule;
    }catch(...){return exception();}
}
PyObject* solve(PyObject*,PyObject* args) {
    PyObject *h,*q;
    if(!PyArg_ParseTuple(args,"OO",&h,&q)) return nullptr;
    try {auto* x=handle<Index>(h,SOLVER);auto pins=read_query(q,x->n,x->cap);Answer a;
        {Released guard;a=x->solve(pins);}
        PyObject* labels=PyBytes_FromStringAndSize(reinterpret_cast<const char*>(a.labels.data()),static_cast<Py_ssize_t>(a.labels.size()));
        if(!labels) return nullptr;
        PyObject* cert=PyBytes_FromStringAndSize(reinterpret_cast<const char*>(a.certificate.data()),static_cast<Py_ssize_t>(a.certificate.size()));
        if(!cert){Py_DECREF(labels);return nullptr;}
        return Py_BuildValue("iNNKK",a.sat?1:0,labels,cert,static_cast<unsigned long long>(a.visits),static_cast<unsigned long long>(a.arcs));
    }catch(...){return exception();}
}
PyObject* solve_bank(PyObject*,PyObject* args) {
    PyObject *solver_handle,*observer_handle,*packet;
    if(!PyArg_ParseTuple(args,"OOO",&solver_handle,&observer_handle,&packet)) return nullptr;
    try {
        auto* solver=handle<Index>(solver_handle,SOLVER);auto* observer=handle<Observer>(observer_handle,OBSERVER);
        if(!PyBytes_CheckExact(packet)) throw Invalid("query bank must be immutable exact bytes");
        const auto* data=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(packet));
        size_t size=static_cast<size_t>(PyBytes_GET_SIZE(packet));BankResult result;
        {Released guard;result=run_bank(*solver,*observer,data,size);}
        PyObject* output=PyBytes_FromStringAndSize(reinterpret_cast<const char*>(result.output.data()),
                                                   static_cast<Py_ssize_t>(result.output.size()));
        if(!output) return nullptr;
        return Py_BuildValue("NKKKKKK",output,
            static_cast<unsigned long long>(result.sat+result.unsat),
            static_cast<unsigned long long>(result.sat),
            static_cast<unsigned long long>(result.unsat),
            static_cast<unsigned long long>(result.proof_bytes),
            static_cast<unsigned long long>(result.visits),
            static_cast<unsigned long long>(result.arcs));
    }catch(...){return exception();}
}
PyObject* solve_session_packet(PyObject*,PyObject* args) {
    PyObject *packet,*mode,*cap;
    if(!PyArg_ParseTuple(args,"OOO",&packet,&mode,&cap)) return nullptr;
    try {
        if(!PyBytes_CheckExact(packet)) throw Invalid("session packet must be immutable exact bytes");
        W which=number(mode,"mode must be 0, 1, 2, 3 or 4",4);
        W limit=number(cap,"invalid memory payload cap");
        if(limit<(1U<<20)) throw Invalid("memory payload cap must be at least 1 MiB");
        const auto* data=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(packet));
        size_t size=static_cast<size_t>(PyBytes_GET_SIZE(packet));
        std::unique_ptr<Index> solver;std::unique_ptr<Observer> observer;BankResult result;
        U bank_queries=0,bank_pins=0;
        {
            Released guard;
            ObserverSessionPacket observer_rows(data,size,limit);
            observer=std::make_unique<Observer>(observer_rows.take_input(limit));
            W observer_bytes=observer->bytes();
            if(observer_bytes>=limit) throw Resource("SBC1 observer consumes session cap");
            SolverSessionPacket solver_rows(data,size,limit-observer_bytes);
            bank_queries=solver_rows.query_count();bank_pins=solver_rows.pin_count();
            solver=std::make_unique<Index>(solver_rows.take_input(limit-observer_bytes),
                                           static_cast<U>(which),bank_queries,bank_pins);
            result=run_rows(*solver,*observer,solver_rows,observer_rows,size,limit,"SBC1");
        }
        PyObject* output=PyBytes_FromStringAndSize(reinterpret_cast<const char*>(result.output.data()),
                                                   static_cast<Py_ssize_t>(result.output.size()));
        if(!output) return nullptr;
        PyObject* metadata=Py_BuildValue("{sI,sI,sK,sK,sK,sK,sK,sK,sO,sO,ss}",
            "vertices",solver->n,"components",solver->nc,
            "original_arcs",static_cast<unsigned long long>(solver->full_arcs),
            "retained_proof_arcs",static_cast<unsigned long long>(solver->original.to.size()),
            "condensation_arcs",static_cast<unsigned long long>(solver->dag.to.size()),
            "retained_array_bytes",static_cast<unsigned long long>(solver->bytes()),
            "routing_array_bytes",static_cast<unsigned long long>(solver->routing_bytes()),
            "native_observer_array_bytes",static_cast<unsigned long long>(observer->bytes()),
            "original_adjacency_retained",solver->routed?Py_False:Py_True,
            "base_satisfiable",solver->bad==NIL?Py_True:Py_False,
            "selected_mode",solver->routed?"route":(solver->sparse?"sparse":(solver->condensed?"scc":"raw")));
        if(!metadata){Py_DECREF(output);return nullptr;}
        PyObject* kernel=PyUnicode_FromString(result.tiled?"tile64":"serial");
        PyObject* tiles=PyLong_FromUnsignedLong(result.tiles);
        PyObject* queries=PyLong_FromUnsignedLong(bank_queries);
        PyObject* pins=PyLong_FromUnsignedLong(bank_pins);
        PyObject* policy=PyUnicode_FromString(which==4?"structural-v2":"explicit");
        PyObject* reason=PyUnicode_FromString(solver->auto_reason);
        if(!kernel || !tiles || !queries || !pins || !policy || !reason
                || PyDict_SetItemString(metadata,"bank_kernel",kernel)<0
                || PyDict_SetItemString(metadata,"bank_tiles",tiles)<0
                || PyDict_SetItemString(metadata,"bank_query_count",queries)<0
                || PyDict_SetItemString(metadata,"bank_pin_count",pins)<0
                || PyDict_SetItemString(metadata,"auto_policy",policy)<0
                || PyDict_SetItemString(metadata,"auto_policy_reason",reason)<0) {
            Py_XDECREF(kernel);Py_XDECREF(tiles);Py_XDECREF(queries);Py_XDECREF(pins);
            Py_XDECREF(policy);Py_XDECREF(reason);Py_DECREF(output);Py_DECREF(metadata);return nullptr;
        }
        Py_DECREF(kernel);Py_DECREF(tiles);Py_DECREF(queries);Py_DECREF(pins);
        Py_DECREF(policy);Py_DECREF(reason);
        return Py_BuildValue("NKKKKKKN",output,
            static_cast<unsigned long long>(result.sat+result.unsat),
            static_cast<unsigned long long>(result.sat),
            static_cast<unsigned long long>(result.unsat),
            static_cast<unsigned long long>(result.proof_bytes),
            static_cast<unsigned long long>(result.visits),
            static_cast<unsigned long long>(result.arcs),metadata);
    }catch(...){return exception();}
}
PyObject* verify(PyObject*,PyObject* args) {
    PyObject *h,*q,*status,*labels,*proof_bytes;
    if(!PyArg_ParseTuple(args,"OOOOO",&h,&q,&status,&labels,&proof_bytes)) return nullptr;
    try {auto* o=handle<Observer>(h,OBSERVER);auto pins=read_query(q,o->n,o->cap);
        if(!PyBytes_CheckExact(labels)||!PyBytes_CheckExact(proof_bytes)||!PyLong_CheckExact(status)) Py_RETURN_FALSE;
        W code=number(status,"bad decision",1);
        const auto* l=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(labels));
        const auto* p=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(proof_bytes));
        size_t nl=PyBytes_GET_SIZE(labels),np=PyBytes_GET_SIZE(proof_bytes);bool ok;
        {Released guard;ok=o->verify(pins,static_cast<U>(code),l,nl,p,np);}
        return PyBool_FromLong(ok);
    }catch(...){return exception();}
}
PyObject* check_bank(PyObject*,PyObject* args) {
    PyObject *observer_handle,*packet,*output;
    if(!PyArg_ParseTuple(args,"OOO",&observer_handle,&packet,&output)) return nullptr;
    try {
        auto* observer=handle<Observer>(observer_handle,OBSERVER);
        if(!PyBytes_CheckExact(packet) || !PyBytes_CheckExact(output)) Py_RETURN_FALSE;
        const auto* request_data=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(packet));
        const auto* answer_data=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(output));
        size_t request_size=static_cast<size_t>(PyBytes_GET_SIZE(packet));
        size_t answer_size=static_cast<size_t>(PyBytes_GET_SIZE(output));bool ok;
        {Released guard;ObserverBank bank(request_data,request_size,observer->n);
            ok=observer->verify_bank(bank,answer_data,answer_size);}
        return PyBool_FromLong(ok);
    }catch(...){return exception();}
}
PyObject* info(PyObject*,PyObject* h) {
    try {auto* x=handle<Index>(h,SOLVER);
        return Py_BuildValue("{sI,sI,sK,sK,sK,sK,sK,sO,sO}","vertices",x->n,"components",x->nc,
            "original_arcs",static_cast<unsigned long long>(x->full_arcs),
            "retained_proof_arcs",static_cast<unsigned long long>(x->original.to.size()),
            "condensation_arcs",static_cast<unsigned long long>(x->dag.to.size()),
            "retained_array_bytes",static_cast<unsigned long long>(x->bytes()),
            "routing_array_bytes",static_cast<unsigned long long>(x->routing_bytes()),
            "original_adjacency_retained",x->routed?Py_False:Py_True,
            "base_satisfiable",x->bad==NIL?Py_True:Py_False);
    }catch(...){return exception();}
}
PyObject* observer_bytes(PyObject*,PyObject* h) {try{return PyLong_FromUnsignedLongLong(handle<Observer>(h,OBSERVER)->bytes());}catch(...){return exception();}}
PyObject* abi(PyObject*,PyObject*){return PyLong_FromLong(3);}
PyMethodDef methods[]={
    {"create",create,METH_VARARGS,nullptr},{"observer_create",observer_create,METH_VARARGS,nullptr},
    {"solve",solve,METH_VARARGS,nullptr},{"solve_bank",solve_bank,METH_VARARGS,nullptr},
    {"solve_session_packet",solve_session_packet,METH_VARARGS,nullptr},
    {"verify",verify,METH_VARARGS,nullptr},{"check_bank",check_bank,METH_VARARGS,nullptr},
    {"info",info,METH_O,nullptr},{"observer_bytes",observer_bytes,METH_O,nullptr},
    {"abi",abi,METH_NOARGS,nullptr},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_exact_binary",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_exact_binary(){return PyModule_Create(&module);}
