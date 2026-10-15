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
#include <tuple>
#include <utility>
#include <vector>

namespace {
using U = uint32_t;
using W = uint64_t;
constexpr U NIL = UINT32_MAX, MAX_N = 100000, MAX_E = 2000000, MAX_P = 1000000;
constexpr const char* SOLVER = "spectra.exact_binary.solver.v2";
constexpr const char* OBSERVER = "spectra.exact_binary.observer.v2";
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
U first(W v) { return static_cast<U>(__builtin_ctzll(v)); }
U bits(W v) { return static_cast<U>(__builtin_popcountll(v)); }
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
struct Answer {bool sat=true;std::vector<uint8_t> labels,certificate;W visits=0,arcs=0;};
struct Index {
    U n,k,nc=0,bad=NIL;bool condensed,sparse,routed;W cap;W full_arcs=0;
    std::vector<W> masks;
    std::vector<uint8_t> lo,hi,base;
    std::vector<U> comp,opposite;
    std::vector<U> inward,outward,representative,cross_from,cross_to;
    CSR original,dag;
    Index(const Input& p,U mode):n(p.n),k(p.k),condensed(mode!=0),sparse(mode==2),routed(mode==3),cap(p.cap),masks(p.masks),lo(n),hi(n,255),base(n) {
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
        std::sort(arcs.begin(),arcs.end());arcs.erase(std::unique(arcs.begin(),arcs.end()),arcs.end());
        full_arcs=arcs.size();original=CSR(2*n,arcs);CSR reverse(2*n,arcs,true);comp=components(original,reverse);
        for(U c:comp) nc=std::max(nc,c+1);
        for(U v=0;v<n;++v) {if(comp[2*v]==comp[2*v+1] && bad==NIL) bad=v;
            base[v]=static_cast<uint8_t>(comp[2*v]<comp[2*v+1]);
        }
        if(condensed && bad==NIL) {
            opposite.assign(nc,NIL);
            for(U v=0;v<2*n;++v) {
                U a=comp[v],b=comp[v^1U];
                if(a==b || (opposite[a]!=NIL && opposite[a]!=b)) throw std::logic_error("complement mapping differs");
                opposite[a]=b;
            }
            std::vector<std::pair<U,U>> links;links.reserve(arcs.size());
            for(auto e:arcs) if(comp[e.first]!=comp[e.second]) links.emplace_back(comp[e.first],comp[e.second]);
            std::sort(links.begin(),links.end());links.erase(std::unique(links.begin(),links.end()),links.end());
            dag=CSR(nc,links);
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
            U previous_a=NIL,previous_b=NIL;
            for(auto e:bridges) {U a=std::get<0>(e),b=std::get<1>(e);
                if(a!=previous_a || b!=previous_b) {
                    if(sparse) keep.emplace_back(std::get<2>(e),std::get<3>(e));
                    if(routed && bad==NIL) {cross_from.push_back(std::get<2>(e));cross_to.push_back(std::get<3>(e));}
                    previous_a=a;previous_b=b;
                }
            }
            std::sort(keep.begin(),keep.end());keep.erase(std::unique(keep.begin(),keep.end()),keep.end());
            if(keep.size()>arcs.size()) throw std::logic_error("proof subgraph grew beyond original");
            if(routed) {
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
    Answer solve(const std::vector<Pin>& pins) const {
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
            auto a=path(2*bad,2*bad+1,answer),b=path(2*bad+1,2*bad,answer);
            answer.certificate=proof(3,a,b);return answer;
        }
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
struct Observer {
    U n,k;W cap;
    std::vector<W> masks;
    std::vector<std::pair<U,U>> edges;
    std::vector<uint8_t> colors;
    explicit Observer(const Input& p):n(p.n),k(p.k),cap(p.cap),masks(p.masks),edges(p.edges),colors(2*n,255) {
        for(U v=0;v<n;++v) {U side=0;for(U c=0;c<k;++c) if((masks[v]>>c)&1) colors[2*v+side++]=static_cast<uint8_t>(c);}
    }
    W bytes() const {return 8*W(masks.capacity()+edges.capacity())+colors.capacity();}
    bool hop(U a,U b) const {
        U u=a/2,v=b/2;
        if(u==v) return (a%2)==1 && (b%2)==0 && bits(masks[u])==1;
        if(colors[a]==255 || colors[b^1U]==255 || colors[a]!=colors[b^1U]) return false;
        return std::binary_search(edges.begin(),edges.end(),std::make_pair(std::min(u,v),std::max(u,v)));
    }
    bool verify(const std::vector<Pin>& pins,U status,const uint8_t* labels,size_t len,const uint8_t* data,size_t size) const {
        if(status==1) {
            if(len!=n || size) return false;
            for(U v=0;v<n;++v) if(labels[v]>=k || !((masks[v]>>labels[v])&1)) return false;
            for(auto e:edges) if(labels[e.first]==labels[e.second]) return false;
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
        std::vector<U> a(na),b(nb);
        for(U i=0;i<na;++i) {a[i]=get(16+4*W(i));if(a[i]>=2*W(n)) return false;}
        for(U i=0;i<nb;++i) {b[i]=get(16+4*(W(na)+i));if(b[i]>=2*W(n)) return false;}
        // Enforce simple, bounded paths. This is a format rule, not needed for logic.
        std::vector<uint8_t> seen(2*n,0);
        for(U x:a) {if(seen[x]) return false;seen[x]=1;}
        std::fill(seen.begin(),seen.end(),0);for(U x:b) {if(seen[x]) return false;seen[x]=1;}
        for(U i=1;i<na;++i) if(!hop(a[i-1],a[i])) return false;
        for(U i=1;i<nb;++i) if(!hop(b[i-1],b[i])) return false;
        if(kind==3) return a.front()==b.back() && a.back()==b.front() && a.back()==(a.front()^1U);
        auto forced=[&](U literal) {return colors[literal]!=255 && effective(literal/2)==(W(1)<<colors[literal]);};
        return a.back()==(b.back()^1U) && forced(a.front()) && forced(b.front());
    }
};
struct Released {
    PyThreadState* state;
    Released():state(PyEval_SaveThread()){} ~Released(){PyEval_RestoreThread(state);}
};
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
        {Released guard;result=std::make_unique<Index>(input,static_cast<U>(which));}
        PyObject* capsule=PyCapsule_New(result.get(),SOLVER,destroy_index);if(capsule) result.release();
        return capsule;
    }catch(...){return exception();}
}
PyObject* observer_create(PyObject*,PyObject* args) {
    PyObject *nv,*kv,*es,*ms,*cap;
    if(!PyArg_ParseTuple(args,"OOOOO",&nv,&kv,&es,&ms,&cap)) return nullptr;
    try {Input input(nv,kv,es,ms,number(cap,"invalid memory payload cap"));
        auto result=std::make_unique<Observer>(input);PyObject* capsule=PyCapsule_New(result.get(),OBSERVER,destroy_observer);
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
PyObject* abi(PyObject*,PyObject*){return PyLong_FromLong(2);}
PyMethodDef methods[]={
    {"create",create,METH_VARARGS,nullptr},{"observer_create",observer_create,METH_VARARGS,nullptr},
    {"solve",solve,METH_VARARGS,nullptr},{"verify",verify,METH_VARARGS,nullptr},
    {"info",info,METH_O,nullptr},{"observer_bytes",observer_bytes,METH_O,nullptr},
    {"abi",abi,METH_NOARGS,nullptr},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_exact_binary",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_exact_binary(){return PyModule_Create(&module);}
