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
// Reasons form an acyclic implication graph over assignments. Root assumptions
// are fixed inputs; a decision refers to its one-based frame; a cover reason
// uses the other false members; an exclusion reason uses its true trigger.
struct Reason {U type=0,index=NONE;};
struct Cause {Reason reason;U owner=NONE;};
// A conservative explanation denotes all choices at levels 1..prior plus the
// choice at top. Prefix closure may lose pruning power but never omit a cause.
struct Failure {U top=0,prior=0;};
struct Answer {
    std::vector<int8_t> value;
    U reason=2;
    uint64_t binary_calls=0,binary_edges=0,binary_solved=0,jumps=0,skipped=0,analysis=0,conflict_bytes=0;
    uint64_t work=0,nodes=0,forced=0,backtracks=0,index_bytes=0,state_bytes=0,trace=14695981039346656037ULL;
};
// Every mutable bank is allocated once from a closed-form linear payload bound.
struct Search {
    const Index& x; bool heap_mode,active_mode,xor_mode,degree_mode,lcv_mode,binary_mode,compact_mode,wdeg_mode,jump_mode,exact_mode; uint64_t limit,state_cap;
    Answer result;
    std::vector<U> left,covered,heap,pos,trail,xor_left,activity;
    U active_count=0;
    std::vector<int32_t> queue;
    std::vector<int8_t> pending;
    std::vector<Frame> frames;
    std::vector<Reason> why,queued_why;
    std::vector<uint8_t> reason_seen;
    std::vector<U> reason_stack,first_prior;
    std::vector<uint64_t> causal,first_causes;size_t cause_words=0;
    Cause conflict_causes[2]; U causes=0;
    size_t head=0;
    U unsatisfied;
    bool conflict=false;
    uint64_t work=0,nodes=0,forced=0,backtracks=0,trace=14695981039346656037ULL;
    Search(const Index& index,U flags,uint64_t cap,uint64_t memory):x(index),heap_mode(flags&1),active_mode(flags&2),xor_mode(flags&4),degree_mode(flags&8),lcv_mode(flags&16),binary_mode(flags&32),compact_mode(flags&64),wdeg_mode(flags&128),jump_mode(flags&256),exact_mode(flags&512),limit(cap),state_cap(memory),unsatisfied(x.p) {
        result.index_bytes=x.bytes;
        result.state_bytes=uint64_t(x.n)*(2+4+4+sizeof(Frame)+(jump_mode?(2*sizeof(Reason)+9):0))+uint64_t(x.p)*(8+((heap_mode||active_mode)?8:0)+(xor_mode?4:0)+(wdeg_mode?4:0));
        if (result.state_bytes>memory) throw Resource("linear search state exceeds payload cap");
        if(exact_mode) {
            cause_words=(size_t(x.n)+63)/64;
            const uint64_t extra=8*(uint64_t(x.n)+1)*cause_words;
            // Exact conflict sets are a capped OPTIONAL departure from linear
            // state. Larger models transparently use conservative prefix closure.
            exact_mode=extra<=4*1024*1024 && extra<=memory-result.state_bytes;
            if(exact_mode) {
                result.state_bytes+=extra;result.conflict_bytes=extra;
                causal.assign(cause_words,0);first_causes.assign(size_t(x.n)*cause_words,0);
            }
        }
        result.value.assign(x.n,0);pending.assign(x.n,0);
        left.resize(x.p);covered.assign(x.p,0);
        if (heap_mode||active_mode) {heap.resize(x.p);pos.resize(x.p);active_count=x.p;}
        if (xor_mode) xor_left.assign(x.p,0);
        if (wdeg_mode) activity.assign(x.p,1);
        if (jump_mode) {why.resize(x.n);queued_why.resize(x.n);reason_seen.resize(x.n);reason_stack.reserve(x.n);first_prior.resize(x.n);}
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
        if(wdeg_mode&&!covered[a]&&!covered[b]) {
            const uint64_t lhs=uint64_t(left[a])*(activity[b]+1);
            const uint64_t rhs=uint64_t(left[b])*(activity[a]+1);
            if(lhs!=rhs)return lhs<rhs;
        }
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
    void conflict_weight(U cause) {
        if(!wdeg_mode||conflict||cause>=x.g)return;
        // Bounded persistent-within-search activity, never cross-query training.
        for(U i=x.offsets[cause];i<x.offsets[cause+1];++i) {
            const U v=x.vars[i];
            for(U j=x.voff[v];j<x.voff[v+1]&&x.groups[j]<x.p;++j) {
                const U g=x.groups[j];if(activity[g]<65535){++activity[g];changed(g);}
            }
        }
    }
    void remember_conflict(Reason a,U owner,Reason b={0,NONE},U other=NONE,U count=1) {
        if(!jump_mode||conflict)return;
        conflict_causes[0]={a,owner};conflict_causes[1]={b,other};causes=count;
    }
    void enqueue(U v,int8_t value,U cause=NONE,Reason reason={0,NONE}) {
        if (result.value[v]) {
            if(result.value[v]!=value){
                remember_conflict(reason,v,jump_mode?why[v]:Reason{},v,2);
                conflict_weight(cause);conflict=true;
            }return;
        }
        if (pending[v]) {
            if(pending[v]!=value){
                remember_conflict(reason,v,jump_mode?queued_why[v]:Reason{},v,2);
                conflict_weight(cause);conflict=true;
            }return;
        }
        pending[v]=value;if(jump_mode)queued_why[v]=reason;
        queue.push_back(value>0?static_cast<int32_t>(v+1):-static_cast<int32_t>(v+1));
    }
    void force_unit(U g) {
        if(covered[g])return;
        if(!left[g]) {
            remember_conflict({2,g},NONE);conflict_weight(g);conflict=true;return;
        }
        if(left[g]==1) {
            if(xor_mode) {enqueue(xor_left[g],1,g,{2,g});return;}
            for(U i=x.offsets[g];i<x.offsets[g+1];++i)
                if(!result.value[x.vars[i]]) {enqueue(x.vars[i],1,g,{2,g});return;}
        }
    }
    // Highest decision on the implication explanation of this conflict. A
    // bounded/unsupported explanation falls back to chronological backtracking.
    // Both-child explanations are combined below with conservative prefix
    // closure; storing only one child's dependency would be unsound.
    Failure jump_target() {
        const U deepest=static_cast<U>(frames.size());
        const Failure fallback{deepest,deepest?deepest-1:0};
        auto make_fallback=[&]() {
            if(exact_mode) {
                std::fill(causal.begin(),causal.end(),0);
                for(U d=0;d<deepest;++d)causal[d/64]|=uint64_t(1)<<(d%64);
            }
            return fallback;
        };
        if(!jump_mode||!causes)return make_fallback();
        if(exact_mode)std::fill(causal.begin(),causal.end(),0);
        std::fill(reason_seen.begin(),reason_seen.end(),0);reason_stack.clear();
        U target=0,prior=0;uint64_t visits=0;bool overflow=false;
        auto push=[&](U v) {
            if(v>=x.n||!result.value[v]) {overflow=true;return;}
            if(!reason_seen[v]) {reason_seen[v]=1;reason_stack.push_back(v);}
        };
        auto tick=[&]() {
            if(visits==1000000||work==limit){overflow=true;return false;}
            ++visits;++work;++result.analysis;return true;
        };
        auto expand=[&](Reason reason,U owner) {
            if(!tick())return;
            if(reason.type==0)return;
            if(reason.type==1){
                if(exact_mode&&reason.index&&reason.index<=deepest)
                    causal[(reason.index-1)/64]|=uint64_t(1)<<((reason.index-1)%64);
                if(!reason.index||reason.index>deepest)overflow=true;
                else if(reason.index>target){prior=target;target=reason.index;}
                else if(reason.index<target)prior=std::max(prior,reason.index);
            }else if(reason.type==3)push(reason.index);
            else if(reason.type==2&&reason.index<x.p) {
                for(U j=x.offsets[reason.index];j<x.offsets[reason.index+1]&&!overflow;++j) {
                    if(!tick())break;
                    const U v=x.vars[j];if(v!=owner)push(v);
                }
            }else overflow=true;
        };
        for(U i=0;i<causes&&!overflow;++i)expand(conflict_causes[i].reason,conflict_causes[i].owner);
        while(!reason_stack.empty()&&!overflow) {
            const U v=reason_stack.back();reason_stack.pop_back();expand(why[v],v);
        }
        return overflow?make_fallback():Failure{target,prior};
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
            if(result.value[v]) {
                if(result.value[v]!=val) {remember_conflict(jump_mode?queued_why[v]:Reason{},v,jump_mode?why[v]:Reason{},v,2);conflict=true;}
                continue;
            }
            if(jump_mode)why[v]=queued_why[v];
            result.value[v]=val;trail.push_back(v);++work;++forced;
            trace=(trace^static_cast<uint64_t>(static_cast<int64_t>(lit)))*1099511628211ULL;
            if(compact_mode) {
                // Per-variable memberships are in ascending group-id order. Counts
                // for a covered group may freeze: later assignments are undone
                // before the first covering true assignment can be undone.
                U j=x.voff[v];
                for(;j<x.voff[v+1]&&x.groups[j]<x.p;++j) {
                    const U g=x.groups[j];
                    if(!covered[g]) {
                        --left[g];if(xor_mode)xor_left[g]^=v;
                        if(val>0){++covered[g];--unsatisfied;}
                        changed(g);force_unit(g);
                    }else if(val>0)++covered[g];
                }
                // Finish the entire atomic assignment even after detecting a
                // conflict, so every partial branch remains exactly reversible.
                if(val>0) for(;j<x.voff[v+1];++j) {
                    const U g=x.groups[j];
                    for(U i=x.offsets[g];i<x.offsets[g+1];++i)
                        if(x.vars[i]!=v)enqueue(x.vars[i],-1,g,{3,v});
                }
            }else {
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
                else if(val>0) for(U i=x.offsets[g];i<x.offsets[g+1];++i) if(x.vars[i]!=v) enqueue(x.vars[i],-1,g,{3,v});
            }
            }
        }
        if(conflict)return 1;
        clear_queue();return 0;
    }
    void rollback(size_t mark) {
        clear_queue();conflict=false;causes=0;
        while(trail.size()>mark) {
            const U v=trail.back();trail.pop_back();const int8_t val=result.value[v];result.value[v]=0;
            for(U j=x.voff[v];j<x.voff[v+1];++j) {
                const U g=x.groups[j];if(g>=x.p)break;
                if(compact_mode) {
                    if(val>0 && --covered[g]==0)++unsatisfied;
                    if(!covered[g]) {++left[g];if(xor_mode)xor_left[g]^=v;changed(g);}
                }else {
                    ++left[g];if(xor_mode)xor_left[g]^=v;
                    if(val>0 && --covered[g]==0)++unsatisfied;
                    changed(g);
                }
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
    // Exact, opportunistic 2-CNF closure. No large clause is approximated.
    // Return: 0 outside admitted residual/payload; 1 SAT; 2 conflict; 3 work cap.
    U binary_residual() {
        uint64_t arcs=0;
        for(U g=0;g<x.p;++g) if(!covered[g]) {
            if(left[g]>2)return 0;
            if(!left[g])return 2;
            arcs+=left[g]==1?1:2;
        }
        // Every surviving exclusion is binary; wide exclusions are conservatively
        // left to ordinary search instead of materializing quadratic pairs.
        for(U g=x.p;g<x.g;++g) {
            U free=0,yes=0;
            for(U j=x.offsets[g];j<x.offsets[g+1];++j) {
                const int8_t a=result.value[x.vars[j]];
                free+=a==0;yes+=a>0;
            }
            if(yes>1)return 2;
            if(yes)arcs+=free;
            else if(free>2)return 0;
            else if(free==2)arcs+=2;
        }
        const uint64_t vertices=2*uint64_t(x.n);
        // Arc list, forward/reverse CSR, DFS stack, finish order, components,
        // visited flags and CSR fill cursors. Counts are logical payload bytes.
        const uint64_t scratch=33*vertices+16*arcs+16;
        if(scratch>state_cap-result.state_bytes)return 0;
        const uint64_t charge=vertices+arcs+1;
        if(charge>limit-work)return 3;
        work+=charge;++result.binary_calls;result.binary_edges+=arcs;
        result.state_bytes+=scratch;
        std::vector<std::pair<U,U>> edges;edges.reserve(arcs);
        auto clause=[&](U a,U b) {
            edges.emplace_back(a^1U,b);
            if(a!=b)edges.emplace_back(b^1U,a);
        };
        for(U g=0;g<x.p;++g) if(!covered[g]) {
            U a=NONE,b=NONE;
            for(U j=x.offsets[g];j<x.offsets[g+1];++j) {
                const U v=x.vars[j];if(result.value[v])continue;
                if(a==NONE)a=2*v;else b=2*v;
            }
            clause(a,b==NONE?a:b);
        }
        for(U g=x.p;g<x.g;++g) {
            U yes=0,a=NONE,b=NONE;
            for(U j=x.offsets[g];j<x.offsets[g+1];++j) {
                const U v=x.vars[j];yes+=result.value[v]>0;
                if(result.value[v])continue;
                if(a==NONE)a=2*v+1;else b=2*v+1;
            }
            if(yes) {
                for(U j=x.offsets[g];j<x.offsets[g+1];++j) {
                    const U v=x.vars[j];if(!result.value[v])clause(2*v+1,2*v+1);
                }
            }else if(b!=NONE)clause(a,b);
        }
        if(edges.size()!=arcs)throw std::logic_error("binary arc inventory differs");
        const U nv=static_cast<U>(vertices);
        std::vector<U> off(size_t(nv)+1,0),rev(size_t(nv)+1,0);
        for(const auto& e:edges) {++off[e.first+1];++rev[e.second+1];}
        for(U v=0;v<nv;++v) {off[v+1]+=off[v];rev[v+1]+=rev[v];}
        std::vector<U> next(off),rnext(rev),to(arcs),from(arcs);
        for(const auto& e:edges) {to[next[e.first]++]=e.second;from[rnext[e.second]++]=e.first;}
        std::vector<uint8_t> visited(nv,0);
        std::vector<U> order,component(nv,NONE);order.reserve(nv);
        std::vector<std::pair<U,U>> stack;stack.reserve(nv);
        for(U root=0;root<nv;++root) if(!visited[root]) {
            visited[root]=1;stack.emplace_back(root,off[root]);
            while(!stack.empty()) {
                auto& f=stack.back();
                if(f.second==off[f.first+1]) {order.push_back(f.first);stack.pop_back();continue;}
                const U u=to[f.second++];
                if(!visited[u]) {visited[u]=1;stack.emplace_back(u,off[u]);}
            }
        }
        U cid=0;
        for(auto it=order.rbegin();it!=order.rend();++it) if(component[*it]==NONE) {
            component[*it]=cid;stack.emplace_back(*it,0);
            while(!stack.empty()) {
                const U v=stack.back().first;stack.pop_back();
                for(U j=rev[v];j<rev[v+1];++j) {
                    const U u=from[j];if(component[u]==NONE) {component[u]=cid;stack.emplace_back(u,0);}
                }
            }
            ++cid;
        }
        for(U v=0;v<x.n;++v) if(component[2*v]==component[2*v+1])return 2;
        // Commit the completed witness only after all SCC contradictions have
        // been excluded. Existing true/false assignments remain untouched.
        for(U v=0;v<x.n;++v) if(!result.value[v]) {
            result.value[v]=component[2*v]>component[2*v+1]?1:-1;
            trace=(trace^(uint64_t(v)+1+(result.value[v]>0?0:uint64_t(x.n))))*1099511628211ULL;
        }
        ++result.binary_solved;
        return 1;
    }

    Answer run(const std::vector<int32_t>& assumptions) {
        uint64_t peak_state=result.state_bytes;
        for(const int32_t a:assumptions) {const int64_t lit=a;enqueue(static_cast<U>((lit<0?-lit:lit)-1),lit>0?1:-1);}
        for(U g=0;g<x.p;++g) force_unit(g);
        while(true) {
            U status=propagate();
            if(status==2) {result.reason=1;break;}
            if(status==1) {
                Failure failure=jump_target();bool found=false;
                while(failure.top) {
                    if(failure.top>frames.size()||failure.prior>=failure.top)
                        throw std::logic_error("invalid conservative conflict prefix");
                    if(failure.top<frames.size()) {
                        ++result.jumps;result.skipped+=frames.size()-failure.top;
                        frames.resize(failure.top);
                    }
                    Frame& f=frames.back();rollback(f.mark);++backtracks;
                    if(!f.second) {
                        f.second=true;if(jump_mode)first_prior[frames.size()-1]=failure.prior;
                        if(exact_mode) {
                            causal[(failure.top-1)/64]&=~(uint64_t(1)<<((failure.top-1)%64));
                            std::copy(causal.begin(),causal.end(),first_causes.begin()+(frames.size()-1)*cause_words);
                        }
                        enqueue(f.v,-1,NONE,{1,static_cast<U>(frames.size())});found=true;break;
                    }
                    // Both values have failed. Resolving their two explanations
                    // eliminates only this decision; retain the entire union's
                    // older prefix. No single-child dependency is reused as a
                    // certificate for the other child.
                    U k=std::max(failure.prior,jump_mode?first_prior[frames.size()-1]:static_cast<U>(frames.size()-1));
                    if(exact_mode) {
                        causal[(failure.top-1)/64]&=~(uint64_t(1)<<((failure.top-1)%64));
                        k=0;
                        for(size_t w=0;w<cause_words;++w) {
                            causal[w]|=first_causes[(frames.size()-1)*cause_words+w];
                            if(causal[w])k=static_cast<U>(64*w+64-__builtin_clzll(causal[w]));
                        }
                    }
                    frames.pop_back();failure={k,k?k-1:0};
                }
                if(!found){result.reason=2;break;}
                continue;
            }
            if(!unsatisfied) {result.reason=0;break;}
            if(work==limit) {result.reason=1;break;}
            if(binary_mode) {
                // result.state_bytes reports the largest simultaneous allocation,
                // while the budget test uses only the live ordinary search banks.
                const uint64_t live=result.state_bytes;
                const U b=binary_residual();
                peak_state=std::max(peak_state,result.state_bytes);result.state_bytes=live;
                if(b==1){result.reason=0;break;}
                if(b==2){conflict=true;continue;}
                if(b==3){result.reason=1;break;}
            }
            const U g=choose();
            if(g==NONE||!left[g]) {conflict=true;continue;}
            const U v=choose_variable(g);
            if(v==NONE)throw std::logic_error("remaining count disagrees with incidence");
            ++work;++nodes;frames.push_back({trail.size(),v,false});enqueue(v,1,NONE,{1,static_cast<U>(frames.size())});
        }
        result.state_bytes=peak_state;
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
        const U flags=static_cast<U>(integer(mode,"search flags",1023));
        if((flags&3)==3)throw Invalid("heap and active strategies cannot be combined");
        if((flags&512)&&!(flags&256))throw Invalid("exact conflict sets require reason-directed backjumping");
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
        return Py_BuildValue("(NIKKKKKKKKKKKKKK)",witness,static_cast<unsigned int>(r.reason),
            static_cast<unsigned long long>(r.work),static_cast<unsigned long long>(r.nodes),
            static_cast<unsigned long long>(r.forced),static_cast<unsigned long long>(r.backtracks),
            static_cast<unsigned long long>(r.index_bytes),static_cast<unsigned long long>(r.state_bytes),
            static_cast<unsigned long long>(r.trace),static_cast<unsigned long long>(r.binary_calls),
            static_cast<unsigned long long>(r.binary_edges),static_cast<unsigned long long>(r.binary_solved),
            static_cast<unsigned long long>(r.jumps),static_cast<unsigned long long>(r.skipped),static_cast<unsigned long long>(r.analysis),static_cast<unsigned long long>(r.conflict_bytes));
    }catch(...){return error();}
}
// Independent observer: reads original Python groups and the returned Boolean
// tuple, never a compiled index, propagation counter, search trail or SCC graph.
PyObject* original_check(PyObject*,PyObject* args) {
    PyObject *nv,*covers,*exclusive,*witness,*assumptions;
    if(!PyArg_ParseTuple(args,"OOOOO",&nv,&covers,&exclusive,&witness,&assumptions))return nullptr;
    try {
        const uint64_t n=integer(nv,"nvars",MAX_VARS);
        if(!PyTuple_CheckExact(covers)||!PyTuple_CheckExact(exclusive)||
           !PyTuple_CheckExact(witness)||!PyTuple_CheckExact(assumptions))Py_RETURN_FALSE;
        if(static_cast<uint64_t>(PyTuple_GET_SIZE(witness))!=n)Py_RETURN_FALSE;
        for(uint64_t v=0;v<n;++v) if(!PyBool_Check(PyTuple_GET_ITEM(witness,v)))Py_RETURN_FALSE;
        const uint64_t nc=PyTuple_GET_SIZE(covers),ne=PyTuple_GET_SIZE(exclusive);
        if(nc+ne>MAX_GROUPS)Py_RETURN_FALSE;
        uint64_t entries=0;
        for(unsigned bank=0;bank<2;++bank) {
            PyObject* rows=bank?exclusive:covers;
            for(Py_ssize_t g=0;g<PyTuple_GET_SIZE(rows);++g) {
                PyObject* row=PyTuple_GET_ITEM(rows,g);
                if(!PyTuple_CheckExact(row))Py_RETURN_FALSE;
                entries+=static_cast<uint64_t>(PyTuple_GET_SIZE(row));
                if(entries>MAX_ENTRIES)Py_RETURN_FALSE;
                unsigned true_count=0;
                for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(row);++j) {
                    const uint64_t v=integer(PyTuple_GET_ITEM(row,j),"group variable",n);
                    if(!v)Py_RETURN_FALSE;
                    if(PyTuple_GET_ITEM(witness,v-1)==Py_True)++true_count;
                    if(bank&&true_count>1)Py_RETURN_FALSE;
                }
                if(!bank&&!true_count)Py_RETURN_FALSE;
            }
        }
        if(static_cast<uint64_t>(PyTuple_GET_SIZE(assumptions))>2*n)Py_RETURN_FALSE;
        for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(assumptions);++j) {
            PyObject* a=PyTuple_GET_ITEM(assumptions,j);
            if(!PyLong_CheckExact(a))Py_RETURN_FALSE;
            const long long v=PyLong_AsLongLong(a);
            if(PyErr_Occurred()) {PyErr_Clear();Py_RETURN_FALSE;}
            if(!v||v>static_cast<int64_t>(n)||v<-static_cast<int64_t>(n))Py_RETURN_FALSE;
            if((PyTuple_GET_ITEM(witness,(v>0?v:-v)-1)==Py_True)!=(v>0))Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    }catch(const Invalid&){Py_RETURN_FALSE;}
    catch(...){return error();}
}

PyObject* abi(PyObject*,PyObject*) {return PyLong_FromLong(9);}
PyMethodDef methods[]={{"check",original_check,METH_VARARGS,"Observe original groups and a Boolean witness independently of search."},{"abi",abi,METH_NOARGS,"Native API version."},{"create",create,METH_VARARGS,"Create immutable sparse constraint index."},{"solve",solve,METH_VARARGS,"Solve with an independent reversible state."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_sparse",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_sparse(){return PyModule_Create(&module);}
