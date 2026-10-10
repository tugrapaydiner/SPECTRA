// SPECTRA cover/exclusion search. MIT, see repository LICENSE.
// Opt-in exact Boolean specialization, not a general CNF or UNSAT-proof engine.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
using Word = uint64_t;
constexpr uint64_t MAX_VARS = 65536, MAX_CLAUSES = 1000000, MAX_LITERALS = 16000000;
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };
void error(char* dst, size_t cap, const char* message) noexcept {
    if (dst && cap) { const size_t n = std::min(cap-1, std::strlen(message));
        std::memcpy(dst, message, n); dst[n] = 0; }
}
unsigned pop(Word x) { return static_cast<unsigned>(__builtin_popcountll(x)); }
unsigned low(Word x) { return static_cast<unsigned>(__builtin_ctzll(x)); }
uint32_t var(int32_t lit) { return static_cast<uint32_t>((lit < 0 ? -int64_t(lit) : lit)-1); }

// 0: tautology; 1: positive cover (including empty); 2: negative unit/pair.
// Literal bounds have already been checked. Mixed tautologies are accepted;
// every other mixed clause is explicitly unsupported, never silently dropped.
int kind(const int32_t* begin, const int32_t* end) {
    bool pos = false, neg = false;
    for (auto p=begin; p!=end; ++p) { pos |= *p > 0; neg |= *p < 0; }
    if (!neg) return 1;
    if (!pos && end-begin <= 2) return 2;
    std::vector<int32_t> unique(begin,end);
    std::sort(unique.begin(),unique.end());
    unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
    if (pos && neg) {
        for (const int32_t x : unique) if (x<0 && std::binary_search(unique.begin(),unique.end(),-x)) return 0;
        throw Unsupported("non-tautological mixed-sign clause is outside cover/exclusion contract");
    }
    if (unique.size() <= 2) return 2;
    throw Unsupported("negative clause has more than two distinct variables");
}

struct Index {
    uint32_t n; size_t words, groups, gwords; uint64_t payload = 0; bool incremental;
    std::vector<Word> conflicts, covers, memberships, available;
    bool contradiction = false;
    Index(uint32_t nv, uint64_t m, const uint64_t* offsets,
          const int32_t* literals, uint64_t nl, uint64_t budget, uint32_t cached, uint64_t positive_groups=UINT64_MAX) : n(nv), incremental(cached!=0) {
        if (cached>1) throw std::invalid_argument("invalid incremental flag");
        if (nv>MAX_VARS || m>MAX_CLAUSES || nl>MAX_LITERALS)
            throw Resource("native cover input exceeds declared geometry limits");
        if (!offsets || (nl && !literals) || offsets[0]!=0 || offsets[m]!=nl)
            throw std::invalid_argument("invalid clause buffers");
        // Make a non-null base even for an empty literal bank; null arithmetic is UB.
        static constexpr int32_t dummy = 0;
        if (!literals) literals = &dummy;
        for (uint64_t i=0; i<m; ++i)
            if (offsets[i]>offsets[i+1] || offsets[i+1]>nl)
                throw std::invalid_argument("nonmonotone clause offsets");
        for (uint64_t i=0; i<nl; ++i)
            if (!literals[i] || int64_t(literals[i])>nv || int64_t(literals[i])<-int64_t(nv))
                throw std::invalid_argument("literal outside declared variables");
        const bool direct=positive_groups!=UINT64_MAX;
        if (direct && positive_groups>m) throw std::invalid_argument("invalid positive group count");
        words = (size_t(n)+63)/64;
        // Admission precedes large bit-matrix allocations. Types occupy one byte/clause.
        std::vector<uint8_t> types; types.reserve(static_cast<size_t>(m));
        groups=0;
        for (uint64_t i=0; i<m; ++i) {
            int t;
            if (direct) {
                for (uint64_t j=offsets[i];j<offsets[i+1];++j)
                    if (literals[j]<1) throw std::invalid_argument("choice groups require positive indices");
                t=i<positive_groups ? 1 : 3;
            } else t=kind(literals+offsets[i],literals+offsets[i+1]);
            types.push_back(static_cast<uint8_t>(t)); groups += t==1;
        }
        gwords=(groups+63)/64;
        const uint64_t required=8*(uint64_t(n)*words+uint64_t(groups)*words+uint64_t(n)*gwords+words);
        if (required>budget) throw Resource("compiled bit matrices exceed max_index_bytes");
        conflicts.assign(size_t(n)*words,0); covers.assign(groups*words,0);
        memberships.assign(size_t(n)*gwords,0); available.assign(words,~Word(0));
        if (words && n%64) available.back()=(Word(1)<<(n%64))-1;
        size_t group=0;
        for (uint64_t i=0; i<m; ++i) {
            const auto first=literals+offsets[i], last=literals+offsets[i+1];
            if (types[i]==1) {
                if (first==last) contradiction=true;
                for (auto p=first; p!=last; ++p) {
                    uint32_t v=var(*p);
                    covers[group*words+v/64] |= Word(1)<<(v%64);
                    memberships[size_t(v)*gwords+group/64] |= Word(1)<<(group%64);
                }
                ++group;
            } else if (types[i]==3) {
                std::vector<Word> mask(words,0);
                for (auto p=first;p!=last;++p) mask[var(*p)/64] |= Word(1)<<(var(*p)%64);
                for (auto p=first;p!=last;++p) {
                    const uint32_t v=var(*p);
                    for (size_t w=0;w<words;++w) conflicts[size_t(v)*words+w] |= mask[w];
                    conflicts[size_t(v)*words+v/64] &= ~(Word(1)<<(v%64));
                }
            } else if (types[i]==2) {
                const uint32_t a=var(*first);
                uint32_t b=a;
                for (auto p=first+1; p!=last; ++p) if (var(*p)!=a) b=var(*p);
                if (a==b) available[a/64] &= ~(Word(1)<<(a%64));
                else {
                    conflicts[size_t(a)*words+b/64] |= Word(1)<<(b%64);
                    conflicts[size_t(b)*words+a/64] |= Word(1)<<(a%64);
                }
            }
        }
        payload=8*(conflicts.capacity()+covers.capacity()+memberships.capacity()+available.capacity());
    }
};
struct State { std::vector<Word> available, selected, uncovered; std::vector<uint32_t> counts; };
void mix(uint64_t& trace,uint64_t event) { trace ^= event; trace *= UINT64_C(1099511628211); }
bool any(const std::vector<Word>& x) { for (Word w:x) if (w) return true; return false; }
void remove_variable(const Index& p,State& s,uint32_t v) {
    const Word bit=Word(1)<<(v%64);
    if (!(s.available[v/64]&bit)) return;
    s.available[v/64]&=~bit;
    if (!p.incremental) return;
    for (size_t gw=0;gw<p.gwords;++gw) {
        Word affected=p.memberships[size_t(v)*p.gwords+gw]&s.uncovered[gw];
        while (affected) { const size_t g=64*gw+low(affected);affected&=affected-1;
            --s.counts[g]; }
    }
}
void execute(const Index& p,uint64_t limit,uint64_t state_budget,uint8_t* witness,uint64_t* stats) {
    // stats: nodes, decisions, propagations, backtracks, reason, payload,
    //        maximum explicitly counted stack/state word bytes, diagnostic FNV64.
    std::fill(stats,stats+8,0); stats[5]=p.payload; stats[7]=UINT64_C(14695981039346656037);
    const uint64_t state_bytes=8*(2*p.words+p.gwords)+(p.incremental?4*p.groups:0);
    if (state_bytes>state_budget) throw Resource("root logical search state exceeds max_state_bytes");
    State s{p.available,std::vector<Word>(p.words,0),std::vector<Word>(p.gwords,~Word(0)),std::vector<uint32_t>(p.incremental?p.groups:0,0)};
    for (size_t g=0;p.incremental && g<p.groups;++g)
        for (size_t w=0;w<p.words;++w) s.counts[g]+=pop(p.covers[g*p.words+w]&s.available[w]);
    if (p.groups%64) s.uncovered.back()=(Word(1)<<(p.groups%64))-1;
    std::vector<State> stack;
    stats[6]=state_bytes;
    if (p.contradiction) stats[4]=2; // exhausted, not proof-certified UNSAT
    else while (true) {
        stats[6]=std::max(stats[6],uint64_t(stack.size()+1)*(8*(2*p.words+p.gwords)+(p.incremental?4*p.groups:0)));
        if (!any(s.uncovered)) { stats[4]=0; break; }
        if (stats[0]==limit) { stats[4]=1; break; }
        ++stats[0];
        uint32_t best_size=p.n+1, best_variable=0;
        bool failed=false;
        for (size_t gw=0; gw<p.gwords && !failed; ++gw) {
            Word bits=s.uncovered[gw];
            while (bits) {
                const size_t group=gw*64+low(bits); bits &= bits-1;
                uint32_t count=0;
                if (p.incremental) count=s.counts[group];
                else for (size_t w=0;w<p.words;++w) count+=pop(p.covers[group*p.words+w]&s.available[w]);
                if (!count) { failed=true; break; }
                if (count<best_size) {
                    best_size=count;
                    for (size_t w=0;w<p.words;++w) {
                        const Word possible=p.covers[group*p.words+w]&s.available[w];
                        if (possible) { best_variable=static_cast<uint32_t>(64*w+low(possible));break; }
                    }
                }
                // A unit can be processed immediately; contradictions elsewhere are
                // detected at the next state, and cannot produce a false SAT return.
                if (count==1) break;
            }
            if (best_size==1) break;
        }
        if (failed) {
            if (stack.empty()) { stats[4]=2; break; }
            s=std::move(stack.back()); stack.pop_back(); ++stats[3]; mix(stats[7],UINT64_MAX);
            continue;
        }
        const uint32_t v=best_variable; const Word bit=Word(1)<<(v%64);
        if (best_size>1) {
            if (state_bytes && uint64_t(stack.size()+2)>state_budget/state_bytes) {
                stats[4]=3;break;
            }
            State alternative=s; remove_variable(p,alternative,v);
            stack.push_back(std::move(alternative)); ++stats[1];
        } else ++stats[2];
        mix(stats[7],uint64_t(v)+1);
        s.selected[v/64] |= bit;
        for (size_t w=0; w<p.gwords; ++w) s.uncovered[w] &= ~p.memberships[size_t(v)*p.gwords+w];
        for (size_t w=0;w<p.words;++w) {
            Word removed=s.available[w]&p.conflicts[size_t(v)*p.words+w];
            while (removed) { const uint32_t z=static_cast<uint32_t>(64*w+low(removed));
                removed&=removed-1;remove_variable(p,s,z); }
        }
        remove_variable(p,s,v);
    }
    for (uint32_t v=0; v<p.n; ++v) witness[v]=static_cast<uint8_t>((s.selected[v/64]>>(v%64))&1);
}
}
API uint32_t spectra_cover_abi() { return 3; }
API int spectra_cover_create(uint32_t n,uint64_t m,const uint64_t* offsets,
    const int32_t* literals,uint64_t nl,uint64_t budget,uint32_t cached,void** out,char* err,size_t cap) noexcept {
    if (!out) { error(err,cap,"null output handle"); return 1; } *out=nullptr;
    try { auto p=std::make_unique<Index>(n,m,offsets,literals,nl,budget,cached); *out=p.release(); return 0; }
    catch (const Unsupported& e) { error(err,cap,e.what()); return 2; }
    catch (const Resource& e) { error(err,cap,e.what()); return 3; }
    catch (const std::bad_alloc&) { error(err,cap,"native allocation failed"); return 3; }
    catch (const std::exception& e) { error(err,cap,e.what()); return 1; }
    catch (...) { error(err,cap,"unknown native exception"); return 1; }
}
API int spectra_choices_create(uint32_t n,uint64_t m,uint64_t positive_groups,const uint64_t* offsets,
    const int32_t* literals,uint64_t nl,uint64_t budget,uint32_t cached,void** out,char* err,size_t cap) noexcept {
    if (!out) { error(err,cap,"null output handle"); return 1; } *out=nullptr;
    try { auto p=std::make_unique<Index>(n,m,offsets,literals,nl,budget,cached,positive_groups); *out=p.release(); return 0; }
    catch (const Resource& e) { error(err,cap,e.what()); return 3; }
    catch (const std::bad_alloc&) { error(err,cap,"native allocation failed"); return 3; }
    catch (const std::exception& e) { error(err,cap,e.what()); return 1; }
    catch (...) { error(err,cap,"unknown native exception"); return 1; }
}
API int spectra_cover_solve(const void* handle,uint64_t limit,uint64_t state_budget,uint8_t* witness,
    uint64_t* stats,char* err,size_t cap) noexcept {
    if (!handle || !stats) { error(err,cap,"invalid search buffers"); return 1; }
    const auto& p=*static_cast<const Index*>(handle);
    if (p.n && !witness) { error(err,cap,"null witness buffer"); return 1; }
    try { execute(p,limit,state_budget,witness,stats); return 0; }
    catch (const Resource& e) { error(err,cap,e.what()); return 3; }
    catch (const std::bad_alloc&) { error(err,cap,"search allocation failed"); return 3; }
    catch (const std::exception& e) { error(err,cap,e.what()); return 1; }
    catch (...) { error(err,cap,"unknown search exception"); return 1; }
}
API void spectra_cover_destroy(void* p) noexcept { delete static_cast<Index*>(p); }
