// Isolated native residual-patch development engine. No legacy runtime changes.
// Stochastic search inspired by probSAT/WalkSAT, not either authors' implementation.
// C ABI is used only by the checked Python owner in runtime.py; pointers are trusted.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rp {
constexpr int F=16, H=16, W=F*H+H+H+1, CAP=24;
using Clock=std::chrono::steady_clock;
thread_local std::string error;
struct RNG {
  uint64_t s;
  uint64_t next() {
    uint64_t z=(s+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
  }
  uint64_t below(uint64_t n) {
    if (!n) throw std::invalid_argument("empty random range");
    uint64_t threshold=-n % n, x;
    do { x=next(); } while(x<threshold);
    return x%n;
  }
  double unit(){return (next()>>11)*0x1.0p-53;}
};
struct Work {
  int64_t moves=0, flips=0, updates=0, feature_visits=0, proposals=0,
          model_calls=0, macs=0, restarts=0, literal_scans=0;
};
struct Patch {int u=-1,v=-1; std::array<double,F> f{};};
struct State {
  int n, m;
  std::vector<std::vector<int>> clauses, occ;
  std::vector<uint8_t> a;
  std::vector<int> counts, unsat, pos, deltas, touched;
  std::vector<uint32_t> marks;
  uint32_t stamp=0;
  std::vector<int64_t> age;
  Work work;
  explicit State(int nv,int mc,const int32_t* offsets,const int32_t* lits,int nl,const uint8_t* assignment):n(nv),m(mc) {
    if(n<0||n>4096||m<0||m>65536||nl<0||nl>1000000||!offsets||!assignment||!lits)
      throw std::invalid_argument("invalid native geometry/buffer");
    if(offsets[0]!=0||offsets[m]!=nl)throw std::invalid_argument("invalid clause offsets");
    clauses.resize(m); occ.resize(n); a.assign(assignment,assignment+n);
    for(auto v:a)if(v>1)throw std::invalid_argument("nonboolean assignment");
    counts.assign(m,0);pos.assign(m,-1);deltas.assign(m,0);marks.assign(m,0);age.assign(n,0);
    for(int c=0;c<m;c++){
      if(offsets[c]<0||offsets[c]>offsets[c+1]||offsets[c+1]>nl)throw std::invalid_argument("nonmonotone offsets");
      for(int j=offsets[c];j<offsets[c+1];j++){
        int64_t l=lits[j]; if(l==0||l>n||l< -int64_t(n))throw std::invalid_argument("literal outside geometry");
        clauses[c].push_back(int(l));
      }
      auto& row=clauses[c];std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
      for(int l:row){occ[std::abs(l)-1].push_back(l>0?c+1:-c-1);counts[c]+=true_lit(l);}
      if(counts[c]==0){pos[c]=int(unsat.size());unsat.push_back(c);}
    }
  }
  bool true_lit(int l)const{return bool(a[std::abs(l)-1])==(l>0);}
  void begin_changes(){
    if(++stamp==0){std::fill(marks.begin(),marks.end(),0);stamp=1;}touched.clear();
  }
  void add_changes(int v,bool feature){
    if(v<0||v>=n)throw std::invalid_argument("variable outside geometry");
    for(int o:occ[v]){
      int c=std::abs(o)-1;
      if(marks[c]!=stamp){marks[c]=stamp;deltas[c]=0;touched.push_back(c);}
      deltas[c]+=(bool(a[v])==(o>0))?-1:1;
    }
    if(feature)work.feature_visits+=int64_t(occ[v].size());else work.updates+=int64_t(occ[v].size());
  }
  std::pair<int,int> makebreak(int u,int v=-1){
    if(u==v)throw std::invalid_argument("duplicate patch variable");
    begin_changes();add_changes(u,true);if(v>=0)add_changes(v,true);
    int make=0,brk=0;
    for(int c:touched){make+=counts[c]==0&&counts[c]+deltas[c]>0;brk+=counts[c]>0&&counts[c]+deltas[c]==0;}
    return {make,brk};
  }
  void flip(int u,int v=-1){
    // Validate entire patch BEFORE mutation; no intermediate accepted assignment.
    if(u<0||u>=n||v< -1||v>=n||u==v)throw std::invalid_argument("invalid atomic patch");
    begin_changes();add_changes(u,false);if(v>=0)add_changes(v,false);
    for(int c:touched){
      int old=counts[c];counts[c]+=deltas[c];
      if(counts[c]<0||counts[c]>int(clauses[c].size()))throw std::logic_error("clause count corruption");
      if(old==0&&counts[c]>0){int i=pos[c],last=unsat.back();unsat[i]=last;pos[last]=i;unsat.pop_back();pos[c]=-1;}
      else if(old>0&&counts[c]==0){pos[c]=int(unsat.size());unsat.push_back(c);}
    }
    a[u]^=1;if(v>=0)a[v]^=1;
    work.moves++;work.flips+=1+(v>=0);age[u]=work.moves;if(v>=0)age[v]=work.moves;
  }
  bool check(){
    for(auto& row:clauses){bool ok=false;for(int l:row){work.literal_scans++;ok|=true_lit(l);}if(!ok)return false;}return true;
  }
  void randomize(RNG& rng){
    unsat.clear();std::fill(pos.begin(),pos.end(),-1);
    for(auto& v:a)v=uint8_t(rng.below(2));
    for(int c=0;c<m;c++){counts[c]=0;for(int l:clauses[c]){counts[c]+=true_lit(l);work.literal_scans++;}
      if(!counts[c]){pos[c]=int(unsat.size());unsat.push_back(c);}}
    std::fill(age.begin(),age.end(),work.moves);work.restarts++;
  }
  bool step(RNG& rng,const std::vector<double>& weights,int mode){
    if(unsat.empty())return false;
    const auto& row=clauses[unsat[rng.below(unsat.size())]];
    if(row.empty())return false;
    std::vector<int> vs,br;
    for(int l:row){int v=std::abs(l)-1;if(std::find(vs.begin(),vs.end(),v)==vs.end())vs.push_back(v);}
    for(int v:vs)br.push_back(makebreak(v).second);
    int picked=0;
    if(mode==1){
      int best=*std::min_element(br.begin(),br.end());
      if(best>0&&rng.unit()<0.5)picked=int(rng.below(vs.size()));
      else {std::vector<int> ties;for(size_t i=0;i<vs.size();i++)if(br[i]==best)ties.push_back(int(i));picked=ties[rng.below(ties.size())];}
    }else{
      double total=0;for(int b:br)total+=weights[b];double r=rng.unit()*total;
      picked=int(vs.size())-1;for(size_t i=0;i<vs.size();i++){r-=weights[br[i]];if(r<0){picked=int(i);break;}}
    }
    flip(vs[picked]);return true;
  }
  std::array<double,F> features(int u,int v){
    auto mb=makebreak(u,v),x=makebreak(u),y=v>=0?makebreak(v):std::pair<int,int>{0,0};
    double du=double(occ[u].size()),dv=v>=0?double(occ[v].size()):0;
    double au=std::min<int64_t>(work.moves-age[u],256)/256.0;
    double av=v>=0?std::min<int64_t>(work.moves-age[v],256)/256.0:au;
    return {n/512.0, m/double(std::max(n,1)*8), unsat.size()/double(std::max(m,1)),
      std::log1p(double(unsat.size()))/8, (1+(v>=0))/2.0, mb.first/8.0,mb.second/8.0,
      (mb.second-mb.first)/8.0,std::min(x.first,y.first)/8.0,std::max(x.first,y.first)/8.0,
      std::min(x.second,y.second)/8.0,std::max(x.second,y.second)/8.0,
      (mb.second-mb.first-(x.second-x.first+y.second-y.first))/8.0,
      (du+dv)/64.0,std::min(au,av),std::max(au,av)};
  }
  std::vector<Patch> pool(RNG& rng,int feature_mode=2){
    std::vector<Patch> pool;if(unsat.empty())return pool;
    const auto& row=clauses[unsat[rng.below(unsat.size())]];if(row.empty())return pool;
    auto add=[&](int u,int v){
      if(v>=0&&v<u)std::swap(u,v);if(u==v||int(pool.size())>=CAP)return;
      for(auto& p:pool)if(p.u==u&&p.v==v)return;
      pool.push_back({u,v,{}});
    };
    std::vector<int> first;
    for(int l:row){int v=std::abs(l)-1;if(std::find(first.begin(),first.end(),v)==first.end())first.push_back(v);}
    // Include every focused single first, then repairs of clauses a first flip breaks.
    for(int u:first)add(u,-1);
    for(int u:first){
      std::vector<int> broken;
      for(int o:occ[u]){work.feature_visits++;int c=std::abs(o)-1;if(counts[c]==1&&bool(a[u])==(o>0))broken.push_back(c);}
      // Three distinct broken clauses, or up to three arbitrary incident clauses.
      if(broken.empty())for(int o:occ[u])broken.push_back(std::abs(o)-1);
      for(int k=0;k<3&&!broken.empty();k++){
        int j=int(rng.below(broken.size())),c=broken[j];broken[j]=broken.back();broken.pop_back();
        for(int l:clauses[c]){work.feature_visits++;int v=std::abs(l)-1;if(v!=u)add(u,v);}
      }
    }
    for(auto& p:pool){
      if(feature_mode==2)p.f=features(p.u,p.v);
      else if(feature_mode==1){auto mb=makebreak(p.u,p.v);p.f[5]=mb.first/8.0;p.f[6]=mb.second/8.0;}
    }
    work.proposals+=int64_t(pool.size());return pool;
  }
};
std::vector<double> weights(int m,double cb){std::vector<double>w(m+1);for(int i=0;i<=m;i++)w[i]=std::pow(1.0+i,-cb);return w;}
double score(const std::array<double,F>& x,const double* p){
  double y=p[W-1];for(int h=0;h<H;h++){double z=p[F*H+h];for(int f=0;f<F;f++)z+=p[h*F+f]*x[f];y+=p[F*H+H+h]*std::max(0.0,z);}return y;
}
int choose(State& st,std::vector<Patch>& pool,RNG& rng,int mode,const double* model){
  if(mode==2)return int(rng.below(pool.size()));
  double best=-std::numeric_limits<double>::infinity();std::vector<int> ties;
  for(size_t i=0;i<pool.size();i++){
    double value;
    if(mode==3)value=pool[i].f[5]-pool[i].f[6];
    else {value=score(pool[i].f,model);st.work.macs+=F*H+H;}
    if(value>best){best=value;ties.clear();ties.push_back(int(i));}
    else if(value==best)ties.push_back(int(i));
  }
  if(mode>=4)st.work.model_calls++;
  return ties[rng.below(ties.size())];
}
void copywork(const Work& w,int64_t* out){
  out[0]=w.moves;out[1]=w.flips;out[2]=w.updates;out[3]=w.feature_visits;out[4]=w.proposals;
  out[5]=w.model_calls;out[6]=w.macs;out[7]=w.restarts;out[8]=w.literal_scans;
}
} // namespace rp

extern "C" {
const char* rp_error(){return rp::error.c_str();}
int rp_dimensions(int* out){out[0]=rp::F;out[1]=rp::H;out[2]=rp::W;out[3]=rp::CAP;return 0;}
void* rp_create(int n,int m,const int32_t* offsets,const int32_t* lits,int nl,const uint8_t* a){
  try {rp::error.clear();return new rp::State(n,m,offsets,lits,nl,a);}catch(const std::exception&e){rp::error=e.what();return nullptr;}}
void rp_destroy(void* p){delete static_cast<rp::State*>(p);}
int rp_inspect(void* ptr,uint8_t* a,int32_t* counts,int64_t* work){try{auto& s=*static_cast<rp::State*>(ptr);
  std::copy(s.a.begin(),s.a.end(),a);std::copy(s.counts.begin(),s.counts.end(),counts);rp::copywork(s.work,work);return int(s.unsat.size());
  }catch(const std::exception&e){rp::error=e.what();return -1;}}
int rp_patch(void* ptr,int u,int v,int32_t* mb){try{auto& s=*static_cast<rp::State*>(ptr);
  if(u<0||u>=s.n||v< -1||v>=s.n||u==v)throw std::invalid_argument("invalid patch");
  auto out=s.makebreak(u,v);mb[0]=out.first;mb[1]=out.second;s.flip(u,v);return 0;
  }catch(const std::exception&e){rp::error=e.what();return -1;}}
int rp_features(void* ptr,int u,int v,double* out){try{auto& s=*static_cast<rp::State*>(ptr);
  if(u<0||u>=s.n||v< -1||v>=s.n||u==v)throw std::invalid_argument("invalid patch");
  auto f=s.features(u,v);std::copy(f.begin(),f.end(),out);return 0;
  }catch(const std::exception&e){rp::error=e.what();return -1;}}
int rp_pool(void* ptr,uint64_t seed,int32_t* pairs,double* features){try{auto& s=*static_cast<rp::State*>(ptr);rp::RNG rng{seed};auto p=s.pool(rng);
  for(size_t i=0;i<p.size();i++){pairs[i*2]=p[i].u;pairs[i*2+1]=p[i].v;std::copy(p[i].f.begin(),p[i].f.end(),features+i*rp::F);}return int(p.size());
  }catch(const std::exception&e){rp::error=e.what();return -1;}}
int rp_scores(const double* features,int count,const double* weights,double* out){try{
  if(count<0||count>100000)throw std::invalid_argument("invalid score inventory");
  for(int i=0;i<count;i++){std::array<double,rp::F>x;std::copy(features+i*rp::F,features+(i+1)*rp::F,x.begin());out[i]=rp::score(x,weights);}return 0;
  }catch(const std::exception&e){rp::error=e.what();return -1;}}
int rp_run(void* ptr,uint64_t seed,int mode,int64_t max_moves,int64_t time_ns,int interval,int restart,const double* model,double cb,int64_t* report){
 try{
  auto& s=*static_cast<rp::State*>(ptr);
  if(mode<0||mode>4||max_moves<0||max_moves>10000000||time_ns<0||interval<1||restart<0||!std::isfinite(cb)||cb<=0||cb>20)
    throw std::invalid_argument("invalid search configuration");
  if(mode==4){if(!model)throw std::invalid_argument("missing model");for(int k=0;k<rp::W;k++)if(!std::isfinite(model[k]))throw std::invalid_argument("nonfinite model");}
  auto start=rp::Clock::now();rp::RNG rng{seed};auto w=mode==1?std::vector<double>{}:rp::weights(s.m,cb);
  int64_t base=s.work.moves;int status=0;
  while(s.work.moves-base<max_moves&&!s.unsat.empty()){
    auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(rp::Clock::now()-start).count();
    if(time_ns&&elapsed>=time_ns){status=2;break;}
    int64_t k=s.work.moves-base;
    if(restart>0&&k>0&&k%restart==0)s.randomize(rng);
    if(s.unsat.empty())break;
    if(mode>=2&&k>0&&k%interval==0){auto pool=s.pool(rng,mode==2?0:mode==3?1:2);if(pool.empty())break;
      int chosen=rp::choose(s,pool,rng,mode,model);s.flip(pool[chosen].u,pool[chosen].v);
    }else if(!s.step(rng,w,mode==1?1:0))break;
  }
  if(s.unsat.empty()){if(!s.check())throw std::logic_error("false native SAT witness");status=1;}
  rp::copywork(s.work,report);report[9]=std::chrono::duration_cast<std::chrono::nanoseconds>(rp::Clock::now()-start).count();
  report[10]=s.unsat.size();report[11]=status;return status;
 }catch(const std::exception&e){rp::error=e.what();return -1;}}
}
