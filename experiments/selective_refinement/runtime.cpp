// Conditional execution of two frozen classifiers; no confidence threshold fitting here.
#ifndef SR_PROTOTYPE
#error Set SR_PROTOTYPE to the canonical prototype runtime.
#endif
#ifndef SR_FINITE
#error Set SR_FINITE to the canonical finite-kernel runtime.
#endif
#include SR_PROTOTYPE
#include SR_FINITE
namespace sr {
struct Engine {
 bp::Model fast;
 fk::Engine strong;
 double threshold, accept_fraction;
 Engine(const unsigned char*f,uint64_t nf,const unsigned char*s,uint64_t ns,double t,double a):
  fast(f,nf),strong(s,ns),threshold(t),accept_fraction(a) {
   lo::require(!std::isnan(t)&&t>=0&&std::isfinite(a)&&a>=0&&a<=1,"invalid routing policy");
   lo::require(fast.d==uint32_t(strong.m.d)&&fast.c==uint32_t(strong.m.c),"classifier geometry mismatch");
 }
 int slow(const uint8_t*q,std::vector<double>&x){
  for(uint32_t j=0;j<fast.d;++j)x[j]=double(q[j])/double(fast.maximum);
  return strong.run(x.data(),1);
 }
 static double blind(const uint8_t*q,uint32_t d){
  uint64_t h=1469598103934665603ull;
  for(uint32_t j=0;j<d;++j){h^=q[j];h*=1099511628211ull;}
  h^=0x9e3779b97f4a7c15ull;h=(h^(h>>30))*0xbf58476d1ce4e5b9ull;
  h=(h^(h>>27))*0x94d049bb133111ebull;h^=h>>31;
  return double(h>>11)*0x1p-53;
 }
};
}
extern "C" {
int sr_abi(){return 1;}
void* sr_create(const unsigned char*f,uint64_t nf,const unsigned char*s,uint64_t ns,double t,double a){
 void*p=nullptr;lo::protect([&]{lo::require(nf<=64ull*1024*1024&&ns<=64ull*1024*1024,"model cap");p=new sr::Engine(f,nf,s,ns,t,a);});return p;
}
void sr_destroy(void*p){delete static_cast<sr::Engine*>(p);}
int sr_info(void*p,uint64_t*out,int n){return lo::protect([&]{
 lo::require(p&&out&&n==5,"invalid refinement info");auto&e=*static_cast<sr::Engine*>(p);
 uint64_t v[]={e.fast.d,e.fast.c,e.fast.p,uint64_t(e.strong.m.nsv),e.fast.storage()+e.strong.m.storage()+e.strong.added_storage()+e.strong.prior.extra_bytes()+e.strong.prior.exact.scratch()-sizeof(spm::Worker)+sizeof(sr::Engine)-sizeof(bp::Model)-sizeof(fk::Engine)};
 std::copy(v,v+5,out);
});}
// modes:0 calibrated,1 cheap only,2 strong only,3 confidence-blind control.
int sr_run(void*p,const uint8_t*q,int rows,int d,int mode,int*out,uint64_t*stats,int count){return lo::protect([&]{
 lo::require(p,"closed refinement engine");auto&e=*static_cast<sr::Engine*>(p);ak::environment();bp::environment();
 lo::require(rows>=0&&rows<=65536&&d==int(e.fast.d)&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=3&&stats&&count==3&&(!rows||(q&&out)),"invalid refinement buffers");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(q[i]<=e.fast.maximum,"input outside declared domain");
 std::fill(stats,stats+3,0);std::vector<uint16_t>x(d);std::vector<double>scores(e.fast.c),slowx(d);
 for(int r=0;r<rows;++r){const uint8_t*row=q+size_t(r)*d;
  if(mode==2||(mode==0&&std::isinf(e.threshold))){out[r]=e.slow(row,slowx);++stats[1];continue;}
  e.fast.score(row,0,scores.data(),x.data());++stats[0];
  int winner=int(std::max_element(scores.begin(),scores.end())-scores.begin());
  bool accept=mode==1;
  if(mode==0){double runner=-INFINITY;for(uint32_t c=0;c<e.fast.c;++c)if(int(c)!=winner)runner=std::max(runner,scores[c]);
   double gap=scores[winner]-runner;lo::require(std::isfinite(gap)&&gap>=0,"nonfinite confidence gap");accept=gap>=e.threshold;
  }else if(mode==3)accept=sr::Engine::blind(row,e.fast.d)<e.accept_fraction;
  if(accept){out[r]=winner;++stats[2];}else{out[r]=e.slow(row,slowx);++stats[1];}
 }
});}
}
