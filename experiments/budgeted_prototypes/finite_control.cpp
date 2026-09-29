// SAME selected SVM; unchanged prior finite/adaptive executor with in-timer input conversion.
#ifndef BP_FINITE_RUNTIME
#error BP_FINITE_RUNTIME required
#endif
#include BP_FINITE_RUNTIME
namespace fc {
struct Model{fk::Engine engine;int maximum;Model(const unsigned char*x,uint64_t n,int D):engine(x,n),maximum(D){lo::require(D>=1&&D<=255,"invalid code range");}};
}
extern "C" {
int ct_abi(){return 1;}
void* ct_svm_create(const unsigned char*x,uint64_t n,int D){void*p=nullptr;lo::protect([&]{p=new fc::Model(x,n,D);});return p;}
void ct_svm_destroy(void*p){delete static_cast<fc::Model*>(p);}
int ct_svm_info(void*p,uint64_t*out,int n){return lo::protect([&]{lo::require(p&&out&&n==5,"invalid finite info");auto&m=*static_cast<fc::Model*>(p);auto&e=m.engine;uint64_t v[]={uint64_t(e.m.d),uint64_t(e.m.c),uint64_t(e.m.nsv),e.m.storage()+e.added_storage()+e.prior.extra_bytes(),e.prior.exact.scratch()+8ull*e.m.d};std::copy(v,v+5,out);});}
int ct_svm_run(void*p,const uint8_t*q,int rows,int d,int mode,int*out){return lo::protect([&]{
 lo::require(p,"closed finite SVM");auto&m=*static_cast<fc::Model*>(p);auto&e=m.engine;ak::environment();
 lo::require(rows>=0&&rows<=65536&&d==e.m.d&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=1&&(!rows||(q&&out)),"invalid finite buffers");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(q[i]<=m.maximum,"invalid raw code");
 std::vector<double>x(size_t(d),0.);
 for(int r=0;r<rows;++r){for(int f=0;f<d;++f)x[f]=double(q[size_t(r)*d+f])/double(m.maximum);out[r]=e.run(x.data(),mode==0?1:3);}
});}
}
