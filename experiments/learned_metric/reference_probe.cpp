// Test-only reference: unmodified binary64 SVM on repeated integer coordinates.
#ifndef LM_BASE_RUNTIME
#error LM_BASE_RUNTIME required
#endif
#include LM_BASE_RUNTIME
struct Reference {spm::Owner model;spm::Worker worker;Reference(const unsigned char*b,size_t n):model(std::make_shared<const spm::Model>(b,n,false)),worker(model){}};
extern "C" void* ref_create(const unsigned char*raw,uint64_t bytes){void*r=nullptr;lo::protect([&]{r=new Reference(raw,bytes);});return r;}
extern "C" void ref_destroy(void*p){delete static_cast<Reference*>(p);}
extern "C" int ref_probe(void* handle,const double* x,int n,int d,double* output,uint64_t count){return lo::protect([&]{
 lo::require(handle && n>=0 && n<=65536,"invalid reference handle/rows");auto&r=*static_cast<Reference*>(handle);
 lo::require(d==r.model->d&&count==uint64_t(n)*r.model->pairs.size()&&(!n||(x&&output)),"reference geometry");
 for(size_t i=0;i<size_t(n)*d;++i)lo::require(std::isfinite(x[i]),"nonfinite reference input");
 for(int row=0;row<n;++row){r.worker.begin(x+size_t(row)*d);r.worker.prepare(r.model->active,x+size_t(row)*d);
  for(size_t k=0;k<r.model->pairs.size();++k){auto&p=r.model->pairs[k];double s=0;
   for(size_t j=0;j<p.ids.size();++j)s+=p.values[j]*r.worker.kernel[p.ids[j]];s+=p.bias;
   lo::require(std::isfinite(s),"nonfinite original score");output[size_t(row)*r.model->pairs.size()+k]=s;}}
});}
