// Matched native baselines. The SVM below uses the unchanged original executor.
// MLP operations preserve input-term order; SIMD lanes are different output units.
#ifndef BP_BASE_RUNTIME
#error BP_BASE_RUNTIME must name the pinned original SPECTRA runtime.cpp
#endif
#include BP_BASE_RUNTIME
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace ct {
void environment(){
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
#if defined(__SSE__)
 lo::require((_mm_getcsr()&0xe040)==0,"SIMD round-to-nearest and gradual underflow required");
#endif
}
struct Layer {uint32_t in,out;std::vector<double> w,b;};
struct Network {
 uint32_t d,c,maximum;std::vector<double> mean,scale;std::vector<Layer> layers;size_t width,parameters=0;
 Network(const unsigned char*raw,uint64_t bytes){
  environment();lo::require(raw&&bytes>=36&&bytes<=64ull*1024*1024,"invalid network byte count");
  lo::require(!std::memcmp(raw,"SPNET001",8),"invalid network magic");
  auto u32=[&](size_t off){uint32_t x;std::memcpy(&x,raw+off,4);return x;};
  d=u32(8);c=u32(12);maximum=u32(16);uint32_t L=u32(20),meta=u32(24),payload=u32(28),crc=u32(32);
  lo::require(d>=1&&d<=256&&c>=2&&c<=128&&maximum>=1&&maximum<=255&&L>=1&&L<=4&&meta>=1&&meta<=65536,"invalid network geometry");
  lo::require(bytes==36ull+payload&&lo::crc32(raw+36,bytes-36)==crc,"network inventory/CRC");
  size_t off=36;
  auto take=[&](void*out,size_t n){lo::require(off<=bytes&&n<=bytes-off,"network bounds");std::memcpy(out,raw+off,n);off+=n;};
  std::vector<uint32_t> dims(L+1);take(dims.data(),4*dims.size());
  lo::require(dims.front()==d&&dims.back()==c,"network dimensions disagree");
  width=0;for(uint32_t n:dims){lo::require(n>=1&&n<=1024,"layer width exceeds cap");width=std::max(width,size_t(n));}
  mean.resize(d);scale.resize(d);take(mean.data(),8*d);take(scale.data(),8*d);
  for(uint32_t j=0;j<d;++j)lo::require(std::isfinite(mean[j])&&std::isfinite(scale[j])&&scale[j]>0,"invalid input scaling");
  for(uint32_t i=0;i<L;++i){Layer a{dims[i],dims[i+1],{}, {}};parameters+=size_t(a.in)*a.out+a.out;
   lo::require(parameters<=4000000,"network parameter cap");
   a.w.resize(size_t(a.in)*a.out);a.b.resize(a.out);take(a.w.data(),8*a.w.size());take(a.b.data(),8*a.b.size());
   for(const auto*v:{&a.w,&a.b})for(double x:*v)lo::require(std::isfinite(x),"nonfinite network weight");
   layers.push_back(std::move(a));
  }
  lo::require(off+meta==bytes,"network trailing bytes");
 }
 void scores(const uint8_t*q,double*a,double*b,int mode)const{
  for(uint32_t f=0;f<d;++f)a[f]=(double(q[f])/double(maximum)-mean[f])/scale[f];
  for(size_t l=0;l<layers.size();++l){const auto&layer=layers[l];uint32_t j=0;
#if defined(__AVX2__)
   if(mode==0){for(;j+4<=layer.out;j+=4){auto sum=_mm256_setzero_pd();
    for(uint32_t i=0;i<layer.in;++i)sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_set1_pd(a[i]),_mm256_loadu_pd(layer.w.data()+size_t(i)*layer.out+j)));
    sum=_mm256_add_pd(sum,_mm256_loadu_pd(layer.b.data()+j));
    if(l+1<layers.size())sum=_mm256_max_pd(sum,_mm256_setzero_pd());
    _mm256_storeu_pd(b+j,sum);
   }}
#endif
   for(;j<layer.out;++j){double sum=0;for(uint32_t i=0;i<layer.in;++i)sum+=a[i]*layer.w[size_t(i)*layer.out+j];sum+=layer.b[j];b[j]=(l+1<layers.size()&&sum<0)?0:sum;}
   for(uint32_t j=0;j<layer.out;++j)lo::require(std::isfinite(b[j]),"nonfinite network activation");
   std::swap(a,b);
  }
  // The caller knows parity and returns the array holding the last layer.
 }
 uint64_t storage()const{uint64_t n=sizeof(*this)+8*(mean.capacity()+scale.capacity())+sizeof(Layer)*layers.capacity();for(const auto&l:layers)n+=8*(l.w.capacity()+l.b.capacity());return n;}
};
struct SVM {
 spm::Owner model;spm::Worker worker;uint32_t maximum;
 SVM(const unsigned char*raw,uint64_t bytes,uint32_t D):model(std::make_shared<const spm::Model>(raw,bytes,false)),worker(model),maximum(D){lo::require(D>=1&&D<=255,"invalid original code range");environment();}
};
}
extern "C" {
int ct_abi(){return 1;}
void* ct_nn_create(const unsigned char*raw,uint64_t bytes){void*p=nullptr;lo::protect([&]{p=new ct::Network(raw,bytes);});return p;}
void ct_nn_destroy(void*p){delete static_cast<ct::Network*>(p);}
int ct_nn_info(void*p,uint64_t*out,int n){return lo::protect([&]{lo::require(p&&out&&n==5,"invalid control info");auto&m=*static_cast<ct::Network*>(p);uint64_t v[]={m.d,m.c,m.parameters,m.storage(),m.width*16};std::copy(v,v+5,out);});}
int ct_nn_run(void*p,const uint8_t*q,int rows,int d,int mode,int*out,double*scores,uint64_t cells){return lo::protect([&]{
 lo::require(p,"closed network");auto&m=*static_cast<ct::Network*>(p);ct::environment();
 lo::require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=1&&(!rows||(q&&out))&&(!scores||cells==uint64_t(rows)*m.c)&&cells<=8000000,"invalid network buffers");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(q[i]<=m.maximum,"control input outside integer range");
 std::vector<double>a(m.width),b(m.width);
 for(int r=0;r<rows;++r){m.scores(q+size_t(r)*d,a.data(),b.data(),mode);const double*final=m.layers.size()%2?b.data():a.data();out[r]=int(std::max_element(final,final+m.c)-final);if(scores)std::copy(final,final+m.c,scores+size_t(r)*m.c);}
});}
void* ct_svm_create(const unsigned char*raw,uint64_t bytes,int maximum){void*p=nullptr;lo::protect([&]{p=new ct::SVM(raw,bytes,uint32_t(maximum));});return p;}
void ct_svm_destroy(void*p){delete static_cast<ct::SVM*>(p);}
int ct_svm_info(void*p,uint64_t*out,int n){return lo::protect([&]{lo::require(p&&out&&n==5,"invalid SVM info");auto&e=*static_cast<ct::SVM*>(p);uint64_t v[]={uint64_t(e.model->d),uint64_t(e.model->c),uint64_t(e.model->nsv),e.model->storage(),e.worker.scratch()+8ull*e.model->d};std::copy(v,v+5,out);});}
int ct_svm_run(void*p,const uint8_t*q,int rows,int d,int mode,int*out){return lo::protect([&]{
 lo::require(p,"closed SVM");auto&e=*static_cast<ct::SVM*>(p);ct::environment();
 lo::require(rows>=0&&rows<=65536&&d==e.model->d&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=1&&(!rows||(q&&out)),"invalid SVM buffers");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(q[i]<=e.maximum,"control input outside integer range");
 std::vector<double>x(size_t(d),0.);
 for(int r=0;r<rows;++r){for(int f=0;f<d;++f)x[f]=double(q[size_t(r)*d+f])/double(e.maximum);out[r]=e.worker.run(x.data(),mode==0?5:0,-1);}
});}
}
