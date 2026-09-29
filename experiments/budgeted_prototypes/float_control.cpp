// Standard FP32 deployment of the SAME fitted MLP. BLAS reduction order differs.
#ifndef BP_BASE_RUNTIME
#error BP_BASE_RUNTIME required
#endif
#include BP_BASE_RUNTIME
extern "C" void scipy_cblas_sgemm(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
extern "C" void scipy_openblas_set_num_threads(int);
extern "C" const char* scipy_openblas_get_config();
namespace nf {
struct Layer {uint32_t in,out;std::vector<float>w,b;};
struct Model {
 uint32_t d,c,D;size_t width=0,parameters=0;std::vector<float>mean,scale;std::vector<Layer>layers;
 Model(const unsigned char*raw,uint64_t bytes){
  lo::require(raw&&bytes>=36&&bytes<=64ull*1024*1024,"FP32 model size");
  lo::require(!std::memcmp(raw,"SPNF0001",8),"wrong FP32 model magic");
  auto u32=[&](size_t off){uint32_t n;std::memcpy(&n,raw+off,4);return n;};
  d=u32(8);c=u32(12);D=u32(16);uint32_t L=u32(20),meta=u32(24),payload=u32(28),crc=u32(32);
  lo::require(d>=1&&d<=256&&c>=2&&c<=128&&D>=1&&D<=255&&L>=1&&L<=4&&meta>=1&&meta<=65536,"FP32 geometry");
  lo::require(bytes==36ull+payload&&lo::crc32(raw+36,bytes-36)==crc,"FP32 bytes/CRC");
  size_t off=36;auto take=[&](void*out,size_t n){lo::require(off<=bytes&&n<=bytes-off,"FP32 bounds");std::memcpy(out,raw+off,n);off+=n;};
  std::vector<uint32_t>dims(L+1);take(dims.data(),4*dims.size());lo::require(dims.front()==d&&dims.back()==c,"FP32 shape");
  for(auto n:dims){lo::require(n>=1&&n<=1024,"FP32 width");width=std::max(width,size_t(n));}
  mean.resize(d);scale.resize(d);take(mean.data(),4*d);take(scale.data(),4*d);
  for(uint32_t f=0;f<d;++f)lo::require(std::isfinite(mean[f])&&std::isfinite(scale[f])&&scale[f]>0,"FP32 scaling");
  for(uint32_t i=0;i<L;++i){Layer l{dims[i],dims[i+1],{}, {}};parameters+=size_t(l.in)*l.out+l.out;lo::require(parameters<=4000000,"FP32 parameter cap");
   l.w.resize(size_t(l.in)*l.out);l.b.resize(l.out);take(l.w.data(),4*l.w.size());take(l.b.data(),4*l.b.size());
   for(const auto*v:{&l.w,&l.b})for(float x:*v)lo::require(std::isfinite(x),"FP32 nonfinite weight");layers.push_back(std::move(l));
  }lo::require(off+meta==bytes,"FP32 trailing bytes");
 }
 uint64_t storage()const{uint64_t n=sizeof(*this)+4*(mean.capacity()+scale.capacity())+sizeof(Layer)*layers.capacity();for(auto&l:layers)n+=4*(l.w.capacity()+l.b.capacity());return n;}
};
}
extern "C" {
int ct_abi(){return 1;}
const char* ct_blas_config(){scipy_openblas_set_num_threads(1);return scipy_openblas_get_config();}
void* ct_nn_create(const unsigned char*raw,uint64_t n){void*p=nullptr;lo::protect([&]{p=new nf::Model(raw,n);});return p;}
void ct_nn_destroy(void*p){delete static_cast<nf::Model*>(p);}
int ct_nn_info(void*p,uint64_t*out,int n){return lo::protect([&]{lo::require(p&&out&&n==5,"FP32 info");auto&m=*static_cast<nf::Model*>(p);uint64_t v[]={m.d,m.c,m.parameters,m.storage(),m.width*8};std::copy(v,v+5,out);});}
int ct_nn_run(void*p,const uint8_t*q,int rows,int d,int mode,int*out,double*scores,uint64_t cells){return lo::protect([&]{
 lo::require(p,"closed FP32 model");auto&m=*static_cast<nf::Model*>(p);
 lo::require(std::fegetround()==FE_TONEAREST,"FP32 rounding");
 lo::require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&mode==0&&(!rows||(q&&out))&&(!scores||cells==uint64_t(rows)*m.c)&&cells<=8000000,"FP32 input bounds");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(q[i]<=m.D,"FP32 raw input outside domain");if(!rows)return;
 std::vector<float>a(size_t(rows)*m.width),b(size_t(rows)*m.width);
 for(int r=0;r<rows;++r)for(uint32_t f=0;f<m.d;++f)a[size_t(r)*m.d+f]=(float(q[size_t(r)*m.d+f])/float(m.D)-m.mean[f])/m.scale[f];
 for(size_t k=0;k<m.layers.size();++k){auto&l=m.layers[k];
  scipy_cblas_sgemm(101,111,111,rows,l.out,l.in,1.f,a.data(),l.in,l.w.data(),l.out,0.f,b.data(),l.out);
  for(int r=0;r<rows;++r)for(uint32_t j=0;j<l.out;++j){float v=b[size_t(r)*l.out+j]+l.b[j];lo::require(std::isfinite(v),"nonfinite FP32 activation");b[size_t(r)*l.out+j]=(k+1<m.layers.size()&&v<0)?0:v;}
  a.swap(b);
 }
 for(int r=0;r<rows;++r){const float*s=a.data()+size_t(r)*m.c;out[r]=int(std::max_element(s,s+m.c)-s);if(scores)for(uint32_t c=0;c<m.c;++c)scores[size_t(r)*m.c+c]=double(s[c]);}
});}
}
