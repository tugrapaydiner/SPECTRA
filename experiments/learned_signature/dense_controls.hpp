// Fixed learned linear/ReLU controls receive the same strict native compilation.
namespace ldc {
struct Model {
 std::vector<uint32_t> widths;std::vector<std::vector<double>> weight,bias;
 std::vector<double> a,b;uint32_t cap;
 Model(const double*values,uint64_t count,const uint32_t*dims,int layers,uint32_t maximum):cap(maximum){
  lo::require(values&&dims&&layers>=1&&layers<=4&&maximum>=1&&maximum<=255,"invalid dense control");
  uint64_t expected=0;uint32_t largest=0;
  for(int i=0;i<=layers;++i){lo::require(dims[i]>=1&&dims[i]<=4096,"dense width");widths.push_back(dims[i]);largest=std::max(largest,dims[i]);if(i)expected+=uint64_t(dims[i-1])*dims[i]+dims[i];}
  lo::require(expected==count&&count<=8000000,"dense inventory");
  uint64_t off=0;for(int l=0;l<layers;++l){uint64_t n=uint64_t(dims[l])*dims[l+1];weight.emplace_back(values+off,values+off+n);off+=n;bias.emplace_back(values+off,values+off+dims[l+1]);off+=dims[l+1];}
  for(uint64_t i=0;i<count;++i)lo::require(std::isfinite(values[i]),"nonfinite dense parameter");a.resize(largest);b.resize(largest);
 }
 int predict(const uint8_t*x){
  for(uint32_t j=0;j<widths[0];++j)a[j]=double(x[j])/cap;
  for(size_t l=0;l<weight.size();++l){uint32_t din=widths[l],dout=widths[l+1],j=0;
#if defined(__AVX2__)
   for(;j+4<=dout;j+=4){__m256d sum=_mm256_setzero_pd();for(uint32_t k=0;k<din;++k)sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_set1_pd(a[k]),_mm256_loadu_pd(weight[l].data()+size_t(k)*dout+j)));sum=_mm256_add_pd(sum,_mm256_loadu_pd(bias[l].data()+j));_mm256_storeu_pd(b.data()+j,sum);}
#endif
   for(;j<dout;++j){double sum=0.;for(uint32_t k=0;k<din;++k)sum+=a[k]*weight[l][size_t(k)*dout+j];b[j]=sum+bias[l][j];}
   for(uint32_t j2=0;j2<dout;++j2){lo::require(std::isfinite(b[j2]),"nonfinite dense output");if(l+1<weight.size())b[j2]=std::max(0.,b[j2]);}
   a.swap(b);
  }
  return int(std::max_element(a.begin(),a.begin()+widths.back())-a.begin());
 }
};
}
extern "C" {
void* ldc_create(const double*v,uint64_t n,const uint32_t*w,int layers,uint32_t cap){void*out=nullptr;lo::protect([&]{out=new ldc::Model(v,n,w,layers,cap);});return out;}
void ldc_destroy(void*p){delete static_cast<ldc::Model*>(p);}
int ldc_run(void*p,const uint8_t*x,int n,int d,int*out){return lo::protect([&]{lo::require(p&&n>=0&&n<=65536&&(!n||(x&&out)),"dense run");auto&m=*static_cast<ldc::Model*>(p);lo::require(d==int(m.widths[0])&&uint64_t(n)*d<=8000000,"dense geometry");lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");for(size_t i=0;i<size_t(n)*d;++i)lo::require(x[i]<=m.cap,"dense input domain");for(int i=0;i<n;++i)out[i]=m.predict(x+size_t(i)*d);});}
}
