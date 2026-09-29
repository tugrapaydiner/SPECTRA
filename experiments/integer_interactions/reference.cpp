// Independent observer: original model/vote objects, original sequential binary64
// distances on externally transformed integer coordinates, scalar system exp.
// Does not include the candidate runtime, projection or lookup-table code.
#ifndef II_BASE_RUNTIME
#error II_BASE_RUNTIME required
#endif
#include II_BASE_RUNTIME
struct Ref {
 spm::Owner model;spm::Worker worker;int bits;
 Ref(const unsigned char* b,size_t n,int p):model(std::make_shared<const spm::Model>(b,n,false)),worker(model),bits(p){
  lo::require(bits>=0&&bits<=24,"invalid reference radix");
 }
};
extern "C" void* ref_create(const unsigned char* b,uint64_t n,int bits){void*r=nullptr;lo::protect([&]{r=new Ref(b,n,bits);});return r;}
extern "C" void ref_destroy(void*p){delete static_cast<Ref*>(p);}
extern "C" int ref_probe(void*p,const double*x,int rows,int d,double*out,uint64_t count){return lo::protect([&]{
 lo::require(p,"null reference");auto&r=*static_cast<Ref*>(p);const auto&m=*r.model;
 lo::require(rows>=0&&rows<=65536&&d==m.d&&count==uint64_t(rows)*m.pairs.size()&&count<=8000000&&(!rows||(x&&out)),"bad reference geometry");
 for(int row=0;row<rows;++row){const double*q=x+size_t(row)*d;r.worker.begin(q);
  for(uint32_t id:m.active){
   double distance=0;for(int j=0;j<d;++j){double delta=q[j]-m.sv[size_t(id)*d+j];distance+=delta*delta;}
   lo::require(std::isfinite(distance)&&distance>=0&&distance<0x1p48&&distance==std::floor(distance),"reference distance not exact integer");
   uint64_t S=uint64_t(distance),mask=(uint64_t(1)<<r.bits)-1;
   double value=std::exp(-m.gamma*double((S>>r.bits)<<r.bits))*std::exp(-m.gamma*double(S&mask));
   r.worker.kernel[id]=value;
  }
  for(const auto& pair:m.pairs){double value=0;for(size_t i=0;i<pair.ids.size();++i)value+=pair.values[i]*r.worker.kernel[pair.ids[i]];value+=pair.bias;*out++=value;}
 }
});}
