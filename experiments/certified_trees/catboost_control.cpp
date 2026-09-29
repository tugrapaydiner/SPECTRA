// Matched raw-uint8 to class-index control. Pinned upstream arithmetic untouched.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(CB_EXPORT_SOURCE)
#include CB_EXPORT_SOURCE
#else
#include "c_api.h"
#endif
namespace control {
thread_local std::string error;
void require(bool x,const char*s){if(!x)throw std::invalid_argument(s);}
template<class F>int protect(F f){try{f();return 0;}catch(const std::exception&e){error=e.what();return 1;}catch(...){error="unknown comparator failure";return 1;}}
struct Model {
 uint32_t d,c,maximum;
#if !defined(CB_EXPORT_SOURCE)
 ModelCalcerHandle* handle=nullptr;
#endif
 Model(const unsigned char*raw,uint64_t bytes,int D){
  require(D>=1&&D<=255,"invalid code range");maximum=D;
#if defined(CB_EXPORT_SOURCE)
  (void)raw;(void)bytes;d=CatboostModelStatic.FloatFeatureCount;c=CatboostModelStatic.Dimension;
#else
  require(raw&&bytes>0&&bytes<=64ull*1024*1024,"CBM source size cap");handle=ModelCalcerCreate();require(handle,"cannot create CatBoost model");
  if(!LoadFullModelFromBuffer(handle,raw,size_t(bytes))){std::string msg=GetErrorString();ModelCalcerDelete(handle);handle=nullptr;throw std::invalid_argument(msg);}
  if(GetCatFeaturesCount(handle)||GetTextFeaturesCount(handle)||GetEmbeddingFeaturesCount(handle)){ModelCalcerDelete(handle);handle=nullptr;throw std::invalid_argument("numeric features only");}
  d=uint32_t(GetFloatFeaturesCount(handle));c=uint32_t(GetDimensionsCount(handle));
#endif
  if(!(d>=1&&d<=256&&c>=2&&c<=64)){
#if !defined(CB_EXPORT_SOURCE)
   ModelCalcerDelete(handle);handle=nullptr;
#endif
   throw std::invalid_argument("unsupported comparator dimensions");
  }
 }
 ~Model(){
#if !defined(CB_EXPORT_SOURCE)
  if(handle)ModelCalcerDelete(handle);
#endif
 }
 void run(const uint8_t*x,int rows,int*out,double*scores)const{
  if(!rows)return;
#if defined(CB_EXPORT_SOURCE)
  std::vector<float>row(d);
  for(int i=0;i<rows;++i){for(uint32_t f=0;f<d;++f)row[f]=float(x[size_t(i)*d+f]);auto result=ApplyCatboostModelMulti(row);
   require(result.size()==c,"export score dimension");out[i]=int(std::max_element(result.begin(),result.end())-result.begin());if(scores)std::copy(result.begin(),result.end(),scores+size_t(i)*c);
  }
#else
  std::vector<float>data(size_t(rows)*d);std::vector<const float*>ptrs(rows);std::vector<double>result(size_t(rows)*c);
  for(int i=0;i<rows;++i){ptrs[i]=data.data()+size_t(i)*d;for(uint32_t f=0;f<d;++f)data[size_t(i)*d+f]=float(x[size_t(i)*d+f]);}
  if(!CalcModelPredictionFlat(handle,size_t(rows),ptrs.data(),d,result.data(),result.size()))throw std::runtime_error(GetErrorString());
  for(int i=0;i<rows;++i){const auto*r=result.data()+size_t(i)*c;out[i]=int(std::max_element(r,r+c)-r);}
  if(scores)std::copy(result.begin(),result.end(),scores);
#endif
 }
};
}
extern "C"{
int cb_abi(){return 1;}
const char*cb_error(){return control::error.c_str();}
void*cb_create(const unsigned char*raw,uint64_t n,int D){void*p=nullptr;control::protect([&]{p=new control::Model(raw,n,D);});return p;}
void cb_destroy(void*p){delete static_cast<control::Model*>(p);}
int cb_info(void*p,uint32_t*out){return control::protect([&]{control::require(p&&out,"invalid info");auto&m=*static_cast<control::Model*>(p);out[0]=m.d;out[1]=m.c;out[2]=m.maximum;});}
int cb_run(void*p,const uint8_t*x,int rows,int d,int*out,double*scores,uint64_t cells){return control::protect([&]{
 control::require(p,"closed CatBoost comparator");auto&m=*static_cast<control::Model*>(p);control::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
 control::require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&(!rows||(x&&out))&&(!scores||(cells==uint64_t(rows)*m.c&&cells<=8000000)),"comparator input/output geometry");
 for(size_t i=0;i<size_t(rows)*d;++i)control::require(x[i]<=m.maximum,"input outside integer domain");m.run(x,rows,out,scores);
});}
}
#if defined(CB_WITH_REFINEMENT)
// Only ambiguous rows reach the unmodified upstream engine. Both allocations
// must remain leased by the caller. Source JSON/CBM pairing is verified offline.
extern "C" int st_info(void*,uint64_t*,int);
extern "C" int st_run(void*,const uint8_t*,int,int,int,int,int,int*,uint32_t*,int*);
extern "C" const char*st_error();
extern "C" int cb_refine(void*p,void*qmodel,const uint8_t*x,int rows,int d,int checkpoint,int*out,uint32_t*steps,uint8_t*fallback){return control::protect([&]{
 control::require(p&&qmodel,"closed refinement models");auto&m=*static_cast<control::Model*>(p);
 control::require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&checkpoint>=0&&checkpoint<=4096&&(!rows||(x&&out&&steps&&fallback)),"invalid refinement geometry");
 uint64_t info[9];if(st_info(qmodel,info,9))throw std::invalid_argument(st_error());
 control::require(info[0]==m.d&&info[1]==m.c&&(info[5]==8||info[5]==16),"incompatible compact model");
 for(size_t i=0;i<size_t(rows)*d;++i)control::require(x[i]<=m.maximum,"refinement domain");
 std::vector<int>approx(rows);if(st_run(qmodel,x,rows,d,checkpoint,1,0,out,steps,approx.data()))throw std::invalid_argument(st_error());
 std::vector<int>indices;indices.reserve(rows);
 for(int i=0;i<rows;++i){fallback[i]=uint8_t(out[i]<0);if(out[i]<0)indices.push_back(i);}
 if(!indices.empty()){
  std::vector<uint8_t>input(indices.size()*d);std::vector<int>answers(indices.size());
  for(size_t i=0;i<indices.size();++i)std::copy(x+size_t(indices[i])*d,x+size_t(indices[i]+1)*d,input.data()+i*d);
  m.run(input.data(),int(indices.size()),answers.data(),nullptr);
  for(size_t i=0;i<indices.size();++i)out[indices[i]]=answers[i];
 }
});}
#endif
#if defined(CB_WITH_REFINEMENT)
extern "C" void*st_create(const unsigned char*,uint64_t);
extern "C" void st_destroy(void*);
namespace fused {
struct Pipeline {
 std::unique_ptr<control::Model> original;
 std::unique_ptr<void,decltype(&st_destroy)> compact;
 Pipeline(const unsigned char*q,uint64_t nq,const unsigned char*cb,uint64_t ncb,int maximum):
  original(new control::Model(cb,ncb,maximum)),compact(st_create(q,nq),st_destroy){
  if(!compact)throw std::invalid_argument(st_error());
  uint64_t info[9];if(st_info(compact.get(),info,9))throw std::invalid_argument(st_error());
  uint32_t qmax;std::memcpy(&qmax,q+16,4);
  control::require(info[0]==original->d&&info[1]==original->c&&(info[5]==8||info[5]==16)&&qmax==uint32_t(maximum),"owning pipeline geometry mismatch");
 }
};
}
extern "C" void*cb_pipeline_create(const unsigned char*q,uint64_t nq,const unsigned char*cb,uint64_t ncb,int D){void*p=nullptr;control::protect([&]{p=new fused::Pipeline(q,nq,cb,ncb,D);});return p;}
extern "C" void cb_pipeline_destroy(void*p){delete static_cast<fused::Pipeline*>(p);}
extern "C" int cb_pipeline_info(void*p,uint64_t*out,int n){return control::protect([&]{control::require(p&&out&&n==9,"owning pipeline info");auto&w=*static_cast<fused::Pipeline*>(p);if(st_info(w.compact.get(),out,n))throw std::invalid_argument(st_error());});}
extern "C" int cb_pipeline_run(void*p,const uint8_t*x,int rows,int d,int checkpoint,int*out,uint32_t*steps,uint8_t*fallback){
 if(!p){control::error="closed owning pipeline";return 1;}
 auto&w=*static_cast<fused::Pipeline*>(p);
 return cb_refine(w.original.get(),w.compact.get(),x,rows,d,checkpoint,out,steps,fallback);
}
#endif
