// Comparison harness only. Calls unmodified upstream LIBSVM 3.37 (BSD-3-Clause).
// SPECTRA and LIBSVM see identical dense binary64 inputs. No cached predictions.
#include "svm.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cfenv>
#include <stdexcept>
#include <string>
#include <limits>

namespace {
thread_local std::string error;
struct Model {
    svm_model* model = nullptr;
    int dimensions;
    std::vector<svm_node> nodes;
    Model(const char* path, int d): dimensions(d), nodes(d+1) {
        if(d<1||d>4096) throw std::invalid_argument("dimensions");
        model=svm_load_model(path);
        if(!model) throw std::runtime_error("LIBSVM load failed");
        for(int j=0;j<d;++j) nodes[j].index=j+1;
        nodes[d].index=-1;
    }
    ~Model(){ if(model) svm_free_and_destroy_model(&model); }
    int predict(const double* x) {
        for(int j=0;j<dimensions;++j) nodes[j].value=x[j];
        return int(svm_predict(model,nodes.data()));
    }
};
template<class F> int guard(F&& f){try{f();error.clear();return 0;}catch(const std::exception&e){error=e.what();return -1;}}
int batch(void* h,const double*x,int rows,int d,int*out){return guard([&]{
    if(!h||!x||!out||rows<1||rows>65536||std::fegetround()!=FE_TONEAREST) throw std::invalid_argument("arguments or rounding");
    auto&m=*static_cast<Model*>(h);
    if(d!=m.dimensions) throw std::invalid_argument("shape");
    for(size_t i=0;i<size_t(rows)*d;++i) if(!std::isfinite(x[i])) throw std::invalid_argument("nonfinite");
    for(int i=0;i<rows;++i) out[i]=m.predict(x+size_t(i)*d);
});}
using Fn=int(*)(void*,const double*,int,int,int,int,int*,int,uint64_t*,int,int);
using Clock=std::chrono::steady_clock;
}
extern "C" {
const char* panel_error(){return error.c_str();}
void* panel_open(const char*path,int d){try{return new Model(path,d);}catch(const std::exception&e){error=e.what();return nullptr;}}
void panel_close(void*h){delete static_cast<Model*>(h);}
int panel_batch(void*h,const double*x,int n,int d,int*out){return batch(h,x,n,d,out);}
uint64_t panel_time_libsvm(void*h,const double*x,int d,int*out){
    auto start=Clock::now();int status=batch(h,x,1,d,out);auto end=Clock::now();
    return status?UINT64_MAX:std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}
uint64_t panel_time_shared(void*fn,void*h,const double*x,int d,int schedule,int*out){
    uint64_t stats[5]{};
    auto start=Clock::now();int status=reinterpret_cast<Fn>(fn)(h,x,1,d,schedule,-1,out,1,stats,5,0);auto end=Clock::now();
    return status?UINT64_MAX:std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}
uint64_t panel_time_noop(){auto start=Clock::now();auto end=Clock::now();return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();}
}
