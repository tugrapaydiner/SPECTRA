// Native baselines for frozen multiclass linear / one-hidden-ReLU models.
// Each AVX2 lane is one output, never a reassociated partial reduction.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif
namespace {
thread_local std::string error;
void require(bool b,const char* m){if(!b)throw std::invalid_argument(m);}
struct Model { uint32_t d,h,c; std::vector<double> values,hidden,scores;
    explicit Model(const char* path) {
        std::ifstream file(path,std::ios::binary|std::ios::ate);require(bool(file),"cannot open weights");
        auto bytes=file.tellg();require(bytes>=20&&bytes<=67108864,"invalid model bytes");file.seekg(0);
        char magic[8];file.read(magic,8);require(std::memcmp(magic,"SPDENSE1",8)==0,"bad model magic");
        file.read(reinterpret_cast<char*>(&d),4);file.read(reinterpret_cast<char*>(&h),4);file.read(reinterpret_cast<char*>(&c),4);
        require(d>=1&&d<=4096&&h<=1024&&c>=2&&c<=128,"invalid shape");
        const uint64_t n=h?uint64_t(d)*h+h+uint64_t(h)*c+c:uint64_t(d)*c+c;
        require(uint64_t(bytes)==20+8*n,"weight inventory mismatch");values.resize(n);file.read(reinterpret_cast<char*>(values.data()),8*n);
        require(bool(file),"truncated weights");for(double x:values)require(std::isfinite(x),"nonfinite weight");
        hidden.resize(h);scores.resize(c);
    }
};
void affine(const double* x,const double* w,const double* b,int d,int c,double* out) {
    int j=0;
#ifdef __AVX2__
    for(;j+4<=c;j+=4){
        __m256d a=_mm256_setzero_pd();
        for(int i=0;i<d;++i) a=_mm256_add_pd(a,_mm256_mul_pd(_mm256_set1_pd(x[i]),_mm256_loadu_pd(w+size_t(i)*c+j)));
        a=_mm256_add_pd(a,_mm256_loadu_pd(b+j));_mm256_storeu_pd(out+j,a);
    }
#endif
    for(;j<c;++j){double a=0;for(int i=0;i<d;++i)a+=x[i]*w[size_t(i)*c+j];out[j]=a+b[j];}
}
}
extern "C" {
const char* dn_error(){return error.c_str();}
void* dn_create(const char* path){try{return new Model(path);}catch(const std::exception& e){error=e.what();return nullptr;}}
void dn_destroy(void* p){delete static_cast<Model*>(p);}
int dn_run(void* p,const double* x,int rows,int d,int* labels,double* scoreout){try{
    require(p,"null model");auto& m=*static_cast<Model*>(p);
    require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000,"invalid input shape");
    require(!rows||(x&&labels),"null buffer");for(size_t i=0;i<size_t(rows)*d;++i)require(std::isfinite(x[i]),"nonfinite input");
    for(int r=0;r<rows;++r){
        const double* w=m.values.data();
        if(m.h){const double* b=w+size_t(m.d)*m.h;affine(x+size_t(r)*d,w,b,m.d,m.h,m.hidden.data());
            for(double& v:m.hidden){require(std::isfinite(v),"hidden overflow");v=std::max(0.,v);}
            w=b+m.h;b=w+size_t(m.h)*m.c;affine(m.hidden.data(),w,b,m.h,m.c,m.scores.data());
        }else affine(x+size_t(r)*d,w,w+size_t(m.d)*m.c,m.d,m.c,m.scores.data());
        int winner=0;for(uint32_t j=0;j<m.c;++j){require(std::isfinite(m.scores[j]),"score overflow");if(m.scores[j]>m.scores[winner])winner=j;}
        labels[r]=winner;if(scoreout)std::copy(m.scores.begin(),m.scores.end(),scoreout+size_t(r)*m.c);
    } return 0;
}catch(const std::exception& e){error=e.what();return 1;}}
}
