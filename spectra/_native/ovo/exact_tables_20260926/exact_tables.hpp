// Exact binary64 feature dictionaries. No rounding, quantization or exp table.
#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace et {
struct FeatureTables {
    size_t dimensions=0, rows=0;
    bool enabled=false, global_codes=false;
    std::vector<uint32_t> offsets;
    std::vector<double> values, squared;
    std::vector<uint8_t> codes, prefix_codes;
    std::vector<std::array<uint8_t,2>> prefix_values;
    std::vector<double> prefix_sums;

    FeatureTables()=default;
    FeatureTables(size_t d,size_t n,const double*raw,bool requested=true,bool want_prefix=false):dimensions(d),rows(n) {
        if(!raw||!d||!n||d>4096||n>100000||d*n>8000000)
            throw std::invalid_argument("invalid feature-bank geometry");
        if(!requested)return;
        std::vector<uint8_t> trial(d*n);std::vector<double> alphabet;
        std::vector<uint32_t> starts;
        for(size_t j=0;j<d;++j){
            starts.push_back(uint32_t(alphabet.size()));
            std::unordered_map<uint64_t,uint16_t> ids;
            for(size_t i=0;i<n;++i){
                double value=raw[i*d+j];if(!std::isfinite(value))throw std::invalid_argument("nonfinite support feature");
                uint64_t bits;std::memcpy(&bits,&value,8);
                auto it=ids.find(bits);
                if(it==ids.end()){
                    if(ids.size()==256)return; // exact direct fallback; never merge distinct values
                    uint16_t id=uint16_t(ids.size());ids.emplace(bits,id);
                    alphabet.push_back(value);trial[i*d+j]=uint8_t(id);
                }else trial[i*d+j]=uint8_t(it->second);
            }
        }
        starts.push_back(uint32_t(alphabet.size()));
        // Count BOTH the dictionary and per-request distance scratch in this decision.
        if(trial.size()+16*alphabet.size()+4*starts.size()>=8*d*n)return;
        offsets=std::move(starts);values=std::move(alphabet);codes=std::move(trial);
        offsets.shrink_to_fit();values.shrink_to_fit();codes.shrink_to_fit();
        squared.resize(values.size());enabled=true;
        global_codes=values.size()<=256;
        if(global_codes)for(size_t i=0;i<n;++i)for(size_t j=0;j<d;++j)
            codes[i*d+j]=uint8_t(offsets[j]+codes[i*d+j]);
        if(want_prefix&&global_codes&&d>=2){
            std::unordered_map<uint16_t,uint16_t> ids;
            std::vector<uint8_t> pc(n);std::vector<std::array<uint8_t,2>> pv;
            bool fits=true;
            for(size_t i=0;i<n;++i){
                uint8_t a=codes[i*d],b=codes[i*d+1];uint16_t key=uint16_t(a)*256+b;
                auto it=ids.find(key);
                if(it==ids.end()){
                    if(ids.size()==256){fits=false;break;}
                    uint16_t id=uint16_t(ids.size());ids.emplace(key,id);pv.push_back({a,b});pc[i]=uint8_t(id);
                }else pc[i]=uint8_t(it->second);
            }
            if(fits){prefix_codes=std::move(pc);prefix_values=std::move(pv);
                prefix_codes.shrink_to_fit();prefix_values.shrink_to_fit();prefix_sums.resize(prefix_values.size());}
        }
    }
    void prepare(const float*x){
        if(!enabled)return;
        for(size_t j=0;j<dimensions;++j)
            for(uint32_t k=offsets[j];k<offsets[j+1];++k){
                double t=double(x[j])-values[k];squared[k]=t*t;
            }
        for(size_t i=0;i<prefix_values.size();++i){
            // Exactly the original initial accumulator and first two additions.
            double sum=0;sum+=squared[prefix_values[i][0]];sum+=squared[prefix_values[i][1]];prefix_sums[i]=sum;
        }
    }
    template<bool Global,bool Prefix,size_t D> double fixed_distance(uint32_t id)const{
        double sum=Prefix?prefix_sums[prefix_codes[id]]:0;
        for(size_t j=Prefix?2:0;j<D;++j)sum+=squared[(Global?0:offsets[j])+codes[id*D+j]];
        return sum;
    }
    double distance(uint32_t id)const{
        if(dimensions==16){
            if(!prefix_codes.empty())return fixed_distance<true,true,16>(id);
            return global_codes?fixed_distance<true,false,16>(id):fixed_distance<false,false,16>(id);
        }
        double sum=prefix_codes.empty()?0:prefix_sums[prefix_codes[id]];for(size_t j=prefix_codes.empty()?0:2;j<dimensions;++j)
            sum+=squared[(global_codes?0:offsets[j])+codes[id*dimensions+j]];
        return sum;
    }
#if defined(__AVX2__)
    template<bool Global,bool Prefix,size_t D> void fixed_batch(const uint32_t*ids,double*out)const{
        __m256d acc=Prefix?_mm256_setr_pd(prefix_sums[prefix_codes[ids[0]]],prefix_sums[prefix_codes[ids[1]]],prefix_sums[prefix_codes[ids[2]]],prefix_sums[prefix_codes[ids[3]]]):_mm256_setzero_pd();
        for(size_t j=Prefix?2:0;j<D;++j){
            // Scalar L1 loads avoid the variable-latency hardware gather.
            const double*bank=squared.data()+(Global?0:offsets[j]);
            __m256d term=_mm256_setr_pd(bank[codes[ids[0]*D+j]],bank[codes[ids[1]*D+j]],
                                      bank[codes[ids[2]*D+j]],bank[codes[ids[3]*D+j]]);
            acc=_mm256_add_pd(acc,term);
        }
        _mm256_storeu_pd(out,acc);
    }
#endif
    void batch(const uint32_t*ids,int count,double*out)const{
#if defined(__AVX2__)
        if(count==4&&dimensions==16){
            if(!prefix_codes.empty())fixed_batch<true,true,16>(ids,out);
            else if(global_codes)fixed_batch<true,false,16>(ids,out);else fixed_batch<false,false,16>(ids,out);
            return;
        }
        if(count==4){
            bool prefix=!prefix_codes.empty();
            __m256d acc=prefix?_mm256_setr_pd(prefix_sums[prefix_codes[ids[0]]],prefix_sums[prefix_codes[ids[1]]],prefix_sums[prefix_codes[ids[2]]],prefix_sums[prefix_codes[ids[3]]]):_mm256_setzero_pd();
            for(size_t j=prefix?2:0;j<dimensions;++j){
                const double*bank=squared.data()+(global_codes?0:offsets[j]);
                __m256d term=_mm256_setr_pd(bank[codes[ids[0]*dimensions+j]],bank[codes[ids[1]*dimensions+j]],
                                          bank[codes[ids[2]*dimensions+j]],bank[codes[ids[3]*dimensions+j]]);
                acc=_mm256_add_pd(acc,term);
            }
            _mm256_storeu_pd(out,acc);return;
        }
#endif
        for(int k=0;k<count;++k)out[k]=distance(ids[k]);
    }
    uint64_t resident()const{return sizeof(*this)+4*offsets.capacity()+8*values.capacity()+codes.capacity()+prefix_codes.capacity()+sizeof(std::array<uint8_t,2>)*prefix_values.capacity();}
    uint64_t scratch()const{return 8*(squared.capacity()+prefix_sums.capacity());}
};
}
