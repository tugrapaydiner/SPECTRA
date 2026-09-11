// Standalone ASan/UBSan contract checks; no PyTorch, CUDA or model data.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include "../../spectra_reliability/native_cpu.cpp"
int main() {
  std::mt19937 rng(160701);std::size_t comparisons=0;
  for (int trial=0;trial<180;++trial) {
    int hidden=1+int(rng()%257),out=1+int(rng()%67),vectors=1+int(rng()%5),rb=(hidden+3)/4;
    std::vector<uint8_t>w(out*rb,0);std::vector<float>s(out),b(out),x(vectors*hidden),a(vectors*out),c(vectors*out);
    for(int o=0;o<out;++o){s[o]=float(rng()%1000)/777.f;b[o]=float(int(rng()%101)-50)/31.f;
      for(int d=0;d<hidden;++d)w[o*rb+d/4]|=uint8_t((rng()%3)<<(2*(d%4)));}
    for(auto&v:x)v=float(int(rng()%2001)-1000)/317.f;
    void*p=spectra_weight_create(w.data(),w.size(),s.data(),s.size(),b.data(),b.size(),out,hidden);assert(p);
    assert(spectra_checked_linear(x.data(),vectors,w.data(),w.size(),s.data(),s.size(),b.data(),b.size(),out,hidden,a.data())==0);
    assert(spectra_weight_linear(p,x.data(),vectors,hidden,c.data(),0)==0);assert(std::memcmp(a.data(),c.data(),a.size()*sizeof(float))==0);
    comparisons+=a.size();
    if(spectra_has_avx2()){assert(spectra_weight_linear(p,x.data(),vectors,hidden,c.data(),1)==0);assert(std::memcmp(a.data(),c.data(),a.size()*sizeof(float))==0);comparisons+=a.size();}
    assert(spectra_weight_linear(p,x.data(),vectors,hidden+1,c.data(),0)==-1);
    assert(spectra_weight_linear(p,x.data(),vectors,hidden,c.data(),9)==-1);
    assert(spectra_weight_create(w.data(),w.size()-1,s.data(),s.size(),b.data(),b.size(),out,hidden)==nullptr);
    auto bad=w;bad[0]|=3;assert(!spectra_weight_create(bad.data(),bad.size(),s.data(),s.size(),b.data(),b.size(),out,hidden));
    auto bad_scale=s;bad_scale[0]=std::numeric_limits<float>::quiet_NaN();assert(!spectra_weight_create(w.data(),w.size(),bad_scale.data(),bad_scale.size(),b.data(),b.size(),out,hidden));
    if(hidden%4){bad=w;bad[rb-1]|=uint8_t(1<<(2*(hidden%4)));assert(!spectra_weight_create(bad.data(),bad.size(),s.data(),s.size(),b.data(),b.size(),out,hidden));}
    spectra_weight_destroy(p);
  }
  const int64_t board[16]={1,2,3,4,3,4,1,2,2,1,4,3,4,3,2,1};int64_t empty[16]={};
  assert(spectra_sudoku_valid(empty,board,16,2)==1);assert(spectra_sudoku_valid(empty,board,15,2)==-1);
  for(int i=0;i<16;++i){int64_t bad[16];std::memcpy(bad,board,sizeof(board));bad[i]=0;assert(spectra_sudoku_valid(empty,bad,16,2)==0);}
  assert(spectra_weight_create(nullptr,0,nullptr,0,nullptr,0,0,0)==nullptr);
  spectra_weight_destroy(nullptr);
  std::cout<<"M16_NATIVE_SANITIZER_PASS comparisons="<<comparisons<<" avx2="<<spectra_has_avx2()<<"\n";
}
