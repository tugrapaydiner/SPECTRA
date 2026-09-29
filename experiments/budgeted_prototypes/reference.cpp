// Independent test observer. Does not include production/candidate headers.
// Recomputes both exponential factors rather than using candidate lookup tables.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
static void ensure(bool x){if(!x)throw std::invalid_argument("reference bounds");}
extern "C" int pr_scores(const unsigned char*raw,uint64_t length,const uint8_t*input,int rows,int dimension,double*out,uint64_t cells){
 try{
  ensure(raw&&length>=48&&std::memcmp(raw,"SPPRO001",8)==0);
  auto number=[&](size_t offset){ensure(offset+4<=length);uint32_t value;std::memcpy(&value,raw+offset,4);return value;};
  uint32_t d=number(8),p=number(12),c=number(16),D=number(20),Q=number(24),U=number(28),bits=number(32),meta=number(36),payload=number(40);
  ensure(d>=1&&d<=256&&p>=1&&p<=4096&&c>=2&&c<=128&&D>=1&&D<=255&&Q>=1&&Q<=16&&U>=1&&U<=16&&bits<32);
  ensure(length==48ull+payload&&payload==8+4ull*p*d+8ull*(uint64_t(p)*c+c)+meta);
  ensure(rows>=0&&dimension==int(d)&&cells==uint64_t(rows)*c&&(!rows||(input&&out)));
  double alpha;std::memcpy(&alpha,raw+48,8);ensure(std::isfinite(alpha)&&alpha>0);
  size_t offset=56;std::vector<uint16_t>centers(size_t(p)*d),weights(size_t(p)*d);std::vector<double>H(size_t(p)*c),b(c);
  auto read=[&](void*dst,size_t n){ensure(offset+n<=length);std::memcpy(dst,raw+offset,n);offset+=n;};
  read(centers.data(),2*centers.size());read(weights.data(),2*weights.size());read(H.data(),8*H.size());read(b.data(),8*b.size());
  uint64_t B=uint64_t(1)<<bits;
  for(int n=0;n<rows;++n){std::fill(out+size_t(n)*c,out+size_t(n+1)*c,0.);
   for(uint32_t j=0;j<p;++j){uint64_t distance=0;
    for(uint32_t f=0;f<d;++f){ensure(input[size_t(n)*d+f]<=D);int64_t difference=int64_t(input[size_t(n)*d+f])*Q-int64_t(centers[size_t(j)*d+f]);distance+=uint64_t(difference*difference)*weights[size_t(j)*d+f];}
    double first=std::exp(-alpha*double((distance/B)*B));double second=std::exp(-alpha*double(distance%B));double value=first*second;
    for(uint32_t k=0;k<c;++k)out[size_t(n)*c+k]+=value*H[size_t(j)*c+k];
   }
   for(uint32_t k=0;k<c;++k){out[size_t(n)*c+k]+=b[k];ensure(std::isfinite(out[size_t(n)*c+k]));}
  }
  return 0;
 }catch(...){return 1;}
}
