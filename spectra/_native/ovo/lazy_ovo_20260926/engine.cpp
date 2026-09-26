// SPECTRA lazy OvO: exact winner certificates over unchanged FP64 pair outcomes.
// No claim of exact real-valued exp, probability calibration, or label correctness.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace lo {
constexpr uint64_t MAX_BYTES=64ull*1024*1024;
thread_local std::string error;
void require(bool yes,const char* message){if(!yes)throw std::invalid_argument(message);}
uint32_t crc32(const unsigned char*p,size_t n){uint32_t crc=~uint32_t(0);for(size_t i=0;i<n;++i){crc^=p[i];for(int j=0;j<8;++j)crc=(crc>>1)^((crc&1)?0xedb88320u:0);}return ~crc;}
template<class F>int protect(F f){try{f();error.clear();return 0;}catch(const std::exception&e){error=e.what();return -1;}catch(...){error="unknown native exception";return -1;}}
struct Work { uint64_t kernels=0,pairs=0,terms=0,cert_checks=0,classes=0,supports=0,kind=0; };

// The scheduler knows only discrete pair winners. All comparisons use the first
// class index to break a tied final vote count, exactly as the exhaustive vote.
struct Votes {
 int c;std::vector<int8_t> known;std::vector<int> low,remain;Work* work;
 explicit Votes(int classes):c(classes),known(c*c,-1),low(c,0),remain(c,c-1),work(nullptr){}
 void reset(Work&w){work=&w;std::fill(known.begin(),known.end(),-1);std::fill(low.begin(),low.end(),0);std::fill(remain.begin(),remain.end(),c-1);}
 void add(int i,int j,int winner){require(i>=0&&j>=0&&i<c&&j<c&&i!=j&&(winner==i||winner==j),"invalid pair result");
  if(known[i*c+j]>=0){require(known[i*c+j]==winner,"inconsistent repeated pair");return;}
  known[i*c+j]=known[j*c+i]=int8_t(winner);++low[winner];--remain[i];--remain[j];++work->pairs;
 }
 int certificate(){++work->cert_checks;int best=0;for(int i=1;i<c;++i)if(low[i]>low[best])best=i;
  for(int j=0;j<c;++j)if(j!=best){int upper=low[j]+remain[j];if(low[best]<upper||(j<best&&low[best]==upper))return -1;}
  return best;
 }
 template<class Pair>int run(Pair&&pair,const std::vector<double>&priority,int mode){
  auto edge=[&](int i,int j){int w=known[i*c+j];if(w<0){w=pair(i,j);add(i,j,w);}return w;};
  if(mode==0||mode==1){for(int i=0;i<c;++i)for(int j=i+1;j<c;++j)edge(i,j);int w=certificate();require(w>=0,"exhaustive vote missing certificate");return w;}
  if(mode==5||mode==6){int candidate=0;for(int i=1;i<c;++i)candidate=edge(candidate,i);if(mode==5)return candidate;
   // Plain knockout is unsafe. Accept only after the vote certificate succeeds.
   for(int j=0;j<c;++j)if(j!=candidate){edge(candidate,j);}
   int found=certificate();if(found>=0)return found;
  }
  int next=0;if(mode!=3&&mode!=6)for(int i=1;i<c;++i)if(priority[i]<priority[next])next=i;
  for(int pass=0;pass<c;++pass){
   if(remain[next])for(int j=0;j<c;++j)if(j!=next)edge(next,j);
   int found=certificate();if(found>=0)return found;
   next=-1;
   // Choose the largest possible winner; model-only centroid distance breaks
   // ties to reduce work, but can never bypass the certificate.
   for(int i=0;i<c;++i)if(remain[i]>0){if(next<0||low[i]+remain[i]>low[next]+remain[next]
      ||(low[i]+remain[i]==low[next]+remain[next]&&(low[i]>low[next]
       ||(low[i]==low[next]&&priority[i]<priority[next]))))next=i;}
   require(next>=0,"unsettled vote without remaining edge");
  }
  throw std::logic_error("vote scheduler failed to terminate");
 }
};
struct Pair {std::vector<uint32_t> ids;std::vector<double> values;double bias;};
struct Engine {
 int c,nsv;double gamma;std::vector<uint32_t> counts,start,active;
 std::vector<double> sv,dense,bias,kernel,centers,priority;std::vector<Pair> pair;
 std::vector<uint64_t> epoch;uint64_t serial=0;Votes vote;Work work;bool has_dense;bool valid_certificate=false;
 // c is learned from the fixed-width pre-header before allocating the scheduler.
 static int read_classes(const char*path){require(path,"null model path");std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"cannot open model");
  auto n=f.tellg();require(n>=24&&uint64_t(n)<=MAX_BYTES,"invalid model size");f.seekg(0);char magic[8];uint32_t c=0;f.read(magic,8);f.read(reinterpret_cast<char*>(&c),4);
  require(bool(f)&&std::memcmp(magic,"SPCSVM01",8)==0&&c>=2&&c<=128,"invalid SVC header");return int(c);}
 explicit Engine(const char*path,bool keep_dense):c(read_classes(path)),vote(c),has_dense(keep_dense){
  const uint16_t one=1;require(*reinterpret_cast<const uint8_t*>(&one)==1,"little endian host required");
  require(std::numeric_limits<double>::is_iec559&&sizeof(double)==8&&sizeof(float)==4,"IEEE binary32/binary64 required");
  std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"cannot reopen model");auto len=f.tellg();require(len>=24&&uint64_t(len)<=MAX_BYTES,"invalid model length");f.seekg(0);
  char magic[8];uint32_t cc,n,bytes,crc;f.read(magic,8);f.read(reinterpret_cast<char*>(&cc),4);f.read(reinterpret_cast<char*>(&n),4);f.read(reinterpret_cast<char*>(&bytes),4);f.read(reinterpret_cast<char*>(&crc),4);
  require(bool(f)&&std::memcmp(magic,"SPCSVM01",8)==0&&cc==uint32_t(c)&&n>=1&&n<=100000,"model header changed or invalid");
  uint64_t doubles=1+16ull*n+(c-1ull)*n+c*(c-1ull)/2;require(bytes==8*doubles+4ull*c&&uint64_t(len)==24ull+bytes,"SVC inventory mismatch");
  std::vector<unsigned char>raw(bytes);f.read(reinterpret_cast<char*>(raw.data()),bytes);require(bool(f)&&crc32(raw.data(),bytes)==crc,"SVC CRC/read mismatch");size_t off=0;
  auto take=[&](void*to,size_t num){require(off+num<=raw.size(),"payload bounds");std::memcpy(to,raw.data()+off,num);off+=num;};
  take(&gamma,8);require(std::isfinite(gamma)&&gamma>0,"invalid gamma");nsv=int(n);counts.resize(c);start.resize(c);take(counts.data(),4*c);uint64_t total=0;
  for(int i=0;i<c;++i){start[i]=uint32_t(total);require(counts[i]>0&&counts[i]<=n,"invalid support count");total+=counts[i];}require(total==n,"support count sum mismatch");
  sv.resize(16ull*n);dense.resize((c-1ull)*n);bias.resize(c*(c-1ull)/2);kernel.resize(n);epoch.resize(n,0);centers.resize(c*16,0);priority.resize(c);
  for(auto*v:{&sv,&dense,&bias}){take(v->data(),v->size()*8);for(double x:*v)require(std::isfinite(x),"nonfinite parameter");}require(off==raw.size(),"unused bytes");
  pair.reserve(bias.size());std::vector<uint8_t>used(n,0);int p=0;
  for(int i=0;i<c;++i)for(int j=i+1;j<c;++j){Pair a;a.bias=bias[p++];
   for(auto half:{std::pair<int,int>{i,j-1},std::pair<int,int>{j,i}})for(uint32_t k=start[half.first];k<start[half.first]+counts[half.first];++k){double v=dense[half.second*n+k];if(v!=0.0){a.ids.push_back(k);a.values.push_back(v);used[k]=1;}}
   double magnitude=std::abs(a.bias);require(magnitude<std::numeric_limits<double>::max()/4,"pair bound too large");
   for(double v:a.values){double term=std::abs(v);require(term<std::numeric_limits<double>::max()/4-magnitude,"pair bound too large");magnitude+=term;}
   // Reject pathological models for which an unevaluated pair could overflow.
   a.ids.shrink_to_fit();a.values.shrink_to_fit();pair.push_back(std::move(a));}
  for(int k=0;k<nsv;++k)if(used[k])active.push_back(k);
  // An overflowed centroid only disables the optional ordering heuristic.
  for(int i=0;i<c;++i)for(int d=0;d<16;++d){double sum=0;for(uint32_t k=start[i];k<start[i]+counts[i];++k)sum+=sv[16*k+d]/double(counts[i]);centers[16*i+d]=sum;}
  if(!keep_dense){std::vector<double>().swap(dense);}
 }
 int pair_index(int i,int j)const{if(i>j)std::swap(i,j);return i*(2*c-i-1)/2+j-i-1;}
 double kvalue(uint32_t id,const float*x){if(epoch[id]==serial)return kernel[id];double dist=0;for(int d=0;d<16;++d){double t=double(x[d])-sv[16*id+d];dist+=t*t;}
  // Finite inputs with overflowing squared distance represent exp(-infinity)=0.
  double v=std::exp(-gamma*dist);require(std::isfinite(v)&&v>=0.0&&v<=1.0,"invalid kernel");kernel[id]=v;epoch[id]=serial;++work.kernels;return v;}
 void kbatch(const uint32_t*ids,int count,const float*x){
#if defined(__AVX2__)
  if(count==4){__m256d sum=_mm256_setzero_pd();for(int d=0;d<16;++d){__m256d a=_mm256_sub_pd(_mm256_set1_pd(double(x[d])),
    _mm256_setr_pd(sv[16*ids[0]+d],sv[16*ids[1]+d],sv[16*ids[2]+d],sv[16*ids[3]+d]));sum=_mm256_add_pd(sum,_mm256_mul_pd(a,a));}
   double dist[4];_mm256_storeu_pd(dist,sum);for(int k=0;k<4;++k){double v=std::exp(-gamma*dist[k]);require(std::isfinite(v)&&v>=0.0&&v<=1.0,"invalid vector kernel");kernel[ids[k]]=v;epoch[ids[k]]=serial;++work.kernels;}return;}
#endif
  for(int k=0;k<count;++k)kvalue(ids[k],x);
 }
 void prepare_kernels(const std::vector<uint32_t>&ids,const float*x,bool vectorized){uint32_t todo[4];int len=0;
  for(uint32_t id:ids)if(epoch[id]!=serial){if(!vectorized)kvalue(id,x);else{todo[len++]=id;if(len==4){kbatch(todo,4,x);len=0;}}}
  if(len)kbatch(todo,len,x);
 }
 int edge(int i,int j,const float*x,int mode){if(i>j)std::swap(i,j);int p=pair_index(i,j);double sum=0;
  if(mode==0){require(has_dense,"dense mode was not prepared");for(auto half:{std::pair<int,int>{i,j-1},std::pair<int,int>{j,i}}){
    for(uint32_t k=start[half.first];k<start[half.first]+counts[half.first];++k){sum+=dense[half.second*nsv+k]*kernel[k];}
    work.terms+=counts[half.first];}sum+=bias[p];}
  else{const Pair&a=pair[p];if(mode!=1&&mode!=4&&mode!=7)prepare_kernels(a.ids,x,mode!=8);
   for(size_t t=0;t<a.ids.size();++t){sum+=a.values[t]*kernel[a.ids[t]];}
   work.terms+=a.ids.size();sum+=a.bias;}
  require(std::isfinite(sum),"nonfinite pair decision");
  if(c==2){return sum>=0?1:0;}
  return sum>0?i:j;
 }
 int run(const float*x,int count,int mode){valid_certificate=false;require(x&&count==16,"expected sixteen features");require(mode>=0&&mode<=8,"invalid mode");for(int i=0;i<16;++i)require(std::isfinite(x[i]),"nonfinite input");
  work=Work{};work.classes=c;work.supports=nsv;work.kind=mode;
  if(++serial==0){std::fill(epoch.begin(),epoch.end(),0);serial=1;}vote.reset(work);
  if(mode==0)for(int k=0;k<nsv;++k)kvalue(k,x);else if(mode==1||mode==4||mode==7)prepare_kernels(active,x,mode!=7);
  std::fill(priority.begin(),priority.end(),0);
  if(mode==2||mode==4||mode==8)for(int i=0;i<c;++i){double q=0;for(int d=0;d<16;++d){double v=double(x[d])-centers[16*i+d];q+=v*v;}priority[i]=std::isfinite(q)?q:std::numeric_limits<double>::infinity();}
  int out=vote.run([&](int i,int j){return edge(i,j,x,mode);},priority,mode==7?1:(mode==8?2:mode));valid_certificate=true;return out;
 }
 uint64_t resident_payload()const{uint64_t n=sizeof(*this)+pair.capacity()*sizeof(Pair)+8*(sv.capacity()+dense.capacity()+bias.capacity()+centers.capacity());n+=4*(counts.capacity()+start.capacity()+active.capacity());for(const auto&p:pair)n+=4*p.ids.capacity()+8*p.values.capacity();return n;}
 uint64_t scratch_payload()const{return 8*(kernel.capacity()+epoch.capacity()+priority.capacity())+vote.known.capacity()+sizeof(int)*(vote.low.capacity()+vote.remain.capacity());}
};
}
extern "C" {
const char*lo_error(){return lo::error.c_str();}
void*lo_create(const char*p,int dense){void*out=nullptr;lo::protect([&]{lo::require(dense==0||dense==1,"invalid preparation mode");out=new lo::Engine(p,bool(dense));});return out;}
void lo_destroy(void*p){delete static_cast<lo::Engine*>(p);}
int lo_info(void*p,uint64_t*out,int n){return lo::protect([&]{lo::require(p&&out&&n==6,"invalid info request");auto&e=*static_cast<lo::Engine*>(p);uint64_t terms=0;for(auto&q:e.pair)terms+=q.ids.size();uint64_t a[]={uint64_t(e.c),uint64_t(e.nsv),terms,uint64_t(e.active.size()),e.resident_payload(),e.scratch_payload()};std::copy(a,a+6,out);});}
int lo_run(void*p,const float*x,int n,int mode,int*out,uint64_t*stats,int cap){return lo::protect([&]{lo::require(p&&out&&stats&&cap==7,"invalid output/stats");auto&e=*static_cast<lo::Engine*>(p);int result=e.run(x,n,mode);auto&w=e.work;uint64_t a[]={w.kernels,w.pairs,w.terms,w.cert_checks,w.classes,w.supports,w.kind};std::copy(a,a+7,stats);*out=result;});}
int lo_certificate(void*p,int8_t*out,int count){return lo::protect([&]{lo::require(p&&out,"invalid certificate buffer");auto&e=*static_cast<lo::Engine*>(p);lo::require(e.valid_certificate,"no successful prediction to certify");lo::require(count==e.c*(e.c-1)/2,"certificate size mismatch");int k=0;for(int i=0;i<e.c;++i)for(int j=i+1;j<e.c;++j)out[k++]=e.vote.known[i*e.c+j];});}
int lo_kernels(void*p,const float*x,int n,int vectorized,double*out,int cap){return lo::protect([&]{lo::require(p&&x&&n==16&&out&&(vectorized==0||vectorized==1),"invalid kernel probe");auto&e=*static_cast<lo::Engine*>(p);lo::require(cap==e.nsv,"kernel capacity mismatch");for(int d=0;d<16;++d)lo::require(std::isfinite(x[d]),"nonfinite input");e.valid_certificate=false;e.work=lo::Work{};if(++e.serial==0){std::fill(e.epoch.begin(),e.epoch.end(),0);e.serial=1;}std::vector<uint32_t>ids(e.nsv);for(int i=0;i<e.nsv;++i)ids[i]=i;e.prepare_kernels(ids,x,bool(vectorized));std::copy(e.kernel.begin(),e.kernel.end(),out);});}
// Test the exact same scheduler with arbitrary abstract tournament outcomes.
int lo_tournament(int c,const int8_t*all,int n,int seed,int mode,int*out,int8_t*known,int*queries){return lo::protect([&]{lo::require(c>=2&&c<=128&&all&&out&&known&&queries&&n==c*(c-1)/2&&seed>=0&&seed<c&&mode>=0&&mode<=6,"invalid tournament arguments");
 lo::Votes v(c);lo::Work w;v.reset(w);std::vector<double>prior(c,1);prior[seed]=0;
 for(int i=0,k=0;i<c;++i)for(int j=i+1;j<c;++j,++k)lo::require(all[k]==i||all[k]==j,"malformed tournament edge");
 int result=v.run([&](int i,int j){if(i>j)std::swap(i,j);return int(all[i*(2*c-i-1)/2+j-i-1]);},prior,mode);
 *out=result;*queries=int(w.pairs);for(int i=0,k=0;i<c;++i)for(int j=i+1;j<c;++j)known[k++]=v.known[i*c+j];});}
}
#ifdef LO_MAIN
int main(int argc,char**argv){try{lo::require(argc==3,"usage: lazy-ovo MODEL MODE < sixteen-feature rows");std::string arg(argv[2]);size_t pos;int mode=std::stoi(arg,&pos);lo::require(pos==arg.size(),"invalid mode");lo::Engine e(argv[1],mode==0);std::array<float,16>x;
 while(std::cin>>x[0]){for(int d=1;d<16;++d)lo::require(bool(std::cin>>x[d]),"truncated input");std::cout<<e.run(x.data(),16,mode)<<'\n';}lo::require(std::cin.eof(),"invalid input token");return 0;
 }catch(const std::exception&e){std::cerr<<"lazy-ovo: "<<e.what()<<'\n';return 2;}}
#endif
