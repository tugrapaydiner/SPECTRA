#ifndef PROBE_RUNTIME
#error PROBE_RUNTIME required
#endif
#include PROBE_RUNTIME
extern "C" int probe_margins(const unsigned char* model,size_t size,int tables,
    const double* inputs,int rows,double* output,int path) {
 return lo::protect([&] {
  auto m=std::make_shared<const spm::Model>(model,size,bool(tables));spm::Worker w(m);
  lo::require(m->c==2 && rows>=0 && path>=0,"binary model required");
  int r=0;
#ifdef PROBE_CANDIDATE
#if defined(__AVX2__)
  if(path==2 && !tables) {
   std::vector<double> packed(size_t(m->d)*4);
   for(;r+4<=rows;r+=4) w.binary_four_scores(inputs+size_t(r)*m->d,packed.data(),output+r);
  }
#endif
#endif
  for(;r<rows;++r) {
   const double* x=inputs+size_t(r)*m->d;w.begin(x);
#ifdef PROBE_CANDIDATE
   if(path) {output[r]=w.binary_stream_score(x);continue;}
#endif
   // Unmodified reference calculation: all kernels first, then original terms.
   const auto& p=m->pairs[0];w.prepare(p.ids,x);double score=0;
   for(size_t t=0;t<p.ids.size();++t)score+=p.values[t]*w.kernel[p.ids[t]];
   output[r]=score+p.bias;
  }
 });
}
