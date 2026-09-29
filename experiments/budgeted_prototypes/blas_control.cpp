// Stronger same-network execution; keep original strict rowwise control separately.
#include "controls.cpp"
extern "C" void scipy_cblas_dgemm(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
extern "C" void scipy_openblas_set_num_threads(int);
extern "C" const char* scipy_openblas_get_config();
extern "C" const char* ct_blas_config(){scipy_openblas_set_num_threads(1);return scipy_openblas_get_config();}
extern "C" int ct_blas_run(void*p,const uint8_t*q,int rows,int d,int mode,int*out,double*scores,uint64_t cells){return lo::protect([&]{
 lo::require(p,"closed BLAS network");auto&m=*static_cast<ct::Network*>(p);ct::environment();
 lo::require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&mode==0&&(!rows||(q&&out))&&(!scores||cells==uint64_t(rows)*m.c)&&cells<=8000000,"invalid BLAS buffers");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(q[i]<=m.maximum,"control input outside integer range");
 if(!rows)return;
 std::vector<double> a(size_t(rows)*m.width),b(size_t(rows)*m.width);
 for(int r=0;r<rows;++r)for(uint32_t f=0;f<m.d;++f)a[size_t(r)*m.d+f]=(double(q[size_t(r)*m.d+f])/double(m.maximum)-m.mean[f])/m.scale[f];
 for(size_t k=0;k<m.layers.size();++k){const auto&l=m.layers[k];
  scipy_cblas_dgemm(101,111,111,rows,int(l.out),int(l.in),1.,a.data(),int(l.in),l.w.data(),int(l.out),0.,b.data(),int(l.out));
  for(int r=0;r<rows;++r)for(uint32_t j=0;j<l.out;++j){double value=b[size_t(r)*l.out+j]+l.b[j];lo::require(std::isfinite(value),"nonfinite BLAS activation");b[size_t(r)*l.out+j]=k+1<m.layers.size()&&value<0?0:value;}
  a.swap(b);
 }
 for(int r=0;r<rows;++r){const double*s=a.data()+size_t(r)*m.c;out[r]=int(std::max_element(s,s+m.c)-s);if(scores)std::copy(s,s+m.c,scores+size_t(r)*m.c);}
});}
