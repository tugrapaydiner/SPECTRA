// Test-only observer. Compile against ORIGINAL_RUNTIME or candidate runtime.
// All buffer extents are supplied and checked; never part of the deployment ABI.
#ifndef OBSERVER_RUNTIME
#error Define OBSERVER_RUNTIME to the absolute runtime.cpp
#endif
#include OBSERVER_RUNTIME
extern "C" {
int bk_observe(const unsigned char* bytes,size_t length,int tables,int mode,
               const double* inputs,int rows,int d,double* distances,double* kernels,
               size_t kernel_count,double* scores,size_t score_count) {
    return lo::protect([&] {
#ifdef OBSERVER_NEW
        auto model=std::make_shared<const spm::Model>(bytes,length,bool(tables),mode);
#else
        lo::require(mode==0,"base observer supports only original mode");
        auto model=std::make_shared<const spm::Model>(bytes,length,bool(tables));
#endif
        lo::require(rows>=0 && rows<=65536 && d==model->d && inputs && distances && kernels && scores,
                    "invalid observer buffers");
        lo::require(kernel_count==size_t(rows)*model->nsv && score_count==size_t(rows)*model->pairs.size(),
                    "observer extent mismatch");
        spm::Worker worker(model);
        std::vector<uint32_t> ids(size_t(model->nsv));
        for (int k=0;k<model->nsv;++k) ids[k]=uint32_t(k);
        for(int row=0;row<rows;++row) {
            const double* x=inputs+size_t(row)*d;
            for(int f=0;f<d;++f) lo::require(std::isfinite(x[f]),"nonfinite observer input");
            worker.begin(x);
            for(int start=0;start<model->nsv;start+=4) {
                int n=std::min(4,model->nsv-start);
#ifdef OBSERVER_NEW
                if(worker.boolean_input) {worker.boolean_distances(ids.data()+start,n,distances+size_t(row)*model->nsv+start);continue;}
#endif
                if(d==16)worker.distance_kind<16>(ids.data()+start,n,x,distances+size_t(row)*model->nsv+start);
                else worker.distance_kind<0>(ids.data()+start,n,x,distances+size_t(row)*model->nsv+start);
            }
            worker.prepare(ids,x);
            std::copy(worker.kernel.begin(),worker.kernel.end(),kernels+size_t(row)*model->nsv);
            for(size_t j=0;j<model->pairs.size();++j) {
                const auto& pair=model->pairs[j];double result=0;
                for(size_t k=0;k<pair.ids.size();++k)result+=pair.values[k]*worker.kernel[pair.ids[k]];
                scores[size_t(row)*model->pairs.size()+j]=result+pair.bias;
            }
        }
    });
}
#ifdef OBSERVER_NEW
unsigned bk_population(uint64_t x){return spm::Worker::population(x);}
int bk_epoch_wrap(const unsigned char* bytes,size_t length,const double* input) {
    return lo::protect([&]{
        auto model=std::make_shared<const spm::Model>(bytes,length,false,2);spm::Worker worker(model);
        int a=worker.run(input,0,-1);
        std::fill(worker.distance_epochs.begin(),worker.distance_epochs.end(),uint64_t(1));
        std::fill(worker.distance_kernels.begin(),worker.distance_kernels.end(),-999.);
        worker.serial=UINT64_MAX;
        lo::require(worker.run(input,0,-1)==a,"stale lookup after epoch wrap");
        lo::require(worker.boolean_exps>0,"wrapped epoch reused stale distances");
    });
}
#endif
}
