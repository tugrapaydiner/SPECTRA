// Experiment adapter only; unmodified upstream LIBSVM performs every decision.
#include "svm.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <string>
static thread_local std::string error;
struct Owner {
    svm_model* model=nullptr;int d=0,c=0;
    std::vector<svm_node> nodes;std::vector<double> margins;
    ~Owner(){if(model)svm_free_and_destroy_model(&model);}
};
extern "C" {
const char* nb_error(){return error.c_str();}
void* nb_create(const char* file,int d){
    try {
        if(!file||d<1||d>4096)throw std::invalid_argument("invalid model dimensions");
        auto p=std::make_unique<Owner>();p->model=svm_load_model(file);
        if(!p->model)throw std::invalid_argument("LIBSVM load failed");
        p->d=d;p->c=svm_get_nr_class(p->model);
        if(p->c<2||p->c>128||p->model->param.svm_type!=C_SVC||p->model->param.kernel_type!=RBF)
            throw std::invalid_argument("unsupported comparison model");
        for(int i=0;i<p->c;++i)if(p->model->label[i]!=i)throw std::invalid_argument("invalid class ordering");
        for(int i=0;i<p->model->l;++i)for(const svm_node* n=p->model->SV[i];n->index!=-1;++n)
            if(n->index<1||n->index>d||!std::isfinite(n->value))throw std::invalid_argument("invalid support vector");
        p->nodes.resize(d+1);p->margins.resize(p->c*(p->c-1)/2);return p.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
void nb_destroy(void* pointer){delete static_cast<Owner*>(pointer);}
int nb_run(void* pointer,const double* x,int rows,int d,int* out,double* scores){
    try {
        auto* p=static_cast<Owner*>(pointer);
        if(!p||rows<0||rows>65536||d!=p->d||1LL*rows*d>8000000|| (rows&&(!x||!out)))throw std::invalid_argument("invalid input shape");
        for(size_t i=0;i<size_t(rows)*d;++i)if(!std::isfinite(x[i]))throw std::invalid_argument("nonfinite input");
        for(int r=0;r<rows;++r){
            int k=0;for(int j=0;j<d;++j)if(x[size_t(r)*d+j]!=0.)p->nodes[k++]={j+1,x[size_t(r)*d+j]};
            p->nodes[k]={-1,0.};
            double label=svm_predict_values(p->model,p->nodes.data(),p->margins.data());
            if(label<0||label>=p->c||!std::isfinite(label))throw std::runtime_error("bad prediction");
            out[r]=int(label);
            if(scores)for(size_t j=0;j<p->margins.size();++j)scores[size_t(r)*p->margins.size()+j]=(p->c==2?-1.:1.)*p->margins[j];
        }return 0;
    }catch(const std::exception& e){error=e.what();return 1;}
}
}
