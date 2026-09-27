// Clean implementation from Beretta et al., Algorithm 1 / Sec.4.4 (2023).
// The optional early certificate is an explicitly labelled strengthening.
// Cache is the existing O(C^2) vote cache, not their linear-space implementation.
#pragma once
#include <array>
#include <algorithm>
#include <stdexcept>
namespace et {
template<class Votes,class Edge>
int beretta(Votes&v,Edge edge,const std::vector<double>&priority,int mode,uint64_t&rounds){
    const int n=v.c;
    std::array<int,128>order{},a{},lost{},next{};
    for(int i=0;i<n;++i)order[i]=i;
    if(mode!=3)std::sort(order.begin(),order.begin()+n,[&](int i,int j){
        return priority[i]<priority[j]||(priority[i]==priority[j]&&i<j);
    });
    auto get=[&](int i,int j){int w=v.known[i*n+j];if(w<0){w=edge(i,j);v.add(i,j,w);}return w;};
    for(int alpha=1;alpha<=2*n;alpha*=2){
        ++rounds;std::fill(lost.begin(),lost.end(),0);
        int alive=n;
        if(mode==3){
            std::copy(order.begin(),order.begin()+n,a.begin());int p=0,q=1;
            while(alive>2*alpha){
                if(q>=alive){++p;q=p+1;}
                if(q>=alive)throw std::logic_error("published array scan exhausted");
                int u=a[p],w=a[q],winner=get(u,w),loser=winner==u?w:u;
                if(++lost[loser]>=alpha){
                    --alive;
                    if(loser==u){std::swap(a[p],a[alive]);q=p+1;}
                    else std::swap(a[q],a[alive]);
                }else ++q;
            }
        }else{
            // Stable linked-list order, O(1) removals, as Sec.4.4 describes.
            for(int i=0;i<n;++i)next[i]=i+1;
            int head=0,prev=-1,p=head;
            while(alive>2*alpha){
                if(p>=n||next[p]>=n)throw std::logic_error("published ordered scan exhausted");
                int before=p,q=next[p];bool removed_p=false;
                while(q<n&&alive>2*alpha){
                    int u=order[p],w=order[q],winner=get(u,w),loser=winner==u?w:u;
                    if(++lost[loser]>=alpha){
                        --alive;
                        if(loser==u){
                            int successor=next[p];if(prev<0)head=successor;else next[prev]=successor;
                            p=successor;removed_p=true;break;
                        }else{next[before]=next[q];q=next[q];}
                    }else{before=q;q=next[q];}
                }
                if(!removed_p&&alive>2*alpha){prev=p;p=next[p];}
            }
            int k=0;for(int i=head;i<n;i=next[i])a[k++]=order[i];
            if(k!=alive)throw std::logic_error("alive list mismatch");
        }
        int best=-1,best_losses=n;
        for(int k=0;k<alive;++k){
            int u=a[k];for(int j=0;j<n;++j)if(j!=u)get(u,j);
            int losses=n-1-v.low[u];
            if(losses<best_losses||(losses==best_losses&&(best<0||u<best))){best=u;best_losses=losses;}
            if(mode==5){int certified=v.certificate();if(certified>=0)return certified;}
        }
        if(best_losses<alpha){
            int certified=v.certificate();
            if(certified!=best)throw std::logic_error("published result failed first-index certificate");
            return best;
        }
    }
    throw std::logic_error("loss-threshold doubling failed to terminate");
}
}
