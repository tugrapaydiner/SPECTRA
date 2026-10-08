// Exact persistent list-colouring quotient. MIT: repository LICENSE.
// Classical parity/SCC compilation; no algorithmic-first or learning claim.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
namespace {
using U=uint32_t; using W=uint64_t;
constexpr U NIL=UINT32_MAX, MAX_N=100000, MAX_E=2000000;
struct Invalid:std::runtime_error{using std::runtime_error::runtime_error;};
struct Resource:std::runtime_error{using std::runtime_error::runtime_error;};
W number(PyObject* p,const char* message,W cap=UINT64_MAX){
 if(!PyLong_CheckExact(p))throw Invalid(message);
 W x=PyLong_AsUnsignedLongLong(p);
 if(PyErr_Occurred()){PyErr_Clear();throw Invalid(message);} if(x>cap)throw Invalid(message);return x;
}
U pop(W x){return static_cast<U>(__builtin_popcountll(x));}
U first(W x){return static_cast<U>(__builtin_ctzll(x));}
W low(W x){return x&(~x+1);}
W full(U k){return k==64?~W(0):((W(1)<<k)-1);}
bool single(W x){return x&&!(x&(x-1));}
struct DSU{
 std::vector<U> p,size;std::vector<uint8_t> parity;
 explicit DSU(U n):p(n),size(n,1),parity(n,0){std::iota(p.begin(),p.end(),0);}
 std::pair<U,U> find(U v){
  U root=v,phase=0;while(p[root]!=root){phase^=parity[root];root=p[root];}
  U accumulated=0;while(p[v]!=v){U next=p[v],old=parity[v];p[v]=root;parity[v]=static_cast<uint8_t>(phase^accumulated);accumulated^=old;v=next;}
  return {root,phase};
 }
 bool join(U a,U b,U diff){auto x=find(a),y=find(b);if(x.first==y.first)return (x.second^y.second)==diff;
  U pa=x.first,pb=y.first;if(size[pa]<size[pb])std::swap(pa,pb);
  p[pb]=pa;parity[pb]=static_cast<uint8_t>(x.second^y.second^diff);size[pa]+=size[pb];return true;
 }
};
struct Arc{U target;W forbidden;};
struct Triple{U source,target;W mask;};
struct Index{
 U n,k,qn=0,atoms=0;W input_edges=0,bytes=0,build_bound=0;bool impossible=false;
 std::vector<W> original,palette,initial;
 std::vector<U> node,start,off,degree;std::vector<uint8_t> colour0,colour1,wide;
 std::vector<Arc> arcs;
 U atom(U q,U colour)const{return start[q]+pop(palette[q]&((W(1)<<colour)-1));}
 std::pair<U,U> map(U v,U colour)const{
  U q=node[v];if(wide[v])return {q,colour};
  if(colour0[v]==colour)return {q,0};
  if(colour1[v]==colour)return {q,1};
  throw std::logic_error("unmapped original colour");
 }
 Index(PyObject* nv,PyObject* kv,PyObject* edges,PyObject* masks,U mode,W cap){
  n=static_cast<U>(number(nv,"n outside [0,100000]",MAX_N));k=static_cast<U>(number(kv,"k outside [0,64]",64));
  if(!PyTuple_CheckExact(edges)||!PyTuple_CheckExact(masks))throw Invalid("exact tuples required");
  input_edges=static_cast<W>(PyTuple_GET_SIZE(edges));if(input_edges>MAX_E)throw Resource("too many edges");
  if(PyTuple_GET_SIZE(masks)&&static_cast<W>(PyTuple_GET_SIZE(masks))!=n)throw Invalid("mask count differs");
  if(64*W(n)+32*input_edges+128>cap)throw Resource("basic construction admission exceeds cap");
  original.assign(n,full(k));node.assign(n,NIL);colour0.assign(n,255);colour1.assign(n,255);wide.assign(n,1);
  std::vector<U> bid(n,NIL),bv;bv.reserve(n);
  for(U v=0;v<n;++v){if(PyTuple_GET_SIZE(masks))original[v]=number(PyTuple_GET_ITEM(masks,v),"invalid mask");
   if(original[v]&~full(k))throw Invalid("mask outside k");
   if(!original[v])impossible=true;
   if(mode&&original[v]&&pop(original[v])<=2){bid[v]=static_cast<U>(bv.size());bv.push_back(v);}
  }
  std::vector<std::pair<U,U>> es;es.reserve(input_edges);W pairs=0;
  for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i){auto* e=PyTuple_GET_ITEM(edges,i);if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)throw Invalid("edge must be exact pair tuple");
   U a=static_cast<U>(number(PyTuple_GET_ITEM(e,0),"bad endpoint",MAX_N)),b=static_cast<U>(number(PyTuple_GET_ITEM(e,1),"bad endpoint",MAX_N));
   if(a>=n||b>=n)throw Invalid("edge endpoint outside n");
   if(a==b)impossible=true;
   es.emplace_back(a,b);pairs+=pop(original[a]&original[b]);
  }
  // Conservative bound for all simultaneously live array payloads (not RSS).
  build_bound=256*W(n)+96*input_edges+128*pairs+4096;
  if(build_bound>cap)throw Resource("complete construction payload bound exceeds cap");
  std::vector<U> comp,lit;
  if(mode){
   U nb=static_cast<U>(bv.size());DSU dsu(nb);
   if(mode==1||mode==3)for(auto e:es){if(bid[e.first]==NIL||bid[e.second]==NIL)continue;
    if(pop(original[e.first])==2&&original[e.first]==original[e.second])if(!dsu.join(bid[e.first],bid[e.second],1))impossible=true;
   }
   std::vector<U> ids(nb,NIL);U nr=0;lit.resize(2*nb);
   for(U b=0;b<nb;++b){auto rp=dsu.find(b);if(ids[rp.first]==NIL)ids[rp.first]=nr++;
    lit[2*b]=2*ids[rp.first]+rp.second;lit[2*b+1]=lit[2*b]^1U;
   }
   comp.resize(2*nr);std::iota(comp.begin(),comp.end(),0);
   if(mode==2||mode==3){
    std::vector<std::pair<U,U>> links;links.reserve(2*pairs+nb);
    for(U b=0;b<nb;++b)if(single(original[bv[b]]))links.emplace_back(lit[2*b+1],lit[2*b]);
    for(auto e:es){U a=e.first,b=e.second;if(bid[a]==NIL||bid[b]==NIL)continue;
     W common=original[a]&original[b];while(common){W c=low(common);common&=common-1;
      U la=lit[2*bid[a]+(c==low(original[a])?0:1)],lb=lit[2*bid[b]+(c==low(original[b])?0:1)];
      if(la!=(lb^1U)){links.emplace_back(la,lb^1U);links.emplace_back(lb,la^1U);}
     }
    }
    std::sort(links.begin(),links.end());links.erase(std::unique(links.begin(),links.end()),links.end());
    U nn=2*nr;std::vector<U> fo(nn+1,0),ro(nn+1,0);
    for(auto e:links){++fo[e.first+1];++ro[e.second+1];}for(U v=0;v<nn;++v){fo[v+1]+=fo[v];ro[v+1]+=ro[v];}
    std::vector<U> fc(fo),rc(ro),ft(links.size()),rt(links.size());
    for(auto e:links){ft[fc[e.first]++]=e.second;rt[rc[e.second]++]=e.first;}
    std::vector<uint8_t> seen(nn,0);std::vector<U> order;order.reserve(nn);std::vector<std::pair<U,U>> stack;stack.reserve(nn);
    for(U root=0;root<nn;++root)if(!seen[root]){seen[root]=1;stack.emplace_back(root,fo[root]);
     while(!stack.empty()){auto& f=stack.back();if(f.second==fo[f.first+1]){order.push_back(f.first);stack.pop_back();continue;}
      U u=ft[f.second++];if(!seen[u]){seen[u]=1;stack.emplace_back(u,fo[u]);}}
    }
    std::fill(comp.begin(),comp.end(),NIL);U nc=0;
    for(auto it=order.rbegin();it!=order.rend();++it)if(comp[*it]==NIL){comp[*it]=nc;stack.emplace_back(*it,0);
     while(!stack.empty()){U v=stack.back().first;stack.pop_back();for(U j=ro[v];j<ro[v+1];++j){U u=rt[j];if(comp[u]==NIL){comp[u]=nc;stack.emplace_back(u,0);}}}++nc;
    }
   }
   std::vector<U> owner(comp.size(),NIL),side(comp.size(),NIL);
   for(U b=0;b<nb;++b){U c0=comp[lit[2*b]],c1=comp[lit[2*b+1]],v=bv[b];
    if(c0==c1){impossible=true;continue;}
    if(owner[c0]==NIL){owner[c0]=owner[c1]=qn++;side[c0]=0;side[c1]=1;palette.push_back(3);initial.push_back(3);}
    if(owner[c0]!=owner[c1]||side[c0]==side[c1])throw std::logic_error("SCC complementation mismatch");
    U q=owner[c0];node[v]=q;wide[v]=0;U lo=first(original[v]);W hi=original[v]^low(original[v]);
    if(side[c0]==0){colour0[v]=static_cast<uint8_t>(lo);if(hi)colour1[v]=static_cast<uint8_t>(first(hi));}
    else{colour1[v]=static_cast<uint8_t>(lo);if(hi)colour0[v]=static_cast<uint8_t>(first(hi));}
    if(!hi)initial[q]&=W(1)<<side[c0];
   }
  }
  // In a contradictory instance unmapped binary vertices are admitted as ordinary
  // nodes solely for geometry integrity; solve never claims them satisfiable.
  for(U v=0;v<n;++v)if(node[v]==NIL){node[v]=qn++;wide[v]=1;palette.push_back(original[v]);initial.push_back(original[v]);}
  start.resize(qn+1,0);for(U q=0;q<qn;++q)start[q+1]=start[q]+pop(palette[q]);atoms=start[qn];
  std::vector<Triple> triples;triples.reserve(2*pairs);
  for(auto e:es){W common=original[e.first]&original[e.second];while(common){U c=first(common);common&=common-1;auto a=map(e.first,c),b=map(e.second,c);
   if(a.first==b.first){if(a.second==b.second)initial[a.first]&=~(W(1)<<a.second);continue;}
   triples.push_back({atom(a.first,a.second),b.first,W(1)<<b.second});triples.push_back({atom(b.first,b.second),a.first,W(1)<<a.second});
  }}
  std::sort(triples.begin(),triples.end(),[](const Triple& a,const Triple& b){return std::tie(a.source,a.target)<std::tie(b.source,b.target);});
  off.assign(size_t(atoms)+1,0);degree.assign(qn,0);arcs.reserve(triples.size());
  for(size_t i=0;i<triples.size();){size_t j=i+1;W mask=triples[i].mask;while(j<triples.size()&&triples[j].source==triples[i].source&&triples[j].target==triples[i].target)mask|=triples[j++].mask;
   ++off[triples[i].source+1];arcs.push_back({triples[i].target,mask});i=j;
  }
  for(U a=0;a<atoms;++a)off[a+1]+=off[a];
  for(U q=0;q<qn;++q)degree[q]=off[start[q+1]]-off[start[q]];
  for(W d:initial)if(!d)impossible=true;
  bytes=8*(original.capacity()+palette.capacity()+initial.capacity())+4*(node.capacity()+start.capacity()+off.capacity()+degree.capacity())+colour0.capacity()+colour1.capacity()+wide.capacity()+sizeof(Arc)*arcs.capacity();
  if(bytes>build_bound)throw std::logic_error("index bound underestimated");
 }
};
struct Change{U q;W old;};struct Frame{size_t mark;U q;W options;};
struct Result{std::vector<W> d;U reason=2;W work=0,branches=0,backtracks=0,reductions=0,peak=0,state=0,trace=14695981039346656037ULL;};
struct Search{
 const Index& x;W limit,trail_bound;bool core_first,conflict=false;Result r;std::vector<Change> trail;std::vector<Frame> frames;std::vector<U> queue;size_t head=0;
 Search(const Index& index,W work,W memory,bool core):x(index),limit(work),core_first(core){
  trail_bound=std::min(64*W(x.qn),W(x.arcs.size())+x.qn);
  r.state=12*W(x.qn)+sizeof(Frame)*W(x.qn)+sizeof(Change)*trail_bound;
  if(r.state>memory)throw Resource("query state payload exceeds cap");
  r.d=x.initial;trail.reserve(trail_bound);frames.reserve(x.qn);queue.reserve(x.qn);
 }
 void event(W v){r.trace=(r.trace^v)*1099511628211ULL;}
 bool spend(){if(r.work==limit)return false;++r.work;return true;}
 void reduce(U q,W d){W old=r.d[q];if(old==d)return;if(d&~old)throw std::logic_error("domain enlargement");
  if(trail.size()==trail_bound)throw std::logic_error("trail cap proof failed");
  trail.push_back({q,old});r.d[q]=d;++r.reductions;r.peak=std::max(r.peak,W(trail.size()));event((W(q)<<32)^d);
  if(!d)conflict=true;else if(single(d))queue.push_back(q);
 }
 U propagate(){while(head<queue.size()&&!conflict){U q=queue[head++];W d=r.d[q];if(!single(d)){conflict=true;break;}U a=x.atom(q,first(d));
   for(U j=x.off[a];j<x.off[a+1]&&!conflict;++j){if(!spend())return 2;const auto& e=x.arcs[j];reduce(e.target,r.d[e.target]&~e.forbidden);}
  }if(conflict)return 1;queue.clear();head=0;return 0;
 }
 void restore(size_t mark){queue.clear();head=0;conflict=false;while(trail.size()>mark){auto c=trail.back();trail.pop_back();r.d[c.q]=c.old;}event(UINT64_MAX);}
 U choose(){U best=NIL,size=65,degree=0;bool have_core=false;
  for(U q=0;q<x.qn;++q){U s=pop(r.d[q]);if(s<2)continue;bool core=pop(x.palette[q])>2;
   if(core_first&&have_core&&!core)continue;
   if((core_first&&core&&!have_core)||s<size||(s==size&&x.degree[q]>degree)){best=q;size=s;degree=x.degree[q];have_core=core;}
  }return best;
 }
 Result run(PyObject* restrictions){
  if(!PyTuple_CheckExact(restrictions))throw Invalid("query restrictions must be tuple");
  if(static_cast<W>(PyTuple_GET_SIZE(restrictions))>2*W(x.n)+1)throw Invalid("too many restrictions");
  for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(restrictions);++i){auto* pair=PyTuple_GET_ITEM(restrictions,i);if(!PyTuple_CheckExact(pair)||PyTuple_GET_SIZE(pair)!=2)throw Invalid("restriction must be (vertex,mask)");
   U v=static_cast<U>(number(PyTuple_GET_ITEM(pair,0),"invalid restriction vertex",MAX_N));W d=number(PyTuple_GET_ITEM(pair,1),"invalid restriction mask");
   if(v>=x.n||(d&~full(x.k)))throw Invalid("restriction outside graph");
   U q=x.node[v];W translated=d&x.original[v];
   if(!x.wide[v])translated=((x.colour0[v]!=255&&(d&(W(1)<<x.colour0[v])))?1:0)|((x.colour1[v]!=255&&(d&(W(1)<<x.colour1[v])))?2:0);
   r.d[q]&=translated;
  }
  if(x.impossible){r.reason=3;return std::move(r);}for(U q=0;q<x.qn;++q){if(!r.d[q]){r.reason=4;return std::move(r);}if(single(r.d[q]))queue.push_back(q);}
  while(true){U state=propagate();if(state==2){r.reason=1;break;}
   if(state==1){bool resumed=false;while(!frames.empty()){auto& f=frames.back();restore(f.mark);++r.backtracks;
     if(f.options){if(!spend()){r.reason=1;return std::move(r);}W c=low(f.options);f.options&=f.options-1;++r.branches;reduce(f.q,c);resumed=true;break;}frames.pop_back();
    }if(!resumed){r.reason=2;break;}continue;}
   U q=choose();if(q==NIL){r.reason=0;break;}if(!spend()){r.reason=1;break;}
   W c=low(r.d[q]);frames.push_back({trail.size(),q,r.d[q]^c});++r.branches;reduce(q,c);
  }return std::move(r);
 }
};
constexpr const char* TAG="spectra.quotient.query.v1";
void destroy(PyObject* p){void* x=PyCapsule_GetPointer(p,TAG);if(x)delete static_cast<Index*>(x);else PyErr_Clear();}
PyObject* fail(){try{throw;}catch(const Invalid& e){PyErr_SetString(PyExc_ValueError,e.what());}catch(const Resource& e){PyErr_SetString(PyExc_MemoryError,e.what());}catch(const std::bad_alloc&){PyErr_NoMemory();}catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());}catch(...){PyErr_SetString(PyExc_RuntimeError,"unknown native error");}return nullptr;}
PyObject* create(PyObject*,PyObject* args){PyObject *n,*k,*e,*m,*mode,*cap;if(!PyArg_ParseTuple(args,"OOOOOO",&n,&k,&e,&m,&mode,&cap))return nullptr;
 try{auto x=std::make_unique<Index>(n,k,e,m,static_cast<U>(number(mode,"mode outside 0..3",3)),number(cap,"invalid build cap"));auto c=PyCapsule_New(x.get(),TAG,destroy);if(c)x.release();return c;}catch(...){return fail();}}
PyObject* solve(PyObject*,PyObject* args){PyObject *cap,*qs,*work,*mem,*core;if(!PyArg_ParseTuple(args,"OOOOO",&cap,&qs,&work,&mem,&core))return nullptr;
 auto* x=static_cast<Index*>(PyCapsule_GetPointer(cap,TAG));if(!x)return nullptr;
 try{if(!PyBool_Check(core))throw Invalid("core_first must be bool");Search s(*x,number(work,"invalid work cap"),number(mem,"invalid state cap"),core==Py_True);Result r=s.run(qs);
  PyObject* labels=PyBytes_FromStringAndSize(nullptr,r.reason==0?x->n:0);if(!labels)return nullptr;
  if(r.reason==0){auto* data=reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(labels));for(U v=0;v<x->n;++v){U c=first(r.d[x->node[v]]);data[v]=x->wide[v]?static_cast<uint8_t>(c):(c==0?x->colour0[v]:x->colour1[v]);if(data[v]>=x->k){Py_DECREF(labels);throw std::logic_error("invalid lifted colour");}}}
  return Py_BuildValue("(NIKKKKKKKKKK)",labels,static_cast<unsigned>(r.reason),
   static_cast<unsigned long long>(r.work),static_cast<unsigned long long>(r.branches),static_cast<unsigned long long>(r.backtracks),static_cast<unsigned long long>(r.reductions),static_cast<unsigned long long>(r.peak),static_cast<unsigned long long>(r.state),static_cast<unsigned long long>(r.trace),static_cast<unsigned long long>(x->qn),static_cast<unsigned long long>(x->bytes),static_cast<unsigned long long>(x->arcs.size()));
 }catch(...){return fail();}}
PyObject* info(PyObject*,PyObject* cap){auto* x=static_cast<Index*>(PyCapsule_GetPointer(cap,TAG));if(!x)return nullptr;return Py_BuildValue("(IIIKKK)",x->n,x->qn,x->atoms,static_cast<unsigned long long>(x->bytes),static_cast<unsigned long long>(x->build_bound),static_cast<unsigned long long>(x->arcs.size()));}
// Uncompiled original-input observer, shared by all experimental arms.
PyObject* check(PyObject*,PyObject* args){PyObject *nv,*kv,*edges,*masks,*query,*answer;if(!PyArg_ParseTuple(args,"OOOOOO",&nv,&kv,&edges,&masks,&query,&answer))return nullptr;
 try{U n=static_cast<U>(number(nv,"invalid n",MAX_N)),k=static_cast<U>(number(kv,"invalid k",64));if(!PyTuple_CheckExact(edges)||!PyTuple_CheckExact(masks)||!PyTuple_CheckExact(query)||!PyBytes_CheckExact(answer))Py_RETURN_FALSE;
  if(static_cast<W>(PyBytes_GET_SIZE(answer))!=n||(PyTuple_GET_SIZE(masks)&&static_cast<W>(PyTuple_GET_SIZE(masks))!=n))Py_RETURN_FALSE;
  auto* a=reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(answer));for(U v=0;v<n;++v){if(a[v]>=k)Py_RETURN_FALSE;if(PyTuple_GET_SIZE(masks)){W d=number(PyTuple_GET_ITEM(masks,v),"invalid mask");if((d&~full(k))||!(d&(W(1)<<a[v])))Py_RETURN_FALSE;}}
  for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i){auto* e=PyTuple_GET_ITEM(edges,i);if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)Py_RETURN_FALSE;W u=number(PyTuple_GET_ITEM(e,0),"bad endpoint"),v=number(PyTuple_GET_ITEM(e,1),"bad endpoint");if(u>=n||v>=n||a[u]==a[v])Py_RETURN_FALSE;}
  for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(query);++i){auto* e=PyTuple_GET_ITEM(query,i);if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)Py_RETURN_FALSE;W v=number(PyTuple_GET_ITEM(e,0),"bad restriction"),d=number(PyTuple_GET_ITEM(e,1),"bad restriction");if(v>=n||(d&~full(k))||!(d&(W(1)<<a[v])))Py_RETURN_FALSE;}
  Py_RETURN_TRUE;
 }catch(const Invalid&){Py_RETURN_FALSE;}catch(...){return fail();}}
PyMethodDef methods[]={{"create",create,METH_VARARGS,"Compile an exact quotient."},{"solve",solve,METH_VARARGS,"Solve restrictions and lift a full answer."},{"check",check,METH_VARARGS,"Check original graph and query independently."},{"info",info,METH_O,"Index resource and geometry information."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_quotient_query",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_quotient_query(){return PyModule_Create(&module);}
