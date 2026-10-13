// MIT, repository LICENSE. Independent original-input migration verifier.
// Does not accept a candidate quotient, reachability index, or reference answers.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
namespace {
using U=uint32_t;using W=uint64_t;
constexpr U NIL=UINT32_MAX,MAX_N=100000;
struct Bad:std::runtime_error{using std::runtime_error::runtime_error;};
struct Large:std::runtime_error{using std::runtime_error::runtime_error;};
W integer(PyObject* p,const char* error,W cap=UINT64_MAX){
 if(!PyLong_CheckExact(p))throw Bad(error);
 W value=PyLong_AsUnsignedLongLong(p);if(PyErr_Occurred()){PyErr_Clear();throw Bad(error);}
 if(value>cap)throw Bad(error);
 return value;
}
struct Observer{
 U n=0;W payload=0;
 std::vector<uint8_t> a,b;
 std::vector<std::pair<U,U>> edges;
 std::vector<U> starts,following;
 Observer(PyObject* ee,PyObject* aa,PyObject* bb,W cap){
  if(!PyTuple_CheckExact(ee)||!PyTuple_CheckExact(aa)||!PyTuple_CheckExact(bb))throw Bad("original inputs must be exact tuples");
  W nv=PyTuple_GET_SIZE(aa),m=PyTuple_GET_SIZE(ee);
  if(!nv||nv>MAX_N||m>2000000||PyTuple_GET_SIZE(bb)!=static_cast<Py_ssize_t>(nv))throw Bad("invalid original geometry");
  if(64*nv+64*m+8192>cap)throw Large("original observer construction exceeds cap");
  n=static_cast<U>(nv);a.resize(n);b.resize(n);edges.reserve(m);
  for(U v=0;v<n;++v){a[v]=static_cast<uint8_t>(integer(PyTuple_GET_ITEM(aa,v),"invalid original colour",63));b[v]=static_cast<uint8_t>(integer(PyTuple_GET_ITEM(bb,v),"invalid target colour",63));}
  for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(ee);++i){
   auto edge=PyTuple_GET_ITEM(ee,i);if(!PyTuple_CheckExact(edge)||PyTuple_GET_SIZE(edge)!=2)throw Bad("edge must be an exact pair");
   U u=static_cast<U>(integer(PyTuple_GET_ITEM(edge,0),"bad endpoint",n-1)),v=static_cast<U>(integer(PyTuple_GET_ITEM(edge,1),"bad endpoint",n-1));
   if(a[u]==a[v]||b[u]==b[v])throw Bad("original plans are not proper");
   edges.emplace_back(std::min(u,v),std::max(u,v));
  }
  std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
  starts.assign(n+1,0);
  for(auto e:edges){if(b[e.first]==a[e.second])++starts[e.first+1];if(b[e.second]==a[e.first])++starts[e.second+1];}
  for(U v=0;v<n;++v)starts[v+1]+=starts[v];
  following.resize(starts[n]);std::vector<U> next(starts);
  for(auto e:edges){if(b[e.first]==a[e.second])following[next[e.first]++]=e.second;if(b[e.second]==a[e.first])following[next[e.second]++]=e.first;}
  payload=a.capacity()+b.capacity()+8*W(edges.capacity())+4*W(starts.capacity()+following.capacity());
 }
};
struct Bank{std::vector<U> starts,vertices;std::vector<W> masks;};
Bank read_queries(PyObject* qs,U n,W cap){
 if(!PyTuple_CheckExact(qs))throw Bad("queries must be an exact tuple");
 W count=PyTuple_GET_SIZE(qs),pins=0;if(count>1000000)throw Large("too many queries");
 for(W i=0;i<count;++i){auto q=PyTuple_GET_ITEM(qs,i);if(!PyTuple_CheckExact(q))throw Bad("query must be an exact tuple");pins+=PyTuple_GET_SIZE(q);if(pins>16000000)throw Large("too many query pins");}
 if(16*pins+8*(count+1)+40*W(n)+8192>cap)throw Large("original observer scratch exceeds cap");
 Bank bank;bank.starts.resize(count+1,0);bank.vertices.reserve(pins);bank.masks.reserve(pins);
 for(W i=0;i<count;++i){auto q=PyTuple_GET_ITEM(qs,i);for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(q);++j){
  auto pin=PyTuple_GET_ITEM(q,j);if(!PyTuple_CheckExact(pin)||PyTuple_GET_SIZE(pin)!=2)throw Bad("pin must be an exact pair");
  bank.vertices.push_back(static_cast<U>(integer(PyTuple_GET_ITEM(pin,0),"bad query vertex",n-1)));bank.masks.push_back(integer(PyTuple_GET_ITEM(pin,1),"bad query mask"));
 }bank.starts[i+1]=static_cast<U>(bank.vertices.size());}
 return bank;
}
U decode(const uint8_t* p){return U(p[0])|(U(p[1])<<8)|(U(p[2])<<16)|(U(p[3])<<24);}
struct Counts{W sat=0,unsat=0,migrations=0,proof_edges=0;};
Counts check(const Observer& x,const Bank& q,const uint8_t* statuses,const uint8_t* labels,const uint8_t* kinds,
             const std::vector<U>& offsets,const uint8_t* paths,bool minimal){
 U count=static_cast<U>(q.starts.size()-1);Counts result;
 std::vector<U> moving(x.n,0),staying(x.n,0),visited(x.n,0),path_seen(x.n,0),queue;queue.reserve(x.n);
 for(U i=0;i<count;++i){
  U stamp=i+1,empty=NIL;queue.clear();const uint8_t* answer=labels+W(i)*x.n;
  for(U j=q.starts[i];j<q.starts[i+1];++j){U v=q.vertices[j];W mask=q.masks[j];
   bool aa=(mask>>x.a[v])&1,bb=(mask>>x.b[v])&1;
   if(!aa&&!bb)empty=v;
   if(x.a[v]!=x.b[v]){if(!aa)moving[v]=stamp;if(!bb)staying[v]=stamp;}
  }
  U begin=offsets[i],end=offsets[i+1];
  if(statuses[i]==0){
   if(kinds[i]!=0||begin!=end||empty!=NIL)throw Bad("SAT output carries a contradiction");
   for(U v=0;v<x.n;++v)if(answer[v]!=x.a[v]&&answer[v]!=x.b[v])throw Bad("answer is outside original plans");
   if(x.edges.size()<=2*W(x.n))
    for(auto e:x.edges)if(answer[e.first]==answer[e.second])throw Bad("answer violates an original conflict");
   for(U j=q.starts[i];j<q.starts[i+1];++j)if(!(q.masks[j]>>answer[q.vertices[j]]&1))throw Bad("answer violates a request");
   if(minimal){
    for(U j=q.starts[i];j<q.starts[i+1];++j){U v=q.vertices[j];if(moving[v]==stamp&&visited[v]!=stamp){visited[v]=stamp;queue.push_back(v);}}
    for(size_t at=0;at<queue.size();++at){U v=queue[at];for(U j=x.starts[v];j<x.starts[v+1];++j){U w=x.following[j];if(visited[w]!=stamp){visited[w]=stamp;queue.push_back(w);}}}
    for(U v=0;v<x.n;++v)if((answer[v]!=x.a[v])!=(visited[v]==stamp))throw Bad("answer is not the minimum-change closure");
   }
   for(U v=0;v<x.n;++v)result.migrations+=answer[v]!=x.a[v];
   ++result.sat;
  }else if(statuses[i]==1){
   for(U v=0;v<x.n;++v)if(answer[v]!=0)throw Bad("UNSAT output padding must be zero");
   if(! (end>begin && W(end)-begin<=x.n))throw Bad("invalid proof path length");
   U first=decode(paths+W(begin)*4),last=decode(paths+W(end-1)*4);
   if(first>=x.n||last>=x.n)throw Bad("proof vertex outside graph");
   if(kinds[i]==2){
    if(end-begin!=1)throw Bad("empty-domain proof must name one vertex");
    bool found=false;
    for(U j=q.starts[i];j<q.starts[i+1];++j)if(q.vertices[j]==first && !(q.masks[j]&((W(1)<<x.a[first])|(W(1)<<x.b[first]))))found=true;
    if(!found)throw Bad("false empty-domain proof");
   }else if(kinds[i]==1){
    if(moving[first]!=stamp||staying[last]!=stamp)throw Bad("contradiction endpoints are not pinned oppositely");
    U prev=NIL;
    for(U j=begin;j<end;++j){
     U v=decode(paths+W(j)*4);if(v>=x.n||path_seen[v]==stamp)throw Bad("invalid or repeated proof vertex");path_seen[v]=stamp;
     if(prev!=NIL){
      if(x.b[prev]!=x.a[v] || !std::binary_search(x.edges.begin(),x.edges.end(),std::make_pair(std::min(prev,v),std::max(prev,v))))throw Bad("proof step is not an original conflict implication");
      ++result.proof_edges;
     }prev=v;
    }
   }else throw Bad("unknown contradiction kind");
   ++result.unsat;
  }else throw Bad("unknown result status");
 }
 // When the original graph is dense, check its mixed-plan equalities 64
 // outputs at once. This uses original edges and plan colours, never solver state.
 // Membership in {A[v],B[v]} was checked above. Proper A and B exclude AA/BB
 // collisions, so these two mixed comparisons are a complete edge check.
 if(x.edges.size()>2*W(x.n)){
  std::vector<W> moved(x.n,0);
  for(U first=0;first<count;first+=64){
   std::fill(moved.begin(),moved.end(),0);U end=std::min<U>(count,first+64);W valid=0;
   for(U i=first;i<end;++i)if(statuses[i]==0){
    W bit=W(1)<<(i-first);valid|=bit;const uint8_t* answer=labels+W(i)*x.n;
    for(U v=0;v<x.n;++v)if(answer[v]!=x.a[v])moved[v]|=bit;
   }
   if(!valid)continue;
   for(auto e:x.edges){U u=e.first,v=e.second;
    if((x.b[u]==x.a[v] && (moved[u]&~moved[v]&valid)) ||
       (x.b[v]==x.a[u] && (moved[v]&~moved[u]&valid)))throw Bad("answer violates an original conflict");
   }
  }
 }
 return result;
}
constexpr const char* TAG="spectra.original.migration.observer.v1";
void destroy(PyObject* c){void* p=PyCapsule_GetPointer(c,TAG);if(p)delete static_cast<Observer*>(p);else PyErr_Clear();}
PyObject* failure(){try{throw;}catch(const Bad& e){PyErr_SetString(PyExc_ValueError,e.what());}catch(const Large& e){PyErr_SetString(PyExc_MemoryError,e.what());}catch(const std::bad_alloc&){PyErr_NoMemory();}catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());}catch(...){PyErr_SetString(PyExc_RuntimeError,"observer failed");}return nullptr;}
PyObject* create(PyObject*,PyObject* args){PyObject *e,*a,*b,*cap;if(!PyArg_ParseTuple(args,"OOOO",&e,&a,&b,&cap))return nullptr;
 try{auto x=std::make_unique<Observer>(e,a,b,integer(cap,"invalid observer cap"));auto c=PyCapsule_New(x.get(),TAG,destroy);if(c)x.release();return c;}catch(...){return failure();}}
PyObject* info(PyObject*,PyObject* c){auto x=static_cast<Observer*>(PyCapsule_GetPointer(c,TAG));if(!x)return nullptr;
 return Py_BuildValue("(IKKK)",x->n,static_cast<unsigned long long>(x->edges.size()),static_cast<unsigned long long>(x->following.size()),static_cast<unsigned long long>(x->payload));}
PyObject* verify(PyObject*,PyObject* args){PyObject *c,*qs,*ss,*ls,*ks,*os,*ps,*min,*cap;
 if(!PyArg_ParseTuple(args,"OOOOOOOOO",&c,&qs,&ss,&ls,&ks,&os,&ps,&min,&cap))return nullptr;
 auto x=static_cast<Observer*>(PyCapsule_GetPointer(c,TAG));if(!x)return nullptr;
 try{
  if(!PyBool_Check(min))throw Bad("minimal must be bool");
  Bank q=read_queries(qs,x->n,integer(cap,"invalid scratch cap"));W count=q.starts.size()-1;
  if(!PyBytes_CheckExact(ss)||!PyBytes_CheckExact(ls)||!PyBytes_CheckExact(ks)||!PyBytes_CheckExact(ps)||!PyTuple_CheckExact(os))throw Bad("invalid output types");
  W path_bytes=PyBytes_GET_SIZE(ps);
  if(W(PyBytes_GET_SIZE(ss))!=count||W(PyBytes_GET_SIZE(ks))!=count||W(PyBytes_GET_SIZE(ls))!=count*x->n||W(PyTuple_GET_SIZE(os))!=count+1||path_bytes%4||path_bytes/4>UINT32_MAX)throw Bad("output geometry differs");
  std::vector<U> offsets(count+1);
  for(W i=0;i<=count;++i){offsets[i]=static_cast<U>(integer(PyTuple_GET_ITEM(os,i),"invalid proof offset",path_bytes/4));if(i&&offsets[i]<offsets[i-1])throw Bad("nonmonotonic proof offsets");}
  if(offsets[0]!=0||W(offsets[count])*4!=path_bytes)throw Bad("unconsumed proof bytes");
  auto bytes=[](PyObject* p){return reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(p));};
  Counts result;PyThreadState* thread=PyEval_SaveThread();
  try{result=check(*x,q,bytes(ss),bytes(ls),bytes(ks),offsets,bytes(ps),min==Py_True);}catch(...){PyEval_RestoreThread(thread);throw;}
  PyEval_RestoreThread(thread);
  return Py_BuildValue("(KKKK)",static_cast<unsigned long long>(result.sat),static_cast<unsigned long long>(result.unsat),static_cast<unsigned long long>(result.migrations),static_cast<unsigned long long>(result.proof_edges));
 }catch(...){return failure();}}
PyMethodDef methods[]={{"create",create,METH_VARARGS,"Own original edges and both proper plans."},{"verify",verify,METH_VARARGS,"Check complete answers, minimum change and original dependency proofs."},{"info",info,METH_O,"Report original observer payload."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_migration_observer",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_migration_observer(){return PyModule_Create(&module);}
