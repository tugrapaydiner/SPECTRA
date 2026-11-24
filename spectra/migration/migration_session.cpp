// Two-plan migration service. MIT, repository LICENSE.
// Reachability is classical. This adapter adds original-address witnesses and
// per-request contradiction paths; no precomputed outcome bank is accepted.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
using U=uint32_t;using W=uint64_t;
constexpr U MAX_N=100000,NIL=UINT32_MAX;
struct Invalid:std::runtime_error{using std::runtime_error::runtime_error;};
struct Resource:std::runtime_error{using std::runtime_error::runtime_error;};
W integer(PyObject* p,const char* message,W cap=UINT64_MAX){
 if(!PyLong_CheckExact(p))throw Invalid(message);
 W value=PyLong_AsUnsignedLongLong(p);if(PyErr_Occurred()){PyErr_Clear();throw Invalid(message);}
 if(value>cap)throw Invalid(message);
 return value;
}
struct Index{
 U n,words;bool compiled;W payload=0;
 std::vector<uint8_t> a,b;
 std::vector<U> offsets,targets;
 std::vector<W> closure;
 Index(PyObject* edges,PyObject* pa,PyObject* pb,bool compile,W cap):compiled(compile){
  if(!PyTuple_CheckExact(edges)||!PyTuple_CheckExact(pa)||!PyTuple_CheckExact(pb))throw Invalid("exact tuples required");
  const W nv=PyTuple_GET_SIZE(pa),m=PyTuple_GET_SIZE(edges);
  if(!nv||nv>MAX_N||m>2000000||PyTuple_GET_SIZE(pb)!=static_cast<Py_ssize_t>(nv))throw Invalid("invalid migration geometry");
  n=static_cast<U>(nv);words=(n+63)/64;
  W required=2*nv+16*(nv+1)+32*m+(compiled?8*nv*words:0)+4096;
  if(required>cap)throw Resource("migration construction exceeds array-payload cap");
  a.resize(n);b.resize(n);offsets.assign(n+1,0);
  for(U v=0;v<n;++v){a[v]=static_cast<uint8_t>(integer(PyTuple_GET_ITEM(pa,v),"bad reference colour",63));b[v]=static_cast<uint8_t>(integer(PyTuple_GET_ITEM(pb,v),"bad target colour",63));}
  std::vector<std::pair<U,U>> arcs;arcs.reserve(m);
  for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(edges);++i){auto e=PyTuple_GET_ITEM(edges,i);
   if(!PyTuple_CheckExact(e)||PyTuple_GET_SIZE(e)!=2)throw Invalid("edge must be an exact pair");
   U u=static_cast<U>(integer(PyTuple_GET_ITEM(e,0),"bad endpoint",n-1)),v=static_cast<U>(integer(PyTuple_GET_ITEM(e,1),"bad endpoint",n-1));
   if(a[u]==a[v]||b[u]==b[v])throw Invalid("both original colour plans must be proper");
   if(b[u]==a[v])arcs.emplace_back(u,v);
   if(b[v]==a[u])arcs.emplace_back(v,u);
  }
  std::sort(arcs.begin(),arcs.end());arcs.erase(std::unique(arcs.begin(),arcs.end()),arcs.end());
  targets.resize(arcs.size());for(auto e:arcs)++offsets[e.first+1];
  for(U v=0;v<n;++v)offsets[v+1]+=offsets[v];
  for(size_t i=0;i<arcs.size();++i)targets[i]=arcs[i].second;
  if(compiled){
   closure.assign(W(n)*words,0);std::vector<U> queue;queue.reserve(n);
   for(U root=0;root<n;++root){
    W* row=closure.data()+W(root)*words;queue.clear();queue.push_back(root);row[root/64]|=W(1)<<(root%64);
    for(size_t head=0;head<queue.size();++head){U v=queue[head];for(U j=offsets[v];j<offsets[v+1];++j){U w=targets[j];W bit=W(1)<<(w%64);if(!(row[w/64]&bit)){row[w/64]|=bit;queue.push_back(w);}}}
   }
  }
  payload=a.capacity()+b.capacity()+4*W(offsets.capacity()+targets.capacity())+8*W(closure.capacity());
  if(payload>required)throw std::logic_error("migration payload admission underestimated");
 }
};
struct Parsed{
 std::vector<U> offsets,vertices;std::vector<W> masks;
};
Parsed parse(PyObject* queries,U n,W budget){
 if(!PyTuple_CheckExact(queries))throw Invalid("query bank must be exact tuple");
 W count=PyTuple_GET_SIZE(queries),entries=0;if(count>1000000)throw Resource("too many queries");
 for(W i=0;i<count;++i){auto q=PyTuple_GET_ITEM(queries,i);if(!PyTuple_CheckExact(q))throw Invalid("query must be exact tuple");entries+=PyTuple_GET_SIZE(q);if(entries>16000000)throw Resource("too many restrictions");}
 // Output labels plus proof paths can both be as large as count*n. Includes
 // temporary native and published buffers, but not Python object/allocator RSS.
 if(count*W(n)>UINT32_MAX)throw Resource("proof offset geometry exceeds uint32 range");
 W required=4*(count+1)+12*entries+14*count*W(n)+64*count+32*W(n)+4096;
 if(required>budget)throw Resource("migration batch exceeds conservative array-payload cap");
 Parsed p;p.offsets.resize(count+1,0);p.vertices.reserve(entries);p.masks.reserve(entries);
 for(W i=0;i<count;++i){auto q=PyTuple_GET_ITEM(queries,i);for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(q);++j){auto r=PyTuple_GET_ITEM(q,j);if(!PyTuple_CheckExact(r)||PyTuple_GET_SIZE(r)!=2)throw Invalid("restriction must be an exact pair");
  p.vertices.push_back(static_cast<U>(integer(PyTuple_GET_ITEM(r,0),"bad query vertex",n-1)));p.masks.push_back(integer(PyTuple_GET_ITEM(r,1),"bad query mask"));}
  p.offsets[i+1]=static_cast<U>(p.vertices.size());}
 return p;
}
struct Worker{
 const Index& x;std::vector<uint8_t> move,stay;std::vector<U> parents,queue;std::vector<W> reached;
 explicit Worker(const Index& index):x(index),move(x.n),stay(x.n),parents(x.n),reached(x.words){queue.reserve(x.n);}
 U bfs(W& operations){
  std::fill(parents.begin(),parents.end(),NIL);queue.clear();
  for(U v=0;v<x.n;++v)if(move[v]){parents[v]=v;queue.push_back(v);}
  U contradiction=NIL;
  for(size_t head=0;head<queue.size();++head){U v=queue[head];if(stay[v]&&contradiction==NIL)contradiction=v;
   for(U j=x.offsets[v];j<x.offsets[v+1];++j){++operations;U w=x.targets[j];if(parents[w]==NIL){parents[w]=v;queue.push_back(w);}}}
  return contradiction;
 }
 // kind 0=SAT, 1=dependency path, 2=no allowed original colour.
 U run(const Parsed& p,U begin,U end,char* labels,std::vector<U>& proof,W& operations){
  std::fill(move.begin(),move.end(),0);std::fill(stay.begin(),stay.end(),0);proof.clear();U empty=NIL;
  for(U i=begin;i<end;++i){U v=p.vertices[i];W mask=p.masks[i];bool allow_a=mask&(W(1)<<x.a[v]),allow_b=mask&(W(1)<<x.b[v]);
   if(!allow_a&&!allow_b){empty=v;continue;}
   if(x.a[v]==x.b[v])continue;
   if(!allow_a)move[v]=1;
   if(!allow_b)stay[v]=1;
  }
  if(empty!=NIL){proof.push_back(empty);return 2;}
  U conflict=NIL;
  if(x.compiled){
   std::fill(reached.begin(),reached.end(),0);
   for(U v=0;v<x.n;++v)if(move[v]){const W* row=x.closure.data()+W(v)*x.words;for(U w=0;w<x.words;++w){reached[w]|=row[w];++operations;}}
   for(U v=0;v<x.n;++v)if(stay[v]&&(reached[v/64]>>(v%64)&1)){conflict=v;break;}
   if(conflict!=NIL){U check=bfs(operations);if(check==NIL)throw std::logic_error("closure conflict has no original dependency path");conflict=check;}
  }else{conflict=bfs(operations);}
  if(conflict!=NIL){
   U v=conflict;while(parents[v]!=v){proof.push_back(v);v=parents[v];if(v==NIL||proof.size()>x.n)throw std::logic_error("invalid dependency predecessor");}
   proof.push_back(v);std::reverse(proof.begin(),proof.end());return 1;
  }
  for(U v=0;v<x.n;++v){bool active=x.compiled?(reached[v/64]>>(v%64)&1):parents[v]!=NIL;labels[v]=static_cast<char>(active?x.b[v]:x.a[v]);}
  return 0;
 }
};
constexpr const char* TAG="spectra.migration.session.v1";
void destroy(PyObject* c){void* p=PyCapsule_GetPointer(c,TAG);if(p)delete static_cast<Index*>(p);else PyErr_Clear();}
PyObject* fail(){try{throw;}catch(const Invalid& e){PyErr_SetString(PyExc_ValueError,e.what());}catch(const Resource& e){PyErr_SetString(PyExc_MemoryError,e.what());}catch(const std::bad_alloc&){PyErr_NoMemory();}catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());}catch(...){PyErr_SetString(PyExc_RuntimeError,"migration failure");}return nullptr;}
PyObject* create(PyObject*,PyObject* args){PyObject *e,*a,*b,*compiled,*cap;if(!PyArg_ParseTuple(args,"OOOOO",&e,&a,&b,&compiled,&cap))return nullptr;
 try{if(!PyBool_Check(compiled))throw Invalid("compiled must be bool");auto x=std::make_unique<Index>(e,a,b,compiled==Py_True,integer(cap,"bad memory cap"));auto c=PyCapsule_New(x.get(),TAG,destroy);if(c)x.release();return c;}catch(...){return fail();}}
PyObject* info(PyObject*,PyObject* c){auto x=static_cast<Index*>(PyCapsule_GetPointer(c,TAG));if(!x)return nullptr;return Py_BuildValue("(IKKK)",x->n,static_cast<unsigned long long>(x->targets.size()),static_cast<unsigned long long>(8*x->closure.size()),static_cast<unsigned long long>(x->payload));}
PyObject* solve(PyObject*,PyObject* args){PyObject *c,*queries,*cap;if(!PyArg_ParseTuple(args,"OOO",&c,&queries,&cap))return nullptr;auto x=static_cast<Index*>(PyCapsule_GetPointer(c,TAG));if(!x)return nullptr;
 try{Parsed p=parse(queries,x->n,integer(cap,"bad batch cap"));W count=p.offsets.size()-1;
  std::vector<char> status(count,0),labels(count*x->n,0),kinds(count,0);std::vector<W> times(count),operations(count);std::vector<U> offsets(count+1,0),paths;
  paths.reserve(count*W(x->n));std::vector<U> proof;proof.reserve(x->n);
  PyThreadState* thread=PyEval_SaveThread();
  try{Worker worker(*x);for(W i=0;i<count;++i){auto begin=std::chrono::steady_clock::now();W work=0;
    U kind=worker.run(p,p.offsets[i],p.offsets[i+1],labels.data()+i*x->n,proof,work);kinds[i]=static_cast<char>(kind);status[i]=kind?1:0;operations[i]=work;
    paths.insert(paths.end(),proof.begin(),proof.end());offsets[i+1]=static_cast<U>(paths.size());
    times[i]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
   }}catch(...){PyEval_RestoreThread(thread);throw;}PyEval_RestoreThread(thread);
  std::vector<char> encoded(paths.size()*4);
  for(size_t i=0;i<paths.size();++i)for(U j=0;j<4;++j)encoded[4*i+j]=static_cast<char>((paths[i]>>(8*j))&255);
  PyObject *s=PyBytes_FromStringAndSize(status.data(),status.size()),*l=PyBytes_FromStringAndSize(labels.data(),labels.size()),*k=PyBytes_FromStringAndSize(kinds.data(),kinds.size()),*pp=PyBytes_FromStringAndSize(encoded.data(),encoded.size());
  PyObject *os=PyTuple_New(offsets.size()),*ts=PyTuple_New(count),*ws=PyTuple_New(count);
  auto cleanup=[&](){Py_XDECREF(s);Py_XDECREF(l);Py_XDECREF(k);Py_XDECREF(pp);Py_XDECREF(os);Py_XDECREF(ts);Py_XDECREF(ws);};
  if(!s||!l||!k||!pp||!os||!ts||!ws){cleanup();return nullptr;}
  for(size_t i=0;i<offsets.size();++i){auto v=PyLong_FromUnsignedLong(offsets[i]);if(!v){cleanup();return nullptr;}PyTuple_SET_ITEM(os,i,v);}
  for(W i=0;i<count;++i){auto t=PyLong_FromUnsignedLongLong(times[i]),w=PyLong_FromUnsignedLongLong(operations[i]);if(!t||!w){Py_XDECREF(t);Py_XDECREF(w);cleanup();return nullptr;}PyTuple_SET_ITEM(ts,i,t);PyTuple_SET_ITEM(ws,i,w);}
  return Py_BuildValue("(NNNNNNN)",s,l,k,os,pp,ts,ws);
 }catch(...){return fail();}}
PyMethodDef methods[]={{"create",create,METH_VARARGS,"Prepare two proper original plans."},{"solve",solve,METH_VARARGS,"Produce complete minimal-change answers or contradiction paths."},{"info",info,METH_O,"Report prepared geometry and payload."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_migration_session",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__spectra_migration_session(){return PyModule_Create(&module);}
