// Export-local memoized fingerprints plus exact structural verification.
// Fingerprint matches NEVER establish expression equality. This is not a digest-only cache.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
struct PythonError {};
struct Unsupported {};
struct Limit {};
struct Cycle {};
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value=nullptr):p(value) {}
    ~Ref(){Py_XDECREF(p);}
    Ref(const Ref&)=delete;
    Ref& operator=(const Ref&)=delete;
};
PyObject *unsupported_error=nullptr, *limit_error=nullptr;
struct Budget:std::pmr::memory_resource {
    size_t used=0,peak=0,limit;
    explicit Budget(size_t n):limit(n){}
    void* do_allocate(size_t n,size_t a) override {
        if(n>limit-used)throw Limit{};
        void* p=std::pmr::new_delete_resource()->allocate(n,a);
        used+=n;if(used>peak)peak=used;return p;
    }
    void do_deallocate(void* p,size_t n,size_t a) override {
        std::pmr::new_delete_resource()->deallocate(p,n,a);used-=n;
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override{return this==&other;}
};
struct Key {
    uint64_t tag=0,a=0,b=0,c=0;
    bool operator==(const Key& k)const noexcept{return tag==k.tag&&a==k.a&&b==k.b&&c==k.c;}
};
uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
struct Hasher {
    bool collide=false;
    size_t operator()(const Key& k)const noexcept {
        return collide?0:size_t(mix(k.tag+0x9e3779b97f4a7c15ULL)^mix(k.a)^mix(k.b+19)^mix(k.c+47));
    }
};
// kind: 0 numeric leaf, 1 unary, 2 binary-op, 3 power, 4 if, 5 vector.
struct Spec {PyTypeObject* type;int kind;PyObject *name1,*name2,*name3;};
struct Frame {
    PyObject* obj;
    Key key;
    std::array<PyObject*,3> children{};
    PyObject* sequence=nullptr;
    uint32_t n=0,next=0,kind=0;
    bool retain=false;
    Frame()=default;
    Frame(const Frame&)=delete;
    Frame& operator=(const Frame&)=delete;
    Frame(Frame&& other)noexcept:obj(other.obj),key(other.key),children(other.children),sequence(other.sequence),n(other.n),next(other.next),kind(other.kind),retain(other.retain){other.children={};other.sequence=nullptr;}
    Frame& operator=(Frame&&)=delete;
    ~Frame(){for(auto* p:children)Py_XDECREF(p);Py_XDECREF(sequence);}
};
struct Slot {PyObject* object=nullptr;uint64_t id=0;};
struct Index {
    Budget budget;
    std::pmr::vector<Spec> specs;
    std::pmr::vector<PyTypeObject*> numeric_types;
    std::pmr::vector<PyObject*> ops;
    std::pmr::vector<Slot> memo;
    uint64_t fingerprint_steps=0,equality_steps=0;
    size_t memo_n=0;
    std::pmr::vector<PyObject*> active;
    size_t active_n=0;
    uint32_t demand_mask=0;
    bool selective=false;
    uint64_t call_visits=0;
    Hasher hasher;
    std::pmr::vector<Frame> stack;

    bool use_memo;
    uint64_t visits=0,memo_hits=0,intern_hits=0;
    uint32_t max_nodes;
    Index(size_t bytes,uint32_t count,bool m,bool collisions):budget(bytes),specs(&budget),numeric_types(&budget),ops(&budget),
        memo(&budget),active(&budget),hasher{collisions},stack(&budget),
        use_memo(m),max_nodes(count){}
    ~Index(){release();for(auto& s:specs){Py_DECREF(s.type);Py_XDECREF(s.name1);Py_XDECREF(s.name2);Py_XDECREF(s.name3);}
        for(auto* t:numeric_types)Py_DECREF(t);for(auto* o:ops)Py_DECREF(o);}
    void release(){
        stack.clear();
        for(auto& entry:memo)if(entry.object){Py_DECREF(entry.object);entry=Slot{};}
        memo_n=0;std::fill(active.begin(),active.end(),nullptr);active_n=0;
    }
    Slot* find(PyObject* p){
        if(memo.empty())return nullptr;
        size_t i=size_t(mix(reinterpret_cast<uintptr_t>(p)))&(memo.size()-1);
        while(memo[i].object&&memo[i].object!=p)i=(i+1)&(memo.size()-1);
        return memo[i].object?&memo[i]:nullptr;
    }
    void reserve_memo(){
        if((memo_n+1)*10<=memo.size()*7)return;
        std::pmr::vector<Slot> fresh(&budget);fresh.resize(memo.empty()?128:memo.size()*2);
        for(const auto& entry:memo)if(entry.object){
            size_t i=size_t(mix(reinterpret_cast<uintptr_t>(entry.object)))&(fresh.size()-1);
            while(fresh[i].object)i=(i+1)&(fresh.size()-1);
            fresh[i]=entry;
        }
        memo.swap(fresh);
    }
    void mark(PyObject* p){
        auto* pos=find(p);
        if(pos){if(!pos->id)throw Cycle{};pos->id=0;return;}
        if(memo_n>=max_nodes)throw Limit{};
        reserve_memo();size_t i=size_t(mix(reinterpret_cast<uintptr_t>(p)))&(memo.size()-1);
        while(memo[i].object)i=(i+1)&(memo.size()-1);
        Py_INCREF(p);memo[i]=Slot{p,0};++memo_n;
    }
    int type_tag(PyObject* p)const {
        for(size_t i=0;i<specs.size();++i)if(Py_TYPE(p)==specs[i].type)return int(i);
        throw Unsupported{};
    }
    bool wanted(PyObject* p,int temporary_refs)const {
        // Refcount is a retention hint, NEVER an equality rule. All AST edges are
        // owning Python references and traversal holds the GIL. A shared child
        // or externally retained node is conservatively memoized. This avoids
        // expanding diamonds whose node type has not yet entered the value cache.
        return use_memo&&(!selective||(demand_mask&(1u<<type_tag(p)))||Py_REFCNT(p)>1+temporary_refs);
    }
    void active_add(PyObject* p){
        if((active_n+1)*10>active.size()*7){
            std::pmr::vector<PyObject*> fresh(&budget);fresh.resize(active.empty()?128:active.size()*2,nullptr);
            for(auto* value:active)if(value){size_t i=mix(reinterpret_cast<uintptr_t>(value))&(fresh.size()-1);while(fresh[i])i=(i+1)&(fresh.size()-1);fresh[i]=value;}
            active.swap(fresh);
        }
        size_t i=mix(reinterpret_cast<uintptr_t>(p))&(active.size()-1);
        while(active[i]){if(active[i]==p)throw Cycle{};i=(i+1)&(active.size()-1);}
        active[i]=p;++active_n;
    }
    void active_erase(PyObject* p){
        size_t mask=active.size()-1,i=mix(reinterpret_cast<uintptr_t>(p))&mask;
        while(active[i]!=p)i=(i+1)&mask;
        size_t j=(i+1)&mask;
        while(active[j]){
            size_t home=mix(reinterpret_cast<uintptr_t>(active[j]))&mask;
            if(((i-home)&mask)<((j-home)&mask)){active[i]=active[j];i=j;}
            j=(j+1)&mask;
        }
        active[i]=nullptr;--active_n;
    }
    uint64_t intern(const Key& k){
        ++fingerprint_steps;
        uint64_t value=uint64_t(hasher(k));
        // Zero is the DFS in-progress marker. Collisions including zero->one
        // are always handled by the exact structural comparison, never accepted.
        return value?value:1;
    }
    PyObject* attr(PyObject* obj,PyObject* name){
        PyObject* value=PyObject_GetAttr(obj,name);
        if(!value){if(PyErr_ExceptionMatches(PyExc_AttributeError)){PyErr_Clear();throw Unsupported{};}throw PythonError{};}
        // Return an owned reference; the frame or a local Ref always releases it.
        // No dangling pointer even if a malformed object supplies a descriptor.
        return value;
    }
    uint64_t scalar(PyObject* p){
        bool known=PyFloat_CheckExact(p)||PyLong_CheckExact(p)||PyBool_Check(p);
        for(auto* t:numeric_types)known=known||(Py_TYPE(p)==t);
        if(!known)throw Unsupported{};
        // NumVal numpy scalar types are admitted only when their value is exactly
        // representable by binary64. Wide integers conservatively fall back.
        if(PyIndex_Check(p)&&!PyFloat_Check(p)){
            Ref q(PyNumber_Index(p));if(!q.p)throw PythonError{};
            int overflow=0;long long v=PyLong_AsLongLongAndOverflow(q.p,&overflow);
            if(PyErr_Occurred())throw PythonError{};
            if(overflow||v>9007199254740992LL||v< -9007199254740992LL)throw Unsupported{};
        }
        double value=PyFloat_AsDouble(p);if(PyErr_Occurred())throw PythonError{};
        if(std::isnan(value))throw Unsupported{}; // Nonreflexive equality is not internable.
        if(value==0.)value=0.; // Match +0 == -0 of the original AST equality.
        uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));return bits;
    }
    Frame describe(PyObject* p){
        size_t i=0;for(;i<specs.size();++i)if(Py_TYPE(p)==specs[i].type)break;
        if(i==specs.size())throw Unsupported{};
        const Spec& s=specs[i];Frame f;f.obj=p;f.kind=uint32_t(s.kind);f.key.tag=i+1;
        if(s.kind==0){Ref value(attr(p,s.name1));f.key.a=scalar(value.p);}
        else if(s.kind==1){f.children[0]=attr(p,s.name1);f.n=1;}
        else if(s.kind==2){
            f.children[0]=attr(p,s.name1);f.children[1]=attr(p,s.name2);f.n=2;
            Ref owned_op(attr(p,s.name3));PyObject* op=owned_op.p;
            // Enum instances are trusted exact values only; identity is stable and
            // distinct enums stay distinct, just as upstream op equality requires.
            bool supported=false;for(auto* value:ops)supported=supported||(value==op);
            if(!supported)throw Unsupported{};
            f.key.c=uint64_t(reinterpret_cast<uintptr_t>(op));
        }else if(s.kind==3){f.children[0]=attr(p,s.name1);f.children[1]=attr(p,s.name2);f.n=2;}
        else if(s.kind==4){f.children[0]=attr(p,s.name1);f.children[1]=attr(p,s.name2);f.children[2]=attr(p,s.name3);f.n=3;}
        else if(s.kind==5){
            f.sequence=attr(p,s.name1);
            if(!PyList_CheckExact(f.sequence)&&!PyTuple_CheckExact(f.sequence))throw Unsupported{};
            Py_ssize_t n=PySequence_Fast_GET_SIZE(f.sequence);
            if(n<0||uint64_t(n)>max_nodes)throw Limit{};
            f.n=uint32_t(n);Ref size(attr(p,s.name2));f.key.b=scalar(size.p);
            double declared=PyFloat_AsDouble(size.p);if(PyErr_Occurred())throw PythonError{};
            if(declared!=double(n))throw Unsupported{};
        }else throw Unsupported{};
        return f;
    }
    void push(PyObject* p,int temporary_refs=0){
        if(visits==std::numeric_limits<uint64_t>::max()||++call_visits>max_nodes)throw Limit{};
        ++visits;active_add(p);bool keep=wanted(p,temporary_refs);if(keep)mark(p);
        Frame f=describe(p);f.retain=keep;stack.push_back(std::move(f));
    }
    uint64_t key(PyObject* p){
        if(selective)demand_mask|=(1u<<type_tag(p));
        auto* old=use_memo?find(p):nullptr;
        if(old){if(!old->id)throw Cycle{};++memo_hits;return old->id;}
        call_visits=0;const size_t depth=stack.size();push(p);uint64_t result=0;
        while(stack.size()>depth){
            Frame& f=stack.back();
            if(f.next<f.n){
                PyObject* child=f.kind==5?PySequence_Fast_GET_ITEM(f.sequence,f.next):f.children[f.next];
                auto* known=use_memo?find(child):nullptr;
                if(known&&!known->id)throw Cycle{};
                if(known){++memo_hits;consume(f,known->id);++f.next;}
                else push(child,f.kind==5?0:1);
            }else{
                uint64_t id=intern(f.key);PyObject* obj=f.obj;bool keep=f.retain;stack.pop_back();active_erase(obj);
                if(keep)find(obj)->id=id;
                if(stack.size()==depth){result=id;break;}
                Frame& parent=stack.back();consume(parent,id);++parent.next;
            }
        }
        return result;
    }
    struct Pair {PyObject* a;PyObject* b;bool operator==(const Pair& p)const noexcept{return a==p.a&&b==p.b;}};
    struct PairHash {size_t operator()(const Pair& p)const noexcept{return mix(reinterpret_cast<uintptr_t>(p.a))^mix(reinterpret_cast<uintptr_t>(p.b)+1);}};
    bool equal(PyObject* a,PyObject* b){
        std::pmr::vector<Pair> pending(&budget);std::pmr::unordered_set<Pair,PairHash> seen(&budget);
        pending.push_back(Pair{a,b});
        while(!pending.empty()){
            Pair pair=pending.back();pending.pop_back();
            if(pair.a==pair.b)continue;
            if(seen.find(pair)!=seen.end())continue;
            if(seen.size()>=max_nodes)throw Limit{};
            seen.insert(pair);++equality_steps;
            if(Py_TYPE(pair.a)!=Py_TYPE(pair.b))return false;
            Frame left=describe(pair.a),right=describe(pair.b);
            if(!(left.key==right.key)||left.n!=right.n)return false;
            for(uint32_t i=0;i<left.n;++i){
                PyObject* x=left.kind==5?PySequence_Fast_GET_ITEM(left.sequence,i):left.children[i];
                PyObject* y=right.kind==5?PySequence_Fast_GET_ITEM(right.sequence,i):right.children[i];
                pending.push_back(Pair{x,y});
            }
        }
        return true;
    }
    void consume(Frame& f,uint64_t id){
        if(f.kind==5){
            // Ordered vector contents form a prefix fingerprint. No O(length)
            // recomputation on an identity-memo hit; collisions are verified.
            f.key.a=intern(Key{0xffffffffffffffffULL,f.key.a,id,0});
        }else if(f.next==0)f.key.a=id;
        else if(f.next==1)f.key.b=id;
        else f.key.c=id;
    }
};
constexpr const char* capname="spectra.structural-index.v1";
Index* unpack(PyObject* c){auto* p=static_cast<Index*>(PyCapsule_GetPointer(c,capname));if(!p)throw PythonError{};return p;}
void destroy(PyObject* c){auto* p=static_cast<Index*>(PyCapsule_GetPointer(c,capname));if(p)delete p;else PyErr_Clear();}
template<class F>PyObject* protect(F f){
    try{return f();}catch(PythonError&){return nullptr;}
    catch(Unsupported&){PyErr_SetString(unsupported_error,"unsupported or nonreflexive AST scalar/type");return nullptr;}
    catch(Limit&){PyErr_SetString(limit_error,"structural index budget exceeded");return nullptr;}
    catch(Cycle&){PyErr_SetString(PyExc_ValueError,"cycle in expression graph");return nullptr;}
    catch(std::bad_alloc&){return PyErr_NoMemory();}
    catch(std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}
}
PyObject* create(PyObject*,PyObject* args){return protect([&]()->PyObject*{
    PyObject *specs,*numbers,*ops;unsigned long long bytes;unsigned int count;int memo,collide;
    if(!PyArg_ParseTuple(args,"OOOKIpp",&specs,&numbers,&ops,&bytes,&count,&memo,&collide))return nullptr;
    if(!PyTuple_CheckExact(specs)||PyTuple_GET_SIZE(specs)<1||PyTuple_GET_SIZE(specs)>31||!PyTuple_CheckExact(numbers)||!PyTuple_CheckExact(ops)||!count||count>10000000||bytes<4096||bytes>2147483648ULL){
        PyErr_SetString(PyExc_ValueError,"invalid structural index configuration");return nullptr;}
    auto* index=new Index(size_t(bytes),count,memo,collide);
    Ref capsule(PyCapsule_New(index,capname,destroy));if(!capsule.p){delete index;return nullptr;}
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(specs);++i){
        PyObject* item=PyTuple_GET_ITEM(specs,i);
        if(!PyTuple_CheckExact(item)||PyTuple_GET_SIZE(item)!=3){PyErr_SetString(PyExc_ValueError,"invalid type spec");return nullptr;}
        PyObject* cls=PyTuple_GET_ITEM(item,0),*kind=PyTuple_GET_ITEM(item,1),*names=PyTuple_GET_ITEM(item,2);
        if(!PyType_Check(cls)||!PyLong_CheckExact(kind)||!PyTuple_CheckExact(names)||PyTuple_GET_SIZE(names)>3){PyErr_SetString(PyExc_ValueError,"invalid type spec");return nullptr;}
        long k=PyLong_AsLong(kind);if(PyErr_Occurred())return nullptr;
        if(k<0||k>5){PyErr_SetString(PyExc_ValueError,"invalid node kind");return nullptr;}
        int needed=k==0||k==1?1:k==3||k==5?2:3;
        if(PyTuple_GET_SIZE(names)!=needed){PyErr_SetString(PyExc_ValueError,"invalid attribute count");return nullptr;}
        for(Py_ssize_t j=0;j<PyTuple_GET_SIZE(names);++j)if(!PyUnicode_CheckExact(PyTuple_GET_ITEM(names,j))){PyErr_SetString(PyExc_ValueError,"invalid attribute name");return nullptr;}
        Spec s{reinterpret_cast<PyTypeObject*>(cls),int(k),nullptr,nullptr,nullptr};
        s.name1=PyTuple_GET_ITEM(names,0);if(needed>=2)s.name2=PyTuple_GET_ITEM(names,1);if(needed>=3)s.name3=PyTuple_GET_ITEM(names,2);
        index->specs.push_back(s);Py_INCREF(cls);Py_XINCREF(s.name1);Py_XINCREF(s.name2);Py_XINCREF(s.name3);
    }
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(numbers);++i){auto* t=PyTuple_GET_ITEM(numbers,i);if(!PyType_Check(t)){PyErr_SetString(PyExc_ValueError,"invalid numeric type");return nullptr;}index->numeric_types.push_back(reinterpret_cast<PyTypeObject*>(t));Py_INCREF(t);}
    for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(ops);++i){auto* op=PyTuple_GET_ITEM(ops,i);index->ops.push_back(op);Py_INCREF(op);}
    Py_INCREF(capsule.p);return capsule.p;
});}
PyObject* getkey(PyObject*,PyObject* args){return protect([&]()->PyObject*{
    PyObject *c,*obj;if(!PyArg_ParseTuple(args,"OO",&c,&obj))return nullptr;Index* p=unpack(c);
    try{return PyLong_FromUnsignedLongLong(p->key(obj));}catch(...){p->release();throw;}
});}
PyObject* clear(PyObject*,PyObject* c){return protect([&]()->PyObject*{unpack(c)->release();Py_RETURN_NONE;});}
PyObject* select(PyObject*,PyObject* c){return protect([&]()->PyObject*{unpack(c)->selective=true;Py_RETURN_NONE;});}
PyObject* equal(PyObject*,PyObject* args){return protect([&]()->PyObject*{
    PyObject *c,*a,*b;if(!PyArg_ParseTuple(args,"OOO",&c,&a,&b))return nullptr;
    return PyBool_FromLong(unpack(c)->equal(a,b));
});}
PyObject* stats(PyObject*,PyObject* c){return protect([&]()->PyObject*{
    auto* p=unpack(c);
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:K}","visits",p->visits,"memo_hits",p->memo_hits,"fingerprint_steps",p->fingerprint_steps,
        "identity_entries",uint64_t(p->memo_n),"equality_steps",p->equality_steps,"native_bytes",uint64_t(p->budget.used),"peak_native_bytes",uint64_t(p->budget.peak));
});}
PyMethodDef methods[]={{"create",create,METH_VARARGS,nullptr},{"key",getkey,METH_VARARGS,nullptr},{"equal",equal,METH_VARARGS,nullptr},{"select",select,METH_O,nullptr},{"clear",clear,METH_O,nullptr},{"stats",stats,METH_O,nullptr},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_spectra_structural",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__spectra_structural(){
    PyObject* m=PyModule_Create(&module);if(!m)return nullptr;
    unsupported_error=PyErr_NewException("_spectra_structural.Unsupported",PyExc_ValueError,nullptr);
    limit_error=PyErr_NewException("_spectra_structural.LimitError",PyExc_MemoryError,nullptr);
    if(!unsupported_error||!limit_error){Py_DECREF(m);return nullptr;}
    PyModule_AddObject(m,"Unsupported",unsupported_error);PyModule_AddObject(m,"LimitError",limit_error);return m;
}
