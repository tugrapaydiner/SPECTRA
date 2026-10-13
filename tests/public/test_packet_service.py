
"""Packed original-address pins and independent direct minimum checking."""
from dataclasses import replace
import os, random, struct, itertools
import pytest
from experiments.real_traffic.packet_migration import PacketMigrationRuntime,build_packet_migration
from experiments.real_traffic.packet_observer import PacketObserverRuntime,build_packet_observer
from experiments.real_traffic.migration_session import MigrationVerifier,MigrationBatch

@pytest.fixture(scope='session')
def solver(tmp_path_factory):
    p=os.environ.get("SPECTRA_PACKET_SOLVER_LIBRARY") or build_packet_migration(tmp_path_factory.mktemp('packet-solver'))
    return PacketMigrationRuntime(p)

@pytest.fixture(scope='session')
def checker(tmp_path_factory):
    p=os.environ.get("SPECTRA_PACKET_OBSERVER_LIBRARY") or build_packet_observer(tmp_path_factory.mktemp('packet-checker'))
    return PacketObserverRuntime(p)

def pack_reference(q):
    counts=[0]
    vertices=[];masks=[]
    for row in q:
        for v,m in row:vertices.append(v);masks.append(m)
        counts.append(len(vertices))
    return b'SPQ1'+struct.pack('<III',len(q),len(vertices),0)+struct.pack('<'+'I'*len(counts),*counts)+struct.pack('<'+'I'*len(vertices),*vertices)+struct.pack('<'+'Q'*len(masks),*masks)

@pytest.mark.parametrize('count',[0,1,2,31,32,63,64,65,127,128,129,257])
def test_wire_and_all_modes(solver,checker,count):
    e=((0,1),);a=(0,1,63);b=(1,0,62)
    source=((),((0,2),),((0,2),(1,2)),((2,0),),((0,1),(0,2)),((2,1<<62),(2,1<<62)),((1,(1<<64)-1),))
    q=tuple(source[i%len(source)] for i in range(count))
    packet=solver.pack_frozen(q)
    assert packet==pack_reference(q)
    with solver.prepare(e,a,b) as p,checker.prepare(e,a,b) as c:
        for mode in ('bfs','tiled','auto'):
            result=p.solve_packet(packet,mode=mode)
            expected=p.solve_frozen(q,mode=mode)
            for field in ('statuses','labels','proof_kinds','proof_offsets','proof_vertices_le'):
                assert getattr(result,field)==getattr(expected,field)
            oracle=MigrationVerifier(e,a,b).verify(q,result,minimal=True)
            for cm in ('bfs','tiled','direct','rows','auto'):
                audit=c.verify_packet(packet,result,mode=cm)
                assert audit.sat==oracle['sat'] and audit.unsat==oracle['unsat']
                assert audit.migrations==oracle['migrations'] and audit.minimum_checked
            # Keep references from previous calls and ensure later writes cannot alias.
            before=bytes(result.labels)
            p.solve_packet(solver.pack_frozen(((),)),mode=mode)
            assert result.labels==before

def test_reject_malformed_packet_independently(solver,checker):
    e=((0,1),);a=(0,1);b=(1,0);q=(((0,2),),(),((1,2),))
    good=solver.pack_frozen(q)
    mutations=[b'',good[:19],good[:-1],good+b'\0',b'bad!'+good[4:]]
    for offset,value in [(4,1000001),(8,16000001),(12,1),(16,1),(20,0xffffffff),(24,0),(28,1),(32,2)]:
        m=bytearray(good);struct.pack_into('<I',m,offset,value);mutations.append(bytes(m))
    with solver.prepare(e,a,b) as p,checker.prepare(e,a,b) as c:
        result=p.solve_packet(good)
        for bad in mutations:
            with pytest.raises((ValueError,MemoryError)):p.solve_packet(bad)
            with pytest.raises((ValueError,MemoryError)):c.verify_packet(bad,result)
        for bad in (bytearray(good),memoryview(good),list(good),True):
            with pytest.raises(ValueError):p.solve_packet(bad)
            with pytest.raises(ValueError):c.verify_packet(bad,result)
        with pytest.raises(MemoryError):p.solve_packet(good,max_bytes=1)
        with pytest.raises(MemoryError):c.verify_packet(good,result,max_bytes=1)

@pytest.mark.parametrize('bad',[[[(0,1)]],(((True,1),),),(((0,False),),),(((0,-1),),),(((0,1<<64),),),(((-1,1),),),(((100000,1),),),(((0,1,2),),),((None,),)])
def test_invalid_frozen_pins(solver,bad):
    with pytest.raises((ValueError,MemoryError)):solver.pack_frozen(bad)

def test_pack_cap(solver):
    with pytest.raises(MemoryError):solver.pack_frozen(((),),max_bytes=1)
    for cap in (True,False,-1,1.5,None):
        with pytest.raises(ValueError):solver.pack_frozen(((),),max_bytes=cap)

@pytest.mark.parametrize('cm',['direct','rows'])
def test_all_small_models_not_only_solver_outputs(solver,checker,cm):
    rng=random.Random(633103)
    states=0
    for _ in range(70):
        n=4;a=tuple(rng.randrange(3) for _ in range(n));b=tuple(rng.randrange(3) for _ in range(n))
        e=tuple((u,v) for u in range(n) for v in range(u+1,n) if a[u]!=a[v] and b[u]!=b[v] and rng.randrange(2))
        q=tuple(tuple((v,1<<rng.choice((a[v],b[v]))) for v in rng.sample(range(n),rng.randrange(n+1))) for _ in range(5))
        with checker.prepare(e,a,b) as c:
            for pins in q:
                packet=solver.pack_frozen((pins,))
                valid=[]
                for bits in itertools.product((0,1),repeat=n):
                    labels=tuple(b[v] if bits[v] else a[v] for v in range(n))
                    if all(labels[u]!=labels[v] for u,v in e) and all(m>>labels[v]&1 for v,m in pins):valid.append(labels)
                # Pure exhaustive reference: minimum is intersection of all feasible changes.
                if valid:
                    required={v for v in range(n) if all(z[v]!=a[v] for z in valid)}
                for labels in itertools.product(*[(a[v],b[v]) if a[v]!=b[v] else (a[v],) for v in range(n)]):
                    r=MigrationBatch(b'\0',bytes(labels),b'\0',(0,0),b'',(),(),0)
                    allowed=labels in valid and {v for v in range(n) if labels[v]!=a[v]}==required
                    if allowed:assert c.verify_packet(packet,r,mode=cm).minimum_checked
                    else:
                        with pytest.raises(ValueError):c.verify_packet(packet,r,mode=cm)
                    states+=1
    assert states>1000

@pytest.mark.parametrize('cm',['direct','rows'])
def test_unseeded_cycles_and_invalid_output(solver,checker,cm):
    e=((0,1),);a=(0,1);b=(1,0);q=((),)*129
    r=MigrationBatch(bytes(129),bytes(b)*129,bytes(129),(0,)*130,b'',(),(),0)
    packet=solver.pack_frozen(q)
    with checker.prepare(e,a,b) as c:
        assert c.verify_packet(packet,r,minimal=False,mode=cm).sat==129
        with pytest.raises(ValueError,match='minimum'):c.verify_packet(packet,r,mode=cm)
        for lane in (0,31,63,64,128):
            z=bytearray(bytes(a)*129);z[lane*2]=255
            with pytest.raises(ValueError):c.verify_packet(packet,replace(r,labels=bytes(z)),mode=cm)

def test_handmade_packet_and_contradiction_without_solver(checker):
    q=(((0,2),),((0,2),(1,2)))
    r=MigrationBatch(b'\0\1',b'\1\0\0\0',b'\0\1',(0,0,2),struct.pack('<II',0,1),(),(),0)
    with checker.prepare(((0,1),),(0,1),(1,0)) as c:
        for mode in ('direct','rows'):
            assert c.verify_packet(pack_reference(q),r,mode=mode).unsat==1
            bad=replace(r,proof_vertices_le=struct.pack('<II',0,0))
            with pytest.raises(ValueError):c.verify_packet(pack_reference(q),bad,mode=mode)

def test_single_byte_and_zero_length_outputs_are_private(solver):
    with solver.prepare((),(0,),(1,)) as p:
        r=p.solve_packet(solver.pack_frozen((((0,2),),)))
        assert r.labels==b'\1'
        for i in range(256):
            other=p.solve_packet(solver.pack_frozen((((0,1),),)))
            assert other.labels==b'\0' and r.labels==b'\1'
            assert bytes([i])==struct.pack('B',i)
        z=p.solve_packet(solver.pack_frozen(()))
        assert z.labels==b'' and z.statuses==b''


def test_packet_service_snapshots_mutable_inputs_and_rechecks(solver,checker):
    from concurrent.futures import ThreadPoolExecutor
    from experiments.real_traffic.packet_service import PacketService
    service=object.__new__(PacketService);service.solver=solver;service.observer=checker
    source=[[[0,2]],[[0,2],[1,2]]]
    packet=service.pack(source)
    source[0][0][1]=1
    with service.prepare(((0,1),),(0,1),(1,0)) as p:
        r=p.solve_packet(packet)
        assert r.audit.sat==r.audit.unsat==1 and r.audit.minimum_checked
        assert r.packing_ns==0 and r.input_form=='packet'
        g=p.solve(((iter(pin) for pin in row) for row in (((0,2),),((0,2),(1,2)))))
        assert g.batch.labels==r.batch.labels and g.input_packet_bytes==len(packet)
        with ThreadPoolExecutor(max_workers=4) as pool:
            repeated=list(pool.map(lambda _:p.solve_packet(packet),range(24)))
        assert all(x.batch.labels==r.batch.labels for x in repeated)
        original=p.solver.solve_packet
        def bad(*a,**kw):
            result=original(*a,**kw)
            return replace(result,labels=bytes(len(result.labels)))
        p.solver.solve_packet=bad
        with pytest.raises(ValueError):p.solve_packet(packet)
    with pytest.raises(RuntimeError):p.solve_packet(packet)
    with pytest.raises(RuntimeError):p.solve(())
    with pytest.raises(RuntimeError):_=p.info
    p.close()

def test_packet_service_caps_and_iterator_failure(solver,checker):
    from experiments.real_traffic.packet_service import PacketService
    service=object.__new__(PacketService);service.solver=solver;service.observer=checker
    with pytest.raises(MemoryError):service.prepare((),(0,),(1,),max_bytes=1)
    def broken():
        yield ((0,1),)
        raise RuntimeError("iterator broke")
    with service.prepare((),(0,),(1,)) as p:
        with pytest.raises(RuntimeError,match="iterator broke"):p.solve(broken())
        with pytest.raises(MemoryError):p.solve(((),),max_bytes=1)
        with pytest.raises(MemoryError):p.solve_packet(service.pack(((),)),max_bytes=1)
