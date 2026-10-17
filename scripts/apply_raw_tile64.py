"""Apply the reviewed raw-tile64 patch to both installed and research sources.

Temporary branch-only transport. Every replacement is exact and counted so a
changed upstream file fails closed instead of receiving an approximate patch.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_native(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """struct Index {
    U n,k,nc=0,bad=NIL;bool condensed,sparse,routed;W cap;W full_arcs=0;
    const char* auto_reason=\"explicit_mode\";
    std::vector<W> masks;
    std::vector<uint8_t> lo,hi,base;
    std::vector<U> comp,opposite;
    std::vector<U> inward,outward,representative,cross_from,cross_to;
    CSR original,dag;
    Index(Input p,U mode,U bank_size=0,U bank_pins=0):n(p.n),k(p.k),condensed(mode!=0),sparse(mode==2),routed(mode==3),cap(p.cap),masks(std::move(p.masks)),lo(n),hi(n,255),base(n) {
""",
        """struct Index {
    U n,k,nc=0,bad=NIL;bool condensed,sparse,routed,raw_tiled;W cap;W full_arcs=0;
    const char* auto_reason=\"explicit_mode\";
    std::vector<W> masks;
    std::vector<uint8_t> lo,hi,base;
    std::vector<U> comp,opposite;
    std::vector<U> inward,outward,representative,cross_from,cross_to;
    CSR original,dag;
    Index(Input p,U mode,U bank_size=0,U bank_pins=0):n(p.n),k(p.k),
        condensed(mode!=0 && mode!=5),sparse(mode==2),routed(mode==3),raw_tiled(mode==5),
        cap(p.cap),masks(std::move(p.masks)),lo(n),hi(n,255),base(n) {
""",
        f"{path}: Index mode fields",
    )

    text = replace_once(
        text,
        """            if(bank_size<64) {
                condensed=false;routed=false;auto_reason=\"bank_below_tile64\";
            } else if(bad!=NIL) {
                condensed=false;routed=false;auto_reason=\"base_infeasible\";
            } else if(W(bank_pins)>=8*W(bank_size)) {
                condensed=false;routed=false;auto_reason=\"high_pin_density\";
            } else {
                condensed=true;routed=nc<=n;
                auto_reason=routed?\"route_component_compression\":\"scc_tile64\";
            }
""",
        """            if(bank_size<64) {
                condensed=false;routed=false;raw_tiled=false;auto_reason=\"bank_below_tile64\";
            } else if(bad!=NIL) {
                condensed=false;routed=false;raw_tiled=false;auto_reason=\"base_infeasible\";
            } else if(W(bank_pins)>=8*W(bank_size)) {
                condensed=false;routed=false;raw_tiled=true;
                auto_reason=\"high_pin_density_raw_tile64\";
            } else {
                condensed=true;routed=nc<=n;raw_tiled=false;
                auto_reason=routed?\"route_component_compression\":\"scc_tile64\";
            }
""",
        f"{path}: structural-v2 raw tile selection",
    )

    text = replace_once(
        text,
        """    const bool use_tiles=solver.condensed && solver.bad==NIL && count>=64;
""",
        """    const bool use_tiles=(solver.condensed || solver.raw_tiled)
        && solver.bad==NIL && count>=64;
""",
        f"{path}: tile admission",
    )

    text = replace_once(
        text,
        """            std::vector<std::vector<U>> roots(lanes);
            std::vector<U> empty(lanes,NIL),conflict(lanes,NIL);
            std::vector<W> reached(solver.nc,0);
            for(U lane=0;lane<lanes;++lane) {
                auto restrictions=solver_rows.row(first_row+lane);
                std::sort(restrictions.begin(),restrictions.end(),[](Pin a,Pin b){return a.vertex<b.vertex;});
                auto& seeds=roots[lane];seeds.reserve(std::min<size_t>(restrictions.size(),solver.n));
                for(size_t i=0;i<restrictions.size();) {
                    U v=restrictions[i].vertex;W domain=solver.masks[v];
                    do {domain&=restrictions[i++].allowed;}
                    while(i<restrictions.size() && restrictions[i].vertex==v);
                    if(!domain) {empty[lane]=v;break;}
                    if(domain!=solver.masks[v])
                        seeds.push_back(2*v+(first(domain)!=solver.lo[v]));
                }
                if(empty[lane]==NIL) {
                    W bit=W(1)<<lane;
                    for(U literal:seeds) reached[solver.comp[literal]]|=bit;
                }
            }
            for(U component=0;component<solver.nc;++component) if(reached[component]) {
                ++result.visits;
                W active=reached[component];
                for(U j=solver.dag.off[component];j<solver.dag.off[component+1];++j) {
                    ++result.arcs;reached[solver.dag.to[j]]|=active;
                }
            }
            for(U component=0;component<solver.nc;++component) {
                W lanes_in_conflict=reached[component]&reached[solver.opposite[component]];
                while(lanes_in_conflict) {
                    U lane=first(lanes_in_conflict);lanes_in_conflict&=lanes_in_conflict-1;
                    if(conflict[lane]==NIL) conflict[lane]=component;
                }
            }
""",
        """            const U nodes=solver.condensed?solver.nc:2*solver.n;
            const CSR& propagation=solver.condensed?solver.dag:solver.original;
            std::vector<std::vector<U>> roots(lanes);
            std::vector<U> empty(lanes,NIL),conflict(lanes,NIL);
            std::vector<W> reached(nodes,0);
            for(U lane=0;lane<lanes;++lane) {
                auto restrictions=solver_rows.row(first_row+lane);
                std::sort(restrictions.begin(),restrictions.end(),[](Pin a,Pin b){return a.vertex<b.vertex;});
                auto& seeds=roots[lane];seeds.reserve(std::min<size_t>(restrictions.size(),solver.n));
                for(size_t i=0;i<restrictions.size();) {
                    U v=restrictions[i].vertex;W domain=solver.masks[v];
                    do {domain&=restrictions[i++].allowed;}
                    while(i<restrictions.size() && restrictions[i].vertex==v);
                    if(!domain) {empty[lane]=v;break;}
                    if(domain!=solver.masks[v])
                        seeds.push_back(2*v+(first(domain)!=solver.lo[v]));
                }
                if(empty[lane]==NIL) {
                    W bit=W(1)<<lane;
                    for(U literal:seeds) {
                        U node=solver.condensed?solver.comp[literal]:literal;
                        reached[node]|=bit;
                    }
                }
            }
            if(solver.condensed) {
                // Component IDs are topological, so every lane reaches closure
                // in one ordered DAG pass.
                for(U component=0;component<nodes;++component) if(reached[component]) {
                    ++result.visits;
                    W active=reached[component];
                    for(U j=propagation.off[component];j<propagation.off[component+1];++j) {
                        ++result.arcs;reached[propagation.to[j]]|=active;
                    }
                }
            } else {
                // Raw implication graphs may contain cycles.  Propagate only the
                // newly arrived lane bits through a bounded worklist.  Every
                // (literal,lane) pair is sent at most once, while 64 requests
                // share each adjacency scan and the same immutable graph.
                std::vector<W> propagated(nodes,0);
                std::vector<uint8_t> queued(nodes,0);
                std::vector<U> queue;queue.reserve(nodes);
                for(U node=0;node<nodes;++node) if(reached[node]) {
                    queued[node]=1;queue.push_back(node);
                }
                for(size_t head=0;head<queue.size();++head) {
                    U node=queue[head];queued[node]=0;
                    W delta=reached[node]&~propagated[node];
                    if(!delta) continue;
                    propagated[node]|=delta;++result.visits;
                    for(U j=propagation.off[node];j<propagation.off[node+1];++j) {
                        ++result.arcs;U target=propagation.to[j];
                        W fresh=delta&~reached[target];
                        if(fresh) {
                            reached[target]|=fresh;
                            if(!queued[target]) {queued[target]=1;queue.push_back(target);}
                        }
                    }
                }
            }
            if(solver.condensed) {
                for(U component=0;component<nodes;++component) {
                    W lanes_in_conflict=reached[component]&reached[solver.opposite[component]];
                    while(lanes_in_conflict) {
                        U lane=first(lanes_in_conflict);lanes_in_conflict&=lanes_in_conflict-1;
                        if(conflict[lane]==NIL) conflict[lane]=component;
                    }
                }
            } else {
                for(U vertex=0;vertex<solver.n;++vertex) {
                    W lanes_in_conflict=reached[2*vertex]&reached[2*vertex+1];
                    while(lanes_in_conflict) {
                        U lane=first(lanes_in_conflict);lanes_in_conflict&=lanes_in_conflict-1;
                        if(conflict[lane]==NIL) conflict[lane]=2*vertex;
                    }
                }
            }
""",
        f"{path}: raw 64-lane closure",
    )

    text = replace_once(
        text,
        """                for(U v=0;v<solver.n;++v) {
                    W low=reached[solver.comp[2*v]],high=reached[solver.comp[2*v+1]];
                    if(low&high&sat_lanes) throw std::logic_error(\"SAT bit lane retains contradiction\");
                    high_choices[v]=(high|(solver.base[v]?~low:0))&sat_lanes;
                }
""",
        """                for(U v=0;v<solver.n;++v) {
                    U low_node=solver.condensed?solver.comp[2*v]:2*v;
                    U high_node=solver.condensed?solver.comp[2*v+1]:2*v+1;
                    W low=reached[low_node],high=reached[high_node];
                    if(low&high&sat_lanes) throw std::logic_error(\"SAT bit lane retains contradiction\");
                    high_choices[v]=(high|(solver.base[v]?~low:0))&sat_lanes;
                }
""",
        f"{path}: raw tiled witness selection",
    )

    text = replace_once(
        text,
        """        Input input(nv,kv,es,ms,limit);std::unique_ptr<Index> result;
        {Released guard;result=std::make_unique<Index>(std::move(input),static_cast<U>(which));}
""",
        """        Input input(nv,kv,es,ms,limit);std::unique_ptr<Index> result;
        {Released guard;result=std::make_unique<Index>(std::move(input),static_cast<U>(which));}
""",
        f"{path}: create body identity guard",
    )
    text = replace_once(
        text,
        """    try {W limit=number(cap,\"invalid memory payload cap\"),which=number(mode,\"mode must be 0, 1, 2 or 3\",3);
""",
        """    try {W limit=number(cap,\"invalid memory payload cap\"),which=number(mode,\"mode must be 0, 1, 2, 3 or 5\",5);
        if(which==4) throw Invalid(\"persistent mode 4 is reserved for one-call auto selection\");
""",
        f"{path}: persistent mode range",
    )
    text = replace_once(
        text,
        """        W which=number(mode,\"mode must be 0, 1, 2, 3 or 4\",4);
""",
        """        W which=number(mode,\"mode must be 0, 1, 2, 3, 4 or 5\",5);
""",
        f"{path}: packet mode range",
    )
    text = replace_once(
        text,
        """            \"selected_mode\",solver->routed?\"route\":(solver->sparse?\"sparse\":(solver->condensed?\"scc\":\"raw\")));
""",
        """            \"selected_mode\",solver->raw_tiled?\"raw-tile\":
                (solver->routed?\"route\":(solver->sparse?\"sparse\":(solver->condensed?\"scc\":\"raw\"))));
""",
        f"{path}: selected mode metadata",
    )

    path.write_text(text, encoding="utf-8")


def patch_python(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """        if mode not in ('scc', 'raw', 'sparse', 'route', 'auto'):
            raise ValueError('mode must be scc, raw, sparse, route or auto')
""",
        """        if mode not in ('scc', 'raw', 'raw-tile', 'sparse', 'route', 'auto'):
            raise ValueError('mode must be scc, raw, raw-tile, sparse, route or auto')
""",
        f"{path}: packet mode validation",
    )
    text = replace_once(
        text,
        """            packet, {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3, 'auto': 4}[mode], max_bytes)
""",
        """            packet, {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3,
                     'auto': 4, 'raw-tile': 5}[mode], max_bytes)
""",
        f"{path}: packet mode mapping",
    )
    text = replace_once(
        text,
        """        if mode not in ('scc', 'raw', 'sparse', 'route'):
            raise ValueError('mode must be scc, raw, sparse or route')
""",
        """        if mode not in ('scc', 'raw', 'raw-tile', 'sparse', 'route'):
            raise ValueError('mode must be scc, raw, raw-tile, sparse or route')
""",
        f"{path}: persistent mode validation",
    )
    text = replace_once(
        text,
        """            self._solver = module.create(p.n, p.palette, p.edges, p.masks,
                                         {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3}[mode], max_bytes - observer_bytes)
""",
        """            self._solver = module.create(p.n, p.palette, p.edges, p.masks,
                                         {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3,
                                          'raw-tile': 5}[mode], max_bytes - observer_bytes)
""",
        f"{path}: persistent mode mapping",
    )
    text = replace_once(
        text,
        """        64 requests, on a base-infeasible graph, or when the immutable bank carries
        at least eight pins per request on average. Otherwise it uses routed proofs
""",
        """        64 requests or on a base-infeasible graph. High-pin banks use a
        64-lane raw-graph kernel; otherwise it uses routed proofs
""",
        f"{path}: auto policy documentation",
    )
    path.write_text(text, encoding="utf-8")


def patch_tests(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_prepared_session_packet_matches_persistent_exact_endpoint(exact_runtime, mode):
""",
        """@pytest.mark.parametrize('mode', ('scc', 'raw', 'raw-tile', 'sparse', 'route'))
def test_prepared_session_packet_matches_persistent_exact_endpoint(exact_runtime, mode):
""",
        f"{path}: persistent packet mode coverage",
    )
    text = replace_once(
        text,
        """@pytest.mark.parametrize('mode', ('scc', 'raw', 'sparse', 'route'))
def test_prepared_session_packet_tile64_matches_individual_exact_answers(exact_runtime, mode):
""",
        """@pytest.mark.parametrize('mode', ('scc', 'raw', 'raw-tile', 'sparse', 'route'))
def test_prepared_session_packet_tile64_matches_individual_exact_answers(exact_runtime, mode):
""",
        f"{path}: raw-tile individual-answer coverage",
    )
    text = replace_once(
        text,
        """    assert prepared.info['bank_kernel'] == ('serial' if mode == 'raw' else 'tile64')
    assert prepared.info['bank_tiles'] == (0 if mode == 'raw' else 3)
""",
        """    assert prepared.info['bank_kernel'] == ('serial' if mode == 'raw' else 'tile64')
    assert prepared.info['bank_tiles'] == (0 if mode == 'raw' else 3)
""",
        f"{path}: raw control identity guard",
    )
    text = replace_once(
        text,
        """@pytest.mark.parametrize('mode', ('scc', 'sparse', 'route'))
def test_mixed_domain_tiles_keep_exact_scalar_proofs(exact_runtime, mode):
""",
        """@pytest.mark.parametrize('mode', ('scc', 'raw-tile', 'sparse', 'route'))
def test_mixed_domain_tiles_keep_exact_scalar_proofs(exact_runtime, mode):
""",
        f"{path}: mixed raw-tile proof coverage",
    )
    text = replace_once(
        text,
        """@pytest.mark.parametrize('mode', ('scc', 'sparse', 'route', 'auto'))
def test_tiled_witnesses_keep_high_colours_and_last_lane(exact_runtime, mode):
""",
        """@pytest.mark.parametrize('mode', ('scc', 'raw-tile', 'sparse', 'route', 'auto'))
def test_tiled_witnesses_keep_high_colours_and_last_lane(exact_runtime, mode):
""",
        f"{path}: high-colour raw-tile coverage",
    )
    text = replace_once(
        text,
        """    # The bank averages well above eight pins per request, so structural-v2
    # intentionally selects raw rather than paying for tile64 propagation.
    expected_mode = 'raw' if mode == 'auto' else mode
""",
        """    # The bank averages well above eight pins per request, so structural-v2
    # selects the raw graph while sharing propagation across 64 request lanes.
    expected_mode = 'raw-tile' if mode == 'auto' else mode
""",
        f"{path}: high-pin expected mode",
    )
    text = replace_once(
        text,
        """            assert answer.info['selected_mode'] == 'raw'
            assert answer.info['auto_policy'] == 'structural-v2'
            assert answer.info['auto_policy_reason'] == 'high_pin_density'
""",
        """            assert answer.info['selected_mode'] == 'raw-tile'
            assert answer.info['auto_policy'] == 'structural-v2'
            assert answer.info['auto_policy_reason'] == 'high_pin_density_raw_tile64'
            assert answer.info['bank_kernel'] == 'tile64'
""",
        f"{path}: auto high-pin metadata",
    )
    text = replace_once(
        text,
        """    assert high.info['selected_mode'] == 'raw'
    assert high.info['auto_policy_reason'] == 'high_pin_density'
    assert high.info['bank_query_count'] == 64 and high.info['bank_pin_count'] == 512
    assert high.output == exact_runtime.solve_session_packet(
        pack_session_packet(p, high_rows), mode='raw').output
""",
        """    assert high.info['selected_mode'] == 'raw-tile'
    assert high.info['auto_policy_reason'] == 'high_pin_density_raw_tile64'
    assert high.info['bank_kernel'] == 'tile64'
    assert high.info['bank_query_count'] == 64 and high.info['bank_pin_count'] == 512
    assert high.output == exact_runtime.solve_session_packet(
        pack_session_packet(p, high_rows), mode='raw-tile').output
    assert high.output == exact_runtime.solve_session_packet(
        pack_session_packet(p, high_rows), mode='raw').output
""",
        f"{path}: frozen density boundary",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    native_paths = [
        ROOT / "spectra/_native/exact_binary.cpp",
        ROOT / "experiments/compiled_query/exact_native.cpp",
    ]
    python_paths = [
        ROOT / "spectra/exact.py",
        ROOT / "experiments/compiled_query/exact.py",
    ]
    if native_paths[0].read_bytes() != native_paths[1].read_bytes():
        raise RuntimeError("native installed/research sources differ before patch")
    if python_paths[0].read_bytes() != python_paths[1].read_bytes():
        raise RuntimeError("Python installed/research sources differ before patch")
    for path in native_paths:
        patch_native(path)
    for path in python_paths:
        patch_python(path)
    patch_tests(ROOT / "experiments/compiled_query/test_exact.py")
    if native_paths[0].read_bytes() != native_paths[1].read_bytes():
        raise RuntimeError("native installed/research sources differ after patch")
    if python_paths[0].read_bytes() != python_paths[1].read_bytes():
        raise RuntimeError("Python installed/research sources differ after patch")
    print("raw tile64 patch applied with all exact replacement guards satisfied")


if __name__ == "__main__":
    main()
