// Add explicit engine controls without sharing any SPECTRA solver code.
// The included file supplies the standalone parser, SCC construction, proof
// machinery, observer, default closure policy, and C ABI.
#include "direct_twosat_native.cpp"

namespace {

std::vector<std::uint8_t> solve_packet_dag(
    const std::uint8_t* data,
    std::size_t size,
    W cap,
    StatsInternal& stats
) {
    Packet packet(data, size, cap);
    Solver solver(packet, cap);
    // The default challenger may build dense component reachability. This entry
    // point deliberately disables it after common construction so the benchmark
    // can choose the stronger independent engine rather than assuming closure wins.
    solver.closure.clear();
    solver.closure.shrink_to_fit();
    solver.closure_enabled = false;

    Observer observer(data, size);
    stats.n = packet.n;
    stats.edges = packet.edge_count;
    stats.queries = packet.query_count;
    stats.pins = packet.pin_count;
    stats.components = solver.component_count;
    stats.dag_edges = static_cast<U>(solver.dag_arcs.size());
    stats.engine = 2U;
    stats.closure_bytes = 0;

    std::vector<std::uint8_t> output;
    W minimum = 16 + W(packet.query_count) * (1 + W(packet.n) + 4);
    if (minimum > cap) throw Resource("minimum SCQ2 output exceeds max_bytes");
    output.reserve(static_cast<std::size_t>(std::min<W>(minimum + 4096, cap)));
    output.insert(output.end(), {'S', 'C', 'Q', '2'});
    put32(output, packet.n, cap);
    put64(output, packet.query_count, cap);

    std::vector<U> touched;
    std::vector<U> marks(packet.n, 0);
    std::vector<W> effective(packet.n, 0);
    U generation = 1;
    for (U row = 0; row < packet.query_count; ++row, ++generation) {
        if (generation == 0) {
            std::fill(marks.begin(), marks.end(), 0);
            generation = 1;
        }
        std::vector<std::uint8_t> labels;
        std::vector<std::uint8_t> certificate;
        solver.solve_row(
            row, labels, certificate, touched, marks, effective, generation
        );
        bool sat = !labels.empty() || (packet.n == 0 && certificate.empty());
        output.push_back(sat ? 1 : 0);
        if (sat) {
            if (labels.size() != packet.n)
                throw std::logic_error("SAT label count differs");
            if (output.size() > cap || W(labels.size()) > cap - output.size())
                throw Resource("SCQ2 output exceeds max_bytes");
            output.insert(output.end(), labels.begin(), labels.end());
            put32(output, 0, cap);
            ++stats.sat;
        } else {
            if (output.size() > cap || W(packet.n) > cap - output.size())
                throw Resource("SCQ2 output exceeds max_bytes");
            output.insert(output.end(), packet.n, 0);
            put32(output, static_cast<U>(certificate.size()), cap);
            put_bytes(output, certificate, cap);
            ++stats.unsat;
            stats.proof_bytes += certificate.size();
        }
    }
    if (!observer.verify(output))
        throw std::logic_error(
            "independent observer rejected direct 2-SAT DAG output"
        );
    stats.output_bytes = output.size();
    return output;
}

}  // namespace

extern "C" {

D2S_EXPORT int direct_twosat_solve_dag(
    const std::uint8_t* packet,
    std::size_t packet_size,
    std::uint64_t max_bytes,
    std::uint8_t** output,
    std::size_t* output_size,
    Direct2SATStats* public_stats
) {
    if (output == nullptr || output_size == nullptr || public_stats == nullptr)
        return 2;
    *output = nullptr;
    *output_size = 0;
    std::memset(public_stats, 0, sizeof(*public_stats));
    try {
        StatsInternal stats;
        std::vector<std::uint8_t> result =
            solve_packet_dag(packet, packet_size, max_bytes, stats);
        void* memory = std::malloc(result.empty() ? 1 : result.size());
        if (memory == nullptr) throw std::bad_alloc();
        if (!result.empty()) std::memcpy(memory, result.data(), result.size());
        *output = static_cast<std::uint8_t*>(memory);
        *output_size = result.size();
        public_stats->n = stats.n;
        public_stats->edges = stats.edges;
        public_stats->queries = stats.queries;
        public_stats->pins = stats.pins;
        public_stats->components = stats.components;
        public_stats->dag_edges = stats.dag_edges;
        public_stats->sat = stats.sat;
        public_stats->unsat = stats.unsat;
        public_stats->engine = stats.engine;
        public_stats->closure_bytes = stats.closure_bytes;
        public_stats->output_bytes = stats.output_bytes;
        public_stats->proof_bytes = stats.proof_bytes;
        LAST_ERROR.clear();
        return 0;
    } catch (const Invalid& error) {
        LAST_ERROR = error.what();
        return 1;
    } catch (const Resource& error) {
        LAST_ERROR = error.what();
        return 2;
    } catch (const std::exception& error) {
        LAST_ERROR = error.what();
        return 3;
    } catch (...) {
        LAST_ERROR = "unknown direct 2-SAT DAG failure";
        return 3;
    }
}

}  // extern "C"
