#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clause = std::vector<std::int32_t>;

struct LiteralLess {
    bool operator()(std::int32_t left, std::int32_t right) const noexcept {
        const std::int32_t la = left < 0 ? -left : left;
        const std::int32_t ra = right < 0 ? -right : right;
        return la != ra ? la < ra : left < right;
    }
};

struct ClauseLess {
    bool operator()(const Clause& left, const Clause& right) const noexcept {
        if (left.size() != right.size()) return left.size() < right.size();
        return std::lexicographical_compare(left.begin(), left.end(), right.begin(), right.end());
    }
};

struct Reduction {
    bool contradiction = false;
    std::uint32_t conflict_code = 0;  // 0 none, 1 binary, 2 unit
    std::uint32_t conflict_variable = 0;
    std::uint32_t normalized_clause_count = 0;
    std::uint32_t unit_input_clauses = 0;
    std::uint32_t binary_input_clauses = 0;
    std::uint32_t long_input_clauses = 0;
    std::uint32_t class_count = 0;
    std::vector<std::int32_t> class_literals;
    std::uint32_t reduced_nvars = 0;
    std::vector<Clause> residual_clauses;
    std::vector<std::int32_t> mapping;
    std::vector<std::uint8_t> fixed;
    std::uint32_t equivalent_variables_removed = 0;
    std::uint32_t propagated_classes = 0;
};

std::uint32_t node_for_literal(std::int32_t literal) noexcept {
    const std::uint32_t variable = static_cast<std::uint32_t>(literal < 0 ? -literal : literal);
    return 2U * (variable - 1U) + (literal < 0 ? 1U : 0U);
}

void sort_unique_clauses(std::vector<Clause>& rows) {
    ClauseLess less;
    std::sort(rows.begin(), rows.end(), less);
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

bool normalize_clause(std::uint32_t nvars, Clause& row, bool validate_range) {
    LiteralLess less;
    if (validate_range) {
        for (std::int32_t literal : row) {
            const std::int64_t wide = literal;
            const std::int64_t absolute = wide < 0 ? -wide : wide;
            if (literal == 0 || absolute > static_cast<std::int64_t>(nvars)) {
                throw std::invalid_argument("CNF literal must be nonzero and in range");
            }
        }
    }
    std::sort(row.begin(), row.end(), less);
    Clause compact;
    compact.reserve(row.size());
    for (std::int32_t literal : row) {
        if (!compact.empty()) {
            if (compact.back() == literal) continue;
            const std::int64_t left = compact.back();
            const std::int64_t right = literal;
            const std::int64_t left_abs = left < 0 ? -left : left;
            const std::int64_t right_abs = right < 0 ? -right : right;
            if (left_abs == right_abs) return false;
        }
        compact.push_back(literal);
    }
    row.swap(compact);
    return true;
}

std::vector<Clause> canonicalize(std::uint32_t nvars, std::vector<Clause> rows) {
    std::vector<Clause> canonical;
    canonical.reserve(rows.size());
    for (Clause& row : rows) {
        if (normalize_clause(nvars, row, true)) canonical.push_back(std::move(row));
    }
    sort_unique_clauses(canonical);
    return canonical;
}

void validate_canonical_row(std::uint32_t nvars, const Clause& row) {
    LiteralLess less;
    for (std::size_t index = 0; index < row.size(); ++index) {
        const std::int32_t literal = row[index];
        const std::int64_t wide = literal;
        const std::int64_t absolute = wide < 0 ? -wide : wide;
        if (literal == 0 || absolute > static_cast<std::int64_t>(nvars)) {
            throw std::invalid_argument("canonical CNF literal must be nonzero and in range");
        }
        if (index) {
            const std::int32_t previous = row[index - 1];
            if (!less(previous, literal)) {
                throw std::invalid_argument("canonical CNF clause is not strictly normalized");
            }
            const std::int64_t previous_wide = previous;
            const std::int64_t previous_abs = previous_wide < 0 ? -previous_wide : previous_wide;
            if (previous_abs == absolute) {
                throw std::invalid_argument("canonical CNF clause is tautological");
            }
        }
    }
}

std::vector<Clause> read_input(
    std::uint32_t nvars,
    std::uint64_t clause_count,
    const std::uint64_t* offsets,
    const std::int32_t* literals,
    std::uint64_t literal_count,
    bool canonical_input) {
    if (!offsets) throw std::invalid_argument("offsets pointer is null");
    if (literal_count && !literals) throw std::invalid_argument("literals pointer is null");
    if (offsets[0] != 0 || offsets[clause_count] != literal_count) {
        throw std::invalid_argument("clause offsets do not span the literal array");
    }
    if (clause_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("clause count is too large");
    }
    std::vector<Clause> rows;
    rows.reserve(static_cast<std::size_t>(clause_count));
    ClauseLess clause_less;
    for (std::uint64_t index = 0; index < clause_count; ++index) {
        const std::uint64_t start = offsets[index];
        const std::uint64_t stop = offsets[index + 1];
        if (start > stop || stop > literal_count) {
            throw std::invalid_argument("clause offsets are not monotone");
        }
        if (stop - start > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::invalid_argument("clause width is too large");
        }
        Clause row;
        row.reserve(static_cast<std::size_t>(stop - start));
        for (std::uint64_t offset = start; offset < stop; ++offset) row.push_back(literals[offset]);
        if (canonical_input) {
            validate_canonical_row(nvars, row);
            if (!rows.empty() && !clause_less(rows.back(), row)) {
                throw std::invalid_argument("canonical CNF clauses are not strictly ordered");
            }
        }
        rows.push_back(std::move(row));
    }
    return canonical_input ? rows : canonicalize(nvars, std::move(rows));
}

void sort_unique(std::vector<std::uint32_t>& row) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
}

std::vector<std::int32_t> strong_components(
    const std::vector<std::vector<std::uint32_t>>& graph,
    const std::vector<std::vector<std::uint32_t>>& reverse) {
    const std::uint32_t size = static_cast<std::uint32_t>(graph.size());
    std::vector<std::uint8_t> seen(size, 0);
    std::vector<std::uint32_t> finish;
    finish.reserve(size);
    for (std::uint32_t root = 0; root < size; ++root) {
        if (seen[root]) continue;
        seen[root] = 1;
        std::vector<std::pair<std::uint32_t, std::size_t>> stack{{root, 0}};
        while (!stack.empty()) {
            auto& frame = stack.back();
            const std::uint32_t vertex = frame.first;
            if (frame.second == graph[vertex].size()) {
                finish.push_back(vertex);
                stack.pop_back();
                continue;
            }
            const std::uint32_t neighbor = graph[vertex][frame.second++];
            if (!seen[neighbor]) {
                seen[neighbor] = 1;
                stack.push_back({neighbor, 0});
            }
        }
    }

    std::vector<std::int32_t> component(size, -1);
    std::int32_t next_component = 0;
    for (auto iterator = finish.rbegin(); iterator != finish.rend(); ++iterator) {
        const std::uint32_t root = *iterator;
        if (component[root] != -1) continue;
        component[root] = next_component;
        std::vector<std::uint32_t> stack{root};
        while (!stack.empty()) {
            const std::uint32_t vertex = stack.back();
            stack.pop_back();
            for (std::uint32_t neighbor : reverse[vertex]) {
                if (component[neighbor] == -1) {
                    component[neighbor] = next_component;
                    stack.push_back(neighbor);
                }
            }
        }
        ++next_component;
    }
    return component;
}

std::vector<std::int32_t> equivalence_classes(
    std::uint32_t nvars,
    const std::vector<std::int32_t>& component,
    std::uint32_t& class_count) {
    std::map<std::pair<std::int32_t, std::int32_t>, std::vector<std::uint32_t>> by_pair;
    for (std::uint32_t variable = 1; variable <= nvars; ++variable) {
        const std::int32_t positive = component[node_for_literal(static_cast<std::int32_t>(variable))];
        const std::int32_t negative = component[node_for_literal(-static_cast<std::int32_t>(variable))];
        by_pair[{std::min(positive, negative), std::max(positive, negative)}].push_back(variable);
    }
    std::vector<std::vector<std::uint32_t>> groups;
    groups.reserve(by_pair.size());
    for (auto& entry : by_pair) groups.push_back(std::move(entry.second));
    std::sort(groups.begin(), groups.end(), [](const auto& left, const auto& right) {
        return left.front() < right.front();
    });

    class_count = static_cast<std::uint32_t>(groups.size());
    std::vector<std::int32_t> class_literals(nvars, 0);
    for (std::uint32_t index = 0; index < class_count; ++index) {
        const auto& members = groups[index];
        const std::uint32_t representative = members.front();
        const std::int32_t positive_component = component[node_for_literal(
            static_cast<std::int32_t>(representative))];
        for (std::uint32_t variable : members) {
            if (component[node_for_literal(static_cast<std::int32_t>(variable))] == positive_component) {
                class_literals[variable - 1] = static_cast<std::int32_t>(index + 1);
            } else if (component[node_for_literal(-static_cast<std::int32_t>(variable))] == positive_component) {
                class_literals[variable - 1] = -static_cast<std::int32_t>(index + 1);
            } else {
                throw std::runtime_error("complementary SCC pairing is inconsistent");
            }
        }
    }
    return class_literals;
}

std::vector<Clause> substitute(
    std::uint32_t class_count,
    std::vector<Clause> clauses,
    const std::vector<std::int32_t>& class_literals) {
    std::vector<Clause> rows;
    rows.reserve(clauses.size());
    for (Clause& row : clauses) {
        for (std::int32_t& literal : row) {
            const std::uint32_t variable = static_cast<std::uint32_t>(literal < 0 ? -literal : literal);
            const std::int32_t target = class_literals[variable - 1];
            literal = literal > 0 ? target : -target;
        }
        if (normalize_clause(class_count, row, false)) rows.push_back(std::move(row));
    }
    sort_unique_clauses(rows);
    return rows;
}

struct UnitResult {
    std::vector<std::int8_t> values;
    bool conflict = false;
};

UnitResult unit_propagate(std::uint32_t nvars, const std::vector<Clause>& clauses) {
    UnitResult result;
    result.values.assign(static_cast<std::size_t>(nvars) + 1, 0);
    bool has_unit = false;
    for (const Clause& row : clauses) {
        if (row.empty()) {
            result.conflict = true;
            return result;
        }
        has_unit = has_unit || row.size() == 1;
    }
    if (!has_unit) return result;

    std::vector<std::vector<std::uint32_t>> occurrences(2ULL * nvars);
    std::vector<std::uint32_t> remaining;
    remaining.reserve(clauses.size());
    std::vector<std::uint8_t> satisfied(clauses.size(), 0);
    std::deque<std::int32_t> pending;
    for (std::uint32_t index = 0; index < clauses.size(); ++index) {
        const Clause& row = clauses[index];
        remaining.push_back(static_cast<std::uint32_t>(row.size()));
        if (row.size() == 1) pending.push_back(row.front());
        for (std::int32_t literal : row) occurrences[node_for_literal(literal)].push_back(index);
    }

    while (!pending.empty()) {
        const std::int32_t literal = pending.front();
        pending.pop_front();
        const std::uint32_t variable = static_cast<std::uint32_t>(literal < 0 ? -literal : literal);
        const std::int8_t sign = literal > 0 ? 1 : -1;
        std::int8_t& value = result.values[variable];
        if (value == sign) continue;
        if (value == -sign) {
            result.conflict = true;
            return result;
        }
        value = sign;

        for (std::uint32_t index : occurrences[node_for_literal(literal)]) satisfied[index] = 1;
        for (std::uint32_t index : occurrences[node_for_literal(-literal)]) {
            if (satisfied[index]) continue;
            std::uint32_t& count = remaining[index];
            --count;
            if (count == 0) {
                result.conflict = true;
                return result;
            }
            if (count == 1) {
                for (std::int32_t candidate : clauses[index]) {
                    const std::uint32_t candidate_variable = static_cast<std::uint32_t>(
                        candidate < 0 ? -candidate : candidate);
                    if (result.values[candidate_variable] == 0) {
                        pending.push_back(candidate);
                        break;
                    }
                }
            }
        }
    }
    return result;
}

void set_identity_result(
    std::uint32_t nvars,
    std::vector<Clause> canonical,
    Reduction& result) {
    result.class_count = nvars;
    result.class_literals.resize(nvars);
    result.mapping.resize(nvars);
    result.fixed.assign(nvars, 0);
    for (std::uint32_t variable = 1; variable <= nvars; ++variable) {
        result.class_literals[variable - 1] = static_cast<std::int32_t>(variable);
        result.mapping[variable - 1] = static_cast<std::int32_t>(variable);
    }
    result.reduced_nvars = nvars;
    result.residual_clauses = std::move(canonical);
}

Reduction reduce_formula(std::uint32_t nvars, std::vector<Clause> canonical) {
    Reduction result;
    if (canonical.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("normalized clause count exceeds uint32");
    }
    result.normalized_clause_count = static_cast<std::uint32_t>(canonical.size());
    bool has_empty_clause = false;
    for (const Clause& row : canonical) {
        if (row.empty()) has_empty_clause = true;
        else if (row.size() == 1) ++result.unit_input_clauses;
        else if (row.size() == 2) ++result.binary_input_clauses;
        else if (row.size() >= 3) ++result.long_input_clauses;
    }

    if (!has_empty_clause && result.unit_input_clauses == 0 && result.binary_input_clauses == 0) {
        set_identity_result(nvars, std::move(canonical), result);
        return result;
    }

    std::vector<std::vector<std::uint32_t>> graph(2ULL * nvars);
    std::vector<std::vector<std::uint32_t>> reverse(2ULL * nvars);
    auto add_arc = [&](std::int32_t source_literal, std::int32_t target_literal) {
        const std::uint32_t source = node_for_literal(source_literal);
        const std::uint32_t target = node_for_literal(target_literal);
        graph[source].push_back(target);
        reverse[target].push_back(source);
    };
    for (const Clause& row : canonical) {
        if (row.size() == 1) {
            add_arc(-row[0], row[0]);
        } else if (row.size() == 2) {
            add_arc(-row[0], row[1]);
            add_arc(-row[1], row[0]);
        } else if (row.size() > 2) {
            break;
        }
    }
    for (auto& row : graph) sort_unique(row);
    for (auto& row : reverse) sort_unique(row);
    const std::vector<std::int32_t> component = strong_components(graph, reverse);

    for (std::uint32_t variable = 1; variable <= nvars; ++variable) {
        if (component[node_for_literal(static_cast<std::int32_t>(variable))] ==
            component[node_for_literal(-static_cast<std::int32_t>(variable))]) {
            result.contradiction = true;
            result.conflict_code = 1;
            result.conflict_variable = variable;
            result.mapping.assign(nvars, 0);
            result.fixed.assign(nvars, 0);
            result.residual_clauses = {Clause{}};
            return result;
        }
    }

    result.class_literals = equivalence_classes(nvars, component, result.class_count);
    result.equivalent_variables_removed = nvars - result.class_count;
    if (has_empty_clause) {
        result.contradiction = true;
        result.conflict_code = 2;
        result.mapping.assign(nvars, 0);
        result.fixed.assign(nvars, 0);
        result.residual_clauses = {Clause{}};
        return result;
    }
    if (result.class_count == nvars && result.unit_input_clauses == 0) {
        result.mapping = result.class_literals;
        result.fixed.assign(nvars, 0);
        result.reduced_nvars = nvars;
        result.residual_clauses = std::move(canonical);
        return result;
    }

    std::vector<Clause> substituted = substitute(
        result.class_count, std::move(canonical), result.class_literals);
    const UnitResult units = unit_propagate(result.class_count, substituted);
    for (std::size_t index = 1; index < units.values.size(); ++index) {
        if (units.values[index] != 0) ++result.propagated_classes;
    }
    if (units.conflict) {
        result.contradiction = true;
        result.conflict_code = 2;
        result.mapping.assign(nvars, 0);
        result.fixed.assign(nvars, 0);
        result.residual_clauses = {Clause{}};
        return result;
    }

    std::vector<std::uint32_t> free(static_cast<std::size_t>(result.class_count) + 1, 0);
    for (std::uint32_t variable = 1; variable <= result.class_count; ++variable) {
        if (units.values[variable] == 0) free[variable] = ++result.reduced_nvars;
    }

    if (result.propagated_classes == 0) {
        result.residual_clauses = std::move(substituted);
    } else if (result.reduced_nvars == 0) {
        result.residual_clauses.clear();
    } else {
        std::vector<Clause> residual_rows;
        residual_rows.reserve(substituted.size());
        for (const Clause& row : substituted) {
            bool satisfied = false;
            for (std::int32_t literal : row) {
                const std::uint32_t variable = static_cast<std::uint32_t>(literal < 0 ? -literal : literal);
                if (units.values[variable] == (literal > 0 ? 1 : -1)) {
                    satisfied = true;
                    break;
                }
            }
            if (satisfied) continue;
            Clause remaining;
            remaining.reserve(row.size());
            for (std::int32_t literal : row) {
                const std::uint32_t variable = static_cast<std::uint32_t>(literal < 0 ? -literal : literal);
                if (units.values[variable] == 0) {
                    const std::int32_t target = static_cast<std::int32_t>(free[variable]);
                    remaining.push_back(literal > 0 ? target : -target);
                }
            }
            residual_rows.push_back(std::move(remaining));
        }
        sort_unique_clauses(residual_rows);
        result.residual_clauses = std::move(residual_rows);
    }

    result.mapping.reserve(nvars);
    result.fixed.reserve(nvars);
    for (std::int32_t target : result.class_literals) {
        const std::uint32_t class_variable = static_cast<std::uint32_t>(target < 0 ? -target : target);
        const std::int32_t relation = target > 0 ? 1 : -1;
        const std::int8_t value = units.values[class_variable];
        if (value != 0) {
            result.mapping.push_back(0);
            result.fixed.push_back(value * relation > 0 ? 1 : 0);
        } else {
            result.mapping.push_back(static_cast<std::int32_t>(free[class_variable]) * relation);
            result.fixed.push_back(0);
        }
    }
    return result;
}

void append_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_i32(std::vector<std::uint8_t>& output, std::int32_t value) {
    append_u32(output, static_cast<std::uint32_t>(value));
}

void append_u64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) output.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::vector<std::uint8_t> serialize(std::uint32_t nvars, const Reduction& result) {
    std::uint64_t residual_literals = 0;
    for (const Clause& row : result.residual_clauses) residual_literals += row.size();
    std::vector<std::uint8_t> output;
    output.reserve(96ULL + 9ULL * nvars + 8ULL * (result.residual_clauses.size() + 1) + 4ULL * residual_literals);
    output.insert(output.end(), {'S', 'G', 'C', '1'});
    append_u32(output, 1);
    append_u32(output, result.contradiction ? 1U : 0U);
    append_u32(output, result.conflict_code);
    append_u32(output, result.conflict_variable);
    append_u32(output, nvars);
    append_u32(output, result.normalized_clause_count);
    append_u32(output, result.unit_input_clauses);
    append_u32(output, result.binary_input_clauses);
    append_u32(output, result.long_input_clauses);
    append_u32(output, result.class_count);
    append_u32(output, static_cast<std::uint32_t>(result.class_literals.size()));
    append_u32(output, result.reduced_nvars);
    append_u32(output, result.equivalent_variables_removed);
    append_u32(output, result.propagated_classes);
    append_u64(output, result.residual_clauses.size());
    append_u64(output, residual_literals);
    for (std::int32_t value : result.class_literals) append_i32(output, value);
    for (std::int32_t value : result.mapping) append_i32(output, value);
    output.insert(output.end(), result.fixed.begin(), result.fixed.end());
    std::uint64_t cursor = 0;
    append_u64(output, 0);
    for (const Clause& row : result.residual_clauses) {
        cursor += row.size();
        append_u64(output, cursor);
    }
    for (const Clause& row : result.residual_clauses) {
        for (std::int32_t literal : row) append_i32(output, literal);
    }
    return output;
}

void write_error(char* buffer, std::uint64_t capacity, const char* message) noexcept {
    if (!buffer || capacity == 0) return;
    const std::size_t length = std::min<std::size_t>(std::strlen(message), static_cast<std::size_t>(capacity - 1));
    std::memcpy(buffer, message, length);
    buffer[length] = '\0';
}

int reduce_entry(
    std::uint32_t nvars,
    std::uint64_t clause_count,
    const std::uint64_t* offsets,
    const std::int32_t* literals,
    std::uint64_t literal_count,
    bool canonical_input,
    std::uint8_t** output_data,
    std::uint64_t* output_size,
    char* error,
    std::uint64_t error_capacity) noexcept {
    if (!output_data || !output_size) {
        write_error(error, error_capacity, "output pointers are null");
        return 1;
    }
    *output_data = nullptr;
    *output_size = 0;
    try {
        std::vector<Clause> canonical = read_input(
            nvars, clause_count, offsets, literals, literal_count, canonical_input);
        const Reduction reduction = reduce_formula(nvars, std::move(canonical));
        const std::vector<std::uint8_t> bytes = serialize(nvars, reduction);
        void* memory = std::malloc(bytes.size());
        if (!memory) throw std::bad_alloc();
        std::memcpy(memory, bytes.data(), bytes.size());
        *output_data = static_cast<std::uint8_t*>(memory);
        *output_size = bytes.size();
        return 0;
    } catch (const std::bad_alloc&) {
        write_error(error, error_capacity, "native binary core ran out of memory");
        return 2;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 1;
    } catch (...) {
        write_error(error, error_capacity, "unknown native binary core failure");
        return 2;
    }
}
}  // namespace

extern "C" std::uint32_t spectra_general_abi() noexcept { return 1; }

extern "C" int spectra_general_reduce(
    std::uint32_t nvars,
    std::uint64_t clause_count,
    const std::uint64_t* offsets,
    const std::int32_t* literals,
    std::uint64_t literal_count,
    std::uint8_t** output_data,
    std::uint64_t* output_size,
    char* error,
    std::uint64_t error_capacity) noexcept {
    return reduce_entry(
        nvars, clause_count, offsets, literals, literal_count, false,
        output_data, output_size, error, error_capacity);
}

extern "C" int spectra_general_reduce_canonical(
    std::uint32_t nvars,
    std::uint64_t clause_count,
    const std::uint64_t* offsets,
    const std::int32_t* literals,
    std::uint64_t literal_count,
    std::uint8_t** output_data,
    std::uint64_t* output_size,
    char* error,
    std::uint64_t error_capacity) noexcept {
    return reduce_entry(
        nvars, clause_count, offsets, literals, literal_count, true,
        output_data, output_size, error, error_capacity);
}

extern "C" void spectra_general_free(void* pointer) noexcept { std::free(pointer); }
