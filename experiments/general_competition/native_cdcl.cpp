#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define SPECTRA_EXPORT extern "C" __declspec(dllexport)
#else
#define SPECTRA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

thread_local std::string last_error;
// Match canonical_cnf's public admission bound before allocating or computing
// signed watch indices. The raw ABI must enforce this without the Python wrapper.
constexpr std::uint32_t max_native_variables = 10000000;

struct Clause {
  std::vector<int> lits;
  int watch0 = 0;
  int watch1 = 0;
  bool learned = false;
  bool deleted = false;
  int lbd = 0;
  double activity = 0.0;
};

struct Stats {
  std::uint64_t decisions = 0;
  std::uint64_t propagations = 0;
  std::uint64_t conflicts = 0;
  std::uint64_t restarts = 0;
  std::uint64_t learned_clauses = 0;
  std::uint64_t learned_literals = 0;
  std::uint64_t reductions = 0;
  std::uint64_t deleted_clauses = 0;
};

struct HeapEntry {
  double activity;
  int negative_variable;
  std::uint64_t generation;

  bool operator<(const HeapEntry &other) const {
    if (activity != other.activity) return activity < other.activity;
    return negative_variable < other.negative_variable;
  }
};

struct Solver {
  int nvars;
  std::vector<Clause> clauses;
  std::vector<std::vector<int>> watches;
  // Keep the watched literals together so satisfied clauses need no clause load.
  std::vector<std::pair<int, int>> watched_literals;
  std::vector<std::int8_t> values;
  std::vector<int> levels;
  std::vector<int> reasons;
  std::vector<std::int8_t> phases;
  std::vector<double> activity;
  // Relevance must survive floating-point activity rescaling and underflow.
  std::vector<std::uint8_t> occurs;
  std::vector<std::int64_t> polarity_bias;
  std::vector<std::uint64_t> heap_generation;
  std::priority_queue<HeapEntry> variable_heap;
  std::vector<std::uint8_t> seen;
  std::vector<std::uint8_t> minimize_seen;
  std::vector<int> trail;
  std::vector<int> trail_limits;
  std::size_t qhead = 0;
  double var_increment = 1.0;
  double var_decay = 0.95;
  bool root_conflict = false;
  Stats stats;
  std::vector<std::vector<int>> proof;

  explicit Solver(int variables)
      : nvars(variables), watches(static_cast<std::size_t>(2 * variables)),
        values(static_cast<std::size_t>(variables + 1), 0),
        levels(static_cast<std::size_t>(variables + 1), 0),
        reasons(static_cast<std::size_t>(variables + 1), -1),
        phases(static_cast<std::size_t>(variables + 1), 1),
        activity(static_cast<std::size_t>(variables + 1), 0.0),
        occurs(static_cast<std::size_t>(variables + 1), 0),
        polarity_bias(static_cast<std::size_t>(variables + 1), 0),
        heap_generation(static_cast<std::size_t>(variables + 1), 0),
        seen(static_cast<std::size_t>(variables + 1), 0),
        minimize_seen(static_cast<std::size_t>(variables + 1), 0) {}

  static int variable(int literal) { return literal < 0 ? -literal : literal; }

  std::size_t watch_index(int literal) const {
    const int var = variable(literal);
    return static_cast<std::size_t>(2 * (var - 1) + (literal < 0));
  }

  int literal_value(int literal) const {
    const int value = values[static_cast<std::size_t>(variable(literal))];
    if (!value) return 0;
    return literal > 0 ? value : -value;
  }

  int decision_level() const { return static_cast<int>(trail_limits.size()); }

  bool enqueue(int literal, int reason) {
    const int var = variable(literal);
    const int sign = literal > 0 ? 1 : -1;
    const auto index = static_cast<std::size_t>(var);
    if (values[index]) return values[index] == sign;
    values[index] = static_cast<std::int8_t>(sign);
    levels[index] = decision_level();
    reasons[index] = reason;
    phases[index] = static_cast<std::int8_t>(sign);
    trail.push_back(literal);
    return true;
  }

  static std::vector<int> normalize_clause(const int *begin, const int *end,
                                           int nvars, bool &tautology) {
    std::vector<int> row(begin, end);
    for (const int literal : row) {
      if (!literal || literal == std::numeric_limits<int>::min() ||
          variable(literal) > nvars) {
        throw std::invalid_argument("literal outside declared variable range");
      }
    }
    std::sort(row.begin(), row.end(), [](int left, int right) {
      const int left_var = variable(left), right_var = variable(right);
      return left_var != right_var ? left_var < right_var : left < right;
    });
    row.erase(std::unique(row.begin(), row.end()), row.end());
    tautology = false;
    for (std::size_t index = 1; index < row.size(); ++index) {
      if (row[index] == -row[index - 1]) {
        tautology = true;
        row.clear();
        break;
      }
    }
    return row;
  }

  int add_clause(std::vector<int> row, bool learned, int lbd = 0) {
    if (row.empty()) {
      if (learned) proof.emplace_back();
      root_conflict = true;
      return -1;
    }
    Clause clause;
    clause.lits = std::move(row);
    clause.learned = learned;
    clause.lbd = lbd;
    clause.watch0 = 0;
    clause.watch1 = clause.lits.size() > 1 ? 1 : 0;
    const int id = static_cast<int>(clauses.size());
    clauses.push_back(std::move(clause));
    Clause &stored = clauses.back();
    watched_literals.emplace_back(
        stored.lits[static_cast<std::size_t>(stored.watch0)],
        stored.lits[static_cast<std::size_t>(stored.watch1)]);
    watches[watch_index(stored.lits[static_cast<std::size_t>(stored.watch0)])]
        .push_back(id);
    if (stored.watch1 != stored.watch0) {
      watches[watch_index(stored.lits[static_cast<std::size_t>(stored.watch1)])]
          .push_back(id);
    }
    if (learned) {
      proof.push_back(stored.lits);
      ++stats.learned_clauses;
      stats.learned_literals += stored.lits.size();
    }
    return id;
  }

  void add_initial(const int *begin, const int *end) {
    bool tautology = false;
    std::vector<int> row = normalize_clause(begin, end, nvars, tautology);
    if (tautology) return;
    const std::size_t width = row.size();
    const std::uint64_t weight =
        std::uint64_t{1} << std::max(0, 16 - std::min(static_cast<int>(width), 16));
    for (const int literal : row) {
      const int var = variable(literal);
      occurs[static_cast<std::size_t>(var)] = 1;
      activity[static_cast<std::size_t>(var)] += static_cast<double>(weight);
      polarity_bias[static_cast<std::size_t>(var)] +=
          literal > 0 ? static_cast<std::int64_t>(weight)
                      : -static_cast<std::int64_t>(weight);
      phases[static_cast<std::size_t>(var)] =
          polarity_bias[static_cast<std::size_t>(var)] >= 0 ? 1 : -1;
    }
    const int id = add_clause(std::move(row), false);
    if (id >= 0 && clauses[static_cast<std::size_t>(id)].lits.size() == 1) {
      if (!enqueue(clauses[static_cast<std::size_t>(id)].lits[0], id)) {
        root_conflict = true;
      }
    }
  }

  int propagate() {
    while (qhead < trail.size()) {
      const int assigned = trail[qhead++];
      ++stats.propagations;
      auto &watching = watches[watch_index(-assigned)];
      std::size_t write = 0;
      for (std::size_t read = 0; read < watching.size(); ++read) {
        const int clause_id = watching[read];
        const auto watched = watched_literals[static_cast<std::size_t>(clause_id)];
        const int false_slot = watched.first == -assigned ? 0
            : watched.second == -assigned ? 1 : -1;
        if (false_slot < 0) continue;
        const int other_literal = false_slot == 0 ? watched.second : watched.first;
        if (literal_value(other_literal) > 0) {
          watching[write++] = clause_id;
          continue;
        }
        Clause &clause = clauses[static_cast<std::size_t>(clause_id)];
        if (clause.deleted) continue;
        const int other_pos = false_slot == 0 ? clause.watch1 : clause.watch0;

        int replacement = -1;
        for (int index = 0; index < static_cast<int>(clause.lits.size()); ++index) {
          if (index == other_pos ||
              index == (false_slot == 0 ? clause.watch0 : clause.watch1)) {
            continue;
          }
          if (literal_value(clause.lits[static_cast<std::size_t>(index)]) >= 0) {
            replacement = index;
            break;
          }
        }
        if (replacement >= 0) {
          const int replacement_literal = clause.lits[static_cast<std::size_t>(replacement)];
          if (false_slot == 0) {
            clause.watch0 = replacement;
            watched_literals[static_cast<std::size_t>(clause_id)].first = replacement_literal;
          } else {
            clause.watch1 = replacement;
            watched_literals[static_cast<std::size_t>(clause_id)].second = replacement_literal;
          }
          watches[watch_index(replacement_literal)].push_back(clause_id);
          continue;
        }

        watching[write++] = clause_id;
        const int other_value = literal_value(other_literal);
        if (other_value < 0 || !enqueue(other_literal, clause_id)) {
          while (++read < watching.size()) watching[write++] = watching[read];
          watching.resize(write);
          return clause_id;
        }
      }
      watching.resize(write);
    }
    return -1;
  }

  void push_heap(int var) {
    const auto index = static_cast<std::size_t>(var);
    const std::uint64_t generation = ++heap_generation[index];
    variable_heap.push(HeapEntry{activity[index], -var, generation});
  }

  void initialize_heap() {
    for (int var = 1; var <= nvars; ++var) {
      if (occurs[static_cast<std::size_t>(var)]) push_heap(var);
    }
  }

  void rebuild_heap() {
    variable_heap = std::priority_queue<HeapEntry>();
    for (int var = 1; var <= nvars; ++var) {
      const auto index = static_cast<std::size_t>(var);
      if (values[index] == 0 && occurs[index]) push_heap(var);
    }
  }

  void bump_variable(int var) {
    activity[static_cast<std::size_t>(var)] += var_increment;
    if (activity[static_cast<std::size_t>(var)] > 1e100) {
      for (int index = 1; index <= nvars; ++index) {
        activity[static_cast<std::size_t>(index)] *= 1e-100;
      }
      rebuild_heap();
      var_increment *= 1e-100;
    } else {
      push_heap(var);
    }
  }

  void decay_activity() { var_increment /= var_decay; }

  void rebuild_watches() {
    watches.assign(static_cast<std::size_t>(2 * nvars), {});
    for (int id = 0; id < static_cast<int>(clauses.size()); ++id) {
      Clause &clause = clauses[static_cast<std::size_t>(id)];
      if (clause.deleted) continue;
      if (clause.lits.empty()) throw std::logic_error("active empty clause");
      if (clause.watch0 < 0 || clause.watch0 >= static_cast<int>(clause.lits.size())) {
        clause.watch0 = 0;
      }
      if (clause.watch1 < 0 || clause.watch1 >= static_cast<int>(clause.lits.size())) {
        clause.watch1 = clause.lits.size() > 1 ? 1 : 0;
      }
      watched_literals[static_cast<std::size_t>(id)] = {
          clause.lits[static_cast<std::size_t>(clause.watch0)],
          clause.lits[static_cast<std::size_t>(clause.watch1)]};
      watches[watch_index(clause.lits[static_cast<std::size_t>(clause.watch0)])]
          .push_back(id);
      if (clause.watch1 != clause.watch0) {
        watches[watch_index(clause.lits[static_cast<std::size_t>(clause.watch1)])]
            .push_back(id);
      }
    }
  }

  void reduce_database() {
    std::vector<std::uint8_t> locked(clauses.size(), 0);
    for (int var = 1; var <= nvars; ++var) {
      const int reason = reasons[static_cast<std::size_t>(var)];
      if (reason >= 0) locked[static_cast<std::size_t>(reason)] = 1;
    }
    std::vector<int> candidates;
    for (int id = 0; id < static_cast<int>(clauses.size()); ++id) {
      const Clause &clause = clauses[static_cast<std::size_t>(id)];
      if (clause.learned && !clause.deleted && clause.lbd > 2
          && !locked[static_cast<std::size_t>(id)] && clause.lits.size() > 2) {
        candidates.push_back(id);
      }
    }
    if (candidates.size() < 32) return;
    std::sort(candidates.begin(), candidates.end(), [&](int left, int right) {
      const Clause &a = clauses[static_cast<std::size_t>(left)];
      const Clause &b = clauses[static_cast<std::size_t>(right)];
      if (a.lbd != b.lbd) return a.lbd > b.lbd;
      if (a.activity != b.activity) return a.activity < b.activity;
      if (a.lits.size() != b.lits.size()) return a.lits.size() > b.lits.size();
      return left < right;
    });
    const std::size_t remove = candidates.size() / 2;
    for (std::size_t index = 0; index < remove; ++index) {
      Clause &clause = clauses[static_cast<std::size_t>(candidates[index])];
      clause.deleted = true;
      watched_literals[static_cast<std::size_t>(candidates[index])] = {0, 0};
      std::vector<int>().swap(clause.lits);
      ++stats.deleted_clauses;
    }
    ++stats.reductions;
    rebuild_watches();
  }

  bool redundant_literal(
      int literal, const std::vector<std::uint8_t> &allowed_levels) {
    std::vector<int> stack;
    std::vector<int> touched;
    stack.push_back(variable(literal));
    while (!stack.empty()) {
      const int current = stack.back();
      stack.pop_back();
      const int reason_id = reasons[static_cast<std::size_t>(current)];
      if (reason_id < 0) {
        for (const int var : touched) minimize_seen[static_cast<std::size_t>(var)] = 0;
        return false;
      }
      const Clause &reason = clauses[static_cast<std::size_t>(reason_id)];
      if (reason.deleted) throw std::logic_error("deleted clause remained a reason");
      for (const int candidate : reason.lits) {
        const int var = variable(candidate);
        if (var == current) continue;
        const int level = levels[static_cast<std::size_t>(var)];
        if (level == 0 || seen[static_cast<std::size_t>(var)]) continue;
        if (level >= static_cast<int>(allowed_levels.size())
            || !allowed_levels[static_cast<std::size_t>(level)]
            || reasons[static_cast<std::size_t>(var)] < 0) {
          for (const int item : touched) {
            minimize_seen[static_cast<std::size_t>(item)] = 0;
          }
          return false;
        }
        if (!minimize_seen[static_cast<std::size_t>(var)]) {
          minimize_seen[static_cast<std::size_t>(var)] = 1;
          touched.push_back(var);
          stack.push_back(var);
        }
      }
    }
    for (const int var : touched) minimize_seen[static_cast<std::size_t>(var)] = 0;
    return true;
  }

  void analyze(int conflict_id, std::vector<int> &learned,
               int &backtrack_level, int &learned_lbd) {
    learned.clear();
    learned.push_back(0);
    int open_at_current_level = 0;
    int resolved_literal = 0;
    int trail_index = static_cast<int>(trail.size()) - 1;
    int clause_id = conflict_id;
    std::vector<int> touched;

    do {
      Clause &clause = clauses[static_cast<std::size_t>(clause_id)];
      clause.activity += 1.0;
      for (const int literal : clause.lits) {
        const int var = variable(literal);
        if (resolved_literal && var == variable(resolved_literal)) continue;
        if (seen[static_cast<std::size_t>(var)] || levels[static_cast<std::size_t>(var)] == 0) {
          continue;
        }
        seen[static_cast<std::size_t>(var)] = 1;
        touched.push_back(var);
        bump_variable(var);
        if (levels[static_cast<std::size_t>(var)] == decision_level()) {
          ++open_at_current_level;
        } else {
          learned.push_back(literal);
        }
      }
      while (trail_index >= 0 &&
             !seen[static_cast<std::size_t>(variable(trail[static_cast<std::size_t>(trail_index)]))]) {
        --trail_index;
      }
      if (trail_index < 0) throw std::logic_error("conflict analysis lost UIP");
      resolved_literal = trail[static_cast<std::size_t>(trail_index--)];
      const int resolved_var = variable(resolved_literal);
      seen[static_cast<std::size_t>(resolved_var)] = 0;
      --open_at_current_level;
      clause_id = reasons[static_cast<std::size_t>(resolved_var)];
    } while (open_at_current_level > 0);

    learned[0] = -resolved_literal;
    seen[static_cast<std::size_t>(variable(learned[0]))] = 1;
    std::vector<std::uint8_t> allowed_levels(
        static_cast<std::size_t>(decision_level() + 1), 0);
    for (std::size_t index = 1; index < learned.size(); ++index) {
      const int level = levels[static_cast<std::size_t>(variable(learned[index]))];
      if (level > 0) allowed_levels[static_cast<std::size_t>(level)] = 1;
    }
    std::size_t write = 1;
    for (std::size_t index = 1; index < learned.size(); ++index) {
      const int literal = learned[index];
      const int var = variable(literal);
      if (reasons[static_cast<std::size_t>(var)] >= 0
          && redundant_literal(literal, allowed_levels)) {
        continue;
      }
      learned[write++] = literal;
    }
    learned.resize(write);

    backtrack_level = 0;
    if (learned.size() > 1) {
      std::size_t best = 1;
      for (std::size_t index = 2; index < learned.size(); ++index) {
        if (levels[static_cast<std::size_t>(variable(learned[index]))] >
            levels[static_cast<std::size_t>(variable(learned[best]))]) {
          best = index;
        }
      }
      std::swap(learned[1], learned[best]);
      backtrack_level = levels[static_cast<std::size_t>(variable(learned[1]))];
    }
    std::fill(allowed_levels.begin(), allowed_levels.end(), 0);
    learned_lbd = 0;
    for (const int literal : learned) {
      const int level = levels[static_cast<std::size_t>(variable(literal))];
      if (level > 0 && !allowed_levels[static_cast<std::size_t>(level)]) {
        allowed_levels[static_cast<std::size_t>(level)] = 1;
        ++learned_lbd;
      }
    }
    learned_lbd = std::max(learned_lbd, 1);
    for (const int var : touched) seen[static_cast<std::size_t>(var)] = 0;
    decay_activity();
  }

  void backtrack(int target_level) {
    if (decision_level() <= target_level) return;
    const std::size_t target = static_cast<std::size_t>(trail_limits[static_cast<std::size_t>(target_level)]);
    for (std::size_t index = trail.size(); index > target; --index) {
      const int var = variable(trail[index - 1]);
      values[static_cast<std::size_t>(var)] = 0;
      levels[static_cast<std::size_t>(var)] = 0;
      reasons[static_cast<std::size_t>(var)] = -1;
      push_heap(var);
    }
    trail.resize(target);
    trail_limits.resize(static_cast<std::size_t>(target_level));
    qhead = trail.size();
  }

  int choose_variable() {
    const std::size_t rebuild_limit =
        static_cast<std::size_t>(8) * static_cast<std::size_t>(nvars) + 1024;
    if (variable_heap.size() > rebuild_limit) rebuild_heap();
    while (!variable_heap.empty()) {
      const HeapEntry entry = variable_heap.top();
      const int var = -entry.negative_variable;
      const auto index = static_cast<std::size_t>(var);
      if (values[index] != 0 || entry.generation != heap_generation[index]
          || entry.activity != activity[index]) {
        variable_heap.pop();
        continue;
      }
      variable_heap.pop();
      return var;
    }
    return 0;
  }

  int solve(std::uint64_t max_conflicts,
            std::uint64_t max_learned_literals) {
    if (root_conflict) {
      proof.emplace_back();
      return 20;
    }
    int conflict = propagate();
    if (conflict >= 0) {
      proof.emplace_back();
      return 20;
    }

    std::uint64_t next_restart = 100;
    std::uint64_t restart_span = 100;
    std::uint64_t next_reduction = 512;
    std::uint64_t reduction_span = 512;
    std::vector<int> learned;
    while (true) {
      conflict = propagate();
      if (conflict >= 0) {
        ++stats.conflicts;
        if (decision_level() == 0) {
          proof.emplace_back();
          return 20;
        }
        if (stats.conflicts > max_conflicts) return 1;
        int backtrack_level = 0;
        int learned_lbd = 0;
        analyze(conflict, learned, backtrack_level, learned_lbd);
        backtrack(backtrack_level);
        if (stats.learned_literals + learned.size() > max_learned_literals) {
          return 2;
        }
        const int learned_id = add_clause(learned, true, learned_lbd);
        if (learned_id < 0 || !enqueue(learned[0], learned_id)) {
          proof.emplace_back();
          return 20;
        }
        if (stats.conflicts >= next_reduction) {
          reduce_database();
          reduction_span = std::min<std::uint64_t>(
              reduction_span * 3 / 2, 100000);
          next_reduction += reduction_span;
        }
        if (stats.conflicts >= next_restart && decision_level() > 0) {
          backtrack(0);
          ++stats.restarts;
          restart_span = std::min<std::uint64_t>(restart_span * 3 / 2, 100000);
          next_restart += restart_span;
        }
        continue;
      }

      const int var = choose_variable();
      // Every variable occurring in an input or learned clause is represented in
      // the heap.  If none remains unassigned and propagation found no conflict,
      // the complete relevant assignment satisfies the formula.  Variables that
      // never occur are left false in the serialized model.
      if (!var) return 10;
      trail_limits.push_back(static_cast<int>(trail.size()));
      ++stats.decisions;
      const int sign = phases[static_cast<std::size_t>(var)] >= 0 ? 1 : -1;
      if (!enqueue(sign * var, -1)) throw std::logic_error("decision assignment failed");
    }
  }
};

void append_u32(std::vector<std::uint8_t> &out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }
}

void append_i32(std::vector<std::uint8_t> &out, std::int32_t value) {
  append_u32(out, static_cast<std::uint32_t>(value));
}

void append_u64(std::vector<std::uint8_t> &out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }
}

std::vector<std::uint8_t> encode_result(const Solver &solver, int status) {
  std::vector<std::uint8_t> out;
  out.reserve(static_cast<std::size_t>(64 + solver.nvars));
  out.insert(out.end(), {'S', 'G', 'C', '1'});
  append_u32(out, static_cast<std::uint32_t>(status));
  append_u32(out, static_cast<std::uint32_t>(solver.nvars));
  append_u64(out, solver.stats.decisions);
  append_u64(out, solver.stats.propagations);
  append_u64(out, solver.stats.conflicts);
  append_u64(out, solver.stats.restarts);
  append_u64(out, solver.stats.learned_clauses);
  append_u64(out, solver.stats.learned_literals);
  append_u64(out, solver.stats.reductions);
  append_u64(out, solver.stats.deleted_clauses);
  if (status == 10) {
    for (int var = 1; var <= solver.nvars; ++var) {
      out.push_back(solver.values[static_cast<std::size_t>(var)] > 0 ? 1 : 0);
    }
  } else if (status == 20) {
    append_u32(out, static_cast<std::uint32_t>(solver.proof.size()));
    for (const auto &clause : solver.proof) {
      append_u32(out, static_cast<std::uint32_t>(clause.size()));
      for (const int literal : clause) append_i32(out, literal);
    }
  }
  return out;
}

}  // namespace

SPECTRA_EXPORT const char *spectra_g_native_signature() {
  return "SPECTRA-G native CDCL v1";
}

SPECTRA_EXPORT const char *spectra_g_native_error() { return last_error.c_str(); }

SPECTRA_EXPORT void spectra_g_native_free(void *pointer) { std::free(pointer); }

SPECTRA_EXPORT int spectra_g_native_solve(
    std::uint32_t nvars, const std::int32_t *literals, std::size_t literal_count,
    const std::uint64_t *offsets, std::size_t clause_count,
    std::uint64_t max_conflicts, std::uint64_t max_learned_literals,
    void **output, std::size_t *output_size) {
  last_error.clear();
  if (!output || !output_size) return 2;
  *output = nullptr;
  *output_size = 0;
  try {
    if (nvars > max_native_variables) {
      throw std::invalid_argument("too many variables");
    }
    if (!offsets || (literal_count && !literals)) {
      throw std::invalid_argument("null CNF buffer");
    }
    if (offsets[0] != 0 || offsets[clause_count] != literal_count) {
      throw std::invalid_argument("clause offsets differ from literal buffer");
    }
    Solver solver(static_cast<int>(nvars));
    for (std::size_t clause = 0; clause < clause_count; ++clause) {
      const std::uint64_t begin = offsets[clause];
      const std::uint64_t end = offsets[clause + 1];
      if (begin > end || end > literal_count) {
        throw std::invalid_argument("clause offsets are not monotonic");
      }
      solver.add_initial(literals + begin, literals + end);
    }
    solver.initialize_heap();
    const int status = solver.solve(max_conflicts, max_learned_literals);
    std::vector<std::uint8_t> encoded = encode_result(solver, status);
    void *buffer = std::malloc(encoded.size());
    if (!buffer) throw std::bad_alloc();
    std::memcpy(buffer, encoded.data(), encoded.size());
    *output = buffer;
    *output_size = encoded.size();
    return 0;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 1;
  } catch (...) {
    last_error = "unknown native exception";
    return 1;
  }
}
