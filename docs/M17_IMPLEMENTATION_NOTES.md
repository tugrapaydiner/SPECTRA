# M17 implementation details, fixed before execution

The numerical recipes, primary comparison, family order and gates in M17_PROTOCOL.md
are unchanged. These details resolve implementation choices before M17 data or
results are generated. They are not additional favorable operating points.

* Source storage: the original accepted ZIP remains pinned. Its durable canonical
  TAR may omit the separately retained historical input ZIPs. In that route the
  exact original hash-inventory bytes must hash to
  `5145184804ac7eecfc4eaa142d7584de9d7d54d184a39231429cc3648047611c`, and every listed
  member must match. The missing inputs are checked against the original M14/M15
  ZIP identities. No model is reconstructed or retrained by this adaptation.
* Sudoku development is run first, then maze development. A family's confirmation
  is generated at most once, only after its declared development gate and freeze.
  All earlier M17 manifests join the content/group exclusion index. The already
  inspected M14/M15/M16 evidence is never called new confirmation.
* Maze decode is argmax over all five logits, first maximum on a tie. Only a PATH
  prediction at an OPEN input cell creates a PATH; immutable input tokens are
  restored. Comparing just OPEN and PATH logits is a different decoder and is
  not used. The existing generator's min_path_len=8 and max_path_len=null remain.
* Maze core initialization and minibatches use the core seed; value initialization
  and minibatches use core_seed+17000 as specified. Targets share the same frozen
  input states, candidate ordering and minibatches. Training uses TRM.forward's
  existing per-supervision-step detached-state graph, not full-depth BPTT.
* The new owning native MazeProblem has parity tests against candidate_success.
  It computes shortest-path distance at construction, so **construction and its
  BFS work are charged inside each solve**, including learned methods. Calling
  this free preprocessing would be invalid. Native BFS with path reconstruction
  is an additional strong classical context; the prescribed Python BFS is also
  retained. Neither is a replacement primary comparator.
* Closed-loop candidates are checked as they are created. Exact first-valid stops
  are not learned halting. For quality/terminal and uniform checked search, the
  four-step identity prefix consumes the same total 24-transition budget.
* In addition to all raw timing rows, summaries use per-core/per-example medians
  over the three rounds. These are not three independent correctness trials.
  All returned answers are stored and independently checked outside timing.
* Prior accepted M16 data can be replayed on another CPU host as an implementation
  replication. That does not reopen candidate selection, erase original timing,
  or turn the old confirmation data into fresh held-out data.
* In the native search adapter, each decode returns the answer and its exact
  native-check result together. The generic search controller consumes that
  already-computed boolean, rather than charging a duplicate check. Every such
  answer is checked again independently outside timing as a test assertion.
* A unit-test fixture revealed that the existing perfect-maze generator accepted
  3x3 geometry despite coincident endpoints. It now rejects that geometry before
  generation; the native checker still supports a general 3x3 grid with distinct
  endpoints. This does not change any 11x11 M17 instance or selection criterion.
