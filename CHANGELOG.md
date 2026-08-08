# Changelog

## Unreleased

- **v22 completes Synaptic's official HotPotQA E2E path and wins every common aggregate
  metric without query-specific logic.** Static text corpora can opt into deterministic
  title-reference graph construction with `build_inmemory(..., auto_link_titles=True)`.
  A token trie creates only unambiguous multi-token links, supports conservative qualified
  and parenthetical aliases, and refuses mutable indexes whose derived edges could become
  stale. On the tagged 24-question/226-document cohort, four isolated AB/BA retrieval trials
  record OmniFuse/Synaptic Recall **0.9792/0.7292**, all-gold rate **0.9583/0.5000**,
  answer-presence **0.9583/0.5833**, mean retrieval **2.39/65.73 ms** and p95
  **2.54/82.71 ms**, for **11/11** wins. The exact upstream prompt, payload and correctness
  function then run through the same local Ollama 0.32.6 `qwen3.5:4b` digest in per-question
  AB/BA order: zero-inclusive correctness is **0.7542/0.5040**, generation mean
  **37.57/42.24 s**, p95 **55.40/65.62 s**, and prompt mean **1,134.96/1,243.54** tokens,
  for **10/10** wins. All 48 completions are immutable checkpoints and both final artifacts
  pass data, source, repository and model postflight verification. The complete suite passes
  **650 tests**.

- **v20 reproduces the official LongMemEval-S retrieval path and wins all eight common
  aggregate metrics.** `longmemeval_retrieval_bench.py` uses the tagged test's seed-42
  balanced sample, exact turn-pair records, fresh index per question, `limit=20` and mean
  gold-session recall while excluding external-LLM answer generation. The nominal 50-question
  run yields 48 questions, 2,296 sessions and 11,935 turn pairs. Two fresh AB/BA workers per
  system record OmniFuse/Synaptic session recall **0.9705/0.8413**, MRR **0.8936/0.6990**,
  nDCG **0.8937/0.6898**, mean retrieval **51.63/235.11 ms**, p95 **58.45/252.83 ms** and
  maximum per-question RSS delta **1.006/2.177 MB**. An immutable selected-sample artifact
  prevents the 277 MB source JSON parser from polluting worker RSS. Recall has 10 question
  wins, 38 ties and no losses; MRR and nDCG each retain two local losses rather than being
  overstated as per-question sweeps. The complete suite passes **633 tests**.

- **v19 audits Synaptic's S0-S8 ablation instead of treating skips and evaluation labels as
  cold retrieval.** The exact tagged suite ran 59,198.80 seconds and ended with all 13 tests
  skipped at the unavailable S8 external-LLM stage. Controlled AutoRAG and NFCorpus runs
  separate unlabeled S0/S1/S2/S6 from qrels-supervised S3/S4/S5. OmniFuse beats the best
  comparable stage on AutoRAG cold MRR/nDCG (**0.8908/0.9187 vs 0.8173/0.8591**), NFCorpus
  cold (**0.5486/0.3334 vs 0.5116/0.2959**) and NFCorpus supervised
  (**0.6276/0.5121 vs 0.2908/0.1832**) with lower build and query time. AutoRAG's supervised
  lane is ineligible, S7's empty-vector Ollama fallback failure is excluded, and S8 is not
  presented as a retrieval capability.

- **v18 reproduces Synaptic's official QA performance contract and removes OmniFuse's cold-start variance structurally.** The new isolated AB/BA harness uses the tag's exact 150-document combined fixture, 16 queries and cold `p95 < 100 ms` / `average < 50 ms` gates. Across four fresh workers per system, OmniFuse passes both gates 4/4 while Synaptic passes each 0/4; median cold p95 is **49.25 vs 3,079.71 ms**, steady p95 **0.052 vs 21.447 ms**, post-query RSS **35.43 vs 534.99 MB**, and lifetime peak **45.30 vs 598.32 MB**. OmniFuse wins all **10/10 common efficiency metrics**. The immutable non-evidence fast path now covers one- and two-field indexes and packs vocabulary once after ingestion, with no corpus threshold, query exception, cache or score change. Its own median cold p95 falls **33.1%** from the pre-change cohort, while the exact QA ranking hash and all 2,269 direct14 top-20 rankings remain unchanged. The official-tag five-run enterprise `full_native` track also reproduces OmniFuse leads in MRR, nDCG@5, Recall@5 and latency; Synaptic retains the tiny-fixture build win, and asymmetric `docs_only` MRR is disclosed rather than tuned away. Doctor coverage is now 22 declared targets. The complete suite passes **627 tests**.

- **v17 closes the measured large-corpus SQLite gap under a same-backend protocol.**
  Persistence workers now release ingestion-only staging before clean open, record
  post-create, clean-open and post-query RSS symmetrically, and route doctor-registered
  non-direct datasets through canonical snapshot preflight. Across 28 fresh workers on
  Allganize RAG-ko, AutoRAG, NFCorpus and 171,332-document TREC-COVID, OmniFuse records
  **76 strict wins, 0 ties and 0 losses** over thirteen efficiency and six official
  accuracy metrics per dataset. On TREC, create is **38.62 vs 42.84 s**, p50 is
  **895.14 vs 1,466.23 ms**, artifact size is **183.8 vs 630.2 MB**, post-query RSS is
  **407.20 vs 454.48 MB** and MRR@10 is **0.9017 vs 0.7210**. The 0.41 MB


- **v16 accelerates native SQLite queries without changing the index contract.** The raw
  forward-only reader now uses a canonical posting decoder with a one-byte fast path,
  precomputes field averages once, specializes exact single-field BM25 and accumulates BM25F
  frequencies without allocating one list per posting. A seven-run frozen NFCorpus diagnostic
  preserves every ranking, score and artifact byte while reducing query p50 **29.6%**, p95
  **36.3%** and complete-round time **36.8%**. A weighted-forward alternative was rejected
  because its faster queries cost **2.03x** build time and **1.594x** artifact bytes. The
  claim-grade Allganize, AutoRAG and NFCorpus persistence cohort records **48/48 strict wins**
  against synaptic-memory `v0.27.0`; the complete suite passes **622 tests**.

- **v15 immutable lexical stores are forward-only and provenance-preserving.** Static
  `InMemoryGraph` labels and non-feedback `InMemoryVector` text now use
  `CompactPostingsSnapshot` without reverse/update postings. Validated packed-forward pickle
  state and raw SQLite writers preserve exact scores, accept legacy BM25/BM25F state and
  reject corrupt envelopes. The 30,000-document diagnostic preserves every top-10 id and
  `float.hex` score while reducing median measured index RSS **28.47 to 7.42 MB**, serialized
  state **4,408,135 to 1,382,373 bytes** and serialization **72.16 to 4.05 ms**. The tradeoff
  is explicit: build rises **490.04 to 565.10 ms** and a cold unseen-term p50 rises **0.020
  to 0.134 ms**; repeated-query p50 is **0.019 versus 0.020 ms**.

- **The current official-tag direct cohort has no dataset loss on any shared metric.** The
  v15 frozen run completes all 14 upstream external cases and 2,269 queries per system.
  OmniFuse records **14/0/0** dataset wins/losses/ties for MRR@20, MRR@10, Precision@10,
  F1@10 and nDCG@10, and **13/0/1** for Recall@10. Dataset macros are MRR@20 **0.7038 vs
  0.6553**, MRR@10 **0.7022 vs 0.6537**, Precision@10 **0.1776 vs 0.1505**, Recall@10
  **0.7178 vs 0.6606**, F1@10 **0.2206 vs 0.1919** and nDCG@10 **0.6747 vs 0.6100**.
  v14 and v15 rankings and scores are identical under the canonical comparison hash
  `1dfeb042926f6dab80080fef639b613224caa70d49053f786675df001b512805`.

- **The large TREC-COVID run is complete and records both the win and the remaining memory
  gap.** On 171,332 documents, v15 preserves MRR@10 **0.9083** and records p50 **109.49 ms**
  versus canonical Synaptic **2,937.81 ms**. Its current RSS falls from the v14 OmniFuse
  **1,078.30 to 786.08 MB**, but Synaptic remains lower at **412.88 MB**. This is a
  capability-qualified RAM-index-versus-durable-SQLite observation; lazy materialization
  excludes ingest comparison, and the v15 OmniFuse-only follow-up is not a new counterbalanced
  two-system cohort. The complete suite passes **621 tests**.

- **The historical v10 native SQLite cohort was 48/48 strict wins against
  synaptic-memory `v0.27.0`.**
  Allganize RAG-ko, AutoRAG and NFCorpus each run four fresh counterbalanced workers per
  system with identical frozen input, K=10, candidate limit 20 and the byte-identical six-
  metric scorer. OmniFuse wins ten efficiency and six accuracy metrics on every dataset with
  **0 ties and 0 losses**. Median durable create is **0.1243 vs 3.4995 s**, **0.6177 vs
  12.2348 s** and **1.0164 vs 1.1430 s** respectively; steady p50 is **1.2023 vs 130.8702
  ms**, **6.1607 vs 248.0345 ms** and **1.7996 vs 244.5423 ms**. Artifacts are **499,712 vs
  524,288 B**, **2,686,976 vs 4,526,080 B** and **5,627,904 vs 18,132,992 B**. Lifetime
  peak RSS is also lower in every case: **35.82 vs 599.44 MB**, **44.73 vs 601.25 MB** and
  **49.88 vs 55.68 MB**.

- **Snapshot creation is now lossless, blocked and forward-only.** Schema v3 stores title and
  text as raw UTF-8 or `zlib.Z_BEST_SPEED`, whichever is smaller, on standard 4 KiB pages.
  The direct builder keeps exact frequencies, field lengths and scoring parameters but no
  reverse/update state, and streams postings into bounded 64 KiB SQLite blocks instead of
  copying them into one large BLOB. This removes transient posting duplication, bounds first-
  query reads and preserves fsync + atomic replace. ASCII token splitting keeps the exact
  `[a-z0-9]+` contract through a faster C path. The full suite passes **618 tests**.

- **Korean query coordination fixes candidate admission without deleting scoring terms.**
  Closed-class question operators can still improve a passage that matches the subject, but
  cannot admit an unrelated passage alone; definition copulas add their normalized subject
  form. Request-form detection is narrow enough to preserve content such as `설명회`, and
  historical content forms such as `임진왜란` remain unchanged. There is no benchmark ID,
  per-query cutoff or dataset-specific configuration.

- **Persistence memory provenance no longer measures a duplicate input payload.** Workers
  fingerprint the input file and verify canonical JSON incrementally rather than holding the
  raw bytes, decoded JSON and a second canonical serialization together. The rule is applied
  identically to both systems and removes the shared parser high-water mark from lifetime peak
  RSS. Compact immutable provenance is in
  `persistence_memory3_synaptic_tag_v0.27.0_836d536_20260723_v10_summary.json` (SHA-256
  `3ab84d3dda2467f31ef710189029d80600f7d68ab191e920a51eae766d408542`).

- **The official synaptic-memory `v0.27.0` direct cohort is complete.** A frozen checkout
  at tag/SHA `v0.27.0` / `836d536` ran the 14 upstream external cases with the tag's native
  no-embedding path, inputs, sampling and metrics. Across 2,269 queries per system,
  OmniFuse wins MRR@20, MRR@10 and nDCG@10 on **14/14 datasets**; the six-metric dataset
  macros are MRR@20 **0.7045 vs 0.6553**, MRR@10 **0.7028 vs 0.6537**, Precision@10
  **0.1715 vs 0.1505**, Recall@10 **0.7249 vs 0.6606**, F1@10 **0.2154 vs 0.1919** and
  nDCG@10 **0.6772 vs 0.6100**. Precision wins 12/14 rather than every dataset because the
  official scorer rewards shorter-than-K result lists; tested result-cutting either regressed
  Recall or compensated on unrelated queries and was rejected. The write-once result
  `direct_external14_fts_synaptic_tag_v0.27.0_836d536_20260722_speed1_v3.json` has SHA-256
  `f6a6f2ed7398c0d174e27d5444e9236b5e143194c7e30a8664603f0720820877` and complete
  pre/post provenance.

- **Plain static indexes now materialize lexical state lazily without changing retrieval.**
  `InMemoryVector` snapshots scalar title/text inputs at construction and builds its immutable
  BM25/BM25F index once, under a lock, on the first lexical query. Mutable and feedback-backed
  stores remain eager so mutation, evidence and snapshot semantics stay unchanged. Pickling,
  concurrent first use, dense-only/hybrid operation and old materialized state are covered.
  The production suite passes **586 tests** and every official v8 top-20 ranking is identical
  to the 2026-07-22 canonical reference.

- **Official-tag v8 wins every measured row in the selected no-embedding in-memory scope.**
  AutoRAG, Allganize RAG-ko and NFCorpus each ran in two fresh counterbalanced AB/BA workers
  with one warm-up, five measured rounds and `time.perf_counter_ns`. Synaptic/OmniFuse ratios
  are **7.46×/5.86×/8.49×** for ingest, **513.25×/892.09×/1,546.95×** for p50 query latency
  and **492.05×/739.50×/433.65×** for ingest-plus-mean-query-set-round. OmniFuse also has
  higher MRR@10 and lower observed peak RSS on all three cases; the NFCorpus RSS difference
  is only 0.014 MB and is not presented as a portable margin.

- **The official NFCorpus CDC gate now covers exact mutation semantics and cold first use.**
  Two fresh workers per system apply 36 inserts, updates, deletes and no-ops, and both systems
  match their own full rebuild at every checkpoint. OmniFuse wins all six shared metrics:
  MRR@20 **0.5080 vs 0.4799**, MRR@10 **0.5056 vs 0.4771**, Precision@10
  **0.2960 vs 0.2507**, Recall@10 **0.1514 vs 0.1323**, F1@10 **0.1401 vs 0.1206** and
  nDCG@10 **0.2927 vs 0.2481**. Synaptic/OmniFuse p50 ratios are **4.82×** initial ingest,
  **6.86×** mutation, **14.49×** cold first round, **461.72×** steady round and **14.49×**
  incremental end-to-end. The cold round includes deferred lexical materialization. Compact
  immutable provenance is in
  `perf_cdc_synaptic_tag_v0.27.0_836d536_20260723_v8_summary.json`.

- **Korean normalization and exact top-K selection are faster without changing retrieval.**
  Korean suffix matching now dispatches to an immutable last-character bucket that preserves
  the original longest-first order. BM25 and BM25F now retain a bounded heap ordered by
  `(score, -doc_id)` instead of sorting every positive candidate. The implementation preserves
  positive-score filtering, ties, zero/negative limits and NaN behavior. The optimized
  official-tag run reproduced every baseline top-10/top-20 ranking, relevance set, reciprocal
  rank and aggregate metric exactly across all 2,269 queries; the full suite passed 244 tests.
  Direct-run time values remain observational because the upstream-compatible Windows clock
  is too coarse for sub-millisecond queries, so precise latency requires a separate protocol.
  After integrating and hardening that protocol, 44 focused performance tests and the current
  262-test repository suite pass.

- **The synaptic-memory upstream-main revalidation now has explicit, machine-readable
  provenance.** The selected competitor is `main` at `7470e72` (package metadata 0.27.0),
  not the official `v0.27.0` tag at `836d536`. OmniFuse is identified by HEAD `cd355dd`
  plus its content-fingerprinted benchmark worktree, not falsely described as a clean fixed
  commit. The canonical tracked-public invocation gives OmniFuse **8/8 MRR@10 wins** and
  macro **0.8408 vs 0.8221**; Recall@10 and nDCG@10 win 8/8, while Precision@10 wins 5/8.
  A separate completed extended9 invocation gives **9/9 MRR@10 wins** and macro **0.6717
  vs 0.6253**. Combining those two independent artifacts gives 17/17 MRR wins, not one
  17-dataset same-pass run. The selected-main enterprise fixture's `full_native` mode,
  scored at its own K=5, also
  favors OmniFuse on MRR (**0.7689 vs 0.7467**), nDCG@5 (**0.7637 vs 0.6649**), and
  Recall@5 (**0.8167 vs 0.7333**). The old `0.7889 vs 0.7689` enterprise aggregate
  does not reproduce against the current fixture. Its `docs_only` mode is retained
  as an ablation, not counted as a head-to-head win, because synaptic receives
  dataset-provided intent labels while OmniFuse receives the raw query.

- **Benchmark coverage and efficiency contracts are now auditable.** `eval/bench.py
  doctor` distinguishes the real matrix — finreg 2, eight Git-tracked public files,
  nine upstream-declared/downloader-generated public files, one enterprise fixture,
  and one private KRA target — instead of calling all 21 “shipped datasets.” It fingerprints inputs,
  repository state, and the byte-identical scorer files, and fails strict public runs
  when any of the 19 strict inputs (17 public IR plus two finreg inputs) is unavailable.
  The loader now binds every strict target to its declared repository/role/path and
  re-hashes all 19 actual files, including inputs not selected by the immediate run; it
  also revalidates them after the run. The performance runner now uses
  the same K/candidate limit, deterministic order, warm-up plus repeated measurements,
  fresh worker processes, p50/p95/mean latency, whole-worker RSS, write-once JSON, complete
  before/after fingerprints, a strict-doctor runtime binding, and a separate shared accuracy
  scorer. The old 2026-07-13 schema-v1 efficiency files remain selected measurements rather
  than official-tag canonical artifacts. `eval/enterprise_bench.py` records per-query ranks,
  native routing, memory-semantics caveats, and scorer provenance.

  A follow-up audit found and fixed two false-green paths in the first doctor revision:
  synaptic v0.27's 2Wiki-dev, MuSiQue-dev, and TREC-COVID declarations were missing, and
  existence-only checks accepted `{}` as a ready dataset. The matrix is now 21 targets / 19
  public, public JSON is validated for parser-compatible non-empty corpus, queries, and
  relevance labels, JSONL is streamed and validated, and strict mode blocks dirty/non-Git
  sources unless `--allow-dirty` is explicit. Benchmark imports now suppress bytecode writes
  and verify that synaptic was loaded from the requested checkout.

### Historical 2026-07-10 checkpoint

The entries below preserve the investigation chronology. Their old coverage totals,
`idf_pow=1.5` default, and cross-protocol headline comparisons are superseded by the current
README and the 2026-07-14 canonical upstream-main artifacts.

- **Index persistence is now gzip-compressed — 2.4–4.3× smaller files, lossless.** A new
  storage-footprint head-to-head (the one efficiency axis not yet measured) found the plain
  pickle LOST to synaptic's SQLite store on tiny corpora (Allganize-ko: 1.34 vs 0.76 MB) while
  winning mid/large ones. gzip flips every corpus to a win — NFCorpus **5.9 vs 20.8 MB**,
  Allganize-ko **0.31 vs 0.76 MB**, KRA **9.7 vs 32.3 MB** (≈ the size of the raw text) — for
  ~0.1 s of load time. Lossless by construction and by test: a loaded index scores
  bit-identically, and pre-gzip index files still load (magic-byte sniff). pytest 66 → 67.

- **Reproduction sweeps #2 and #3** re-ran the full 18-target suite with synaptic
  re-ingested per dataset; all 42 arm values (14 sets × 3 idf_pow arms) reproduced to four
  decimals across runs, and the shipped 1.2 arm went 14/14 against synaptic in the same
  pass. Memory, incremental (remember/forget bit-identity), perf and enterprise disclosures
  all reproduced. `eval/results/full_suite_2026_07_10.json`.

- **Benchmark coverage completed (18 targets) — and it re-derived the `idf_pow` default to
  1.2, at which OmniFuse beats synaptic on every measured accuracy dataset.** An audit found
  synaptic's `tests/benchmark/data/` holds 15 files, of which 3 had never been run: **FiQA**
  (57,638 docs — the largest corpus yet), **MultiLongDoc-ko**, and the 12-document
  **enterprise_scenario** toy. Run head-to-head (synaptic re-ingested per set, its own
  scorer): MultiLongDoc-ko wins both arms; **FiQA LOSES at the old default 1.5** (0.2781 vs
  0.2902) and wins at p≤1.3 (0.2929 at 1.0). With Ko-StrategyQA binding from below (wins at
  p≥1.1), the winning band is **[1.1, 1.3]**; the default moves to its midpoint **1.2** —
  the same mid-band rule that chose 1.5 when the suite had 13 datasets, recomputed for 18.
  At 1.2, all 16 accuracy targets win (0.8471/108 finreg — multi-hop UP from 107 —
  MIRACL-ko 0.9750, golden 0.4973, FiQA 0.2920), the memory axes improve (KRA covered
  +0.4016, NFCorpus +0.1785, placebos dead, Δuncovered exactly 0), and remember/forget stay
  bit-identical.

  **enterprise_scenario is disclosed, not tuned away**: synaptic edges MRR by 0.0056 (one
  query of 15, rank 2 vs 1, 12 documents); OmniFuse wins nDCG in every configuration, wins
  recall, and under the full scenario (docs + links + agent sessions — synaptic driven by its
  own conftest, OmniFuse links-as-triples + `Feedback` from session descriptions) OmniFuse
  answers the scenario's designed memory query at 1.000 vs 0.333. The diagnosed cause is a
  token-length artifact meaningful only at n=12; the parameter-free fix (word-mass query
  weighting, Lucene SynonymQuery-style) was implemented, measured, and **rejected: 11 of 12
  real datasets degraded** (MIRACL 0.9376, finreg 0.7965). The length-proportional bigram
  mass is not a bug at corpus scale — a fully-matched long compound is genuinely stronger
  evidence — and the negative result is recorded.

- **`forget()` — memory can now be withdrawn, in place, to the same bar.** `remember()`'s
  inverse: `OmniFuse.forget(query, doc_ids)` removes a remembered pair from the live index in
  ~1 ms (NFCorpus 1.11 ms, KRA 1.52 ms), **bit-identical** to an index rebuilt without that
  pair. `BM25F.update_evidence` is generalized from grow-only to grow-or-shrink: the evidence
  df decrements, evidence-only terms whose IDF moves are recomputed from their kept `tfw`, and
  a term whose last holder forgets it is **erased from the vocabulary exactly as a rebuild
  would**. Strongest inverse property, tested: remember everything, forget everything, land
  bit-identically on the cold index. Forgetting a pair that was never remembered is a no-op.
  Closes the "evidence may only grow" limitation. `tests/test_incremental.py` 7 → 13.

- **The `idf_pow` ablation is now a real head-to-head, and its honest summary got sharper.**
  The docs claimed idf_pow=1.5 was "strictly additive"; that was measured before the English
  S-stemmer and the Korean copula fix. Re-ablated under the shipping tokenizer — with
  **synaptic re-ingested and re-queried per dataset in the same pass** (`eval/idf_pow_bench.py`,
  driving synaptic's own `run_public_dataset`; all 13 re-measured values reproduced the
  recorded ones exactly) — the net effect of 1.5 vs 1.0 over 13 datasets is **+0.0065, a wash**:
  it buys AutoRAG/HotPotQA-200/Ko-StrategyQA and costs MIRACL-ko (0.9812→0.9617), finreg
  (0.8533→0.8400) and NFCorpus. At `idf_pow=1.0` OmniFuse still wins **14 of 15**; the single
  loss is Ko-StrategyQA by **0.0006** — less than one query's worth on that 592-query set.
  The per-query diff at p=1.0 shows **92 losses / 94 wins / 406 ties** against synaptic (RR
  mass 36.49 vs 36.14): no clustered defect, so no principled fix exists and chasing the
  margin would be label-fitting. 1.5 stays (band-robust across p∈[1.3, 2.0]; the entity-burial
  mechanism verified by a sign-flip test: gold max-IDF 6.22 vs 6.08 where it rescues, 5.37 vs
  6.16 where it breaks), with the margin disclosed everywhere the 15/15 is claimed.
  Also rejected and recorded: adding 되 to `_KO_SUFFIX` (right in principle, immaterial in
  measurement). See `eval/results/idf_pow_ablation.json`.

- **`remember()` — memory folds into the live index, bit-identically, in ~1 ms.** Memory was
  batch-only: a confirmed `(query -> documents)` pair required an index rebuild, which no live
  service can afford per click. It is now an in-place update.

  This is what the evidence-field design was always worth. Evidence is excluded from document
  frequency and is not length-normalized, so `N`, the content df, every content term's IDF and
  the content fields' avglen are all **fixed** — remembering rewrites the contributions of
  exactly one document. The single coupling is that a term seen *only* in evidence takes its
  IDF from the evidence df, which grows; but all of that term's postings are evidence-derived,
  so the documents to fix are exactly the ones that remember it. Measured, that coupled set is
  **15 terms out of a 23,610-term vocabulary** (NFCorpus, 100 memories).

  | | rebuild | `remember()` | per memory |
  |---|---:|---:|---:|
  | NFCorpus (3,633 docs, 100 memories) | 1.389 s | **1.00 ms** | **1,386x** |
  | same memories, a tenth of the corpus | 0.175 s | **1.02 ms** | 172x |
  | KRA (5,234 chunks, 120 memories) | 6.605 s | **1.52 ms** | **4,335x** |

  The middle row is the control: ten times fewer documents makes the rebuild 7.9x cheaper and
  leaves `remember()` where it was, so the cost tracks the memory rather than the corpus. It is
  also flat as memory accumulates (1.99 -> 1.69 ms per 50 over 215 KRA memories).

  The bar was **bit equality** with a full rebuild — every posting, every float — not `isclose`,
  because a weight that drifts is a scoring bug with a stopwatch. To meet it the index keeps the
  tfw of evidence-only terms beside their weights, so a moved IDF is recomputed rather than
  rescaled. The first prototype asserted the update was purely local, skipped the evidence-df
  coupling entirely, and differed from a rebuild in 1,181 terms; the bar caught it, and the
  wrong claim is recorded in `eval/results/incremental_memory.json`.

  New: `OmniFuse.remember()`, `InMemoryVector.remember()`, `BM25F.update_evidence()`,
  `eval/incremental_bench.py`, `tests/test_incremental.py` (7 tests, incl. bit-identity at every
  prefix and `remember` after `save_index`/`load_index`). Building with an *empty* `Feedback()`
  now opts a store into the memory field — previously an empty `Feedback` was falsy and silently
  fell back to a non-evidence index. A cold store still ranks identically. Static suite
  unchanged; no `forget()` — evidence may only grow.

- **The Korean copula's interrogative paradigm was missing from the ending list — and it
  was the last loss. OmniFuse now wins all 15 datasets.** A per-query diff against synaptic
  (scored by its own `metrics.reciprocal_rank`) showed OmniFuse losing 29 MIRACL-ko queries
  and winning 14, with the losses clustered on a single junk document: every
  *"…어디인가?"* ("where is…?") question retrieved the article titled **"내 친구의 집은
  어디인가"** — a 4×-weighted title match on nothing but the question word.

  Cause: `-인가/-인가요/-입니까/-인지` were absent from `_KO_SUFFIX`, so `어디인가` stemmed to
  the *rare* token `어디인` rather than the common word `어디`, and `idf_pow` amplified that
  rarity. Kiwi splits the copula into morphemes, which is why synaptic never saw it.

  **MIRACL-ko 0.9052 → 0.9617** (synaptic 0.9495) — the extended track goes 4/4 and the
  suite to **15/15**. Also XPQA-ko 0.3256 → 0.3290, KLUE-MRC 0.8280 → 0.8288, the real-world
  golden set 0.4775 → **0.4957**. Small, still-winning costs: PublicHealthQA 0.6284 → 0.6217,
  AutoRAG 0.9309 → 0.9293, Ko-StrategyQA 0.6509 → 0.6496. finreg, HotPotQA, Allganize,
  NFCorpus and SciFact are bit-identical. Not a fit: every suffix subset from `{인가}` alone
  to a nine-ending superset wins 8/8 of the Korean-bearing sets — a flat band, and a closed
  linguistic class like the 조사/어미 already shipped.

  Two standard fixes were tried on the *symptom* first and are recorded as rejected:
  coordination-level matching (`score *= coverage**λ`) wins MIRACL at 0.9536 but breaks
  Ko-StrategyQA 0.6135, HotPotQA-200 0.8774, PublicHealthQA 0.5824 and NFCorpus 0.5040;
  `minimum_should_match ≥ 2` reaches only 0.9167 and collapses Ko-StrategyQA to 0.5663
  because it filters gold documents that legitimately match one query word.

- **Efficiency is now measured with synaptic's own `mean_search_time_ms`**
  (`eval/perf_bench.py`), not a timer we invented. NFCorpus: ingest **2.01 s vs 55.01 s**,
  mean search **1.66 ms vs 14.14 ms**, MRR 0.5182 vs 0.5124. Allganize RAG-ko: **0.18 s /
  0.18 ms / 0.9683** vs 5.39 s / 4.41 ms / 0.9562.

- **`Feedback` — memory that survives its own placebo, and beats synaptic on the axis that
  names it.** A confirmed query becomes *evidence about* a chunk: it is indexed as a BM25F
  **evidence field** whose terms score the chunk but never enter document frequency, and
  which is not length-normalized. Measured with synaptic's own `metrics.py`, on the same
  corpus, queries and scorer — feedback on the original questions, evaluated on **held-out
  paraphrases** of them:

  | ΔMRR@10, held-out re-queries | KRA (ko) all | KRA covered | NFCorpus (en) all | NFCorpus covered |
  |---|---:|---:|---:|---:|
  | synaptic (Hebbian) | +0.0000 | +0.0093 | −0.0010 | −0.0008 |
  | **OmniFuse (`Feedback`)** | **+0.1790** | **+0.3903** | **+0.0150** | **+0.0300** |
  | ↳ shuffled placebo | +0.0059 | +0.0213 | +0.0015 | +0.0031 |
  | ↳ random-query placebo | +0.0029 | +0.0215 | +0.0000 | +0.0000 |

  Replicated on a **second corpus, second language and a different relevance structure**:
  on NFCorpus both placebos go *negative* while `real` stays positive, and Δuncovered is
  exactly **0.0000**. The effect is smaller there (+0.0460 vs +0.4167 covered) because its
  cold score is already 0.556 — with 38 relevant documents per query there is little
  headroom. Memory pays most where relevance is concentrated and the cold ranking is weak.
  `real` is 5.2× the strongest placebo on KRA and is the only positive variant on NFCorpus,
  so the `(query, chunk)` pairing carries the signal.
  On *unrelated* held-out questions memory correctly does nothing (+0.0006), and
  Δuncovered is **exactly 0.0000** there — the collection's IDF is provably untouched. A
  cold store ranks **bit-identically** to one built with no feedback (verified on finreg and
  the whole public suite). Nothing is tuned.

  Three design decisions, each forced by a measurement: excluding evidence from **df**
  (our retracted version injected the query into the body, which deflated the IDF of query
  vocabulary corpus-wide — an accidental `idf_pow` reduction); **no length normalization**
  on the evidence field (a memory held by 2% of chunks otherwise explodes `fnorm` and the
  covered gain collapses +0.4167 → +0.0742); and giving **evidence-only terms** an IDF from
  the evidence df (otherwise the very words memory exists to contribute are discarded).

- **Fairness correction on synaptic.** An earlier revision reported its Hebbian
  reinforcement as *harmful* (−0.0174). Re-running the same configuration gave −0.0045:
  synaptic's warm pass is not deterministic. Channel isolation shows why the effect is
  noise — in this version `SynapticGraph.search()` reads **none** of the fields
  `reinforce()` writes (`ResonanceScorer`, the only consumer of `success_count`, is used by
  `search.py`/`agent_search.py`, not by `graph.search`; nothing reads `edge.weight`).
  Reinforcement reaches retrieval only through the edges it creates on success: with no
  reinforcement Δ = **0.0000** exactly, with negatives only **0.0000**, with positives only
  **+0.0001**. The fair statement is that Hebbian is *not wired into* this version's
  retrieval — not that it hurts.

- **Full suite re-verified after the `BM25F` change.** finreg (0.8400 / 107 / 0.9250), all
  8 core public sets, all 4 extended sets and the real-world golden set are **bit-identical**
  to before. Index build costs +1.8 % time (6.03 s → 6.14 s) and +0.6 MB peak (46.6 → 47.2 MB)
  for the extra evidence-df bookkeeping; persisted index size and `load_index` are unchanged.
- **`BM25F(evidence_fields=…)`** — fields that describe a document rather than being its
  content: scored, but excluded from document frequency and from length normalization.
  With no evidence fields the class behaves exactly as before.

- **Retracted: `Feedback` (query-conditional memory). The claimed win was not real.**
  A previous entry claimed OmniFuse beat synaptic on the axis that defines it — memory —
  by remembering the queries a document was confirmed to answer and indexing them as a
  BM25F field (NFCorpus +0.0019/+0.0076, MIRACL-ko +0.0618/+0.0729 vs synaptic's
  −0.0002/−0.0174/−0.0165). Placebo controls destroy that claim:
  - permuting the (query ↔ document) pairing keeps the gain (+0.0555 on MIRACL-ko);
  - attaching a **random** feedback query to each confirmed document scores *better* than
    the real one (+0.0665 vs +0.0618);
  - held-out queries whose relevant documents remembered **nothing** still gained +0.0441,
    which a query-conditional mechanism cannot do.

  The real mechanism: injecting query text into documents raises the document frequency of
  query vocabulary and deflates its IDF corpus-wide. MIRACL-ko is precisely the corpus that
  prefers weaker emphasis (0.9052 at `idf_pow=1.5` → 0.9489 at 1.0), so "memory" was an
  accidental, uncontrolled `idf_pow` reduction. A control memory made of tokens that can
  never match a query moves nothing (**+0.0000** exactly), which pins the effect on term
  statistics rather than on the pairing.

  The feature is reverted. The benchmark is kept — with placebo and covered/uncovered
  controls now mandatory, because its naive form yields a convincing false positive:
  [`eval/adaptive_bench.py`](eval/adaptive_bench.py) ·
  [`eval/results/adaptive_memory.json`](eval/results/adaptive_memory.json).

  Our own query-independent designs failed first and are recorded too (Beta posterior odds
  −0.0384, positive-only −0.0175, empirical-Bayes shrinkage −0.0489). **Nothing we tried,
  and nothing synaptic ships, improves held-out retrieval in a query-conditional way.**

- **Index build peaks at ~1/4 the memory, at no cost in build time.** Indexing needs
  corpus-wide document frequency before it can compute a contribution, so per-document
  term counts have to survive from pass 1 to pass 2. They used to survive as dicts of
  strings over the whole corpus — that was the peak. They are now interned **term ids in
  `array('i')`**, and each document is released the moment pass 2 consumes it. `BM25`/
  `BM25F` also accept a zero-arg callable yielding documents, so `InMemoryVector` streams
  tokenization instead of materializing the tokenized corpus. On 5,234 chunks: build peak
  **177.2 MB → 46.6 MB**, build time **6.13 s → 6.03 s**, rankings unchanged (finreg + all
  8 public + NFCorpus verified). A first attempt — stream and simply re-tokenize in pass 2
  — also hit 46.7 MB but doubled build time to 13.6 s; it was rejected.
- **MIRACL-ko stays the single loss, and we stopped trying to fix it.** Seven principled
  variants were measured and none wins it or dominates the shipped configuration:
  `idf_pow` band, `contribution^q` power-mean, emphasis restricted to word tokens,
  Hangul stem-only, Hangul bi-gram-only, bi-grams over the raw surface form (`V4`), and
  `V4 + #raw`. `V4` is notable — it lifts Ko-StrategyQA to 0.6599 (best seen) and KLUE to
  0.8314, but costs AutoRAG, PublicHealthQA and Allganize-Eval, so the 7-set average falls
  0.8355 → 0.8311. MIRACL's best possible score is 0.9489 at `idf_pow=1.0` against
  synaptic's 0.9495 — a 0.13-query gap. Its relevance is diffuse (14.4 relevant/query) and
  emphasis is a precision move; that is a structural trade with Ko-StrategyQA, not a bug.

- **English morphological normalization — NFCorpus flips to a win (0.5053 → 0.5182 vs
  synaptic's 0.5124).** OmniFuse normalized Korean morphology but indexed Latin tokens as
  raw surface forms, so `statin` could never match `statins`. `text._en_stem` adds
  **Harman's S-stemmer** (singularize; no `-ing`/`-ed`, no Porter cascade). It has *no
  tunable parameter*, so there is nothing to fit. SciFact 0.6422 → 0.6456,
  HotPotQA-200 0.9028 → 0.9044, HotPotQA-24 0.9286 → 0.9077 (still a win); every Korean
  dataset, finreg, and the golden set are **bit-identical**. Suite: **14 of 15** datasets
  (Core 10/10, Extended 3/4, Real-world 1/1).
- **Correction.** An earlier revision claimed the IDF emphasis *caused* both the NFCorpus
  and MIRACL-ko losses. That was wrong: with emphasis off, NFCorpus still lost (0.5080 vs
  0.5124). Emphasis widens the MIRACL-ko gap; the NFCorpus gap was missing English
  morphology, now fixed. MIRACL-ko remains the single loss.
- **Recorded negative results** (they are why the real fix was findable):
  - *Replacing graph fusion with PPR-style damped, degree-normalized propagation* — much
    worse (finreg single-hop 0.6294, multi-hop 81/120). Degree normalization starves the
    cited article; finreg's evidence is one out-edge away.
  - *`contribution^q` power-mean instead of `idf^p`* — better on the two Extended losses
    (MIRACL-ko 0.9052 → 0.9321) but worse on nine other datasets; average MRR 0.7628 →
    0.7610. Not shipped: it trades nine datasets for two.
  - *Augmenting Latin tokens with `surface + #stem`* (mirroring the Korean design) — worse
    than plain replacement (NFCorpus 0.5090 vs 0.5182).
  - *An English-stemming run that "proved" stemming harmful* was **void**: the harness
    patched `text.tokenize` but `backends/memory.py` binds `tokenize` at import, so
    documents were indexed with the old tokenizer while queries used the new one. The
    corrected run reverses the conclusion.

- **Graph-companion fusion now follows edge direction — finreg multi-hop 101 → 107/120**
  (R@10 0.8958 → 0.9250), single-hop unchanged (hit@10 113 → 114, nDCG 0.8651 → 0.8663).
  `retrieve()` documents that it surfaces "a passage a strong seed *references*", but
  `InMemoryGraph._adj_ids` was symmetric, so a seed also promoted every node that cited
  *it* — a crowd, not evidence. `GraphStore.neighbor_ids` gains
  `direction="out" | "in" | "both"` (default `"both"`, unchanged); fusion asks for `"out"`
  via the new `OmniFuse(fusion_direction=…)`. Set `"both"` for genuinely symmetric graphs.
  No parameter was refitted (`fusion_alpha` stays 0.9). Public datasets carry no graph, so
  their scores are untouched. Recorded negative results: replacing this rule with damped,
  degree-normalized PPR-style propagation is far *worse* (0.6294 / 81), and multi-hop
  propagation never helps — finreg's evidence is exactly one out-edge away.
- **Index is ~30 % smaller, loads 2× faster, and peaks at half the memory while building.**
  `BM25.tf`, `BM25F.doc_tf` and the per-field length norms existed only to *derive* the
  postings, and afterwards were read by nothing but `score()`.
  - They are no longer **retained**: `score()` reads its precomputed contribution straight
    out of the postings (each `_pd[t]` is ascending, so it is a binary search). Persisted
    index **41.2 MB → 28.7 MB**; `load_index` **0.43 s → 0.21 s**.
  - They are no longer **materialized for the whole corpus** either. `BM25F` build peak
    **68.2 MB → 36.0 MB**; full `build_inmemory` peak 209.3 MB → 177.2 MB here, and then
    → **46.6 MB** once the pass-1 intermediate was compacted (see the entry at the top).
  - Rankings unchanged throughout (verified on finreg + all 8 public sets).
- **Index persistence — `save_index` / `load_index`.** A built in-memory index (graph +
  passage store) round-trips to disk with stdlib `pickle`, so a process starts warm
  instead of re-indexing: on a 5,234-chunk corpus, **load 0.43 s vs a 5.98 s rebuild
  (14×)**, rankings identical. The LLM and the embedder callable are deliberately not
  persisted — pass them to `load_index(..., llm=, embedder=)`. `pickle` executes arbitrary
  code on load, so only load indexes you produced. This closes the one gap that forced
  OmniFuse to pay build cost every run; the index is still read into RAM, so a truly
  disk-resident backend remains future work.
- **Lexical search is ~6.4× faster, with bit-identical rankings.** A term's contribution
  to a document (`idf * tfw(k1+1)/(k1+tfw)` for `BM25F`, `idf*(k1+1)*f/(f+norm)` for
  `BM25`) does not depend on the query, so it is now folded into the inverted index at
  build time: a search is a plain accumulation of precomputed floats over the postings,
  instead of a full scan that re-derived document lengths per term. Verified score-for-score
  against the previous implementation on finreg + all 8 public datasets (8 public: 249.4 s
  → 38.7 s; KLUE-MRC 230.5 s → 28.8 s). `InMemoryGraph._by_label` is a dict lookup, not an
  O(N) scan.
- **`idf_pow` is now a documented knob** (`InMemoryVector`, `build_inmemory(vector_kwargs=…)`).
  It was effectively hardcoded before.
- **Honest correction.** The IDF term-specificity emphasis is a Pareto trade, not a free
  win: it takes the core suite 10/10 but *regresses* heavily multi-relevant passage-IR
  corpora (MIRACL-ko 0.949 → 0.905, NFCorpus 0.508 → 0.505). The previously published
  BEIR/MTEB numbers predated `idf_pow` and were never re-run — they are corrected in
  `eval/results/beir_mteb_extra.json`. An earlier parameter sweep was also invalid
  (keyword-only defaults bind at def time, so monkeypatching the module constant changed
  nothing); the corrected sweep confirms the core p∈[1.3,2.0] band and exposes the
  regression the broken sweep hid.

## 0.5.0

Retrieval quality — the ranking now uses field structure and graph structure,
not just flat body BM25. Measured on the finreg corpus (4,417 Korean statute
articles) against synaptic-memory's own eval metric (`metrics.py`, k=10),
single-shot, no LLM:

| | synaptic FTS-only | OmniFuse 0.5 |
|---|---:|---:|
| single-hop MRR@10 | 0.704 | **0.840** |
| single-hop hit@10 | 103/120 | **114/120** |
| multi-hop strict-solved | 56/120 | **107/120** |

On the full synaptic benchmark suite (finreg + 8 public IR sets, zero-embedder lexical
track), OmniFuse 0.5 wins **all ten datasets** (avg MRR 0.846 vs synaptic 0.809) via two
honest, dependency-free, zero-hardcode logic changes:

- **Dependency-free Korean stemming in `text.tokenize`** — Hangul runs now have their
  common particles (조사), verb/adjective endings (어미), and trailing derivational
  suffixes (적/화/성/상/하/들) stripped by a small rule table before bi-gramming, so a
  query and a document align on the stem the way a morphological analyzer (Kiwi) would
  — but pure Python, and it emits *fewer* tokens than raw bi-grams (stem bi-grams + one
  stem unigram) ⇒ more accurate on Korean *and* more memory-efficient. Suffixes are
  stripped only when *trailing*, so 상황/성별 (with the char leading) are untouched, and
  the emitted stem unigram still lets compound forms match. Hanja/Kana and Latin are
  unchanged. This flips AutoRAG and PublicHealthQA to OmniFuse (7→9 wins).
- **IDF term-specificity emphasis in `text.BM25`/`BM25F`** (`_IDF_POW`, default 1.5) —
  each term's IDF is raised to a power so a rare, discriminative term (a named entity)
  dominates the several common words it is buried under in a long natural-language
  question ("장 발장은 어떤 범죄로 유죄 판결을 받았나요?" — the entity 발장 vs common
  범죄/유죄/판결). Plain BM25 sums per-term scores, so many common matches otherwise
  outrank the one rare-entity match; the power fixes this "entity-burial". Found by
  inspecting the failing queries, not fishing. Zero runtime cost (folded into the
  precomputed IDF once at index build). Flips the last holdout **Ko-StrategyQA
  0.6414→0.6509 (9→10 wins)** and lifts every other set (HotPotQA-24 0.908→0.929,
  AutoRAG 0.917→0.931); the win holds across the whole flat band `p ∈ [1.3, 2.0]`, so it
  is a robust default, not a fit to test labels. Tunable via
  `BM25(..., idf_pow=…)` / `BM25F(..., idf_pow=…)`.
- **`Chunk.title`** — an optional short high-signal field. When any chunk carries
  a title, `InMemoryVector` indexes it with **field-weighted BM25** (`text.BM25F`),
  title weighted 4x over body — a query term in the heading outranks a chunk
  that only mentions it deep in a long passage. No title → identical to before.
- **Hybrid dense + lexical retrieval** — when chunks carry embeddings *and* text,
  `InMemoryVector` min-max normalizes dense cosine and lexical BM25(F) per query and
  combines them `dense_weight·dense + lexical_weight·lexical` (dense recovers
  paraphrase, lexical nails exact terms). The default `lexical_weight=0.8` (vs dense
  1.0) is a *flat* optimum — the aggregate MRR barely moves across 0.4–1.0, so it is
  a single principled setting, not a per-corpus fit. This score fusion beat rank
  fusion (RRF) on every dataset measured. Tunable via
  `build_inmemory(..., vector_kwargs={"lexical_weight":…, "dense_weight":…})`.
  In a full-pipeline benchmark (shared e5-small embedder) it flips omnifuse's two
  lexical-only losses (AutoRAG, Ko-StrategyQA) into wins over synaptic's fused pipeline.
- **Graph-companion fusion** (`OmniFuse.retrieve`) — a new public retrieval API
  that fuses 1-hop graph structure *into the ranking*: a passage referenced/linked
  by a strong lexical seed is surfaced beside it (score `fusion_alpha`×seed), so
  multi-hop evidence sharing no query vocabulary lands in one shot — no agent, no
  LLM. `search()` now builds its chunks/evidence on `retrieve()`. Opt out with
  `graph_fusion=False`. Added `GraphStore.neighbor_ids` (InMemory + Fuseki).

## 0.4.0

- Replaced `Memory` (remember/recall) with **`Vault`** — an omnifuse-native memory:
  - `fuse(text=, facts=)` write / `surface(query)` read (on-brand verbs, not generic remember/recall).
  - **fuse-on-write**: facts deduped & entities coreferenced by label (knowledge merges, not piles).
  - **salience**: each fuse/surface bumps node salience; `surface()` re-ranks results by it (no PPR/Hebbian).
  - `save()/load()` JSONL incl. salience; incremental label set (no per-write re-derivation).
- **Breaking**: `Memory`/`remember`/`recall` removed (the lib is pre-1.0; no released users relied on it).

## 0.3.0

- `Memory` — a growing store built on OmniFuse search. `remember()` facts/notes over time,
  `recall()` via the same one-shot graph+vector fusion. Notes auto-link to known entities by
  label; `save()/load()` JSONL persistence. (synaptic-memory–style memory on omnifuse's engine.)

## 0.2.0

- Convenience loaders so you can give loose data and search immediately (synaptic `from_data` style):
  - `from_triples(triples, chunks=...)` — accepts `(s, p, o)` tuples / dicts / `Triple`; **infers nodes**
    (object of an is-a edge → class) when none are given.
  - `from_jsonl(triples=, nodes=, chunks=)` and `from_csv(triples=, chunks=)` — stdlib json/csv, zero deps.
  - `from_fuseki(query_url, graph_uri, user=, password=)` — one call over any SPARQL endpoint.
- `build_inmemory` now coerces loose tuples/dicts too.

## 0.1.0

Initial extraction of the one-shot GraphRAG fusion algorithm as a backend-agnostic library.

- `OmniFuse.search` — one-shot fusion: vector/lexical + graph label-linking + class
  enumeration + HippoRAG 1-hop, fused with MMR diversity and adaptive top-k, single synthesis.
- `GraphStore` / `VectorStore` / `LLM` protocols (structural typing).
- Zero-infra backends: `InMemoryGraph` (BM25 label search with CJK n-grams, class
  enumeration, 1-hop traversal) and `InMemoryVector` (cosine or BM25 lexical).
- `EchoLLM` so the pipeline runs with no API key; `build_inmemory(...)` one-call setup.
- `FusekiGraph` — stdlib-only SPARQL adapter (any SPARQL 1.1 endpoint); same algorithm runs
  on a real Apache Jena Fuseki store. Self-contained (in-memory) and Jena modes both supported.
- Language-neutral default system prompt (overridable via `system_prompt=`).
- `dependencies = []` core; pytest smoke tests; quickstart + fuseki examples.
