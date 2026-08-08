# eval/ — OmniFuse retrieval benchmark

Standalone evaluation harness (not shipped in the package). The public IR tracks compare
single-shot OmniFuse and **synaptic-memory** on identical corpus, queries, qrels, K=10, and
the byte-identical `eval/metrics.py` scorer. synaptic runs through its own
`eval.run_all` FTS-only driver (`embedder=None, reranker=None`). The enterprise fixture is
reported separately as a K=5 native-capability track because its intent routing and memory
semantics are intentionally not equal-input equivalents. Performance and native artifact
footprint likewise carry their capability caveats beside the numbers.

The current canonical direct artifact targets the official synaptic-memory `v0.27.0` tag
at `836d53640e520c88910dd57e098167a4defe50d2`. The older public8 and extended9
artifacts target upstream `main` at `7470e728a7d728dedea5363aabbb73adf6ac666f`;
that checkout's package metadata also says `0.27.0`, but it is not the tag. The two
provenance cohorts remain separate.

## HotPotQA official E2E result (2026-08-08, v22)

`e2e_qa_retrieval_bench.py` reproduces the tagged
`tests/benchmark/test_e2e_qa.py` corpus, seed-42 cohort, search limit 10, evidence-step limit
8 and 2,048-token context budget. Synaptic uses its native classifier, relation detector,
`PhraseExtractor(5)` and `build_evidence`. OmniFuse uses native `retrieve` over an opt-in
title-reference graph followed by product MMR. The graph derives 94 edges from unambiguous
multi-token title mentions across 226 documents; it never reads queries, answers or qrels.

| retrieval worker median | **OmniFuse** | synaptic-memory |
|---|---:|---:|
| Recall / Hit / all-gold rate | **0.979167 / 1.000000 / 0.958333** | 0.729167 / 0.958333 / 0.500000 |
| MRR / nDCG | **0.907738 / 0.900579** | 0.845139 / 0.690834 |
| answer exact presence / token recall | **0.958333 / 0.875000** | 0.583333 / 0.525000 |
| build / mean query / p95 query ms | **14.487 / 2.390 / 2.541** | 58.271 / 65.725 / 82.714 |
| RSS delta MB | **1.572864** | 1.695744 |

The retrieval verdict is OmniFuse **11**, Synaptic **0**, ties **0**. Four isolated workers
run AB/BA/AB/BA without warm-up; lexical materialization is charged to first retrieval.

`e2e_qa_answer_bench.py` then changes only the retrieved context. It preserves the official
prompt, `/api/chat` messages, `stream:false`, `think:false` and simple correctness function.
The same local Ollama 0.32.6 `qwen3.5:4b` digest serves both systems in alternating
per-question order. The 120-second timeout changes no prompt or model payload.

| complete 24-question answer run | **OmniFuse** | synaptic-memory |
|---|---:|---:|
| cohort mean correctness, zeroes included | **0.754239** | 0.504040 |
| upstream nonzero-only mean correctness | **0.905087** | 0.864069 |
| accuracy@0.5 / exact / positive | **0.750000 / 0.708333 / 0.833333** | 0.500000 / 0.500000 / 0.583333 |
| mean / p95 / total generation ms | **37,573 / 55,404 / 901,760** | 42,237 / 65,621 / 1,013,679 |
| mean retrieval ms / prompt tokens | **2.308 / 1,134.96** | 71.914 / 1,243.54 |

The answer verdict is OmniFuse **10**, Synaptic **0**, ties **0**. This is one complete
controlled local-model run; stochastic generation is not presented as a universal model
constant. The upstream correctness mean drops zero-score rows, so the honest cohort mean is
also reported.

Evidence:
[`results/e2e_qa_retrieval_synaptic_tag_v0.27.0_836d536_20260808_v22.json`](results/e2e_qa_retrieval_synaptic_tag_v0.27.0_836d536_20260808_v22.json)
(SHA-256 `860ea0c2e9b4caa1d7598fcda269cb017c24d095fd39038965883a86399e4ded`)
and
[`results/e2e_qa_answer_synaptic_tag_v0.27.0_836d536_qwen3.5_4b_20260808_v22.json`](results/e2e_qa_answer_synaptic_tag_v0.27.0_836d536_qwen3.5_4b_20260808_v22.json)
(SHA-256 `0c96d722f07700603ad0a44d4ca825954da964323b870f28eca112c905677961`).

Reproduce into new immutable paths:

```powershell
python -I eval/e2e_qa_retrieval_bench.py `
  --synaptic-repo <clean-v0.27.0-checkout> `
  --data <clean-v0.27.0-checkout>/tests/benchmark/data/hotpotqa_24.json `
  --python <frozen-python> --trials 4 `
  --workers-dir <new-workers-dir> --out <new-retrieval-result.json>

python -I eval/e2e_qa_answer_bench.py `
  --synaptic-repo <clean-v0.27.0-checkout> `
  --data <clean-v0.27.0-checkout>/tests/benchmark/data/hotpotqa_24.json `
  --model qwen3.5:4b --base-url http://127.0.0.1:11434 `
  --timeout-seconds 120 --work-dir <new-checkpoint-dir> `
  --out <new-answer-result.json>
```

## LongMemEval-S retrieval result (2026-08-03, v20)

`longmemeval_retrieval_bench.py` reproduces the retrieval-owned portion of the tagged
`tests/benchmark/test_longmemeval.py`: seed-42 balanced sampling, fresh memory per question,
turn-pair records, 2,000-character truncation, `limit=20`, deduplicated session rankings and
mean gold-session recall. The upstream answer-generation stage is excluded because it calls
an external LLM; this result makes no answer-accuracy claim.

The upstream default requests 50 questions but selects 8 per each of 6 types, so the actual
cohort is 48 questions, 2,296 sessions, 23,668 turns and 11,935 turn-pair records. The
controller materializes only that sample for isolated workers. This prevents the 277 MB
source JSON and Python's retained parse allocator from contaminating product RSS. Two fresh
workers per system run in AB/BA order, and both systems preserve identical rankings across
their two trials.

| worker median | **OmniFuse** | synaptic-memory |
|---|---:|---:|
| mean session Recall@20 | **0.970486** | 0.841319 |
| session Hit@20 | **1.000000** | 0.916667 |
| session MRR@20 | **0.893601** | 0.699007 |
| session nDCG@20 | **0.893737** | 0.689795 |
| total build ms | **20.4873** | 82.1059 |
| mean retrieval ms | **51.6333** | 235.1101 |
| retrieval p95 ms | **58.4504** | 252.8300 |
| max per-question RSS delta MB | **1.0056** | 2.1770 |

OmniFuse's build measurement is its native lazy-store construction. Because every question
has exactly one retrieval and no warm-up, full lexical materialization is charged to the
reported retrieval latency. Synaptic's native `graph.add` and `graph.search` costs remain in
their respective phases.

OmniFuse wins 8/8 aggregate metrics. Per question, upstream session recall is 10 wins,
38 ties and 0 losses. MRR and nDCG each retain two local losses, which remain visible in
the artifact's complete rankings.

Evidence:
[`results/longmemeval_retrieval_synaptic_tag_v0.27.0_836d536_20260803_v20.json`](results/longmemeval_retrieval_synaptic_tag_v0.27.0_836d536_20260803_v20.json)
(SHA-256 `f5d8c551cf658f8b8d2fdeb64f0b46d04542c14f0f8149eedc9c711ced488536`).
The source dataset SHA-256 is
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`;
the 48-question worker sample SHA-256 is
`f22c0ea8a5faa91138e2870ed86298bbe666634fd6eeb1203a5f2fb0235077d0`.
LongMemEval is not silently added to the 22-target doctor matrix because the tagged checkout
does not contain its 277 MB external file. This runner requires an explicit `--data` path and
performs its own source/sample/source-tree preflight and postflight fingerprints.

Reproduce with new output and worker paths:

```powershell
python -I -B -X utf8 eval/longmemeval_retrieval_bench.py `
  --synaptic-repo <clean-v0.27.0-checkout> `
  --data <longmemeval_s_cleaned.json> --python <frozen-python> `
  --max-questions 50 --limit 20 --trials 2 `
  --out <new-result.json> --workers-dir <new-workers-dir>
```

## Official ablation-stage investigation (2026-07-30, v19)

The exact tagged `test_ablation.py` completed after 59,198.80 seconds with 13/13 tests
skipped at S8 because no configured external or local LLM was reachable. Empty stderr and
the raw completion record are fingerprinted in
[`../worklogs/2026-08-03-v19-v20-ablation-longmemeval.md`](../worklogs/2026-08-03-v19-v20-ablation-longmemeval.md).
The skip occurs after S0-S7 work, but the upstream test asserts no per-stage improvement and
publishes no result when S8 skips; it is not counted as a pass or benchmark win.

The controlled comparison separates unlabeled S0/S1/S2/S6 from qrels-supervised S3/S4/S5.
On AutoRAG's unlabeled lane, OmniFuse records MRR 0.8908 and nDCG 0.9187 versus the best
comparable Synaptic values 0.8173 and 0.8591. On NFCorpus, cold OmniFuse records MRR 0.5486,
nDCG 0.3334 and mean query 5.274 ms versus best Synaptic 0.5116, 0.2959 and 184.993 ms.
The supervised NFCorpus lane records OmniFuse MRR 0.6276, nDCG 0.5121 and query 0.287 ms
versus best Synaptic S3/S4/S5 values 0.2908, 0.1832 and 12,128.912 ms.

AutoRAG has zero feedback-eligible queries under the upstream simulator rule. S7 is excluded
from the fair table because the Ollama implementation returns an empty vector on connection
failure instead of raising, so the test's mock fallback never runs. S8 requires an external
LLM and remains outside the retrieval claim.

Evidence:
[`results/ablation_autorag_synaptic_tag_v0.27.0_836d536_20260730_v19.json`](results/ablation_autorag_synaptic_tag_v0.27.0_836d536_20260730_v19.json)
(SHA-256 `98910fff89c28c90b13cadd901a446f213496df8f94eefeaeac9aa89d1a8516d`) and
[`results/ablation_nfcorpus_synaptic_tag_v0.27.0_836d536_20260730_v19.json`](results/ablation_nfcorpus_synaptic_tag_v0.27.0_836d536_20260730_v19.json)
(SHA-256 `e2bd7173586c397f229b924441e939d88bee460905b12e9d2eed4221ea320ac0`).

## Official `v0.27.0` direct14 result (2026-07-28, v18 revalidation)

`direct_external_bench.py` imports the official tag's dataset preparation and native
no-embedding graph construction, preserves its 2,000-character truncation, seed-42
sampling and 20-candidate search, and records every query ranking. Both systems are scored
through the official driver's `tests/benchmark/metrics.py::BenchmarkResult`; the local
scorer copy is verified byte-identical. The upstream MRR is
calculated over those 20 candidates; MRR@10 and Precision/Recall/F1/nDCG use K=10.

The 2026-07-28 v18 frozen revalidation completed all **14/14 executable cases** declared by the tag's
`test_external_datasets.py`, with 2,269 queries per system:

| metric, unweighted dataset macro | synaptic | **OmniFuse** | dataset W/L/T |
|---|---:|---:|---:|
| upstream MRR@20 | 0.6553 | **0.7038** | **14/0/0** |
| MRR@10 | 0.6537 | **0.7022** | **14/0/0** |
| Precision@10 | 0.1505 | **0.1776** | **14/0/0** |
| Recall@10 | 0.6606 | **0.7178** | **13/0/1** |
| F1@10 | 0.1919 | **0.2206** | **14/0/0** |
| nDCG@10 | 0.6100 | **0.6747** | **14/0/0** |

The v18 result has no dataset losses on MRR@20, MRR@10, Precision@10, F1@10 or nDCG@10; Recall@10 has one tie. No result-list cutoff or dataset/query-specific branch is used. Result and full per-query provenance:
[`results/direct_external14_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json`](results/direct_external14_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json)
(SHA-256 `b8b0d333c4fb44adb8b32d3fbea01abea232f42f34a0ae71cd667a302e9e51ef`).

The v15 and v18 runs have identical macro metrics, dataset verdicts and all 2,269 OmniFuse top-20 rankings (canonical `[case_id, query_id, retrieved_top_20]` SHA-256 `3b6bd4f9d3213036adb128637bff52adfe08803ec00389d1e8a7ec3c742c25c6`). Direct latency is observational because the
upstream-compatible Windows `time()` clock quantizes sub-millisecond calls; claim-grade
latency is reported by the separate repeated high-resolution performance protocol below.

## Official QA performance contract (2026-07-28, v18)

`qa_performance_bench.py` reproduces the exact 150-document `combined_graph` fixture and
16-query sequence in the tagged `tests/qa/test_performance.py`. It runs four fresh workers
per system in AB/BA/AB/BA order, records cold official and steady repeated timings separately,
and verifies source, data, environment, worker and doctor provenance before publication.

| worker median | **OmniFuse** | synaptic-memory |
|---|---:|---:|
| build ms | **0.2015** | 65.4525 |
| official p95 ms | **49.2506** | 3,079.7073 |
| official average ms | **3.1906** | 205.0199 |
| steady p50 / p95 ms | **0.0252 / 0.0524** | 8.2988 / 21.4472 |
| steady 16-query round ms | **0.4643** | 159.6240 |
| post-query / peak RSS MB | **35.43 / 45.30** | 534.99 / 598.32 |

OmniFuse wins 10/10 common efficiency metrics and passes the upstream `p95 < 100 ms` and
`average < 50 ms` gates in 4/4 cold workers; Synaptic passes each in 0/4. Both rankings are
deterministic in 4/4 workers. The accepted forward-only build change lowers OmniFuse cold p95
from 73.6668 to 49.2506 ms (-33.1%) without changing its exact ranking SHA-256. Raw batch
storage and full lexical materialization are recorded but excluded from the verdict because
Synaptic's `save_nodes_batch` contract is not capability-equivalent to both OmniFuse phases.

Evidence:
[`results/qa_memory_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json`](results/qa_memory_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json)
(SHA-256 `bcb3e213f4271a341f198d38db925e58c9cc1ab50fc8c85065ffa02ea6ef62ed`).

Reproduce with new, unused output and worker paths:

```powershell
python -I -B -X utf8 eval/qa_performance_bench.py `
  --synaptic-repo <clean-v0.27.0-checkout> `
  --doctor-manifest <doctor.json> --trials 4 --repeats 3 `
  --out <new-result.json> --workers-dir <new-workers-dir>
```

## Static immutable-index footprint (2026-07-24, v15)

Static graph/vector lexical search now uses a forward-only `CompactPostingsSnapshot`; validated
packed-forward pickle and raw SQLite writers avoid rebuilding reverse records. A three-run
30,000-document diagnostic preserved every top-10 id and `float.hex` score. Median index RSS
fell 28.47 to 7.42 MB, state bytes 4,408,135 to 1,382,373, and serialization 72.16 to 4.05 ms.
Build rose 490.04 to 565.10 ms; unseen-term p50 rose 0.020 to 0.134 ms, while repeated-query
p50 was 0.019 versus 0.020 ms. This diagnostic is not a Synaptic capability comparison.
Artifact: [`results/static_index_micro_20260724_v15_static_compact_v1.json`](results/static_index_micro_20260724_v15_static_compact_v1.json)
(SHA-256 `87c4eb4f2215dbe68bbc7147080b346e30611175fcc0a1099a0d885687c8aae8`).

## Official `v0.27.0` MemoryBackend performance and CDC result (2026-07-23, v8)

`perf_bench.py --protocol official-external-memory` uses the official driver selection and
preprocessing, native no-embedding `MemoryBackend`, K=10 and candidate limit 20. It runs two
fresh counterbalanced AB/BA trials per system, one warm-up plus five measured rounds, and times
fully materialized raw top-20 results with `time.perf_counter_ns`. A worker fails if a ranking
changes inside the run. UTF-8 mode, the official `uv.lock`, installed-distribution manifest and
`uv sync --check` are checked before and after every worker. Repository/input/source
fingerprints and doctor binding are checked at controller preflight/postflight before
write-once publication. A separate post-run audit matched every canonical ranking hash to the
direct14 reference.

New performance schema-v5 and CDC schema-v3 runs retain the canonical frozen input and every
raw worker JSON in a new, non-reused `--workers-dir`. Direct14 schema-v4 retains every raw
worker JSON and binds the official-tag input files with controller and worker fingerprints; it
does not create a separate frozen-input copy. Each controller assigns a unique UUIDv4
`worker_run_id` to every launch, verifies the exact echo, and records launcher PID and worker
PID separately. PID equality is observational because a Windows virtual-environment launcher
can hand off to a child Python process; duplicate run IDs or any raw-artifact hash change block
report publication. Completed raw worker files (and the performance/CDC frozen input) remain
available after an interrupted run, but launcher-PID summaries are currently published only in
a successful final report. Historical artifacts linked above remain unchanged and are not
retrofitted with evidence that was not captured when they ran.

The table reports the mean of the two worker-level values. In each worker, end-to-end adds
native ingest to the mean duration of one complete measured query-set round; the table then
averages those two worker values.

| dataset | system | ingest s | query p50 ms | query p95 ms | end-to-end s | peak RSS MB | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| AutoRAG | **OmniFuse** | **0.0010** | **0.4030** | **0.6016** | **0.0483** | **63.50** | **0.8919** |
| | synaptic | 0.0074 | 206.8121 | 260.0951 | 23.7816 | 614.38 | 0.8310 |
| Allganize RAG-ko | **OmniFuse** | **0.0003** | **0.1047** | **0.1884** | **0.0226** | **43.25** | **0.9683** |
| | synaptic | 0.0016 | 93.3795 | 105.0590 | 16.7036 | 611.19 | 0.9468 |
| NFCorpus | **OmniFuse** | **0.0044** | **0.1758** | **1.5593** | **0.0532** | **74.06** | **0.5058** |
| | synaptic | 0.0371 | 271.9918 | 349.2337 | 23.0643 | 74.07 | 0.4771 |

Synaptic/OmniFuse ratios are **7.46× / 5.86× / 8.49×** for ingest,
**513.25× / 892.09× / 1,546.95×** for p50 query latency, and
**492.05× / 739.50× / 433.65×** for end-to-end. The ingest deficit was removed structurally:
plain static stores snapshot scalar title/text inputs and materialize the immutable lexical
index once on first use; mutable and feedback-backed stores keep eager snapshot semantics.

The NFCorpus CDC run applies 36 inserts, updates, deletes and no-ops and verifies each
checkpoint against a full rebuild. Both systems are scored with the same six upstream metrics:

| metric | **OmniFuse** | synaptic |
|---|---:|---:|
| MRR@20 | **0.5080** | 0.4799 |
| MRR@10 | **0.5056** | 0.4771 |
| Precision@10 | **0.2960** | 0.2507 |
| Recall@10 | **0.1514** | 0.1323 |
| F1@10 | **0.1401** | 0.1206 |
| nDCG@10 | **0.2927** | 0.2481 |

Its p50 Synaptic/OmniFuse ratios are **4.82×** for initial ingest, **6.86×** for a mutation
group, **14.49×** for the cold first query-set round, **461.72×** for the steady round and
**14.49×** for incremental end-to-end. Both systems exactly match their own full-rebuild
rankings and `float.hex` scores at every checkpoint.

Compact evidence and hashes:
[`results/perf_cdc_synaptic_tag_v0.27.0_836d536_20260723_v8_summary.json`](results/perf_cdc_synaptic_tag_v0.27.0_836d536_20260723_v8_summary.json)
(SHA-256 `a4dd6e22df1bac13c5ff5d360dfd86cf2b4ff67ad8bb3ee6f2a3e26be48e6019`).
These are current-host observations for three selected official static cases and one official
CDC case. They do not claim cross-machine constants or cover persistent backends, embeddings,
reranking or unrelated agent-memory features. Pre-existing host workloads were not stopped.

## TREC-COVID large-corpus observation (2026-07-24)

The v14 AB/BA run completed four fresh workers on 171,332 documents; the v15 follow-up reran
two OmniFuse workers on the byte-identical frozen input. v15 OmniFuse preserves MRR@10 0.9083,
records p50 109.49 ms versus canonical Synaptic 2,937.81 ms, and reduces its own current RSS
1,078.30 to 786.08 MB. Synaptic current RSS remains lower at 412.88 MB, and both lifetime peaks
are about 2.13 GB because frozen-input parsing dominates. Ingest is not compared because the
RAM index materializes during warm-up. The follow-up is not a new counterbalanced two-system run.
Artifacts: [`results/perf_trec_covid_sqlite_native_trials2_20260724_v14_raw_df_v1.json`](results/perf_trec_covid_sqlite_native_trials2_20260724_v14_raw_df_v1.json)
and [`results/perf_trec_covid_v15_omnifuse_followup_20260724_v1.json`](results/perf_trec_covid_v15_omnifuse_followup_20260724_v1.json).

## Official native SQLite persistence cohort (2026-07-28, v17)

`persistence_bench.py` schema v2 measures both systems from byte-identical frozen inputs
through closed, durable SQLite artifacts. It releases ingestion-only staging before clean
open and records post-create, clean-open and post-query RSS at the same phases. Allganize,
AutoRAG and NFCorpus use four workers per system in AB/BA/AB/BA order, one warm-up and three
measured rounds. TREC-COVID uses two workers per system in AB/BA order and two measured rounds.
K=10, candidate limit 20 and the byte-identical six-metric scorer are shared. Each cell below
is the worker median, OmniFuse / Synaptic.

| efficiency metric | Allganize RAG-ko | AutoRAG | NFCorpus | TREC-COVID |
|---|---:|---:|---:|---:|
| durable create (s) | **0.0897 / 3.6958** | **0.6646 / 13.3143** | **1.0521 / 1.1533** | **38.6159 / 42.8383** |
| clean open (ms) | **1.0438 / 2.5157** | **0.9809 / 2.5457** | **1.0310 / 2.4883** | **1.9352 / 2.5216** |
| first query (ms) | **1.1161 / 143.2841** | **4.8718 / 264.8729** | **0.7301 / 81.7479** | **864.7729 / 1,577.6638** |
| steady p50 (ms) | **0.9335 / 137.2669** | **3.9969 / 262.8316** | **1.5333 / 271.5704** | **895.1377 / 1,466.2267** |
| steady p95 (ms) | **1.5236 / 153.5278** | **6.2968 / 324.3645** | **13.1850 / 315.8520** | **1,252.1812 / 1,727.1442** |
| query round (s) | **0.1910 / 24.7631** | **0.4771 / 30.4265** | **0.4028 / 21.2739** | **43.8405 / 74.2288** |
| artifact (bytes) | **499,712 / 524,288** | **2,686,976 / 4,526,080** | **5,627,904 / 18,132,992** | **183,779,328 / 630,202,368** |
| post-run RSS (MB) | **34.92 / 521.45** | **38.70 / 529.04** | **45.51 / 50.02** | **407.19 / 454.45** |
| post-create RSS (MB) | **34.67 / 537.26** | **38.91 / 543.30** | **45.67 / 48.64** | **375.60 / 453.35** |
| clean-open RSS (MB) | **34.78 / 537.29** | **39.00 / 543.33** | **45.74 / 48.82** | **375.94 / 453.38** |
| post-query RSS (MB) | **35.50 / 521.48** | **39.85 / 533.32** | **46.21 / 55.22** | **407.20 / 454.48** |
| lifetime peak RSS (MB) | **38.65 / 599.63** | **44.85 / 601.37** | **49.88 / 55.58** | **1,813.81 / 1,814.21** |
| workload peak RSS (MB) | **35.58 / 538.74** | **44.57 / 547.60** | **49.84 / 55.58** | **463.53 / 595.19** |

| official accuracy metric | Allganize RAG-ko | AutoRAG | NFCorpus | TREC-COVID |
|---|---:|---:|---:|---:|
| MRR@20 | **0.9733 / 0.9585** | **0.8977 / 0.8911** | **0.5173 / 0.5088** | **0.9029 / 0.7210** |
| MRR@10 | **0.9733 / 0.9585** | **0.8977 / 0.8905** | **0.5147 / 0.5080** | **0.9017 / 0.7210** |
| Precision@10 | **0.1227 / 0.1058** | **0.1000 / 0.0991** | **0.3010 / 0.2844** | **0.6800 / 0.5420** |
| Recall@10 | **1.0000 / 0.9950** | **1.0000 / 0.9912** | **0.1521 / 0.1448** | **0.0173 / 0.0132** |
| F1@10 | **0.2088 / 0.1877** | **0.1818 / 0.1802** | **0.1412 / 0.1329** | **0.0334 / 0.0255** |
| nDCG@10 | **0.9802 / 0.9673** | **0.9237 / 0.9155** | **0.2980 / 0.2807** | **0.7135 / 0.5510** |

The arithmetic verdict is **76 strict wins, 0 ties, 0 losses**. The 0.41 MB TREC lifetime-
peak difference is parser-dominated and not an operational margin; phase RSS and workload peak
exclude that interpretation. v17 changes measurement attribution and TREC preflight routing,
not retrieval scores or the raw SQLite format.

Compact evidence:
[`results/persistence_memory4_synaptic_tag_v0.27.0_836d536_20260728_v17_summary.json`](results/persistence_memory4_synaptic_tag_v0.27.0_836d536_20260728_v17_summary.json)
(SHA-256 `60411946ca671f8e3f1f402786e4fb7aa703a70671426d4374b3a972ac0ec11b`).
The v16 scorer diagnostic remains in
[`results/sqlite_raw_scorer_micro_20260727_v16.json`](results/sqlite_raw_scorer_micro_20260727_v16.json)




Reproduction requires a new output path and worker directory. Use `--trials 4 --repeats 3`
for the three smaller direct cases and `--trials 2 --repeats 2` for TREC-COVID:

```powershell
python -I -B -X utf8 eval/persistence_bench.py `
  --synaptic-repo <clean-v0.27.0-checkout> `
  --data-dir <clean-v0.27.0-checkout>/tests/benchmark/data `
  --dataset trec_covid.json --doctor-manifest <strict-doctor.json> `
  --out <new-result.json> --workers-dir <new-workers-dir> `


```

## Datasets

**finreg** (self-contained here) — 4,417 Korean financial-statute articles from
[law.go.kr](https://www.law.go.kr) (public-domain law). `data/finreg/raw.jsonl` +
`data/queries/finreg{,_multihop}.json`. Corpus + query GT reused from
synaptic-memory's public eval (Apache-2.0). Single-hop = 1 relevant article;
multi-hop = article + the article it cites (both must be retrieved).

**Tracked public IR sets** — the eight BEIR-style datasets synaptic commits in
`tests/benchmark/data/*.json` (HotPotQA, Allganize RAG-ko/Eval, KLUE-MRC,
PublicHealthQA, AutoRAG, Ko-StrategyQA). Run via `public_bench.py --synaptic-repo`.
Nine additional public sets are upstream-declared/downloader-generated, gitignored inputs;
`bench.py doctor`
keeps them visible and fingerprints the exact frozen files used by a run.

## Reproducibility doctor

Before a benchmark sweep, write a machine-readable manifest of all 22 harness-declared
targets (including unavailable download-only and private data):

```bash
python eval/bench.py doctor --synaptic-repo PATH --out eval/results/doctor.json
python eval/bench.py doctor --synaptic-repo PATH --out eval/results/doctor.json --strict-public
```

The manifest records every target's explicit status, SHA-256 and byte size, both
Git SHAs and dirty flags, Python/platform details, and whether the local and
synaptic scorer files are byte-identical. `--strict-public` writes the manifest
first, then exits with status 2 unless all 19 strict inputs (17 public IR plus two
local finreg inputs) are ready and the
scorers match. Pass a local private corpus with `--kra-golden PATH`; otherwise
KRA remains visible as `skipped_private` rather than disappearing from coverage.
Strict mode also blocks non-Git or dirty repositories. During an intentional development
run, `--allow-dirty` records that override explicitly instead of silently treating the
worktree as clean.

## Run

```bash
pip install -e .
python eval/finreg_bench.py                              # finreg (self-contained)
python eval/compare_synaptic.py --synaptic-repo PATH --synaptic-graph PATH  # finreg, unequal index conditions
python eval/public_bench.py --synaptic-repo PATH         # 8 public datasets
python eval/direct_external_bench.py --synaptic-repo TAG_PATH --synaptic-python FROZEN_PYTHON --doctor-manifest DOCTOR --workers-dir NEW_DIRECT_WORKERS --out NEW.json
python eval/adaptive_bench.py --data-dir PATH            # does memory improve retrieval?
FROZEN_PYTHON -I -X utf8 -B eval/perf_bench.py --protocol official-external-memory --data-dir PATH --dataset DATASET.json --synaptic-repo TAG_PATH --k 10 --candidate-limit 20 --warmup 1 --repeats 5 --trials 2 --doctor-manifest DOCTOR --workers-dir NEW_PERF_WORKERS --out NEW.json
FROZEN_PYTHON -I -X utf8 -B eval/cdc_bench.py --data-dir PATH --dataset DATASET.json --synaptic-repo TAG_PATH --steady-repeats 5 --trials 2 --doctor-manifest DOCTOR --workers-dir NEW_CDC_WORKERS --out NEW_CDC.json
python eval/incremental_bench.py --data-dir PATH        # is remember() exact, what does it cost?
python eval/idf_pow_bench.py --synaptic-repo PATH       # p=1.0/1.1/1.2/1.3/1.5
```

## Historical upstream-main public8 — single-shot, no LLM, identical scorer

| dataset | lang | task | synaptic (FTS) | **OmniFuse** | Δ |
|---|---|---|---:|---:|---:|
| HotPotQA-24 | EN | multi-hop | 0.8879 | **0.9077** | +0.020 |
| HotPotQA-200 | EN | multi-hop | 0.8775 | **0.8958** | +0.018 |
| Allganize RAG-ko | KO | enterprise RAG | 0.9595 | **0.9683** | +0.009 |
| Allganize RAG-Eval | KO | domain RAG | 0.9303 | **0.9371** | +0.007 |
| KLUE-MRC | KO | machine reading | 0.7718 | **0.8293** | +0.058 |
| PublicHealthQA | KO | paraphrase QA | 0.6065 | **0.6133** | +0.007 |
| AutoRAG | KO | passage retrieval | 0.8994 | **0.9282** | +0.029 |
| Ko-StrategyQA | KO | strategy QA | 0.6440 | **0.6466** | +0.003 |

The 2026-07-14 upstream-main run wins all **8/8 tracked public datasets** on MRR@10
(mean MRR 0.8408 vs 0.8221). Those eight files are the public datasets actually
committed by synaptic-memory; finreg and the nine downloader-generated extended
inputs are separate tracks. Recall@10 and nDCG@10 also win 8/8; Precision@10 wins
5/8 datasets while improving in macro average. OmniFuse has no required runtime dependencies. The
synaptic comparison environment enabled its optional `korean` extra (Kiwi), so
Kiwi is part of this run but is not a mandatory synaptic dependency. Exact SHAs,
driver, and values:
[`results/public_v027_20260714_canonical.json`](results/public_v027_20260714_canonical.json).

> **Exploratory full-pipeline (dense) track**: historical runs cover seven datasets and
> the best observed OmniFuse variant leads six. The embedder/configuration is not one fixed,
> predeclared head-to-head across all seven, so this is context rather than a canonical 6/7
> result. See
> [`docs/comparison/omnifuse_vs_synaptic.md`](../docs/comparison/omnifuse_vs_synaptic.md)
> and [`results/full_pipeline_e5.json`](results/full_pipeline_e5.json).

On finreg, OmniFuse's one-shot graph-companion run solves 108/120 at the current 1.2
default with no LLM. Synaptic's historical five-turn agent reports 88/120, but the two
runners use different index construction and query protocols; those values are context,
not a controlled head-to-head.

### Why the shipped `idf_pow` is 1.2 — suite-level constraint, not per-dataset tuning

Ko-StrategyQA was long the sole holdout. Six lexical levers (Kiwi / stemmer variants /
expanded endings / corpus compound-splitting / same-article graph) each failed or traded
a bigger win. Inspecting the worst-ranked queries identified an entity-burial failure:

- **"장 발장은 어떤 범죄로 유죄 판결을 받았나요?"** → the relevant "Jean Valjean" doc
  (matching the rare entity 발장) ranked *below* generic legal docs matching the common
  attribute words 범죄/유죄/판결.

The failure mode is **entity burial**: a natural-language question carries one rare,
discriminative entity under several common words; plain BM25 sums term scores, so a
document matching many common words can outrank the entity match. IDF emphasis addresses
that structurally and is folded into precomputed IDF, so it adds no query-time work.
Completing the extended coverage exposed the other side of the trade: FiQA requires
`p ≤ 1.3`, while Ko-StrategyQA requires `p ≥ 1.1`. The shipped value is therefore the
midpoint **`idf_pow=1.2`** of the global overlap `[1.1, 1.3]`, not the old 1.5 and not a
per-dataset switch. See [`results/idf_pow_ablation.json`](results/idf_pow_ablation.json).

### What makes OmniFuse win (ablation, finreg)

| config | single-hop MRR | multi-hop strict |
|---|---:|---:|
| field-weighted BM25F only (`--no-graph`) | **0.8585** | 21/120 |
| + graph-companion fusion (default) | 0.8471 | **108/120** |

- **Field-weighted BM25 (`Chunk.title`→`text.BM25F`, title 4× body)** — a query
  term in the heading beats a deep body mention. Lifts flat-body 0.797 → 0.85.
- **Graph-companion fusion (`OmniFuse.retrieve`)** — folds 1-hop graph structure
  into the ranking: a cited passage sharing no query vocabulary is surfaced beside
  the seed that references it. One shot, no LLM. Multi-hop 21 → 108. The trade-off is
  explicit: graph fusion gives up 0.0114 single-hop MRR to gain 87 strict multi-hop solves.

Current machine reports: [`results/finreg_omnifuse_v027_20260713.json`](results/finreg_omnifuse_v027_20260713.json)
and [`results/finreg_no_graph_v027_20260713.json`](results/finreg_no_graph_v027_20260713.json).

### Extended coverage — download-only BEIR/MTEB sets

The current upstream surface contains **nine** non-tracked public inputs: 2Wiki-dev,
MuSiQue-dev, TREC-COVID, SciFact, XPQA-ko, NFCorpus, MIRACL-retrieval-ko, FiQA, and
MultiLongDoc-ko. They are generated/downloaded and gitignored, so a result is accepted only
when `bench.py doctor` validates corpus/query/qrels structure and links the exact file hash.

`idf_pow_bench.py` evaluates all 17 public IR datasets (eight tracked + nine extended) with
five global arms: the no-emphasis control 1.0, both previously claimed band edges 1.1/1.3,
the shipped midpoint 1.2, and the former default 1.5. It loads each large JSON once, rebuilds
OmniFuse for every arm, and re-ingests/re-queries synaptic through its current native public
driver. No per-dataset switch is allowed.

The completed 2026-07-14 extended9 invocation records MRR@10:

| dataset | synaptic FTS | **OmniFuse p=1.2** | Δ |
|---|---:|---:|---:|
| 2Wiki-dev | 0.8244 | **0.9517** | +0.1273 |
| MuSiQue-dev | 0.7316 | **0.7791** | +0.0476 |
| TREC-COVID | 0.7437 | **0.9083** | +0.1647 |
| SciFact | 0.6317 | **0.6441** | +0.0124 |
| XPQA-ko | 0.3115 | **0.3277** | +0.0162 |
| NFCorpus | 0.5124 | **0.5175** | +0.0051 |
| MIRACL-ko | 0.9495 | **0.9750** | +0.0254 |
| FiQA | 0.2902 | **0.2920** | +0.0018 |
| MultiLongDoc-ko | 0.6326 | **0.6501** | +0.0176 |
| **macro** | 0.6253 | **0.6717** | **+0.0464** |

The shipped p=1.2 arm wins 9/9. The p=1.1 arm has a slightly higher extended9 macro,
so p=1.2 is the disclosed stable midpoint, not the macro-optimal arm for this cohort.
Combined after the fact with public8, the two independent artifacts give 17/17 MRR wins
and macro **0.7513 vs 0.7179**; they are not one 17-dataset same-pass invocation. Extended9
does not record Precision/Recall/nDCG. Artifact:
[`results/extended9_v027_20260714_canonical.json`](results/extended9_v027_20260714_canonical.json).

### Real-world golden set — a live xgen domain corpus (dev-xgen)

This section is a **2026-07-10 historical private-corpus result**. The raw KRA input was
not available to the 2026-07-13 doctor and is `skipped_private`; none of these values is
included in the current same-pass win count.

Beyond the academic IR sets, we built a **golden benchmark from a real production
corpus**: an xgen retrieval collection on `dev-xgen.x2bee.com` — 한국마사회 (KRA)
institutional documents (동반성장 / ESG / 청렴 / 경마운영 …). We downloaded 220
documents (**5,234 chunks**, mean 1,143 chars) via the retrieval API, then generated
**215 natural Korean questions** with `gpt-4o-mini` from each document's richest chunk
— *body only* (the repeated document-metadata header stripped), paraphrased, neutral to
both retrievers. Both systems then retrieve over the identical corpus/queries/qrels.

| system | MRR@10 | nDCG@10 | R@10 | wall |
|---|---:|---:|---:|---:|
| synaptic (FTS) | 0.2547 | 0.2956 | 0.4279 | 98.0 s |
| **OmniFuse** | **0.4957** | **0.5446** | **0.7535** | **6.6 s** |

Wall time is end-to-end from raw data on both sides (OmniFuse: 6.1 s index build + 0.5 s
for all 215 queries = 2.3 ms/query). **OmniFuse wins by +0.2410 MRR (~1.95×)** on every metric —
on genuinely out-of-distribution real documents. This is exactly the long-institutional-
document regime the retrieval logic targets: a specific entity buried in pages of
boilerplate. Ablation of the two logic improvements on this corpus (every config still
beats synaptic):

| OmniFuse config | MRR@10 |
|---|---:|
| plain CJK bi-gram, `idf_pow=1.0` | 0.4579 |
| + dependency-free Korean stemmer | 0.4775 |
| + IDF emphasis `idf_pow=1.5` (then-current) | 0.4775 |

The field-weighted BM25F (title 4×) + pipeline already dominate; the Korean stemmer adds
+0.020; **IDF emphasis is neutral out of distribution** (0.4775, neither helps nor hurts)
— evidence about that historical corpus, not a claim that 1.5 remains the current default.

The raw KRA documents are a **private domain corpus and are not committed**. Reproduce
from the live collection (needs dev-xgen + OpenAI credentials, read from env):

```bash
python eval/golden_devxgen_bench.py --collection-id 42 --max-docs 220 --num-queries 215
```

Numbers + methodology: [`results/golden_devxgen.json`](results/golden_devxgen.json).

### Memory — does either system's memory improve retrieval?

This is the axis that names synaptic-**memory**: it is stateful and learns. Its upstream
suite already tests reinforcement and consolidation contracts; `adaptive_bench.py` asks the
narrower held-out `graph.search` retrieval question, with controls that make the result
interpretable. A naive run reported a win that was not there, and was retracted.

    python eval/adaptive_bench.py --data-dir <synaptic tests/benchmark/data>          # disjoint queries
    python eval/adaptive_bench.py --corpus c.json --golden g.json --paraphrase p.json # re-queries

Every run prints two placebos (**shuffled** = the (query, chunk) pairing permuted;
**random-q** = a random other feedback query) and splits the eval set into **covered**
(relevant chunk remembers something) and **uncovered** (it remembers nothing). Memory
cannot move uncovered queries; if it does, the effect is a corpus-wide scoring artifact.

**Re-query axis** — feedback on the original questions, evaluation on held-out paraphrases
of them (token Jaccard 0.43). Same corpus, same queries, scored by synaptic's own
`metrics.py`:

| ΔMRR@10, held-out re-queries | KRA (ko) all | KRA covered | NFCorpus (en) all | NFCorpus covered |
|---|---:|---:|---:|---:|
| synaptic (Hebbian) | +0.0000 | +0.0093 | −0.0010 | −0.0002 |
| **OmniFuse (`Feedback`)** | **+0.1843** | **+0.4016** | **+0.1133** | **+0.1785** |
| ↳ shuffled placebo | −0.0184 | −0.0217 | +0.0028 | +0.0044 |
| ↳ random-query placebo | +0.0263 | +0.0590 | +0.0002 | +0.0006 |

On covered KRA queries `real` is 6.8× the strongest placebo, and on NFCorpus it is about
40×, so the pairing carries the signal. **Disjoint-query axis** — a different question,
not a rephrasing: memory correctly does nothing (+0.0006),
with Δuncovered **exactly 0.0000**, because a `Feedback` memory is indexed as an *evidence
field* (scored, but excluded from document frequency and from length normalization) so the
collection's IDF is provably untouched. A cold store ranks bit-identically.

synaptic scores ~0 because in the benchmarked version `graph.search()` reads none of the
fields `reinforce()` writes. Numbers, controls and the full retraction history:
[`results/adaptive_memory.json`](results/adaptive_memory.json).

### Historical upstream-main efficiency — isolated and capability-qualified

The earlier performance runner separates accuracy from timing. Both systems use the byte-identical
`BenchmarkResult` for MRR, while direct `perf_counter` calls measure p50/p95/mean latency
after one warm-up and across five rounds. Each system runs in a fresh process; K, candidate
limit, ranking semantics, and query order are identical. Whole-worker current/peak RSS includes
imports, build, warm-up, and measured queries. The current runner requires a strict doctor
for machine output, verifies inputs/repos/sources before and after both workers, and refuses
to overwrite an existing result path.

Ingest is the same raw corpus to each system's native queryable index, but the outputs are
not capability-equivalent: OmniFuse builds a RAM index and synaptic builds a durable,
disk-queryable SQLite graph. `footprint_bench.py` therefore reports native artifact bytes
separately and carries the same caveat. The table below preserves selected 2026-07-13
upstream-main measurements. Their schema-v1 artifacts predate the stricter provenance
contract, so they are not described as official-tag canonical evidence.

| dataset | system | ingest s | p50 ms | p95 ms | peak RSS MB | MRR@10 |
|---|---|---:|---:|---:|---:|---:|
| PublicHealthQA | **OmniFuse** | **0.06** | **0.13** | **0.23** | **31.7** | **0.6133** |
| | synaptic | 5.52 | 4.20 | 5.27 | 599.8 | 0.6065 |
| NFCorpus | **OmniFuse** | **1.69** | **0.38** | **5.00** | **62.4** | **0.5175** |
| | synaptic | 50.52 | 16.92 | 31.04 | 97.3 | 0.5124 |
| Allganize RAG-ko | **OmniFuse** | **0.21** | **0.19** | **0.39** | **35.6** | **0.9683** |
| | synaptic | 5.80 | 4.57 | 6.19 | 599.9 | 0.9595 |

Native artifact footprint on the two explicitly selected regression fixtures: NFCorpus
5.90 MB vs 20.76 MB (**3.52× smaller**) and Allganize 0.31 MB vs 0.76 MB (**2.45×
smaller**) for OmniFuse. This is not a representative sample or a total capability/memory
claim. Selected machine reports: [`results/perf_publichealth_v027_20260713.json`](results/perf_publichealth_v027_20260713.json),
[`results/perf_nfcorpus_v027_20260713.json`](results/perf_nfcorpus_v027_20260713.json),
[`results/perf_allganize_v027_20260713.json`](results/perf_allganize_v027_20260713.json), and
[`results/footprint_v027_20260713.json`](results/footprint_v027_20260713.json).

### Incremental memory

Memory used to be batch: folding a confirmed pair in meant rebuilding the index, which is
not something a live service can do per click. `remember()` now updates the index in place.

```python
of = build_inmemory(nodes, triples, chunks, feedback=Feedback())   # an empty Feedback opts in
of.remember("statin side effects", ["doc7"])                       # ~1 ms, no rebuild
```

This is what the evidence-field design buys. Evidence never enters document frequency, so
`N`, the content df and every content term's IDF are **fixed** — remembering rewrites the
contributions of exactly one document. The single coupling is that a term seen *only* in
evidence takes its IDF from the evidence df; but every posting of such a term is
evidence-derived, so the documents to fix are the ones that remember it. The blast radius
is the memory, not the corpus — measured, **15 such terms out of a 23,610-term vocabulary**.

| | rebuild | `remember()` | per memory |
|---|---:|---:|---:|
| NFCorpus (3,633 docs, 100 memories) | 1.389 s | **1.00 ms** | **1,386x** |
| same memories, a tenth of the corpus | 0.175 s | **1.02 ms** | 172x |
| KRA (5,234 chunks, 120 memories) | 6.605 s | **1.52 ms** | **4,335x** |

The middle row is the control: ten times fewer documents makes the *rebuild* 7.9x cheaper
and leaves `remember()` where it was. Cost tracks the memory, not the corpus, and it stays
flat as memory accumulates.

The bar is that the updated index is **bit-identical** to a full rebuild — every posting,
every float — not merely close, because a weight that drifts is a scoring bug with a
stopwatch. The first prototype claimed the update was purely local, skipped the evidence-df
coupling, and differed from a rebuild in 1,181 terms; the bar caught it.
[`eval/incremental_bench.py`](incremental_bench.py) ·
[`eval/results/incremental_memory.json`](results/incremental_memory.json) ·
[`tests/test_incremental.py`](../tests/test_incremental.py). `forget(query, doc_ids)` is
the exact inverse (~1 ms, bit-identical to a rebuild without the pair; forget-everything lands
bit-identically on the cold index). The bench's forget pass withdraws half the memories in
place and compares against a rebuild with the other half.

### Reproducibility notes

- Both systems are non-neural in the canonical lexical track. Historical dense runs are
  exploratory and are not folded into its win count.
- Eight public dataset JSONs are tracked by synaptic-memory; nine more are generated or
  downloaded and gitignored. The doctor fingerprints the exact local inputs instead of
  treating an absent file as an omitted benchmark. finreg (public-domain law) is included here.
- synaptic's **private** corpora (krra/assort/x2bee) were not present in the selected
  checkout, so they were not runnable. The dev-xgen golden corpus is likewise private; its reproducer
  requires the original service and credentials, and the documents are not committed.
