# DeepSeek-V4.1-Flash TP2 DSpark drafter: x86 NVLinked RTX 3090 pair for DGX Spark GB10

Split the remote DSpark worker across two NVLinked RTX 3090s while keeping the same Spark-side client and wire contract. The worker reduces drafting time, and its strongest measured engine result combines TP2 with the async process proxy.

**Hugging Face:** [Card and files](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) · **Stack overview:** [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · **Collection:** [All entries](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

## Status

2026-10-23 — opt-in; TP2 production promotion is pending. GB10 engine comparisons, greedy checks and supervised recovery drills passed on the reviewed build; production uses one card with async rounds.

The earlier allocator and capture-stream failures are reproduced and fixed. The production layout and GPU ownership checks were independently reviewed; that clearance is separate from a production switch. No promotion date is projected.

## What it is

One worker process drives both cards (`--tp 2`) and splits the drafter the way the Spark target splits itself:

- **Vocabulary-sharded** head and Markov, with an argmax exchange each Markov step.
- **Head-split** `wq_b` and attention, **group-split** `wo_a`, **row-parallel** `wo_b`.
- **Channel-split** routed experts (the EXL3 bank at tp_size 2), and a column/row-split shared expert.
- **Replicated:** the embedding, mHC, norms, `wq_a`, `wkv`, the KV ring, routing and confidence.

Collectives use a small custom CUDA extension (`code/dspark/worker/p2p/`): a one-shot push over NVLink with epoch flags,
double buffering, and a sticky error word on timeout. Partial sums are fp32, and a + b is commutative, so both ranks hold
identical bits. Each rank's CUDA graphs are captured separately and replayed in lockstep.

On the wire it is the same contract (CONTRACT v1.8) on the same port. The Spark side doesn't change: swap the worker unit
and the engine reconnects.

Peer state is allocated outside Torch's expandable-segment allocator, and every capture site supplies a stream owned by the correct device. Peer failures are mirrored, successful exchanges wait for consumption acknowledgments, and a poisoned CUDA context refuses reuse. A pinned prebuilt extension removes compilation from worker startup.

## Why

The worker can split the head, attention and experts while retaining replicated context state. That saves worker time, but the remote drafter occupies only part of a target decode round. The measured worker gain is therefore larger than the engine gain. The async process proxy overlaps more of the remaining round.

## When to use it

Use it when the one-card remote drafter is already qualified, an NVLinked pair is available, and the measured engine improvement justifies dedicating another GPU. Pair it with the process proxy if you need the best measured combined result. Qualify the exact prebuilt extension, image, weights and cost calibration you will serve.

## When not to use it

Avoid it if you need the second GPU for other work, lack peer access or an NVLink bridge, or cannot sustain the pair's power and cooling. PCIe-only performance is not qualified by these tests. A successful CPU suite does not qualify peer ordering, graph replay or sticky-fault handling on hardware.

## Requirements

Two RTX 3090s with an NV4 bridge, peer access enabled, the same receipt-verified drafter tensors as the one-card worker, and the tested PyTorch/CUDA/exllamav3 environment. The target is two DGX Spark (GB10) nodes with the capacity-mode remote client and an x86 offload host. No target overlay change is needed solely to select TP2. Provision memory using the later per-card observations below rather than packed checkpoint size alone.

## Measured effect

All engine tests use the [EXL3 2.75 bpw build](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark), capacity-mode remote drafting, remote Engram and expanded target KV. These combined offloads and the expanded KV pool are the measured configuration throughout this entry. Baseline means one-card synchronous remote drafting at the time of the comparison. It does not mean today's one-card async production configuration. Each table states its repeats and hardware; context and concurrent request count are in the workload row.

### Controlled GB10 comparison (2026-10-23)

On 2026-10-23, two DGX Spark GB10 nodes used the x86 one-card RTX 3090 worker or NVLinked pair as shown. The one-card synchronous control was already running; the TP2 synchronous and process-proxy tests used fresh engine starts. Short and 32K single-stream workloads had three repeats, 150K had two, and concurrent 32K was one group of three. Tests used cold prefixes, temperature 1, high thinking and 1,500-token budgets. Values are direct per-step p50 per request; percentages compare against one-card sync.

| workload | one card, sync | two cards, sync | two cards + async process proxy |
|---|---|---|---|
| short × 3, step (ms) | 39.6 / 40.1 / 39.4 | 39.0 / 38.9 / 38.9 (−1.8 %) | 37.1 / 37.4 / 36.9 (**−6.3 %**) |
| 32K × 3, step (ms) | 41.2 / 42.2 / 42.3 | 41.6 / 41.5 / 41.6 (−0.8 %) | 39.6 / 39.5 / 40.2 (**−5.2 %**) |
| 32K, 3 concurrent, step (ms) | 75.9 / 76.1 / 76.1 | 74.9 / 75.1 / 74.8 (−1.4 %) | 72.8 / 73.5 / 73.0 (**−3.9 %**) |
| 150K × 2, step (ms) | 41.5 / 42.2 | 40.7 / 41.4 (−2.0 %) | 39.6 / 39.8 (**−5.1 %**) |

Worker step p50 2.55 ms, p99 3.47 ms, over 6,284 steps.

### Reviewed-code repeat and four concurrent requests (2026-10-23)

The reviewed-code repeat used the same model, hardware, combined offloads, contexts, sampling and budgets with TP2 async. Exact per-request step times, acceptance and TTFT appear in the full repeat table below. The rounded step ranges were short 37.2–37.3, 32K 40.4–40.5, concurrent 32K 72.7–73.6 and 150K 39.4–39.9 ms, within the earlier comparison's run-to-run spread.

- **New: 4 × 32K concurrent.** 51.7 tok/s overall, prefill included; about 119 tok/s combined decode (31.0 + 31.6 +
  32.7 + 23.7); step p50 87–88 ms.
- **Greedy outputs:** 6 / 6 fixed prompts identical between the one-card and two-card workers.
- **Rollback rehearsal:** switching from two cards to one card took 12.2 s in the supervised test; this is an observation, not a restart-time guarantee.

### Per-card memory and power (2026-10-23)

On 2026-10-23, the x86 NVLinked RTX 3090 pair used the EXL3-quantized worker, with 325 W caps and a 1 s sampler. Values are sampled peaks over one load and serving sequence; engine workloads were short/32K/150K at concurrency 1 or 3.

| phase | card 0 | card 1 |
|---|---|---|
| load | 5,852 MiB / 146 W | 5,212 MiB / 141 W |
| serving | 5,862 MiB / 124 W | 5,222 MiB / 122 W |

The later supervised trace measured load-time peaks of 5,862 / 5,222 MiB and serving-plus-drill peaks of 5,862 / 5,222 MiB at ≤ 124 / 121 W. Conditions: 2026-10-23; same x86 pair and GB10 configuration, one load and serving/drill sequence, sampled each second. The earlier 5.16 / 4.54 GiB reserved-memory snapshot used a different context allocation; the later worker ready line reported 5.33 / 4.72 GiB reserved. Reserved allocator memory and device-wide sampled memory are different measures.

### Earlier interleaved engine history (2026-10-23)

Earlier on 2026-10-23, the same two DGX Spark GB10 nodes and model used synchronous rounds with combined offloads and expanded KV. Only the x86 RTX 3090 worker was swapped, interleaving one card and the NVLinked pair. Short/32K had three single-stream repeats, 150K two, and concurrent 32K one group of three; temperature 1, high thinking and 1,500-token budgets. Step time is direct p50 per request from SSE event gaps.

| workload | TP2 step (ms) | one-card step (ms) | change |
|---|---|---|---|
| short × 3 | 38.5 / 38.0 / 38.4 | 38.5 / 40.0 / 39.4 | −2.5 % |
| 32K × 3 | 41.1 / 41.9 / 41.6 | 42.6 / 41.9 / 43.2 | −2.5 % |
| 32K, 3 concurrent | 73.3 / 73.6 / 74.2 | 75.6 / 75.3 / 76.0 | −2.6 % |
| 150K × 2 | 40.2 / 40.9 | 41.9 / 41.2 | −2.4 % |

6,142 drafted steps, worker p50 2.44 ms (one card ~3.23 ms), 0 TP communication errors.

### Worker in isolation (2026-10-23)

On 2026-10-23, one RTX 3090 and the NVLinked pair were compared at 325 W caps, using a 128-row worker window. Values are CUDA-graph replay p50 from one replay benchmark and its repeat. Request count and `n_out` (number of draft positions returned) vary by row.

| requests, n_out | one card | TP2 | change |
|---|---|---|---|
| 1, 5 | 2.94 ms | 1.94 ms | −34 % |
| 2, 5 | 3.67 ms | 2.42 ms | −34 % |
| 3, 4 | 4.10 ms | 2.67 ms | −35 % |
| 4, 4 | 4.89 ms | 3.17 ms | −35 % |

- Collectives are 3.6–4.7 % of the step.
- Full STEP over RDMA loopback: 3.14 → 2.34 ms at 1 request, 5.01 → 3.67 ms at 4 requests.

### Draft parity (2026-10-23)

**Parity:**

- Real pair vs single-device emulation: bit-identical.
- Agreement with the engine's own drafter on live captures: 92.6–93.4 % of blocks, the same as the one-card worker. Every
  first mismatch is a numeric near-tie.
- One-card vs TP2: drafts agree 1.000 at positions 0 and 900,000. Confidence is identical.
- During the earlier isolation test the highest total across the four installed GPUs was 709.5 W; this is not a pair-only measurement. NVLink replay/recovery/CRC errors were 0. The drafter pair used 325 W caps.

### Full reviewed-code repeat table (2026-10-23)

On 2026-10-23, the reviewed-code repeat used the same two GB10 nodes, x86 NVLinked RTX 3090 pair, combined offloads, expanded KV and async process proxy. Sampling was temperature 1 with high thinking and 1,500-token budgets. Short/32K had three single-stream repeats, 150K two; concurrent 32K had one group each of three and four. The earlier column is the preceding controlled TP2 async test, with matching repeats except no four-request group. There was no new matched one-card control.

| workload | Reviewed TP2 async step p50 (ms) | Earlier TP2 async step p50 (ms) | Reviewed acc/round | Earlier acc/round | Reviewed TTFT |
|---|---|---|---|---|---|
| short × 3 | 37.2 / 37.3 / 37.3 | 37.1 / 37.4 / 36.9 | 0.98 / 1.04 / 0.94 | 0.98 / 0.99 / 0.88 | 0.60 (first after start) / 0.38 / 0.37 s |
| 32K × 3 | 40.4 / 40.4 / 40.5 | 39.6 / 39.5 / 40.2 | 1.64 / 1.50 / 1.69 | 1.53 / 1.46 / 1.65 | 15.73 (first) / 14.55 / 14.50 s |
| 32K, 3 concurrent | 73.3 / 73.6 / 72.7 | 72.8 / 73.5 / 73.0 | 1.64 | 1.68 | 42.82 / 52.29 / 52.29 s |
| 150K × 2 | 39.9 / 39.4 | 39.6 / 39.8 | 1.35 / 1.09 | 1.19 / 1.20 | 70.69 / 68.80 s |
| **32K, 4 concurrent** | 87.1–88.2 | — | 1.68 | — | 52.74 / 66.01 / 66.01 / 66.01 s |

### Illegal-address and stream-capture diagnostics (2026-10-23)

Conditions: x86 NVLinked RTX 3090 pair, offline model/collective checks in a supervised window; context/concurrency are not engine workloads. The allocator fixed control ran two full cycles; the push/copy check used 100 repeats; other diagnostic counts are stated below.

An earlier independent review saw an illegal memory access in the first collective in some test sequences. We reproduced it
on the real pair and found two separate causes. Both are now fixed.

**1. Peer-mapped state carved from PyTorch's expandable segments.**

- The collective's peer-accessed buffers came from the Torch caching allocator. With `expandable_segments` on, old code
  crashed with Xid 31 on both cards.
- The same old code passed with expandable segments off.
- **Fix:** the peer state now gets its own `cudaMalloc` allocation outside the Torch allocator, made outside graph capture.
  The fixed code passed two full create/use/destroy cycles with expandable segments on, leaving 0 bytes in the allocator.

**2. A foreign capture stream.**

- PyTorch 2.10's graph helper keeps a class-wide default capture stream. A TP2 capture on card 0 can seed it, and a later
  one-card capture on card 1 then runs on a stream that belongs to the other device, so the capture is invalidated.
- **Fix:** every capture site passes its own stream. The regression test passes on the pair.

**Also checked on the pair:**

- **Bad-ID refusal.** A bad id is now refused symmetrically on both ranks, and each collective waits for the peer's
  consumption ack.
- **Deliberate fault.** A null store faulted as intended (Xid 31). The pair was marked poisoned and reuse was refused; both
  cards were healthy afterwards.
- **Push collective vs a peer copy:** 68.7–100.8 µs vs 187–196 µs at 5–20 rows, exact on both ranks.

## Upstream claims

No upstream performance claim is included in our measured tables. DeepSeek supplies the model and DSpark design; HeNryous and sudoingX provide the DGX Spark image and patch lineage; vLLM supplies the target runtime; turboderp/exllamav3 supplies EXL3 kernels used by the worker. A kernel capability or upstream benchmark alone does not qualify this TP2 implementation.

## Deploy

Use a reviewed immutable copy and the same model receipt as the one-card worker. Build the peer extension ahead of service startup with the worker's own build function. From the repository root, on the x86 build host with PyTorch, a compatible CUDA toolkit and a C++ compiler installed:

```sh
cd code/dspark
export DSPARK_P2P_BUILD=/data/build/dspark-p2p
mkdir -p "$DSPARK_P2P_BUILD"
env -u DSPARK_P2P_PREBUILT CUDA_VISIBLE_DEVICES='' \
  python -c 'from worker.p2p import ext; print(ext().__file__)'
source_key=$(sha256sum worker/p2p/p2p.cu | cut -c1-16)
prebuilt_dir="/data/p2p-prebuilt/$source_key"
mkdir -p "$prebuilt_dir"
cp "$DSPARK_P2P_BUILD/dspark_p2p.so" "$prebuilt_dir/dspark_p2p.so"
(cd "$prebuilt_dir" && sha256sum dspark_p2p.so > dspark_p2p.so.sha256)
```

Record the source digest, binary digest and build environment. `DSPARK_P2P_PREBUILT` names the parent directory, not the binary. The loader uses the source digest to select the subdirectory and refuses a missing or mismatched binary; it never silently falls back to compiling. Build and qualify against your exact runtime before switching workers.

Stop the one-card worker before starting a TP2 service on the same endpoint. Adapt `DIRECT_LINK_IPS` to your direct-link addresses, set `WORKER_BIND` to your listener address and port, and use an exclusive hold for both devices throughout the worker lifetime:

```sh
export CUDA_VISIBLE_DEVICES=GPU-DRAFTER0-UUID,GPU-DRAFTER1-UUID
export DSPARK_P2P_PREBUILT=/data/p2p-prebuilt
export OFL_PY_DIR=/data/ofl
export OFL_LIB=/data/ofl/libofl.so
mkdir -p /data/locks
worker/run/gpu_locked.sh /data/locks/drafter0.lock /data/locks/drafter1.lock -- \
  python -m worker.server --tp 2 --root /data/dspark \
    --ofl "$WORKER_BIND" --max-model-len 327680 \
    --status-file /data/run/dspark-worker.status.json
```

Confirm readiness and per-card memory before letting the engine reconnect. Recalibrate `VLLM_DSPARK_REMOTE_DRAFT_MS`; the earlier measured worker-time saving was about 0.8 ms, but reuse of that estimate without measuring your full remote round is not a qualification. The async settings and its exact sync rollback are in the [async sibling](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff).

## Rollback

Stop the TP2 worker, retain exclusive ownership while checking that its GPU work has drained, and start the one-card worker on the same endpoint. The engine reconnects with a new worker session; capacity mode decodes target-only during the gap. If you adapted the supplied service templates, the worker swap is:

```sh
systemctl --user stop dsv41-dspark-worker-tp2-prod.service
systemctl --user start dsv41-dspark-worker.service
```

If you changed enablement during your own promotion, also disable the TP2 unit and enable the one-card unit so a reboot restores the selected worker. Restoring one card need not disable async. To restore synchronous rounds as well, remove `VLLM_DSPARK_REMOTE_ASYNC` from both ranks' persistent environment and restart the engine. A failed TP2 start should return to the one-card worker rather than leave the service without drafting indefinitely.

## Limits and known issues

The bundled `prod_tp2.sh` and systemd templates assume the original release layout. Configure their archive paths, pins and direct-link addresses for your installation. The generic [gpu_locked.sh](code/dspark/worker/run/gpu_locked.sh) accepts caller-supplied lock paths. Full maintenance-window scripts are outside this package. Keep a hold while revoking work, draining jobs and checking both cards; retain it if a process remains or a check fails.

The later controlled test measured only −0.8 to −2.0 % per round for TP2 alone; the larger combined gain includes async overlap. Inserting context rows is about 0.1 ms slower on TP2 from host overhead. Both cards are dedicated to drafting while the worker runs. Do not equate reserved allocator memory with device-wide sampled memory or compare isolated worker replay directly with end-to-end token throughput.

The allocator and stream fixes cover the reproduced sequences; they do not justify recovering a poisoned CUDA context in place. A sticky failure requires process replacement. Maintenance jobs must not daemonize or escape their ownership hold. Independent review accepted a narrow start-up race for a process without a visible CUDA context and a bounded close wait for orphaned work; the hold remains on refusal. Pair safety requires operational ownership as well as a passing numerical test.

## Repository layout

| Path | Purpose |
|---|---|
| [code/dspark/worker/tp2.py](code/dspark/worker/tp2.py) · [model.py](code/dspark/worker/model.py) | TP2 sharding, replicated state, one-card reference and explicit capture streams. |
| [code/dspark/worker/p2p](code/dspark/worker/p2p) | CUDA peer collectives, allocation/lifetime rules, eager reference, prebuilt loader and optional fault probe. |
| [code/dspark/worker/server.py](code/dspark/worker/server.py) | Shared contract server selecting the worker backend. |
| [code/dspark/worker/run](code/dspark/worker/run) | Deployment-specific service and production helpers; generic exclusive lock wrapper. |
| [code/dspark/worker/exl3](code/dspark/worker/exl3) | Plugin wrappers, EXL3 checks and accumulation patch. |
| [code/dspark/worker/tools](code/dspark/worker/tools) | Replay, parity, tuning and conformance tools; donor code is external. |
| [code/dspark/worker/tests](code/dspark/worker/tests) | Contract, identity, peer protocol and capture-stream regressions. |
| [code/dspark/worker/TP2.md](code/dspark/worker/TP2.md) · [ENV.md](code/dspark/worker/ENV.md) | Design and historical deployment environment. |
| [code/dspark/CONTRACT.md](code/dspark/CONTRACT.md) · [code/transport](code/transport) | Shared wire/state rules and ofl transport. |

The Spark-side client is distributed in the one-card and async sibling repositories.

## Tests

For the CPU suite, from the repository root with pytest, NumPy, torch and Triton installed:

```sh
cd code/dspark
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -p no:cacheprovider -q worker/tests --ignore=worker/tests/test_fused.py
```

The excluded file contains GPU cases without no-device guards. Other GPU/model cases skip when CUDA is hidden. The CPU suite tests contract/identity logic, prebuilt-loader refusals and host-side poison handling; it does not execute NVLink collectives.

Publication checks on 2026-10-23 recorded 29 passed for the original TP2 contract subset, and later 64 passed / 40 skipped for the shared worker CPU suite with the GPU-only file excluded. Those are historical recorded counts, not a fresh hardware qualification. GPU checks in `test_p2p.py`, `test_tp2.py` and `test_capture_stream.py` require a separately supervised exclusive pair window. Intentional poison probes must never run against a serving worker.

## Credits and licences

Our code is Apache-2.0. The `vllm_exl3_moe` wrappers in [VENDORED.md](code/dspark/worker/exl3/VENDORED.md) are Apache-2.0 and call turboderp/exllamav3's MIT extension; `exl3_ref.py` records MIT-derived decoding logic. The included exllamav3 patch modifies accumulation and its build script. HeNryous and sudoingX supply the image and patch lineage, DeepSeek supplies the model and DSpark drafter, and vLLM supplies the target runtime. See [NOTICE](NOTICE) and [LICENSES](LICENSES) for bundled third-party licence texts. No model files are distributed.

## Related

- dsv41-remote-dspark-drafter-3090: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090).
- dsv41-async-hostflag-handoff: [GitHub](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).
- ofl-rdma-transport: [GitHub](https://github.com/ProprietaryLegal/ofl-rdma-transport) · [Hugging Face](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport).
- dsv41-native-1017-drafter-sm86: [GitHub](https://github.com/ProprietaryLegal/dsv41-native-1017-drafter-sm86) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86).
- dsv41-spark-offload-findings: [GitHub](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).
- DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload: [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [Hugging Face](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
- dsv41-remote-engram-rdma: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma).
- dsv41-longctx-bench-tools: [GitHub](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools).
- dsv41-exact-acceptance-ceiling: [GitHub](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling).
- Image forks: [HeNryous lineage](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark) · [sudoingX lineage](https://github.com/ProprietaryLegal/dgx-spark-deepseek-v41).
- [Research hub](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload) · [Hugging Face collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e).

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
