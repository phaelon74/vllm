# Installing the fidelity suite

Read this before [`README.md`](README.md). The README is the operating manual and
assumes a working install; this document is how you get one, and it opens with the
thing that surprises people.

## 1. This is a fork of vLLM, not a package you add to vLLM

There is no `pip install fidelity`. The suite is `fidelity/` plus a scorer in
`examples/offline_inference/score_mode_kld.py`, and both import machinery that
exists only in this branch. `vllm/v1/sample/kld.py` — 1,594 lines carrying the
runtime manifest, the numerics digest, the model inspection hooks, the KV-cache
policy, and the exact-repeat certification set — was added by this fork's first
commit and has no upstream counterpart. Neither do the routing-replay hooks that
BxQ depends on, which thread `enable_return_routed_experts` from
`SamplingParams` through the scheduler, the request, the engine, the model
runner, and back out through the output path.

Measured against the upstream commit this branch was cut from, the fork changes:

| Surface | Extent |
|---|---|
| Compiled sources | 3 files, all in `csrc/libtorch_stable/moe/marlin_moe_wna16/` |
| New Python modules | `vllm/v1/sample/kld.py`, `fused_moe/forced_routing.py`, `fused_moe/router/base_router.py`, `quantization/utils/nvfp4_activation_scales.py`, five TRT-LLM MoE expert backends |
| Modified Python | ~50 files under `vllm/`, concentrated in `fused_moe/`, `v1/worker/`, and `v1/engine/` |
| Additions outside `vllm/` | three scorers under `examples/offline_inference/`, seventeen probes under `scripts/`, and `fidelity/` itself |

Two consequences worth internalizing. Upstream vLLM cannot run this suite, so
"install the fidelity suite onto vLLM" is not a thing you can do — you install
*this* vLLM. And because `vllm/**/*.py` is inside the numerics digest (§6), an
install built from a different commit of this fork measures under a different
binding, which the pipeline detects rather than trusts.

Section 8 records what would have to move to make a plugin possible, so that work
can start from a written surface rather than a rediscovery.

## 2. What you need

**A CUDA GPU whose architecture has a certified kernel path.** This is stricter
than "vLLM runs here." Exact-repeat certification is a property of specific expert
implementations on specific architectures, and a candidate whose loader builds an
uncertified kernel is refused before its first forward pass rather than scored
with a caveat. The published results were taken on SM120 Blackwell hardware; read
`runtime_binding.gpu_names` out of any published `report.json` for the exact
identity. The published campaigns score 30B-class candidates at tensor parallel 2
on 96 GiB cards, and the planned TP follows from the larger of the candidate and
its reference, so a smaller card means a wider split rather than a refusal.

**CUDA toolkit with `nvcc` on `PATH`**, because §3 compiles one extension. CMake
finds the compiler through `CMAKE_CUDA_COMPILER`; `which nvcc` is the quick check.

**`uv`, `cmake`, `ninja`, and ideally `ccache`.** The suite never calls system
`python3` or bare `pip`.

**Disk.** The reference tensors for one model family are a substantial cache and
one worth keeping, since every candidate in the family scores against them. Add one
candidate checkpoint at a time under `fetch: lease`, or all of them under
`fetch: upfront`; `sweep.py` reports what the published library is not using and
reclaims it.

**A Hugging Face token, and two dataset terms accepted.** `HF_TOKEN` must belong
to an account that has accepted `bigcode/starcoderdata` and `Idavidrein/gpqa` on
the Hub, or suite building reports them unreachable. Publishing additionally
needs write scope.

## 3. Install

Clone the fork and pin it. If you are reproducing a published artifact, pin to
that artifact's own commit, which every report records as
`runtime_binding.vllm_commit`:

```bash
git clone https://github.com/phaelon74/vllm.git
cd vllm
git checkout feature/glm53-kld-determinism   # or the commit you are reproducing
```

Build the environment. The first install deliberately uses upstream's precompiled
extensions, so you pay for one kernel rather than for all of them:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install 'huggingface_hub[cli]' datasets matplotlib
```

That leaves you with every compiled extension from upstream's wheel, including a
`_moe_C_stable_libtorch` built from *unpatched* Marlin MoE sources. Replace that
one extension and nothing else:

```bash
python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target _moe_C_stable_libtorch
cmake --install cmake-build-release --prefix "$PWD" \
  --component _moe_C_stable_libtorch
```

`cmake-build-release` is the preset's `binaryDir`; use whatever yours reports.
The component name is not a convention this document invented — `cmake/utils.cmake`
gives every extension its own install component named after the target, so
installing one by name is a supported operation rather than a copy by hand.
`--prefix "$PWD"` matters: the target's destination is `vllm`, so the prefix must
be the repo root for the library to land where the editable install reads it.

Both stages are what `bootstrap.sh` does for you, along with fetching the
checkpoints a campaign names:

```bash
bash fidelity/bootstrap.sh fidelity/campaigns/qwen3.6-35b-a3b.json
```

## 4. Why the kernel rebuild is not optional

Read this section even if you plan to skip the rebuild, because skipping it fails
quietly rather than loudly.

**What the patch does.** It is the MoE counterpart of upstream's batch-invariant
WNA16 reduction (vllm-project/vllm#46639). Under `VLLM_BATCH_INVARIANT=1`, which
the scorer pins for itself, it threads a `use_full_k` flag into the Marlin MoE
kernel so each block reduces the whole of K, and it stops the thread-config
heuristic from varying with `prob_m`. Both changes remove a dependence on how
tokens happened to be grouped into a batch. The flag is
`vllm::vllm_is_batch_invariant()`, so with batch invariance off the patched kernel
is the upstream kernel; this is not a general behavior change and it is not a
performance change.

**Why it must be in the build.** Exact-repeat certification is keyed on the
kernel's *class name* — `_EXACT_REPEAT_CERTIFIED_EXPERTS` in
`vllm/v1/sample/kld.py` lists `MarlinExperts`, and that name is what a build
reports whether or not the patch is compiled in. So an unpatched extension is
certified by the pipeline and then produces arithmetic that moves with batching.
Nothing refuses it. The only thing that would notice is
`compiled_extensions_sha256` disagreeing with a published report, which helps only
if you are comparing against one. In the two published routed families this
affects four of nine candidates each — every INT4, AWQ, and GPTQ MoE export.

If you have an install of unknown provenance, this is the check:

```bash
python -c "
import vllm, glob, os
print(glob.glob(os.path.join(os.path.dirname(vllm.__file__),
                             '_moe_C_stable_libtorch*')))"
ls -l cmake-build-release/_moe_C_stable_libtorch* 2>/dev/null \
  || echo "no local build of this extension exists"
```

A `_moe_C_stable_libtorch` in the package with no local build behind it came from
upstream's wheel and does not carry the patch.

**What the rebuild costs.** More than three files: `kernel.h` and
`marlin_template.h` are included by every generated Marlin MoE translation unit,
so touching them invalidates all of them. Set `TORCH_CUDA_ARCH_LIST` to your own
architecture — `12.0` for SM120, `9.0a` for Hopper — and you compile one
architecture's worth rather than every architecture CMake would otherwise pick. It
remains a small fraction of a full build, which additionally compiles CUTLASS,
Machete, and every `scaled_mm` variant into `_C_stable_libtorch`.

**Why mixing builds is sound here.** The patch lives under
`csrc/libtorch_stable/`, the stable-ABI tree, which exists precisely so an
extension can be built separately from the torch it loads against. The resulting
install still gets its own `compiled_extensions_sha256`, so it never silently
claims comparability with numbers taken against a different build — see §6.

## 5. Verify the install

Every module that can be checked offline can be checked before you spend a GPU
hour. The invocation is unfortunately not uniform: most take a flag, two take a
subcommand.

```bash
python fidelity/campaign.py --selftest      # also exercises the laws
python fidelity/qdq.py --selftest
python fidelity/publish.py --selftest
python fidelity/curate.py --selftest
python fidelity/provenance.py --selftest
python fidelity/strata.py --selftest
python fidelity/artifact.py selftest
python fidelity/suite.py selftest
```

`compliance.py`, `sweep.py`, `tails.py`, and `redaction.py` have no selftest of
their own; the laws are exercised through `campaign.py --selftest`, which is why
that one is worth running even when you have changed nothing.

Then prove the fork's own runtime imports and can describe itself:

```bash
python -c "
from vllm.v1.sample.kld import capture_runtime_manifest
m = capture_runtime_manifest()
for key in ('numerics_digest', 'compiled_extensions_sha256', 'torch',
            'flashinfer', 'driver', 'gpu_names'):
    print(f'{key:28} {m.get(key)}')
"
```

Those six fields are exactly what a measurement is bound to. If
`numerics_digest` is `None`, the checkout is not where the runtime thinks it is;
set `KLD_REPO_ROOT`. Computing it walks every `.py` under `vllm/`, so a few
seconds is normal and a minute means you are on a network filesystem.

The end-to-end check is one window on one routed candidate, which loads a real
checkpoint, builds real kernels, and reports which ones it built:

```bash
python fidelity/campaign.py smoke \
  --config fidelity/campaigns/qwen3.6-35b-a3b.json \
  --only-candidate Qwen3.6-35B-A3B-NVFP4
```

A pass prints the expert backend it chose, `SMOKE PASS`, and a natural control of
`max=0.000e+00`. Anything other than exactly zero there means the install is not
deterministic and no number it produces is worth publishing.

## 6. What you can reproduce

There are two honest answers, and which one applies to you is a question about
your hardware rather than about your install.

### Mode A: your own numbers, which is the default and is not a lesser result

Score whatever you like on whatever you have. Every candidate you measure is
measured by the identical code path, which is the property that makes a comparison
between two of *your* candidates mean something. Your absolute KLDs will differ
from the published ones, and that is correct rather than a bug: a measured number
is bound to the runtime that produced it, and `gpu_names` alone is enough to make
your binding a different binding. The comparability key printed with every
leaderboard group exists to stop the accidental cross-comparison.

What does not automatically transfer is the *ranking*. All your candidates being
measured identically makes your ordering valid for your deployment; whether it
agrees with ours is an empirical question this pipeline does not promise an answer
to, and finding out is a more interesting result than assuming.

Two things you should still take from the published artifact rather than rebuild.
Use the **published token suite**, which ships in each family's dataset repo under
`suite/` — token IDs are not portable across tokenizers, and minting your own
suite means your numbers are not comparable to the published ones even in
principle. Do not expect to reuse the **published reference capture**: a capture
is bound to the runtime that produced it, so a different numerics digest or a
different compiled extension makes the scorer recapture locally, which is a
forward pass of the reference over the whole suite. Budget for that once per
family.

### Mode B: bit-matching the published numbers

Achievable if you have the same GPU model and can produce the same build. Six
fields have to agree, and `runtime_binding_view` in `vllm/v1/sample/kld.py` is the
authoritative list rather than this table. Check before spending GPU time, against
any published `report.json`:

```bash
python - path/to/published/report.json <<'PY'
import json, sys
from vllm.v1.sample.kld import capture_runtime_manifest, runtime_binding_view
theirs = runtime_binding_view(json.load(open(sys.argv[1])).get("runtime_binding"))
mine = runtime_binding_view(capture_runtime_manifest())
for key in theirs:
    if theirs[key] == mine[key]:
        print(f"ok    {key}")
    else:
        print(f"DIFF  {key}\n      published {theirs[key]!r}\n"
              f"      local     {mine[key]!r}")
PY
```

What each difference means:

| Field | Cause | Fixable by |
|---|---|---|
| `numerics_digest` | different commit, or local edits to `vllm/**/*.py` or the scorer | `git checkout` the published `vllm_commit`, and commit or discard local edits |
| `compiled_extensions_sha256` | a different build of the extensions | same toolkit, same arch flags, same precompiled wheel — the hardest to match, and often the one that cannot be |
| `torch`, `flashinfer` | version drift | pin to what the published `environment/` records |
| `driver` | host NVIDIA driver | update or downgrade the host |
| `gpu_names` | hardware | nothing; this is not a configuration |

If all six agree, a rescore should land on the published number and the pipeline
will treat your report as current. If they do not, Mode A is your mode, and the
numbers you get are real numbers about your deployment.

## 7. Your first campaign

[`README.md`](README.md) is the operating manual from here. The shortest path that
produces something publishable:

```bash
# 1. A config. Interactive, or driven entirely by flags.
python fidelity/curate.py --base Qwen/Qwen3.6-35B-A3B \
  --suite-dir /path/to/suite --out fidelity/campaigns/mine.json

# 2. Environment, install check, and checkpoint fetch.
bash fidelity/bootstrap.sh fidelity/campaigns/mine.json

# 3. One window per routed candidate, before committing to the sweep.
python fidelity/campaign.py smoke --config fidelity/campaigns/mine.json

# 4. The sweep, then the artifact.
python fidelity/campaign.py all --config fidelity/campaigns/mine.json
```

Run step 3. It costs one window per candidate and it is where a checkpoint that
cannot be scored announces itself, which is much cheaper than discovering it six
hours into step 4. If your reference model is new enough that step 3 fails in a
way this document has not prepared you for, [`EXTENDING.md`](EXTENDING.md) is the
triage order.

Minting a suite for a tokenizer nobody has minted one for is its own procedure,
documented under "Building the token suite" in the README. You need it whenever
your reference model's vocabulary differs from any published suite's.

## 8. What a plugin would require

This section exists so a future attempt to make the suite installable against
upstream vLLM starts from a written surface. It is not a plan and nothing here is
implemented.

**Separable today.** `vllm/v1/sample/kld.py` is a library: it computes digests,
captures the runtime manifest, and inspects a loaded model through
`LLM.apply_model`, which is public API. The scorers under
`examples/offline_inference/` and the whole of `fidelity/` are ordinary Python
that imports it. This group could live out of tree with no changes to vLLM.

**The real design work: routing replay.** BxQ needs the student to return the
expert IDs it selected, and needs to be forced to use the teacher's. That threads
`enable_return_routed_experts` through `SamplingParams`, `Request`, the scheduler
output, the model runner, the MoE routers, and the output path — roughly twenty
files whose only common property is that they sit on the request lifecycle. A
plugin would need either an upstream capability for returning and forcing routed
experts, or a documented MoE-layer hook plus a side channel for per-layer
selections. Everything else in the suite works without it; only BxQ and Law 14 do
not.

**Cannot be a plugin at all: the fixes that change numbers.** The Marlin MoE
canonical-order and full-K patch (three compiled files and three Python ones), the
NVFP4 per-expert activation-scale handling, the GDN determinism work, and the
full-precision prompt-logprob path are corrections to vLLM's arithmetic. A plugin
cannot supply them; they belong upstream as pull requests, and
[`PR54444-Adendum.md`](PR54444-Adendum.md) is the argument for one of them.

The practical order, if this is ever attempted: land the arithmetic fixes
upstream, then negotiate a routed-experts capability, then extract the library.
Doing it in the other order produces a plugin that installs cleanly and measures
the wrong thing.
