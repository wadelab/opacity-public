# Honest-LoRA runner: operator handoff

**27 October 2026: first execution.** The runner was supplied unexecuted
earlier the same day. It has now been run for the first time: the CPU
suite on d2-1 and Viking, the runtime image and environment built and
locked, the site configured, and the three GPU stages run one at a time.
Each passed on its first attempt.
Running it for the first time turned up the problems listed below. Each now
has a fix and a test. Measured results are in
[viking-lora-pilot-results.md](viking-lora-pilot-results.md). The design and
pilot scope documents (not in this public release) explain the experiment; this document is the command interface as it works.

## What the runner does

`opacity.lora_pilot` provides CPU acceptance checks, input preparation, one-stage
submission, an off-budget rehearsal of the launch path, isolated runtime
preflight, a two-update smoke, twenty-update honest training, sequential
fresh-process reload/disable checks and evidence reports. Heavy dependencies
are imported only by commands that need them. Importing the package does not
load a model or submit work.

The recipe uses the characterised bf16 Qwen3.5-27B revision, fp32 rank-8
adapters on the 32 enumerated text full-attention q/v projections (q is
12288 x 5120 because of Qwen3.5's attention output gate, v is 1024 x 5120),
and token-weighted accumulation over four examples. The 64 training and 16
diagnostic examples are synthetic and oracle-scored. Under the real tokenizer
and template every exchange is 163 to 190 tokens, with 17 to 22 supervised
(the answer JSON and `<|im_end|>`). Existing study prompts, labels and
historical runs are not modified.

Stages reserve 30/90/30 minutes on one H100. A locked ledger reserves before
submission, retains ambiguous/failed reservations, and refuses a repeated
stage or more than 150 reserved GPU-minutes. So any failed GPU stage uses up
the budget for this bundle; a retry needs Wade's explicit decision. Training
requires a passing, matching smoke record plus confirmed Slurm completion;
reload likewise requires the pilot. Nothing submits the next stage or retries.

## Faults found on first execution (27 October)

Each was fixed, and each fix has a test.

| Fault | Effect had it reached the H100 | Fix |
|---|---|---|
| `one_group` assigned `weighted_loss = 0.0`, shadowing the `weighted_loss()` function | the smoke crashes at its first microstep, after loading and three diagnostic passes | renamed the local; a tiny real Qwen3.5 now runs the stage functions on CPU |
| The lock check compared the distribution version with `2.13.0+cu130` | every preparation and preflight refuses: the PyPI wheel's metadata says `2.13.0`, and only `torch.__version__` carries `+cu130` | check both, plus `torch.version.cuda == "13.0"` |
| `uv run --offline --no-project <python>` found no interpreter in the image or under the job's replaced HOME | the container and the Slurm wrapper exit before Python starts | set `UV_PYTHON` explicitly |
| Interfaces were read from `/sys/class/net` | always refuses: Apptainer binds the host's sysfs, which lists the host's NICs | `socket.if_nameindex()`, the process's own network namespace; the sysfs view is recorded |
| Read-only was judged from per-mount flags only | refuses: Apptainer's root overlay is `rw` per mount but read-only in its superblock (writes fail with EROFS) | read both flag sets |
| On equal paths the lower mount won, and shadowed mounts were checked | `/tmp` judged by Apptainer's covered tmpfs, not our bind | the upper mount wins; only visible mounts are checked |
| `/dev/shm` was a 270 GB tmpfs | refuses the 8 GiB cap, and was a writable area outside the bound | bound from the attempt filesystem |
| Apptainer writes `/etc/passwd`, `/etc/group`, `/etc/resolv.conf` on a writable session tmpfs | refuses: writable mounts outside the bound | read-only replacements bound over them; `resolv.conf` is the host's, so the DNS probe tests the namespace |
| A failed egress probe was taken as isolation | a probe that also fails outside proves nothing | the launcher first checks both targets answer from the host and refuses otherwise |
| The output volume had to be a site-provisioned mount of at most 8 GiB | Viking offers none without administrators | each attempt gets its own 8 GiB ext4 image, below |
| The credential-file check called `Path.exists()` on `/root/.cache/...` | the gpu_short dry run (job 38051107) crashed with EACCES: the image's `/root` is mode 0700 | each path is recorded as absent, inaccessible or present; only present refuses |
| Only `/usr/bin/nvidia-smi` was an allowed `--nv` bind | the same dry run showed four more NVIDIA tools and four driver data files, all read-only | read-only mounts whose path names `nvidia` are allowed; a writable one still refuses; the policy is tested against the observed table |
| Greedy diagnostics generated without a KV cache | up to 128 full forwards per case, 16 cases, four or more passes per stage: a risk to the 30-minute smoke | cached generation in every compared condition; one forward keeps both diagnostic positions |

Smaller ones: `HOME` is no longer passed with `--env` (Apptainer refuses the
override; `--home` sets it); `check-cpu` now puts `src/` on `PYTHONPATH` (the package
is never installed, because an installed `opacity` distribution would break
the exact package-set match); and `sacct`'s `CANCELLED by <uid>` is read
as `CANCELLED`. Writing the tiny-model test also showed that a base built
with `.to(bfloat16)` differs from one loaded with `from_pretrained(dtype=...)`,
which keeps some modules and buffers in fp32. The runner already loaded every
copy the second way; that load is now one function, `load_model`, used for the
fit, clean reload and adapter reload alike.

## The bounded output filesystem

The handoff required that everything the training runtime can write live on
one filesystem of at most 8 GiB in total. Viking cannot provide that as a
mount without administrators, but it has `fuse2fs`, `/dev/fuse` on every
node checked and Apptainer 1.4.4 in user-namespace mode. So:

- At submission, `mkfs.ext4 -d` makes `attempts/<run-id>/output.img`, an
  8 GiB ext4 image (statvfs capacity 8,455,188,480 bytes), holding `run/`,
  `home/`, `tmp/` and `shm/`. No mount is needed to build it.
- Inside the job the launcher mounts it with `fuse2fs` and binds those
  directories as `/run`, `/home/pilot`, `/tmp`, `/var/tmp` and `/dev/shm`.
  The preflight checks they share one device and its capacity. A prototype
  (job 38050632) wrote to a 64 MiB image until ENOSPC at 54 MB.
- After the container exits the launcher unmounts it, remounts it read-only
  and copies `run/` to `attempts/<run-id>/evidence/`. If a job dies first,
  `report` does the same read-only extraction later. The image is never
  mounted writable outside its job.
- The ledger, lock, `invocation.json`, Slurm records and launcher records
  live beside the image in the private study root (`site.study_root`,
  mode 0700), written only by host code outside the container.
- A predecessor's `evidence/` is bound read-only as `/previous`.

This is a per-run bound, as the design
(section 7) specifies, rather than one volume for the whole study.

**Confirmed by Wade, 27 October**, after the pilot: this per-attempt image
in place of a site-provisioned volume; the egress probe targets (login1's
sshd and `www.york.ac.uk`); allowing read-only mounts that name `nvidia`;
KV-cached greedy generation in the diagnostics; and off-budget rehearsals on
`gpu_short`. These are the runner's settings now, not open questions.

## Runtime as configured on Viking

| Item | Value |
|---|---|
| Image recipe | `configs/lora-pilot-runtime/Dockerfile`: Debian bookworm-slim (pinned digest), Viking's uv 0.12.2 binary, uv-managed CPython 3.12.13, `/opt/venv` from the hashed lock |
| Lock | `configs/lora-pilot-runtime/requirements.lock`: 59 hash-pinned PyPI wheels, compiled with Viking's characterised `.venv` as constraints, so every shared package keeps that version; no `causal-conv1d` or `flash-linear-attention` |
| Runtime lock | `configs/lora-pilot-runtime/runtime-lock-h-v1.json`, read from inside the SIF by `package_versions()`; identical to the d2-1 test environment and the Viking host environment |
| SIF | `opacity-lora-pilot-h-v1.sif`, 2,855,485,440 bytes, sha256 `ca7c659b…c589`, built from the Docker archive by Viking job 38050998 |
| Host environment | `/mnt/scratch/users/<user>/opacity-lora-pilot-20261028/host-venv`, the same lock |
| Site config | `configs/lora-pilot-h-v1-viking.yaml` (copied to the pilot root as `site-h-v1.yaml`) |
| Study root | `/mnt/scratch/users/<user>/opacity-lora-pilot-20261028/study` |
| Backup root | `/mnt/longship/users/<user>/opacity-lora-pilot-backup` (York's longship storage, outside scratch) |
| Egress probes | `10.10.0.21:22` (login1's sshd) and `www.york.ac.uk`; both answer from a compute node outside the container and fail inside it |
| Checkout | `~/git/opacity-lora-pilot-20261028`, a clone of this branch; the existing `~/git/opacity`, its `.venv`, plans and runs are untouched |

The image is rebuilt with:

```bash
# d2-1: build context = Dockerfile, requirements.lock and the uv binary
docker build -t opacity-lora-pilot:h-v1 <context>
docker save opacity-lora-pilot:h-v1 | ssh viking 'cat > <pilot root>/image/opacity-lora-pilot-h-v1.tar'
# Viking, CPU job: apptainer build <...>.sif docker-archive://<...>.tar
```

## Commands

Run from a clean, committed checkout with the host environment. Set
`PILOT_PYTHON` to its interpreter and `PILOT_ROOT` to the pilot root; load
`module load Apptainer` before `submit` and `rehearse`. `UV_PYTHON` must name
the interpreter, because uv otherwise looks for managed Pythons under HOME.

```bash
export UV_PYTHON="$PILOT_PYTHON"
# CPU checks (in a CPU job). Every test must pass; a skip refuses preparation.
uv run --offline --no-project "$PILOT_PYTHON" -m opacity.lora_pilot check-cpu \
  --out "$PILOT_ROOT/cpu-checks-01"

# CPU preparation in a CPU job; this copies and hashes the 54 GB cached base.
uv run --offline --no-project "$PILOT_PYTHON" -m opacity.lora_pilot prepare \
  --config "$PILOT_ROOT/site-h-v1.yaml" \
  --cpu-checks "$PILOT_ROOT/cpu-checks-01" --out "$PILOT_ROOT/bundle-01"

# Off-budget rehearsal on a gpu_short A40: the full launch path, preflight only.
uv run --offline --no-project "$PILOT_PYTHON" -m opacity.lora_pilot rehearse \
  --bundle "$PILOT_ROOT/bundle-01" --run-id rehearsal-01 --owner-note '<coordination>'

# Submit only after checking current jobs and coordinating with their owner.
uv run --offline --no-project "$PILOT_PYTHON" -m opacity.lora_pilot submit \
  --bundle "$PILOT_ROOT/bundle-01" --stage smoke --run-id honest-smoke-01 \
  --owner-note '<record actual coordination>'
```

A fourth stage, `profile` (added 27 October, after the pilot), takes no
predecessor: it loads the base, attaches a fresh adapter, takes two honest
updates so AdamW state is resident, then runs one stress cycle at each of
2,048, 3,072 and 4,096 tokens (`STUDY_PROFILE_LENGTHS`; the bundle's
`stress-study.json`), stopping at the first length without headroom. It
saves nothing. It exists because the study's own training examples run to
3,927 tokens (`scripts/study_lengths.py`). It ran under its own study root
and ledger (`configs/lora-pilot-h-v1-viking-lengths.yaml`) on 27 October:
2,048 tokens fit under the 85% gate and 3,072 did not (results document).

The rehearsal runs the stage launcher unchanged (image, mounts, binds,
boundary check, bundle hash, runtime lock, GPU query and resource monitor)
but its payload is `preflight`: it loads no model and never enters the
ledger. Its stage name cannot satisfy a predecessor gate, and the training
stages refuse any GPU but an H100.

Submission prints the exact job ID. Inspect `slurm.log`, `launcher.json` and
`evidence/{result,preflight,checks}.json` under
`<study_root>/attempts/honest-smoke-01`. A passing smoke must include finite
adapter gradients, a changed effective adapter contribution, unchanged
in-memory base weights, no new diagnostic errors and memory headroom at 512
actual non-padding tokens. An adapter file, decreasing loss or a zero process
exit alone is not sufficient evidence.

After that job has completed successfully, invoke the next stage explicitly:

```bash
uv run --offline --no-project "$PILOT_PYTHON" -m opacity.lora_pilot submit \
  --bundle "$PILOT_ROOT/bundle-01" --stage train --run-id honest-train-01 \
  --predecessor "$PILOT_ROOT/study/attempts/honest-smoke-01" \
  --owner-note '<record actual coordination>'

uv run --offline --no-project "$PILOT_PYTHON" -m opacity.lora_pilot submit \
  --bundle "$PILOT_ROOT/bundle-01" --stage reload --run-id honest-reload-01 \
  --predecessor "$PILOT_ROOT/study/attempts/honest-train-01" \
  --owner-note '<record actual coordination>'
```

The pilot starts from a fresh base/adapter, not smoke weights. It saves only
validated adapter tensors/configuration; the 1,024/2,048-token profiles do
backward accumulation without optimiser updates. Insufficient headroom stops
length escalation and is reported; an OOM or other invariant failure stops
the process and fails the stage. No automatic resubmission or size reduction
follows. Failed attempts retain their output and reservation.
AdamW state remains resident during stress measurements so the memory figure
includes the training optimiser's footprint.

Reload launches clean-base and adapter processes sequentially. Each repeats
preflight, loads only one base, and checks the saved adapted outputs and the
disabled adapter against pre-training references. Strict numeric comparison
rules are fixed before fitting. A mismatch remains a failed instrument check;
the code does not adjust tolerances after observing it.

Once the jobs are terminal, reconcile their exact Slurm records and archive:

```bash
uv run --offline --no-project "$PILOT_PYTHON" -m opacity.lora_pilot report \
  --bundle "$PILOT_ROOT/bundle-01" \
  --attempt "$PILOT_ROOT/study/attempts/honest-smoke-01" \
  --attempt "$PILOT_ROOT/study/attempts/honest-train-01" \
  --attempt "$PILOT_ROOT/study/attempts/honest-reload-01" \
  --out "$PILOT_ROOT/report-01" --backup
```

Reports are created exclusively at new paths. `--backup` copies the host
records and the extracted evidence once to the configured private durable
root, verifies hashes and writes a receipt into the new report. The 8 GiB
image, mount points and disposable caches are excluded. Base weights are
identified by the frozen manifest and retained separately; they are not copied
into every attempt archive. A repeated backup destination is refused.

## What the CPU tests do and do not show

`tests/test_lora_pilot.py` runs in the locked environment with no downloads.
Besides data, masking, gates, manifests, mount-table parsing, container
arguments, the real bounded image and a stub scheduler, one test builds a tiny
Qwen3.5 (the real classes, full attention every fourth layer, bf16 base) and
drives the runner's own `targets`, `meta_targets`, `attach`, `diagnostics`,
`train_updates`, `save_adapter`, a fresh `load_model`, adapter reload and
disable, with the strict comparison rule. That is the test that found the
crash. Another holds the mount table seen inside the container on a Viking
GPU node (`tests/fixtures/lora_pilot_viking_gpu_mounts.json`) and requires the
policy to accept exactly that. The tests cannot show memory, speed or
numerical behaviour on the 27B or on CUDA. Only the H100 smoke can.

`--nv` binds every GPU's device node on the host, not only the allocated
one (three on the A40 node). Only the allocated GPU is visible to CUDA: the
preflight requires `torch.cuda.device_count() == 1` under Slurm's
`CUDA_VISIBLE_DEVICES`. Whether Slurm's device cgroup also blocks opening
the others was not tested.

API references consulted:
[PEFT model API](https://huggingface.co/docs/peft/v0.21.0/package_reference/peft_model),
[Qwen Transformers API](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[Apptainer exec flags](https://apptainer.org/docs/user/latest/cli/apptainer_exec.html).
