# Inference CPU/CUDA and source-authority audit — 2026-09-28

## Repository role

This repository is an inference/data-processing desktop application, not a model-training repository. Its retained Transformer models are pretrained Hugging Face classifiers, with additional VADER, spaCy, Ollama, CSV/MySQL and Tkinter processing. The correct account-wide training-control behavior is therefore to **audit and fail closed if authored training appears**, not to invent optimizer jobs or synthetic experiments.

## Native device defects found

The two retained sentiment application copies previously selected GPUs independently of the account scheduler:

- `DataProcessor.getBestGpu()` enumerated `torch.cuda` directly;
- Hugging Face pipelines ignored the already-selected `device` object and called the GPU selector again;
- fallback logic trusted `torch.cuda.is_available()` without executing a CUDA operation;
- generic exception handlers could convert GPU initialization or inference failure into a normal return or partial/zero emotion output.

This meant a centrally CPU-admitted child could still touch CUDA, while a GPU-admitted child could fail CUDA work without reliably surfacing failure to a supervising scheduler.

## Implemented device policy

`inference_device_policy.py` is now the single inference placement boundary. It:

- recognizes `CPU_ONLY`, `TRAINING_CONTROL_CPU_ONLY`,
  `OPF_ADP_DISABLE_GPU_ACCELERATORS`,
  `TRAINING_CONTROL_BACKEND=cpu`, and empty/-1
  `CUDA_VISIBLE_DEVICES`;
- rejects contradictory central CPU and GPU admission;
- treats CUDA indices as **local masked indices**, never as physical IDs;
- requires a Torch allocation, tensor operation and synchronization on each
  candidate local device before selecting it;
- chooses among usable local GPUs using memory information;
- permits standalone CPU fallback when CUDA is not usable;
- fails closed when `TRAINING_CONTROL_BACKEND=gpu` has no usable CUDA device;
- converts the resolved Torch device to Hugging Face Pipeline's
  `-1`/integer convention.

Both `Sentiment Analysis.py` and
`Sentiment Analysis (Stable last code).py` use that one resolved device
for all three retained Hugging Face classifiers. They no longer contain
direct `torch.cuda.*` device selection. GPU-admitted model setup and
inference exceptions are re-raised through every previously swallowing
boundary so a failed GPU workload cannot be reported as a successful
partial analysis.

## Inference-only authority corrections

The previous authority parsed only `.py` even though the central profile
claimed `*.txt` coverage and `Models.txt` contains retained Python-like
model examples. The authority now:

- requires the canonical application, local LLM, inference-device and
  application-configuration sources to exist;
- parses every retained Python source;
- scans retained TXT/Markdown for code-signature training primitives such as
  optimizer construction, `.fit(`, `.backward(`, Trainer and
  TrainingArguments while avoiding generic prose false positives;
- fails closed if those signatures or Python training primitives appear.

The v41 controller's broad scientific-registry heuristics also classified
ordinary inference variables such as dataset objects, a local user-agent
list, model response text and derived sentiment-output dictionaries as
training registries. The profile now carries exact, reasoned non-training
coverage/exemptions for only those known surfaces. The general
`require_workload_surface_accounting` heuristic is disabled because it
cannot distinguish these application variables from trainable registries,
while semantic training-surface, retained-source reachability, source
contracts, training contracts, dynamic registry accounting and registry
member accounting remain enabled. Native/exact checkpoint resume is no
longer required for the non-training audit job; exact resume remains
required for any genuine training job discovered in the future.

## Public credential remediation

Both public application scripts contained a literal MySQL password. Current
source now calls `application_config.database_config_from_env()`; database
authentication requires `SENTIMENT_DB_PASSWORD`. Host, user and database
may also be supplied through `SENTIMENT_DB_HOST`,
`SENTIMENT_DB_USER`, and `SENTIMENT_DB_NAME`.

Removing a secret from current source does **not** revoke a value already
present in Git history. The old database credential must be rotated outside
GitHub. Historical commits are not rewritten by this audit.

## Executed evidence

GitHub Actions run `36396787122` executed the focused policy/source suite
and the strict v41 audit:

- 14 focused tests passed;
- `coverage_ok: true`;
- dynamic scientific registry accounting passed;
- registry-member accounting passed;
- retained training closure passed;
- semantic training-surface accounting passed;
- source contracts passed;
- training contracts passed.

The separate estate certificate run `36396787097` also passed. A later
configuration-secret regression was wired into the same workflow and must
remain green on subsequent heads.

These are CPU/fake-CUDA/source tests. They are **not** physical NVIDIA CUDA
execution, Hugging Face model download/inference parity, multi-GPU behavior,
database integration, Ollama integration or GUI acceptance.

## Remaining repository closure work

- run the retained Hugging Face models on real supported CUDA and compare
  outputs with CPU within a declared tolerance;
- exercise masked multi-GPU selection and CUDA allocation/runtime failure;
- lock and validate the Python dependency environment rather than relying on
  floating `pip install` versions;
- test actual model downloads/cache/offline behavior and the required
  `en_core_web_sm`/NLTK assets;
- integration-test MySQL and Ollama without embedding credentials;
- reconcile the duplicated application source so one canonical implementation
  cannot silently drift from the other;
- test the Tkinter GUI and end-to-end CSV generation on a supported desktop
  environment.


## 2026-10-01 continuation — dataframe/spaCy acceleration and scanner correction

The inference source audit itself exposed a false positive after this document
described forbidden signatures such as `.fit(` and `.backward(` in prose.
The authority now treats plain `.txt` as retained code-like material but
scans Markdown training signatures only inside fenced code blocks. This keeps
`Models.txt` fail-closed while preventing documentation prose from being
misclassified as executable training. Tests cover both prose and fenced-code
cases.

The retained applications now also honor the account-wide "GPU-first where an
actual compatible library exists" requirement beyond Torch model inference:

- before importing pandas, `enable_optional_dataframe_acceleration()` checks
  central CPU admission, probes an independently usable CuPy CUDA device with
  allocation + operation + synchronization, and installs `cudf.pandas` when
  available; otherwise it truthfully keeps pandas;
- after importing spaCy but before loading `en_core_web_sm`,
  `configure_optional_spacy_gpu()` calls `spacy.prefer_gpu()` only outside
  CPU admission. spaCy GPU unavailability is non-fatal because Transformer
  placement has its own Torch CUDA authority;
- CPU-only workers never import/probe the optional CuPy/cuDF stack through
  these helpers;
- the two retained application copies are asserted byte-identical so the
  historical "Stable last code" copy cannot silently diverge from the active
  script.

The universal controller conservatively classified a test-local variable named
`model_load` as a scientific registry. The test variable was renamed; no
controller check, production path or registry-accounting requirement was
disabled.

Executed evidence on the current source:
- strict training-control run `36879845905`: **success**, including focused
  device/source tests and the exhaustive v41 controller audit;
- estate certificate run `36879845917`: **success**.

These remain CPU/fake-CUDA/source-contract tests. Physical NVIDIA execution,
actual cuDF/spaCy GPU behavior, Hugging Face CPU/GPU output parity, dependency
locking, model/cache/offline integration, MySQL/Ollama integration and GUI
acceptance remain separate open qualifications.
