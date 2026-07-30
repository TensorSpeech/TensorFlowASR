# Train a language model on Kaggle

Terraform that generates a Kaggle notebook, pushes it, waits for it to run, and downloads the
weights. The notebook runs [`train_lm`](../../../tensorflow_asr/scripts/train_lm.py), which fits
the language models beam search fuses in. See [decoders.md](../../../docs/decoders.md) for what
those models do and [training.md](../../../docs/tutorials/training.md) for the plain CLI version.

## What this is, honestly

**There is no Kaggle Terraform provider** — the registry has none and no `terraform-provider-kaggle`
exists. So this is not Terraform managing a Kaggle resource. It renders files from your tfvars and
shells out to the Kaggle CLI.

What you still get from Terraform: one declarative place for every knob, a plan that shows what
changed before anything is pushed, and a real `terraform destroy` (the CLI does have
`kernels delete`, so the notebook is genuinely removed).

What you do not get: drift detection. Terraform cannot tell you someone edited the notebook on
Kaggle. `terraform apply` overwrites it either way.

## Requirements

- Terraform >= 1.5
- The Kaggle CLI: `pip install kaggle`
- `python3` on PATH (used to write the credentials file safely)

## Use it

```bash
cd terraform/kaggle/lm
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars

terraform init
terraform apply
```

`apply` renders `build/train_lm.ipynb` and pushes it. By default it returns straight away —
`terraform output status_command` gives you the command to check on the run, and `logs_command`
fetches the output once it finishes. Set `wait_for_completion = true` to make `apply` block until
the run ends and download the result to `output/` itself.

Then point `test` at the weights:

```bash
tensorflow_asr test ... --internal-lm-h5=output/internal_lm.weights.h5
```

`terraform output test_command` prints this with your paths filled in.

## Read this before a long run

**Kaggle caps a session at roughly 9–12 hours**, with a weekly GPU quota on top. The external LM
setups in the papers use the LibriSpeech LM corpus, about 800M words — that will not finish in one
session, and Kaggle kills the kernel rather than saving what it had. Start with
`lm_dataset_config.max_lines` (in the config) and `epochs` small enough to finish, confirm the
weights are usable, and scale up from there.

**A run that is cut short loses everything by default.** `/kaggle/working` starts empty on every
run, and `train_lm` only writes the weights once `fit` returns — so a kernel killed at the session
cap leaves nothing behind. Set `kaggle_model_handle` and the training state is checked into a
Kaggle model after each epoch and pulled back at the start of the next run, so re-pushing continues
rather than restarts.

**Setting `kaggle_model_handle` puts your API token in the notebook.** A Kaggle notebook has no
write credentials of its own: the implicit auth is a read-only data proxy that serves public models
and cannot create a model version at all. So the generated notebook exports `KAGGLE_USERNAME` and
`KAGGLE_KEY` from the tfvars, in a cell that appears only when the handle is set.

Know what that costs. The token ends up in three more places than the push credentials do:

| Where | Mitigation |
| --- | --- |
| `terraform.tfstate` | gitignored; Terraform prints it as `(sensitive value)` |
| `build/notebook.ipynb` | gitignored; written `0600` |
| the notebook Kaggle stores | keep `is_private = true` |

If any of those get out, rotate at <https://www.kaggle.com/settings>.

The clean alternative is a **Kaggle Secret** (Add-ons → Secrets on the notebook), which keeps the
token off disk entirely — but it cannot be automated, so it is a manual step on every new kernel.
There is no API for it at all: the kernel push request carries 21 fields and none is a secret, the
whole `kaggle` package contains no occurrence of the string, and the generated SDK exposes 22
services with no secrets RPC among them. Kaggle's own UI drives an internal endpoint that
authenticates with a browser session rather than an API token. To switch, replace the two
`os.environ[...]` lines in `templates/train_lm.py.tftpl` with `UserSecretsClient().get_secret(...)`
and drop `kaggle_username`/`kaggle_key` from `notebook.tf`.

The same cell sets `DISABLE_KAGGLE_CACHE=1`. Without it, `model_download` goes through the notebook
data proxy, which answers a model that does not exist yet — every first run — with a generic
`{"errors":["Internal error"],"error":{"code":13}}`. That surfaces as a `kagglehub.BackendError`,
which the restore path does not recognise (it handles the REST 404), so it escapes and kills the run
before the first step. The REST client returns a real 404, understood as "nothing to restore yet".

This is why `wait_for_completion` defaults to `false`: turning it on ties up a terminal for the
whole run, and interrupting a blocked `apply` leaves the push recorded in state while you have no
idea what the kernel is doing. Waiting is most useful for a short run you want to fetch
automatically, or from CI.

## Where the API key goes

You chose to keep it in `terraform.tfvars`. Consequences worth knowing:

- `terraform.tfvars` is **gitignored** by `.gitignore` in this directory. Check `git status`
  before committing anyway.
- The key is **not** in `terraform.tfstate`. It only travels through provisioner `environment`
  blocks, which Terraform does not persist. Nothing in this module puts it in a resource attribute.
- It **is** written to `build/kaggle.json` (mode 0600, gitignored). It has to be: `terraform destroy`
  needs to authenticate, and destroy-time provisioners may only read `self` — Terraform rejects
  `var.kaggle_key` there. This is the same file the Kaggle CLI keeps in `~/.kaggle` anyway.

To keep the key off disk entirely, drop it from the tfvars and export it instead:

```bash
export TF_VAR_kaggle_key=...
```

## How it fits together

```
terraform.tfvars
      |
      v
notebook.tf     renders templates/train_lm.py.tftpl, splits it on `# %%`,
                builds the .ipynb with jsonencode
      |
      v
build/train_lm.ipynb + build/kernel-metadata.json
      |
      v
scripts/push_and_wait.sh     kaggle kernels push -> poll status -> kernels output
      |
      v
output/                      the .h5, plus the run log
```

Two `terraform_data` resources, split on purpose:

- `terraform_data.kernel` owns the kernel's *existence* and has no triggers. It only ever runs
  its destroy provisioner.
- `terraform_data.push` re-runs whenever the notebook or the settings change.

If they were one resource, changing the notebook would replace it, and replacing runs destroy
first — so every edit would delete the kernel and its version history before pushing it back.

## TPU

Supported, and **untested on real hardware**. Everything below was verified with a
`MirroredStrategy` over virtual CPU devices, which shares the strategy machinery but not the XLA
compiler. Read this before spending a session.

```hcl
accelerator = "TpuV5E8"
device_type = "tpu"
tpu_address = "local"
tpu_vm      = true
```

Hardware is picked by an accelerator ID, written to `machine_shape` and passed as
`kaggle kernels push --accelerator`. Kaggle's older `enable_gpu` / `enable_tpu` booleans are not
exposed by this module: they cannot say *which* GPU, and `enable_tpu` maps to the v3-8 that Kaggle
has phased out, so a TPU asked for that way comes up with nothing attached. The metadata booleans
are derived from the ID's prefix.

Valid IDs as of Feb 2026, from the
[CLI docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) — some are restricted to
competition participants or admins:

```
NvidiaTeslaP100  NvidiaTeslaT4  NvidiaTeslaT4Highmem  NvidiaTeslaA100
NvidiaL4  NvidiaL4X1  NvidiaH100  NvidiaRtxPro6000
TpuV38  Tpu1VmV38  TpuV5E8  TpuV6E8
```

`accelerator` is what Kaggle attaches; `device_type` is what TensorFlow is told to use. They are
separate knobs and disagreeing is silent — the kernel boots, installs the wrong extra and trains on
the CPU — so a precondition rejects the mismatch at plan time.

**Sizing.** A v5e-8 is 8 chips x 16 GB HBM. `bs` is per replica, so `bs = 32` is a global batch of
256 — 8x what the same number means on one GPU, so scale `learning_rate` up (~2.5e-3 by sqrt
scaling) or the run does one-eighth the updates per epoch at the old rate. Weights plus Adam are
only ~0.7 GB per core; activations dominate, and they scale with `bs x max_length`. Since TPU pads
every sequence to `lm_dataset_config.max_length` (which it requires — the run stops without one),
lowering it to about the 99th percentile of your token lengths cuts memory *and* compute, which is
a bigger lever than `bs`. Out-of-memory shows up at compile or the first step, so bisecting `bs`
costs minutes.

**`bs` is per replica.** A v3-8 has 8 cores, so `bs = 32` is a global batch of 256. `steps_per_epoch`
counts steps at that global batch, so switching to a TPU without dividing it by 8 turns one pass
into eight.

**Two pipeline changes are forced on TPU**, because XLA compiles per input shape and the default
pipeline pads each batch to its own longest sequence — a new shape almost every step, so the run
would recompile rather than train. Sequences are padded to `max_length`, and the short final batch
is dropped. Both cost something: short sequences carry padding out to `max_length`, and up to
`bs x replicas - 1` sequences are skipped per pass (different ones each pass, since the shuffle is
upstream). Neither is worth paying on a GPU, so neither happens there.

**`spx` stays at 1.** `steps_per_execution` above 1 is the usual TPU throughput lever, but on
keras 3 / tensorflow 2.19 any value above 1 combined with a distribution strategy fails during
`fit` with an `InvalidArgumentError` about a `while/cond` placeholder. Reproduced with a plain
Dense model and with the stock Keras loss, so it is not this repository. Whether `TPUStrategy`
shares the fault is unknown. Raise it if you like, but check a few steps run first.

**The TensorFlow build is swapped, not added.** There is no `tpu` extra, and there cannot be:
`tensorflow` is a base dependency and `tensorflow-tpu` ships its own `tensorflow` distribution, so
an extra adding it resolves to `tensorflow==2.19.1` *and* `tensorflow-tpu==2.19.1` and whichever
lands last wins. The notebook installs the project normally, then runs
[`scripts/install_tpu.sh`](../../../scripts/install_tpu.sh) from the clone:

```
uv pip uninstall --system --python <py> tensorflow
uv pip install   --system --python <py> tensorflow-tpu~=<minor of the tensorflow it replaced>
```

The version is read off the `tensorflow` being replaced, so it tracks whatever `uv sync` resolved
and cannot drift from the pin in `pyproject.toml`. Only the minor is pinned, because the patch
levels do not line up between the two distributions — `tensorflow` has a 2.19.0 where
`tensorflow-tpu` jumps from 2.19.0rc0 straight to 2.19.1. Set `TPU_PACKAGE` in the environment to
force something else.

The logic lives in the repository rather than in this template so the same command works outside a
notebook — `uv sync && ./scripts/install_tpu.sh` locally.

`tensorflow-text` keeps working — the module it imports is still there, now provided by
`tensorflow-tpu`.

`setup.sh` also passes `-f https://storage.googleapis.com/libtpu-tf-releases/index.html`. That was
needed for the 2.18 line, whose `libtpu~=2.18.0` existed only on Google's index — that index still
tops out at 2.18. From 2.19 the dependency is `libtpu==0.0.10`, which is on PyPI, and
`tensorflow-tpu~=2.19.0` resolves for linux with no extra index. Pinning back to 2.18 means
adding that `--find-links` to the install command in `scripts/install_tpu.sh`.

**An LSTM is a poor fit for a TPU.** It is sequential over timesteps, which is what TPUs are worst
at. Benchmark a few hundred steps against the P100 before committing — the GPU may simply win.

## Things that will bite you

**The branch has to be pushed, and it cannot be `main`.** The notebook clones over the network, so
it sees GitHub, not your working tree. `train_lm` and `tensorflow_asr/models/lm` are not on `main`
yet — `repo_ref` defaults to `feat/beamsearch` for that reason. Commit and push before applying, or
the run fails at the train step with a missing command.

**The config is jinja2 and its imports resolve against `repodir`.** `load_yaml` builds a
`FileSystemLoader` over it, so `{% import "examples/datasets/..." %}` needs the repository on disk.
That is why the notebook clones the repo and passes `--repodir`, and why installing the package
alone is not enough. It also means `config_file` (a config from your machine) still works — the
imports resolve against the clone, not against the config's own directory.

**The repo is cloned to `/tmp`, not `/kaggle/working`.** Everything under `/kaggle/working` is
collected as notebook output, so cloning there would drag the whole repository into every
`kaggle kernels output` download.

**`extra_args` is where jinja variables go.** The example configs interpolate `{{ vocabprefix }}`,
`{{ vocabsize }}` and friends; `train_lm` forwards unknown CLI flags into the jinja context. An
undefined jinja variable renders as an empty string instead of failing, so a forgotten one usually
surfaces as a path with a hole in it rather than an error.

**Installing replaces the image's TensorFlow** and takes a while. The notebook bootstraps `uv`
and runs `uv pip install --system`, which resolves the same dependencies as pip but much faster —
and `pyproject.toml` pins `tensorflow~=2.19.0`, so a differing image version gets replaced.

**The accelerator picks the extra.** An `Nvidia*` ID installs `-e .[cuda]`, which is what pulls
`tensorflow[and-cuda]` and the twelve nvidia wheels; a `Tpu*` ID runs `scripts/install_tpu.sh`
instead. Plain `tensorflow` on a GPU box runs fine, sees no device, and trains on the CPU — so the
notebook checks `tf.config.list_physical_devices("GPU")` right after installing and stops there if
`device_type` is `gpu` and nothing showed up.

Terraform builds the target as `.[cuda]` rather than passing `--extra cuda`: uv rejects `--extra`
next to `-e .` with *"Requesting extras requires a pyproject.toml … use `<dir>[extra]` syntax"*.

Note `uv pip install` reads `pyproject.toml`, **not `uv.lock`**. Only `uv sync` uses the lock, and
that builds a virtualenv and re-downloads TensorFlow every session — a poor trade inside a
time-boxed notebook. So the versions here are the pins, not the locked resolution.

**The training text is a dataset setting, not a flag.** Point
`data_config.lm_dataset_config.data_paths` at ASR transcript `.tsv` for `target = "internal"`, or a
large `.txt`/`.txt.gz` corpus for `target = "external"`; `max_length` and `max_lines` live on the
same config block. `train_lm` reads it all from there.

## Files

| File | What it does |
| --- | --- |
| `variables.tf` | every knob, with the constraints as `validation` blocks |
| `notebook.tf` | renders the Python template and assembles the `.ipynb` |
| `main.tf` | writes `build/`, owns push and destroy |
| `outputs.tf` | kernel URL and ready-made CLI commands |
| `templates/train_lm.py.tftpl` | the notebook source, cells split on `# %%` |
| `scripts/push_and_wait.sh` | push, poll, download |
| `scripts/delete.sh` | `terraform destroy` |
| `scripts/write_credentials.sh` | writes `build/kaggle.json` from the provisioner environment |
