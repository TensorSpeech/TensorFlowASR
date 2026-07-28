# ---------------------------------------------------------------------------
# Kaggle account and the notebook itself
# ---------------------------------------------------------------------------

variable "kaggle_username" {
  description = "Kaggle username. Also the owner half of the kernel id."
  type        = string
}

variable "kaggle_key" {
  description = <<-EOT
    Kaggle API key, from https://www.kaggle.com/settings ("Create New Token").

    This is a live credential. It is passed to the Kaggle CLI through a provisioner
    `environment` block, which Terraform does not persist, so it never reaches
    terraform.tfstate. It is written to build/kaggle.json (mode 0600) because
    `terraform destroy` needs it and destroy-time provisioners cannot read variables.
    Both that file and *.tfvars are gitignored -- keep them that way.
  EOT
  type        = string
  sensitive   = true
}

variable "kernel_slug" {
  description = <<-EOT
    Kernel slug. The notebook lives at kaggle.com/code/<username>/<slug>, and the slug doubles as
    the title.

    The 5-character minimum comes from the title: Kaggle rejects anything shorter. Slugs alone
    could be 3.
  EOT
  type        = string
  default     = "tensorflowasr-train-lm"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{4,}$", var.kernel_slug))
    error_message = "kernel_slug must be lowercase letters, digits and hyphens, at least 5 characters (Kaggle rejects titles shorter than that, and the slug is used as the title)."
  }
}

variable "is_private" {
  description = "Keep the notebook private."
  type        = bool
  default     = true
}

variable "accelerator" {
  description = <<-EOT
    Accelerator ID, written to `machine_shape` and passed as `kaggle kernels push --accelerator`.
    This is what picks the hardware. "" means no accelerator; pair it with device_type = "cpu".

    Kaggle also has older `enable_gpu` / `enable_tpu` booleans, which this module does not expose:
    they cannot say *which* GPU, and `enable_tpu` maps to the v3-8 that Kaggle has phased out, so
    a TPU asked for that way comes up with nothing attached. The metadata booleans are derived
    from the prefix of this ID instead, so they can never contradict it.

    Valid IDs as of Feb 2026, from https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
    -- some are restricted to competition participants or Kaggle admins:

      NvidiaTeslaP100  NvidiaTeslaT4  NvidiaTeslaT4Highmem  NvidiaTeslaA100
      NvidiaL4  NvidiaL4X1  NvidiaH100  NvidiaRtxPro6000
      TpuV38  Tpu1VmV38  TpuV5E8  TpuV6E8
  EOT
  type        = string
  default     = "NvidiaTeslaP100"

  validation {
    condition = contains(
      [
        "", "NvidiaTeslaP100", "NvidiaTeslaT4", "NvidiaTeslaT4Highmem", "NvidiaTeslaA100",
        "NvidiaL4", "NvidiaL4X1", "NvidiaH100", "NvidiaRtxPro6000",
        "TpuV38", "Tpu1VmV38", "TpuV5E8", "TpuV6E8",
      ],
      var.accelerator
    )
    error_message = "Unknown accelerator ID. See https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md for the current list."
  }
}

variable "dataset_sources" {
  description = <<-EOT
    Kaggle datasets to mount, as "owner/dataset-slug". Each appears at
    /kaggle/input/<dataset-slug>, which is what `datadir` should point at.
  EOT
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for source in var.dataset_sources : can(regex("^[^/]+/[^/]+$", source))])
    error_message = "Each dataset source must be \"owner/dataset-slug\"."
  }
}

variable "competition_sources" {
  description = "Kaggle competitions to mount, as competition slugs."
  type        = list(string)
  default     = []
}

variable "model_sources" {
  description = "Kaggle models to mount, as \"owner/model/framework/variation/version\"."
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# Where the code comes from
# ---------------------------------------------------------------------------

variable "repo_url" {
  description = "Git URL cloned inside the notebook. Must be reachable without credentials."
  type        = string
  default     = "https://github.com/TensorSpeech/TensorFlowASR.git"
}

variable "repo_ref" {
  description = <<-EOT
    Branch or tag to clone.

    Not "main": `train_lm` and `tensorflow_asr/models/lm` are not on main yet, and the notebook
    fails at the train step if they are missing. Point this at whichever branch carries them.

    Whatever you choose has to be pushed -- the notebook clones over the network and cannot see
    your working tree.
  EOT
  type        = string
  default     = "feat/beamsearch"
}

# ---------------------------------------------------------------------------
# Config and data
# ---------------------------------------------------------------------------

variable "config_path" {
  description = <<-EOT
    Config file to train against, as a path inside the cloned repository.
    Ignored when `config_file` is set.
  EOT
  type        = string
  default     = "examples/models/transducer/conformer/small.yml.j2"
}

variable "config_file" {
  description = <<-EOT
    Optional path to a config on this machine. Its contents are embedded in the notebook
    and written to /kaggle/working/config.yml.j2, so a config that is not committed to the
    repository can still be used. Jinja imports inside it still resolve against the clone.
  EOT
  type        = string
  default     = ""
}

variable "datadir" {
  description = "Value for --datadir inside the notebook, normally /kaggle/input/<dataset-slug>."
  type        = string
}

variable "modeldir" {
  description = "Value for --modeldir. Only matters if the config interpolates {{ modeldir }}."
  type        = string
  default     = "/kaggle/working/model"
}

variable "dataset_type" {
  description = "One of tfrecord, slice, generator."
  type        = string
  default     = "slice"

  validation {
    condition     = contains(["tfrecord", "slice", "generator"], var.dataset_type)
    error_message = "dataset_type must be tfrecord, slice or generator."
  }
}

# ---------------------------------------------------------------------------
# train_lm arguments
# ---------------------------------------------------------------------------

variable "target" {
  description = <<-EOT
    Which language model to train.

    "internal" fits lm_config.internal_config, the low-order LM that LODR subtracts. It must
    be fitted on the ASR training transcripts, so it ignores `text_path`.

    "external" fits lm_config.external_config, the LM fused in. Point `text_path` at a large
    corpus; without one it falls back to the transcripts and warns.
  EOT
  type        = string
  default     = "internal"

  validation {
    condition     = contains(["external", "internal"], var.target)
    error_message = "target must be external or internal."
  }
}

variable "output_path" {
  description = "Where the notebook writes the weights. Keep it under /kaggle/working so it is collected as output."
  type        = string
  default     = "/kaggle/working/lm.weights.h5"
}

variable "text_path" {
  description = <<-EOT
    Text corpus for the external LM, one sentence per line, optionally gzipped. A path inside
    the Kaggle machine, normally a mounted dataset.

    The published setups use the LibriSpeech LM corpus (openslr.org/resources/11,
    librispeech-lm-norm.txt.gz). Upload it as a Kaggle dataset and add it to `dataset_sources`.
  EOT
  type        = string
  default     = ""
}

variable "max_lines" {
  description = "Stop after this many lines of text_path. null reads the whole corpus."
  type        = number
  default     = null
}

variable "bs" {
  description = <<-EOT
    Batch size **per replica**. The dataset is batched at `bs x replicas`.

    One replica on CPU or a single GPU, so there it is simply the batch size. A TPU v3-8 has 8
    cores, so bs = 32 means a global batch of 256.
  EOT
  type        = number
  default     = 32
}

variable "epochs" {
  description = "Training epochs."
  type        = number
  default     = 10
}

variable "max_length" {
  description = "Truncate sequences to this many tokens."
  type        = number
  default     = 256
}

variable "learning_rate" {
  description = "Adam learning rate."
  type        = number
  default     = 0.001
}

variable "device_type" {
  description = "cpu, gpu or tpu. Must line up with `accelerator`; a precondition rejects a mismatch."
  type        = string
  default     = "gpu"

  validation {
    condition     = contains(["cpu", "gpu", "tpu"], var.device_type)
    error_message = "device_type must be cpu, gpu or tpu."
  }
}

variable "mxp" {
  description = "Mixed precision: none, auto, strict."
  type        = string
  default     = "none"
}

variable "tpu_address" {
  description = <<-EOT
    Cluster address for --tpu-address. Only used when device_type is "tpu".

    "local" is what `docs/tutorials/training.md` uses and what a TPU VM wants, since the chips are
    attached to the machine running the code rather than reached over the network. Set it to ""
    to let the resolver auto-detect.

    If the TPU runtime complains

        Could not find SliceBuilder port 8471 in any of the 0 ports provided in
        tpu_process_addresses="local"

    then "local" is being read as a list of process addresses and finding none. Try "" first,
    then "" together with tpu_vm = false, which restores the `experimental_connect_to_cluster`
    call that the VM path skips. Which combination Kaggle wants is untested here.
  EOT
  type        = string
  default     = "local"
}

variable "tpu_vm" {
  description = <<-EOT
    Pass --tpu-vm. True for Kaggle, whose TPUs are TPU VMs: the flag skips
    `experimental_connect_to_cluster`, which is for reaching a remote TPU node and is wrong when
    the chips are local.
  EOT
  type        = bool
  default     = true
}

variable "spx" {
  description = <<-EOT
    --spx, `steps_per_execution`: batches per device call. Raising it cuts host round trips and is
    the usual TPU throughput lever.

    Left at 1 because it could not be verified. On keras 3 / tensorflow 2.19, any value above 1
    combined with a distribution strategy fails during `fit` with an InvalidArgumentError about a
    `while/cond` placeholder -- reproduced with a plain Dense model and with the stock Keras loss,
    so it is not specific to this repository. Single-device runs are fine at any value. Whether
    TPUStrategy shares the fault is untested. If you raise it, confirm a few steps run before
    spending a session on it.
  EOT
  type        = number
  default     = 1
}

variable "extra_args" {
  description = <<-EOT
    Extra flags appended to the train_lm command, verbatim.

    The example configs interpolate jinja variables that train_lm forwards from the CLI, so
    this is where they go, for example ["--vocabprefix=/kaggle/input/vocab/sp", "--vocabsize=1000"].
    An undefined jinja variable renders empty rather than failing, which usually shows up as a
    path that is missing a component.
  EOT
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# What apply does after pushing
# ---------------------------------------------------------------------------

variable "wait_for_completion" {
  description = <<-EOT
    Poll until the kernel finishes. `terraform apply` blocks for the whole training run --
    hours, for a real corpus. Set false to push and return immediately.
  EOT
  type        = bool
  default     = false
}

variable "poll_interval_seconds" {
  description = "Seconds between status checks."
  type        = number
  default     = 60
}

variable "timeout_minutes" {
  description = <<-EOT
    Give up waiting after this long. The kernel keeps running on Kaggle; only the wait stops.
    Kaggle caps a session at around 9-12 hours, so a longer timeout than that buys nothing.
  EOT
  type        = number
  default     = 720
}

variable "download_output" {
  description = "Run `kaggle kernels output` after a successful run to fetch the weights."
  type        = bool
  default     = true
}

variable "output_dir" {
  description = "Where to download the output. Defaults to output/ next to this module."
  type        = string
  default     = ""
}
