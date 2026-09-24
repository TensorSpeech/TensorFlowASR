locals {
  # Cloned outside /kaggle/working deliberately: everything under /kaggle/working is collected as
  # notebook output, so a clone there would be dragged into every `kaggle kernels output`.
  repo_dir = "/tmp/TensorFlowASR"

  # The accelerator decides the extra. `cuda` pulls tensorflow[and-cuda] with the nvidia wheels;
  # plain `tensorflow` cannot see a GPU. There is no TPU extra -- `scripts/install_tpu.sh` swaps
  # the build in afterwards (see run.sh). Bracket syntax rather than `--extra cuda`: uv rejects
  # `--extra` alongside `-e .`.
  install_extras = compact([local.use_gpu ? "cuda" : ""])
  install_target = length(local.install_extras) > 0 ? ".[${join(",", local.install_extras)}]" : "."

  # Flags for the chosen trainer. `train_internal_lm` and `train_kenlm` fit by counting, so the
  # gradient-descent knobs (bs, epochs, learning_rate, ...) do not apply to them and are not passed.
  # The dataset itself -- data_paths, max_length, max_lines -- lives in the config's
  # data_config.lm_dataset_config, not in flags.
  external_flags = concat(
    [
      "--bs=${var.bs}",
      "--learning-rate=${var.learning_rate}",
      "--lr-schedule=${var.lr_schedule}",
      "--device-type=${var.device_type}",
      "--mxp=${var.mxp}",
    ],
    # epochs and steps_per_epoch are guarded rather than inlined. This whole local is evaluated even
    # for a counting trainer -- it is one branch of `trainer_flags` below, and Terraform evaluates a
    # referenced local eagerly, not lazily -- so inlining `${var.steps_per_epoch}` makes a kenlm or
    # internal run fail at plan time on the null default ("Cannot include a null value in a string").
    # Omitting them when null lets the trainer fall back to its own default (epochs) and, for
    # steps_per_epoch, leaves the real requirement to the main.tf precondition, which fires only for
    # train_external_lm and with a far clearer message.
    var.epochs != null ? ["--epochs=${var.epochs}"] : [],
    var.steps_per_epoch != null ? ["--steps-per-epoch=${var.steps_per_epoch}"] : [],
    var.kaggle_model_handle != "" ? ["--kaggle-model-handle=${var.kaggle_model_handle}"] : [],
    var.spx > 1 ? ["--spx=${var.spx}"] : [],
    # Only meaningful on a TPU; passing them on a GPU run would put misleading flags in the script.
    var.device_type == "tpu" && var.tpu_address != "" ? ["--tpu-address=${var.tpu_address}"] : [],
    var.device_type == "tpu" && var.tpu_vm ? ["--tpu-vm=True"] : [],
  )
  kenlm_flags = concat(
    var.max_lines != null ? ["--max-lines=${var.max_lines}"] : [],
    var.text_path != "" ? ["--text-path=${var.text_path}"] : [],
    length(var.prune) > 0 ? ["--prune=[${join(",", [for p in var.prune : tostring(p)])}]"] : [],
    # Same as the external trainer: with a handle set, the built model is pushed to the Kaggle model
    # so a later run (or `test`) can pull it. train_kenlm uploads once at the end rather than per epoch.
    var.kaggle_model_handle != "" ? ["--kaggle-model-handle=${var.kaggle_model_handle}"] : [],
  )
  trainer_flags = (
    var.trainer == "train_external_lm" ? local.external_flags :
    var.trainer == "train_kenlm" ? local.kenlm_flags :
    [] # train_internal_lm: config and dirs only, which run.sh already passes
  )
  # extra_args carries jinja variables the config interpolates and the trainer forwards from the
  # CLI, e.g. ["--vocabprefix=/kaggle/input/vocab/sp", "--vocabsize=1000"].
  train_flags = join(" ", concat(local.trainer_flags, var.extra_args))

  # The whole run, as one bash script. Rendered here, written to build/run.sh (main.tf) for
  # inspection, and embedded verbatim in the single notebook cell below.
  run_sh = templatefile("${path.module}/templates/run.sh.tftpl", {
    repo_url       = var.repo_url
    repo_ref       = var.repo_ref
    repo_dir       = local.repo_dir
    install_target = local.install_target
    datadir        = var.datadir
    modeldir       = var.modeldir
    device_type    = var.device_type
    trainer        = var.trainer
    config_path    = var.config_path
    train_flags    = local.train_flags

    # An embedded config is base64-encoded so no quoting or `$` inside it can break the script;
    # empty means "use the repo path config_path instead". config_file wins over config_path.
    config_b64 = var.config_file == "" ? "" : base64encode(file(var.config_file))

    # Only rendered when kaggle_model_handle is set -- nothing else in the run authenticates. This
    # puts the API token in build/run.sh and the notebook (and therefore terraform.tfstate); it is
    # the deliberate trade for not attaching a Kaggle Secret by hand, which no API can automate.
    # jsonencode, not bare quotes: a JSON string is a valid shell double-quoted word, so a token
    # with an odd character cannot break the assignment.
    kaggle_model_handle = var.kaggle_model_handle
    kaggle_username     = jsonencode(var.kaggle_username)
    kaggle_key          = jsonencode(var.kaggle_key)
  })

  # The notebook is a single Python cell that writes run.sh and runs it. run.sh is the readable
  # artifact; the cell is just a launcher.
  launch_py = templatefile("${path.module}/templates/launch.py.tftpl", {
    run_sh = local.run_sh
  })

  # Built with jsonencode rather than a JSON template on purpose: notebook JSON is one escaped
  # string per source line, and hand-escaping it is exactly where this normally breaks.
  notebook = {
    cells = [
      {
        cell_type       = "code"
        execution_count = null
        metadata        = {}
        outputs         = []
        # nbformat wants a list of lines, each keeping its trailing newline except the last.
        source = [
          for index, line in split("\n", trimspace(local.launch_py)) :
          index == length(split("\n", trimspace(local.launch_py))) - 1 ? line : "${line}\n"
        ]
      }
    ]
    metadata = {
      kernelspec = {
        display_name = "Python 3"
        language     = "python"
        name         = "python3"
      }
      language_info = {
        name = "python"
      }
    }
    nbformat       = 4
    nbformat_minor = 5
  }
}
