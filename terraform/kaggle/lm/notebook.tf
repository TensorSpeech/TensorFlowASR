locals {
  # Cloned outside /kaggle/working deliberately: everything under /kaggle/working is collected as
  # notebook output, so a clone there would be dragged into every `kaggle kernels output`.
  repo_dir = "/tmp/TensorFlowASR"

  # Flags that only appear when they are set. train_lm rejects --text-path unless
  # --target=external, so it is gated here too rather than left to fail on Kaggle.
  optional_flags = concat(
    var.text_path != "" ? ["--text-path=${var.text_path}"] : [],
    var.max_lines != null ? ["--max-lines=${var.max_lines}"] : [],
    var.spx > 1 ? ["--spx=${var.spx}"] : [],
    var.kaggle_model_handle != "" ? ["--kaggle-model-handle=${var.kaggle_model_handle}"] : [],
    # Only meaningful on a TPU, and train_lm ignores them otherwise, but passing them anyway would
    # put misleading flags in the notebook for a GPU run.
    var.device_type == "tpu" && var.tpu_address != "" ? ["--tpu-address=${var.tpu_address}"] : [],
    var.device_type == "tpu" && var.tpu_vm ? ["--tpu-vm=True"] : [],
  )

  # The accelerator decides the extra. `cuda` pulls tensorflow[and-cuda] with the nvidia wheels;
  # plain `tensorflow` cannot see a GPU.
  #
  # There is no TPU extra, in pyproject or here, and there cannot be: `tensorflow` is a base
  # dependency and `tensorflow-tpu` ships its own `tensorflow` distribution, so an extra adding it
  # installs the pair and whichever lands last wins. The TPU build is swapped in afterwards by
  # `scripts/install_tpu.sh` -- see the install cell.
  #
  # Bracket syntax rather than `--extra cuda`: uv rejects `--extra` alongside `-e .` with
  # "Requesting extras requires a pyproject.toml ... use <dir>[extra] syntax instead".
  install_extras = compact([local.use_gpu ? "cuda" : ""])
  install_target = length(local.install_extras) > 0 ? ".[${join(",", local.install_extras)}]" : "."

  notebook_source = templatefile("${path.module}/templates/train_lm.py.tftpl", {
    repo_url        = var.repo_url
    repo_ref        = var.repo_ref
    repo_dir        = local.repo_dir
    install_target  = local.install_target
    config_path     = var.config_path
    datadir         = var.datadir
    modeldir        = var.modeldir
    output_path     = var.output_path
    dataset_type    = var.dataset_type
    target          = var.target
    bs              = var.bs
    epochs          = var.epochs
    steps_per_epoch = var.steps_per_epoch
    max_length      = var.max_length
    learning_rate   = var.learning_rate
    device_type     = var.device_type
    mxp             = var.mxp

    # Only rendered when kaggle_model_handle is set, since nothing else in the notebook needs to
    # authenticate -- the repo clone and the dataset mount do not.
    #
    # This puts the API token in `local.notebook_source`, so it reaches terraform.tfstate,
    # build/notebook.ipynb and the notebook Kaggle stores. That is the deliberate trade for not
    # having to attach a Kaggle Secret by hand: no Kaggle API can create or attach one. The push
    # credentials in `main.tf` still travel through provisioner `environment`, which stays out of
    # state; this is the one path that does not.
    #
    # jsonencode, not quotes: a JSON string is a valid Python string literal, so Terraform does
    # the escaping and a key containing a quote or backslash cannot break the cell.
    kaggle_model_handle = var.kaggle_model_handle
    kaggle_username     = jsonencode(var.kaggle_username)
    kaggle_key          = jsonencode(var.kaggle_key)

    # A JSON list is also a valid Python list literal, and a JSON string is a valid Python
    # string literal, so both drop straight into the source with Terraform doing the escaping.
    extra_flags   = jsonencode(concat(local.optional_flags, var.extra_args))
    config_inline = var.config_file == "" ? "None" : jsonencode(file(var.config_file))
  })

  # The notebook is authored as one readable Python file and split into cells on `# %%`, the
  # jupytext convention. Prefixing a newline makes the leading marker split cleanly; the empty
  # first chunk is dropped.
  cells = [
    for chunk in split("\n# %%\n", "\n${local.notebook_source}") :
    trimspace(chunk) if trimspace(chunk) != ""
  ]

  # Built with jsonencode rather than a JSON template on purpose. Notebook JSON is one escaped
  # string per source line, and hand-escaping it is exactly where this normally breaks.
  notebook = {
    cells = [
      for cell in local.cells : {
        cell_type       = "code"
        execution_count = null
        metadata        = {}
        outputs         = []
        # nbformat wants a list of lines, each keeping its trailing newline except the last.
        source = [
          for index, line in split("\n", cell) :
          index == length(split("\n", cell)) - 1 ? line : "${line}\n"
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
