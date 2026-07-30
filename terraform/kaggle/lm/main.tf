locals {
  code_file  = "train_lm.ipynb"
  build_dir  = abspath("${path.module}/build")
  output_dir = var.output_dir != "" ? abspath(var.output_dir) : abspath("${path.module}/output")
  kernel_id  = "${var.kaggle_username}/${var.kernel_slug}"

  # Kaggle still reads the older enable_gpu / enable_tpu booleans, so the metadata carries them --
  # derived from the accelerator ID rather than configured separately, which is what stops a stale
  # boolean from contradicting `machine_shape`.
  use_tpu = startswith(var.accelerator, "Tpu")
  use_gpu = var.accelerator != "" && !local.use_tpu
}

resource "local_file" "notebook" {
  filename = "${local.build_dir}/${local.code_file}"
  content  = jsonencode(local.notebook)

  # 0600, not 0644: with `kaggle_model_handle` set the notebook carries the API token, the same
  # reason build/kaggle.json is owner-only.
  file_permission = "0600"

  lifecycle {
    precondition {
      condition     = var.config_file == "" || fileexists(var.config_file)
      error_message = "config_file does not exist: ${var.config_file}"
    }
    precondition {
      # Two knobs that have to agree: the accelerator Kaggle attaches, and the one TensorFlow is
      # told to use. Disagreeing is silent -- the kernel boots, installs the wrong extra, and
      # trains on the CPU.
      condition     = (var.device_type == "gpu") == local.use_gpu || var.device_type == "cpu"
      error_message = "device_type=\"gpu\" needs an Nvidia* accelerator, and an Nvidia* accelerator needs device_type = \"gpu\" (or \"cpu\" to deliberately ignore it)."
    }
    precondition {
      condition     = (var.device_type == "tpu") == local.use_tpu || var.device_type == "cpu"
      error_message = "device_type=\"tpu\" needs a Tpu* accelerator (accelerator = \"TpuV5E8\"), and a Tpu* accelerator needs device_type = \"tpu\" (or \"cpu\" to deliberately ignore it)."
    }
    precondition {
      condition     = startswith(var.output_path, "/kaggle/working")
      error_message = "output_path must be under /kaggle/working, otherwise Kaggle does not keep it and `kaggle kernels output` cannot fetch the weights."
    }
  }
}

resource "local_file" "metadata" {
  filename        = "${local.build_dir}/kernel-metadata.json"
  file_permission = "0644"

  # Field names come from the CLI's own `kernels init` template
  # (kaggle/api/kaggle_api_extended.py), not from the website docs, which lag it.
  #
  # `machine_shape` is omitted entirely rather than sent empty: the CLI reads it with
  # `get_or_default(meta_data, "machine_shape", None)`, and an empty string is not None.
  content = jsonencode(merge({
    id          = local.kernel_id
    title       = var.kernel_slug # Kaggle needs >= 5 characters, which the slug validation enforces
    code_file   = local.code_file
    language    = "python"
    kernel_type = "notebook"
    is_private  = var.is_private
    enable_gpu  = local.use_gpu
    enable_tpu  = local.use_tpu
    # Always on: the notebook clones the repository, downloads uv and installs the dependencies.
    enable_internet     = true
    dataset_sources     = var.dataset_sources
    competition_sources = var.competition_sources
    kernel_sources      = []
    model_sources       = var.model_sources
    }, var.accelerator != "" ? { machine_shape = var.accelerator } : {}),
  )
}

# Owns the kernel's existence and nothing else.
#
# It deliberately has no triggers. If it were replaced whenever the notebook changed, the
# destroy provisioner would delete the kernel -- and its version history -- on every edit.
# Pushing an existing slug adds a version, so updates belong on `terraform_data.push` below.
resource "terraform_data" "kernel" {
  # Destroy-time provisioners may only read `self`, so everything the delete needs is carried
  # here. Credentials are not: they stay out of state and are read from build/kaggle.json.
  input = {
    kernel_id     = local.kernel_id
    config_dir    = local.build_dir
    delete_script = abspath("${path.module}/scripts/delete.sh")
  }

  provisioner "local-exec" {
    when    = destroy
    command = self.input.delete_script

    environment = {
      KERNEL_ID         = self.input.kernel_id
      KAGGLE_CONFIG_DIR = self.input.config_dir
    }
  }
}

resource "terraform_data" "push" {
  # Re-push whenever the notebook or the kernel settings change.
  triggers_replace = {
    notebook = sha256(local_file.notebook.content)
    metadata = sha256(local_file.metadata.content)
  }

  depends_on = [terraform_data.kernel]

  # The key reaches the CLI through `environment`, which Terraform does not write to state.
  # It lands in build/kaggle.json so that `terraform destroy` can authenticate later.
  provisioner "local-exec" {
    command = abspath("${path.module}/scripts/write_credentials.sh")

    environment = {
      KAGGLE_USERNAME   = var.kaggle_username
      KAGGLE_KEY        = var.kaggle_key
      KAGGLE_CONFIG_DIR = local.build_dir
    }
  }

  provisioner "local-exec" {
    command = abspath("${path.module}/scripts/push_and_wait.sh")

    environment = {
      KAGGLE_CONFIG_DIR     = local.build_dir
      KERNEL_ID             = local.kernel_id
      BUILD_DIR             = local.build_dir
      ACCELERATOR           = var.accelerator
      OUTPUT_DIR            = local.output_dir
      WAIT_FOR_COMPLETION   = tostring(var.wait_for_completion)
      POLL_INTERVAL_SECONDS = tostring(var.poll_interval_seconds)
      TIMEOUT_MINUTES       = tostring(var.timeout_minutes)
      DOWNLOAD_OUTPUT       = tostring(var.download_output)
    }
  }
}
