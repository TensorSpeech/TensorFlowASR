locals {
  code_file  = "train_lm.ipynb"
  build_dir  = abspath("${path.module}/build")
  output_dir = var.output_dir != "" ? abspath(var.output_dir) : abspath("${path.module}/output")
  kernel_id  = "${var.kaggle_username}/${var.kernel_slug}"
}

resource "local_file" "notebook" {
  filename        = "${local.build_dir}/${local.code_file}"
  content         = jsonencode(local.notebook)
  file_permission = "0644"

  lifecycle {
    precondition {
      condition     = var.text_path == "" || var.target == "external"
      error_message = "text_path applies to target=\"external\" only. The internal LM must be fitted on the ASR training transcripts, so train_lm rejects it."
    }
    precondition {
      condition     = var.enable_internet
      error_message = "enable_internet must be true: the notebook clones the repository and installs it with pip."
    }
    precondition {
      condition     = var.config_file == "" || fileexists(var.config_file)
      error_message = "config_file does not exist: ${var.config_file}"
    }
    precondition {
      condition     = !(var.enable_gpu && var.enable_tpu)
      error_message = "Enable either a GPU or a TPU, not both."
    }
    precondition {
      # Two separate knobs that have to agree: the accelerator Kaggle attaches, and the one
      # TensorFlow is told to use. Disagreeing is silent -- the kernel boots, installs the wrong
      # extra, and trains on the CPU.
      condition     = (var.device_type == "gpu") == var.enable_gpu || var.device_type == "cpu"
      error_message = "device_type=\"gpu\" needs enable_gpu = true, and enable_gpu = true needs device_type = \"gpu\" (or \"cpu\" to deliberately ignore the accelerator)."
    }
    precondition {
      condition     = (var.device_type == "tpu") == var.enable_tpu || var.device_type == "cpu"
      error_message = "device_type=\"tpu\" needs enable_tpu = true, and enable_tpu = true needs device_type = \"tpu\" (or \"cpu\" to deliberately ignore the accelerator)."
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
  content = jsonencode({
    id                  = local.kernel_id
    title               = var.kernel_title
    code_file           = local.code_file
    language            = "python"
    kernel_type         = "notebook"
    is_private          = var.is_private
    enable_gpu          = var.enable_gpu
    enable_tpu          = var.enable_tpu
    enable_internet     = var.enable_internet
    dataset_sources     = var.dataset_sources
    competition_sources = var.competition_sources
    kernel_sources      = []
    model_sources       = var.model_sources
  })
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
      OUTPUT_DIR            = local.output_dir
      WAIT_FOR_COMPLETION   = tostring(var.wait_for_completion)
      POLL_INTERVAL_SECONDS = tostring(var.poll_interval_seconds)
      TIMEOUT_MINUTES       = tostring(var.timeout_minutes)
      DOWNLOAD_OUTPUT       = tostring(var.download_output)
    }
  }
}
