output "kernel_id" {
  description = "Kernel id, as the Kaggle CLI wants it."
  value       = local.kernel_id
}

output "kernel_url" {
  description = "The notebook on Kaggle."
  value       = "https://www.kaggle.com/code/${var.kaggle_username}/${var.kernel_slug}"
}

output "notebook_path" {
  description = "The generated .ipynb, for inspection before or after a push."
  value       = local_file.notebook.filename
}

output "weights_path_in_kernel" {
  description = "Where the notebook writes the weights inside the Kaggle machine."
  value       = var.output_path
}

output "local_output_dir" {
  description = "Where `kaggle kernels output` downloads to when download_output is true."
  value       = local.output_dir
}

output "status_command" {
  description = "Check the run by hand."
  value       = "KAGGLE_CONFIG_DIR=${local.build_dir} kaggle kernels status ${local.kernel_id}"
}

output "logs_command" {
  description = "Fetch the run log, which is where a failed run explains itself."
  value       = "KAGGLE_CONFIG_DIR=${local.build_dir} kaggle kernels output ${local.kernel_id} -p ${local.output_dir}"
}

output "test_command" {
  description = "How to use the downloaded weights, once they are pulled down."
  value = join(" ", [
    "tensorflow_asr test",
    "--config-path=<config.yml.j2>",
    "--dataset-type=slice",
    "--datadir=<datadir>",
    "--outputdir=<outputdir>",
    "--h5=<transducer weights.h5>",
    var.target == "external" ? "--lm-h5=${local.output_dir}/${basename(var.output_path)}" : "--internal-lm-h5=${local.output_dir}/${basename(var.output_path)}",
  ])
}
