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
  description = "Where the trainer writes the language model inside the Kaggle machine."
  value = (
    var.trainer == "train_internal_lm" ? "${var.modeldir}/lm/internal.weights.h5" :
    var.trainer == "train_kenlm" ? "${var.modeldir}/lm/kenlm.weights.h5 (plus lm.arpa)" :
    "${var.modeldir}/lm/external.weights.h5"
  )
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
  description = "How to use the downloaded language model, once it is pulled down."
  value = join(" ", [
    "tensorflow_asr test",
    "--config-path=<config.yml.j2>",
    "--dataset-type=slice",
    "--datadir=<datadir>",
    "--outputdir=<outputdir>",
    "--h5=<transducer weights.h5>",
    # `kaggle kernels output` mirrors /kaggle/working, so <modeldir>/lm lands under output_dir at
    # the same path minus the /kaggle/working prefix.
    var.trainer == "train_internal_lm" ?
    "--internal-lm-h5=${local.output_dir}/${trimprefix(var.modeldir, "/kaggle/working/")}/lm/internal.weights.h5" :
    "--lm-h5=${local.output_dir}/${trimprefix(var.modeldir, "/kaggle/working/")}/lm/${var.trainer == "train_kenlm" ? "kenlm" : "external"}.weights.h5",
  ])
}
