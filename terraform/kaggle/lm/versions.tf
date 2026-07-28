terraform {
  # `terraform_data` needs 1.4; `precondition` needs 1.2.
  required_version = ">= 1.5"

  required_providers {
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}
