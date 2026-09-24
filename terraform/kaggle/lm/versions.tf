terraform {
  # `terraform_data` needs 1.4; `precondition` needs 1.2.
  required_version = ">= 1.5"

  # Keep the state inside .terraform/ (already gitignored) rather than at the module root, so the
  # module directory stays free of tfstate files. Still the ordinary local backend -- state is a
  # plain file on disk, just under .terraform/. The filename is `state.tfstate`, not
  # `terraform.tfstate`: Terraform reserves `.terraform/terraform.tfstate` for its own backend
  # record, and pointing the workspace state at that exact path collides and breaks every command.
  backend "local" {
    path = ".terraform/state.tfstate"
  }

  required_providers {
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}
