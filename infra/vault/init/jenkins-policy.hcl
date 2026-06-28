# Jenkins CI/CD policy — read-only access to ml_api secrets only
path "yamwisdom/data/develop/ml_api" {
  capabilities = ["read"]
}

path "yamwisdom/data/prod/ml_api" {
  capabilities = ["read"]
}
