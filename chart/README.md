# Disease Prediction

![Version: 0.2.0](https://img.shields.io/badge/Version-0.2.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| image | string | `"intel/ai-workflows:eap-disease-prediction"` |  |
| inputs.artifacts.s3.key | string | `"datasets/disease-prediction"` | path to preprocessed dataset in s3 |
| metadata.name | string | `"disease-prediction"` |  |
| proxy | string | `"nil"` |  |
