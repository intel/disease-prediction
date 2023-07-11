# Disease Prediction

![Version: 0.3.0](https://img.shields.io/badge/Version-0.3.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataset.nfs.configSubPath | string | `"nil"` | Path to configs in Local NFS |
| dataset.nfs.datasetSubPath | string | `"nil"` | Path to dataset in Local NFS |
| dataset.nfs.path | string | `"nil"` | Path to Local NFS Share in Cluster Host |
| dataset.nfs.server | string | `"nil"` | Hostname of NFS Server |
| dataset.s3.key | string | `"nil"` | Path to Dataset in S3 Bucket |
| dataset.type | string | `"<nfs/s3>"` | `nfs` or `s3` dataset input enabler |
| image.base | string | `"intel/ai-workflows"` | base container repository |
| image.hf_nlp | string | `"beta-hf-nlp-disease-prediction"` | hf nlp workflow container tag |
| image.use_case | string | `"beta-disease-prediction"` | ensemble container tag |
| image.vision_tlt | string | `"beta-vision-tlt-disease-prediction"` | vision tlt workflow container tag |
| metadata.name | string | `"disease-prediction"` |  |
| proxy | string | `"nil"` |  |
| serviceAccountName | string | `argo` |  |
| workflow.config.hf_nlp | string | `"nlp_finetune"` | hf nlp finetuning config file name |
| workflow.config.use_case | string | `"disease_prediction_container"` | ensemble inference config file name |
| workflow.config.vision_tlt | string | `"vision_finetune"` | vision tlt finetuning config file name |
| workflow.script.hf_nlp | string | `"run.py"` | |
| workflow.script.use_case | string | `"breast_cancer_prediction.py"` |  |
| workflow.script.vision_tlt | string | `"run.py"` |  |
