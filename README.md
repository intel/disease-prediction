# Multi-Modal Disease Prediction
Many reference kits in the bio-medical domain focus on a single-model and single-modal solution. Exclusive reliance on a single method has some limitations, such as impairing the design of robust and accurate classifiers for complex datasets. To overcome these limitations, we provide this multi-modal disease prediction reference kit.

Multi-modal disease prediction is an Intel optimized, end-to-end reference kit for fine-tuning and inference. This reference kit implements a multi-model and multi-modal solution that will help to predict diagnosis by using categorized contrast enhanced mammography data and radiologists’ notes.
 
Check out more workflow and reference kit examples in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Table of Contents
- [Solution Technical Overview](#solution-technical-overview)
- [Validated Hardware Details](#validated-hardware-details)
- [Software Requirements](#software-requirements)
- [How it Works?](#how-it-works)
    - [Architecture](#architecture)
- [Get Started](#get-started)
- [Run Using Docker](#run-using-docker)
- [Run Using Argo Workflows on K8s using Helm](#run-using-argo-workflows-on-k8s-using-helm)
- [Run Using Bare Metal](#run-using-bare-metal) 
- [Run Using Jupyter Lab](#run-using-jupyter-lab)
- [Expected Output](#expected-output)
- [Summary and Next Steps](#summary-and-next-steps)
- [Learn More](#learn-more)
- [Support](#support)


## Solution Technical Overview
This reference kit demonstrates one possible reference implementation of a multi-model and multi-modal solution. While the vision workflow aims to train an image classifier that takes in contrast-enhanced spectral mammography (CESM) images, the natural language processing (NLP) workflow aims to train a document classifier that takes in annotation notes about a patient’s symptoms. Each pipeline creates prediction for the diagnosis of breast cancer. In the end, weighted ensemble method is used to create final prediction.

The goal is to minimize an expert’s involvement in categorizing samples as normal, benign, or malignant, by developing and optimizing a decision support system that automatically categorizes the CESM with the help of radiologist notes.

### DataSet
The dataset is a collection of 2,006 high-resolution contrast-enhanced spectral mammography (CESM) images (1003 low energy images and 1003 subtracted CESM images) with annotations of 326 female patients. See Figure-1. Each patient has 8 images, 4 representing each side with two views (Top Down looking and Angled Top View) consisting of low energy and subtracted CESM images. Medical reports, written by radiologists, are provided for each case along with manual segmentation annotation for the abnormal findings in each image. As a preprocessing step, we segment the images based on the manual segmentation to get the region of interest and group annotation notes based on the subject and breast side. 

  ![CESM Images](assets/cesm_and_annotation.png)

*Figure-2: Samples of low energy and subtracted CESM images and Medical reports, written by radiologists from the Categorized contrast enhanced mammography dataset. [(Khaled, 2022)](https://www.nature.com/articles/s41597-022-01238-0)*

For more details of the dataset, visit the wikipage of the [CESM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8) and read [Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research](https://www.nature.com/articles/s41597-022-01238-0).

## Validated Hardware Details
There are workflow-specific hardware and software setup requirements depending on how the workflow is run. Bare metal development system and Docker image running locally have the same system requirements. 

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors| FP32, BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |

To execute the reference solution presented here, use CPU for fine tuning.

## Software Requirements 
Linux OS (Ubuntu 22.04) is used to validate this reference solution. Make sure the following dependencies are installed.

1. `sudo apt-get update`
2. `sudo apt-get install -y build-essential gcc git libgl1-mesa-glx libglib2.0-0 python3-dev`
3. `sudo apt-get install -y python3.9 python3-pip`, and some virtualenv like python3-venv or [conda](#1-set-up-system-software) 
4. `pip install dataset-librarian`

## How It Works?
### Architecture
![Use_case_flow](assets/e2e_flow_HLS_Disease_Prediction.png)
*Figure-1: Architecture of the reference kit*

- Uses real-world CESM breast cancer datasets with “multi-modal and multi-model” approaches.
- Two domain toolkits (Intel® Transfer Learning Toolkit and Intel® Extension for Transformers), Intel® Neural Compressor and other libs/tools and uses Hugging Face model repo and APIs for [ResNet-50](https://huggingface.co/microsoft/resnet-50) and [ClinicalBert](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) models. 
- The NLP reference Implementation component uses [HF Fine-tuning and Inference Optimization workload](https://github.com/intel/intel-extension-for-transformers/tree/main/workflows/hf_finetuning_and_inference_nlp), which is optimized for document classification. This NLP workload employs Intel® Neural Compressor and other libraries/tools and utilizes Hugging Face model repository and APIs for ClinicalBert models. The ClinicalBert model, which is pretrained with a Masked-Language-Modeling task on a large corpus of English language from MIMIC-III data, is fine-tuned with the CESM breast cancer annotation dataset to generate a new BERT model.
- The Vision reference Implementation component uses [TLT-based vision workload](https://github.com/IntelAI/transfer-learning), which is optimized for image fine-tuning and inference. This workload utilizes Intel® Transfer Learning Tool and tfhub's ResNet-50 model to fine-tune a new convolutional neural network model with subtracted CESM image dataset. The images are preprocessed by using domain expert-defined segmented regions to reduce redundancies during training.
- Predict diagnosis by using categorized contrast enhanced mammography images and radiologists’ notes separately and weighted ensemble method applied to results of sub-models to create the final prediction.

## Get Started
Start by defining an environment variable that will store the workspace path, this can be an existing directory or one to be created in further steps. This ENVVAR will be used for all the commands executed using absolute paths.

E. g.
```bash
export WORKSPACE=/mydisk/mtw/mywork
```

### Download the Reference Kit Repository
Create a working directory for the reference kit and clone the [Breast Cancer Prediction Reference Kit](https://github.com/intel/disease-prediction) repository into your working directory.
```
git clone https://github.com/intel/disease-prediction.git $WORKSPACE/brca_multimodal
cd $WORKSPACE/brca_multimodal
git submodule update --init --recursive
```


### Download and Preprocess the Datasets
Use the links below to download the image datasets. Or skip to the [Docker](#run-using-docker) section to download the dataset using a container.

- [High-resolution Contrast-enhanced spectral mammography (CESM) images](https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/260?passcode=5335d2514638afdaf03237780dcdfec29edf4238#)

Once you have downloaded the image files and placed them into the data directory, proceed by executing the following command. This command will initiate the download of segmentation and annotation data, followed by the application of segmentation and preprocessing operations.

Command-line Interface:
- -d : Directory location where the raw dataset will be saved on your system. It's also where the preprocessed dataset files will be written. If not set, a directory with the dataset name will be created.
- --split_ratio: Split ratio of the test data, the default value is 0.1.

More details of the dataset_librarian can be found [here](https://pypi.org/project/dataset-librarian/).


```
python -m dataset_librarian.dataset -n brca --download --preprocess -d data/ --split_ratio 0.1
```

**Note:** See this dataset's applicable license for terms and conditions. Intel Corporation does not own the rights to this dataset and does not confer any rights to it.


## Supported Runtime Environment
This reference kit offers three options for running the fine-tuning and inference processes:

- Docker
- Argo Workflows on K8s Using Helm
- Bare Metal
- Jupyter Workspace

Details about each of these methods can be found below.

## Run Using Docker
Follow these instructions to set up and run our provided Docker image. For running on bare metal, see the [bare metal](#run-using-bare-metal) instructions.

### 1. Set Up Docker Engine and Docker Compose
You'll need to install Docker Engine on your development system. Note that while **Docker Engine** is free to use, **Docker Desktop** may require you to purchase a license. See the [Docker Engine Server installation instructions](https://docs.docker.com/engine/install/#server) for details.

If the Docker image is run on a cloud service, mention they may also need
credentials to perform training and inference related operations (such as these
for Azure):
- [Set up the Azure Machine Learning Account](https://azure.microsoft.com/en-us/free/machine-learning)
- [Configure the Azure credentials using the Command-Line Interface](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli)
- [Compute targets in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
- [Virtual Machine Products Available in Your Region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=us-east)


To build and run this workload inside a Docker Container, ensure you have Docker Compose installed on your machine. If you don't have this tool installed, consult the official [Docker Compose installation documentation](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually).


```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
docker compose version
```

### 2. Set Up Docker Image
Enter the docker directory.

```bash
cd $WORKSPACE/brca_multimodal/docker
```

Build or Pull the provided docker image.
```bash
docker compose build hf-nlp-wf vision-tlt-wf
docker compose build dev
```
OR
```bash
docker pull intel/ai-workflows:pa-hf-nlp-disease-prediction
docker pull intel/ai-workflows:pa-vision-tlt-disease-prediction
docker pull intel/ai-workflows:pa-disease-prediction
```

### 3. Preprocess Dataset with Docker Compose
Prepare dataset for Disease Prediction workflows and accept the legal agreement to use the Intel Dataset Downloader.

```bash
USER_CONSENT=y docker compose run preprocess
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| DATASET_DIR | `$PWD/../data` | Unpreprocessed dataset directory |
| SPLIT_RATIO | `0.1` | Train/Test Split Ratio |
| USER_CONSENT | n/a | Consent to legal agreement |

### 4. Run Pipeline with Docker Compose

Both NLP and Vision Fine-tuning containers must complete successfully before the Inference container can begin. The Inference container uses checkpoint files created by both the nlp and vision fine-tuning containers stored in the `${OUTPUT_DIR}` directory to complete inferencing tasks.


```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart RL
  VDATASETDIR{{"/${DATASET_DIR"}} x-. "-$PWD/../data}" .-x hf-nlp-wf[hf-nlp-wf]
  VCONFIGDIR{{"/${CONFIG_DIR"}} x-. "-$PWD/../configs}" .-x hf-nlp-wf
  VOUTPUTDIR{{"/${OUTPUT_DIR"}} x-. "-$PWD/../output}" .-x hf-nlp-wf
  VDATASETDIR x-. "-$PWD/../data}" .-x vision-tlt-wf[vision-tlt-wf]
  VCONFIGDIR x-. "-$PWD/../configs}" .-x vision-tlt-wf
  VOUTPUTDIR x-. "-$PWD/../output}" .-x vision-tlt-wf
  VDATASETDIR x-. "-$PWD/../data}" .-x ensemble-inference[ensemble-inference]
  VCONFIGDIR x-. "-$PWD/../configs}" .-x ensemble-inference
  VOUTPUTDIR x-. "-$PWD/../output}" .-x ensemble-inference
  ensemble-inference --> hf-nlp-wf
  ensemble-inference --> vision-tlt-wf

  classDef volumes fill:#0f544e,stroke:#23968b
  class VDATASETDIR,VCONFIGDIR,VOUTPUTDIR,VDATASETDIR,VCONFIGDIR,VOUTPUTDIR,VDATASETDIR,VCONFIGDIR,VOUTPUTDIR volumes
```

Run entire pipeline to view the logs of different running containers.

```bash
docker compose run ensemble-inference &
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| CONFIG | `disease_prediction_container` | Config file name |
| CONFIG_DIR | `$PWD/../configs` | Disease Prediction Configurations directory |
| DATASET_DIR | `$PWD/../data` | Preprocessed dataset directory |
| OUTPUT_DIR | `$PWD/../output` | Logfile and Checkpoint output |

#### 4.1 View Logs
Follow logs of each individual pipeline step using the commands below:

```bash
docker compose logs vision-tlt-wf -f
docker compose logs hf-nlp-wf -f
```

To view inference logs
```bash
fg
```

### 5. Run One Workflow with Docker Compose
Create your own script and run your changes inside of the container or run inference without waiting for fine-tuning.

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart RL
  VDATASETDIR{{"/${DATASET_DIR"}} x-. "-$PWD/../data}" .-x dev
  VCONFIGDIR{{"/${CONFIG_DIR"}} x-. "-$PWD/../configs}" .-x dev
  VOUTPUTDIR{{"/${OUTPUT_DIR"}} x-. "-$PWD/../output}" .-x dev

  classDef volumes fill:#0f544e,stroke:#23968b
  class VDATASETDIR,VCONFIGDIR,VOUTPUTDIR volumes
```

Run using Docker Compose.

```bash
docker compose run dev
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| CONFIG | `disease_prediction_container` | Config file name |
| CONFIG_DIR | `$PWD/../configs` | Disease Prediction Configurations directory |
| DATASET_DIR | `$PWD/../data` | Preprocessed Dataset |
| OUTPUT_DIR | `$PWD/output` | Logfile and Checkpoint output |
| SCRIPT | `src/breast_cancer_prediction.py` | Name of Script |

#### 5.1 Run Docker Image in an Interactive Environment

If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment:
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

Run the workflow with the ``docker run`` command, as shown:

```bash
export CONFIG_DIR=$PWD/../configs
export DATASET_DIR=$PWD/../data
export OUTPUT_DIR=$PWD/../output
docker run -a stdout ${DOCKER_RUN_ENVS} \
           -v /$PWD/../hf_nlp:/workspace/hf_nlp \
           -v /$PWD/../vision_wf:/workspace/vision_wf \
           -v /${CONFIG_DIR}:/workspace/configs \
           -v /${DATASET_DIR}:/workspace/data \
           -v /${OUTPUT_DIR}:/workspace/output \
           --privileged --init -it --rm --pull always \
           intel/ai-workflows:pa-disease-prediction \
           bash
```

Run the command below for fine-tuning and inference:
```bash
python src/breast_cancer_prediction.py --config_file /workspace/configs/disease_prediction_baremetal.yaml
```

### 6. Clean Up Docker Containers
Stop containers created by docker compose and remove them.

```bash
docker compose down
```

## Run Using Argo Workflows on K8s Using Helm
### 1. Install Helm
- Install [Helm](https://helm.sh/docs/intro/install/)
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
chmod 700 get_helm.sh && \
./get_helm.sh
```
### 2. Setting up K8s
- Install [Argo Workflows](https://argoproj.github.io/argo-workflows/quick-start/) and [Argo CLI](https://github.com/argoproj/argo-workflows/releases)
- Configure your [Artifact Repository](https://argoproj.github.io/argo-workflows/configure-artifact-repository/)
- Ensure that your dataset and config files are present in your chosen artifact repository.
### 3. Install Workflow Template
```bash
export NAMESPACE=argo
helm install --namespace ${NAMESPACE} --set proxy=${http_proxy} disease-prediction ./chart
argo submit --from wftmpl/disease-prediction --namespace=${NAMESPACE}
```
### 4. View 
To view your workflow progress
```bash
argo logs @latest -f
```
## Run Using Bare Metal
### 1. Set Up System Software 

Users are encouraged to use python virtual environments for consistent package management

Using virtualenv:

```
cd $WORKSPACE/brca_multimodal
python3.9 -m venv hls_env
source hls_env/bin/activate
```

Or conda: If you don't already have conda installed, see the [Conda Linux installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).
```
cd $WORKSPACE/brca_multimodal
conda create --name hls_env python=3.9
conda activate hls_env
```

### 2. Set Up Workflow 
This step involves the installation of the following  workflows:

- HF Fine-tune & Inference Optimization workflow
- Transfer Learning based on TLT workflow

```
bash setup_workflows.sh
```

### 3. Model Building Process

To train the multi-model disease prediction, utilize the 'breast_cancer_prediction.py' script along with the arguments outlined in the 'disease_prediction_baremetal.yaml' configuration file, which has the following structure:

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    disease_prediction_baremetal.yaml --> output_dir
    disease_prediction_baremetal.yaml --> write
    disease_prediction_baremetal.yaml --> nlp
    nlp --> nlp-finetune-params
    nlp --> nlp-inference-params
    nlp --> other-nlp-params
    disease_prediction_baremetal.yaml --> vision
    vision --> vision-finetune-params
    vision --> vision-inference-params
    vision --> other-vision-params
```

The 'disease_prediction_baremetal.yaml' file includes the following parameters:

- output_dir: specifies the location of the output model and inference results
- write: a container parameter that is set to false for bare metal
- nlp:
  - finetune: runs nlp fine-tuning
  - inference: runs nlp inference
  - additional parameters for the HF fine-tune and inference optimization workflow (more information available [here](https://github.com/intel/intel-extension-for-transformers/tree/main/workflows/hf_finetuning_and_inference_nlp/config))

- vision:
  - finetune: runs vision fine-tuning
  - inference: runs vision inference
  - additional parameters for the Vision: Transfer Learning Toolkit based on TLT workflow (more information available [here](https://github.com/IntelAI/transfer-learning/tree/f2e83f1614901d44d0fdd66f983de50551691676/workflows/disease_prediction))


To solely perform the fine-tuning process, set the 'finetune' parameter to true in the 'disease_prediction.yaml' file and execute the following command:

```
python src/breast_cancer_prediction.py --config_file configs/disease_prediction_baremetal.yaml
```

### 4. Running Inference
After the models are trained and saved using the script from step 4, load the NLP and vision models using the inference option. This applies a weighted ensemble method to generate a final prediction. To only run inference, set the 'inference' parameter to true in the 'disease_prediction.yaml' file and run the command provided in step 4.

> Alternatively, you can combine the training and inference processes into one execution by setting both the 'finetune' and 'inference' parameters to true in the 'disease_prediction.yaml' file and running the command provided in step 4.

## Run Using Jupyter Lab
To be able to run the code inside ```brca_multimodal_notebook.ipynb``` user should first create a virtual environment. For conda use the following commands:
```
conda create --name hls_env python=3.9 ipykernel jupyterlab  tornado==6.2 ipywidgets==8.0.4 -y -q
```
To activate it run the following command:
```
conda activate hls_env
python -m ipykernel install --user --name hls_env
```
Once Jupyter Lab is installed run from the project root directory inside ```hls_env``` environment:
```
jupyter lab
```
When jupyter lab is running click on the provided link and follow the instructions inside the notebook. If needed change the kernel to ```hls_env```.

## Expected Output
A successful execution of inference returns the confusion matrix of the sub-models and ensembled model, as shown in these example results: 
```
------ Confusion Matrix for Vision model ------
           Benign  Malignant  Normal  Precision
Benign       18.0     11.000   1.000      0.486
Malignant     5.0     32.000   0.000      0.615
Normal       14.0      9.000  25.000      0.962
Recall        0.6      0.865   0.521      0.652

------ Confusion Matrix for NLP model ---------
           Benign  Malignant  Normal  Precision
Benign     25.000      4.000     1.0      0.893
Malignant   3.000     34.000     0.0      0.895
Normal      0.000      0.000    48.0      0.980
Recall      0.833      0.919     1.0      0.930

------ Confusion Matrix for Ensemble --------
           Benign  Malignant  Normal  Precision
Benign     26.000      4.000     0.0      0.897
Malignant   3.000     34.000     0.0      0.895
Normal      0.000      0.000    48.0      1.000
Recall      0.867      0.919     1.0      0.939

```



## Summary and Next Steps
This Github repo describes a reference kit for multi-modal disease prediction in the biomedical domain. The kit provides an end-to-end solution for fine-tuning and inference using categorized contrast-enhanced mammography data and radiologists' notes to predict breast cancer diagnosis. The reference kit includes a vision workload that trains an image classifier using CESM images, and a NLP pipeline that trains a document classifier using annotation notes about a patient's symptoms. Both pipelines create predictions for the diagnosis of breast cancer, which are then combined using a weighted ensemble method to create a final prediction. The ultimate goal of the reference kit is to develop and optimize a decision support system that can automatically categorize samples as normal, benign, or malignant, thereby reducing the need for expert involvement.

As a future work, we will use a postprocessing method that will ensemble the different domain knowledge at feature level and fine-tune a model that would increase the accuracy of the prediction, speed up the end-to-end execution time for fine-tuning and inference, and reduce the cost of computation. 

* If you want to enable distributed training on k8s for your use case, please follow steps to apply that configuration mentioned in the [Intel® Transfer Learning Tools](https://github.com/IntelAI/transfer-learning/docker/README.md#kubernetes) which provides insights into k8s operators and yml file creation.

### How to customize this use case
Tunable configurations and parameters are exposed using yaml config files allowing users to change model training hyperparameters, datatypes, paths, and dataset settings without having to modify or search through the code.

#### Adopt to your dataset
To deploy this reference use case on a different or customized dataset, you can easily modify the disease_prediction_baremetal.yaml file. For instance, if you have a new text dataset, simply update the paths of finetune_input and inference_input and adjust the dataset features in the disease_prediction_baremetal.yaml file, as demonstrated below.

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    nlp --> args
    args --> local_dataset
    local_dataset --> |../data/annotation/training.csv| finetune_input
    local_dataset --> |../data/annotation/testing.csv| inference_input
    local_dataset --> features
    features --> |label| class_label
    features --> |symptoms| data_column
    features --> |Patient_ID| id
    local_dataset --> |Benign, Malignant, Normal| label_list
```    

#### Adopt to your model

To implement this reference use case on a different or customized pre-training model, modifications to the disease_prediction_baremetal.yaml file are straightforward. For instance, to use an alternate model, one can update the path of the model by modifying the 'model_name_or_path' and 'tokenizer_name' fields in the disease_prediction_baremetal.yaml file structure. The following example illustrates this process:

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart LR
    nlp --> args
    args --> |emilyalsentzer/Bio_ClinicalBERT| model_name_or_path
    args --> |emilyalsentzer/Bio_ClinicalBERT| tokenizer_name
```


## Learn More
For more information or to read about other relevant workflow examples, see these guides and software resources:
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
- [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel® Transfer Learning Tool](https://github.com/IntelAI/models/tree/master/docs/notebooks/transfer_learning)
- [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers)

## Troubleshooting
<!--- Validation Team please fill out --->
Currently mixing workflows on bare-metal and Docker from same path on the host is not supported. Please start each workflow run from scratch to minimize the chances of previously cached data.

## Support
The end-to-end multi-modal disease prediction tea tracks both bugs and enhancement requests using [disease prediction GitHub repo](https://github.com/intel/disease-prediction). We welcome input, however, before filing a request, search the GitHub issue database. 

\*Other names and brands may be claimed as the property of others.
[Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html).
