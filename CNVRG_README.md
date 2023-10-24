# HLS Disease Prediction Blueprint

Many reference kits in the bio-medical domain focus on a single-model and single-modal solution. Exclusive reliance on a single method has some limitations, such as impairing the design of robust and accurate classifiers for complex datasets. To overcome these limitations, we provide this multi-modal disease prediction reference kit.

Multi-modal disease prediction blueprint is an Intel optimized, end-to-end reference kit for fine-tuning and inference. This reference kit implements a multi-model and multi-modal solution that will help to predict diagnosis by using categorized contrast enhanced mammography data and radiologists’ notes

## How to Use 

### [CNVRG IO FLows] 

> Note: you can experience the workflow step by step and view each step logs and results. 

* How to execute: Flows -> Multi-modal Disease Prediction-> Click ‘Run’.

* How to view results: Experiments -> Click and Check each experiment result.


* Step-by-step explanation: 

   1) vision finetuning: Runs vision finetuning on the images with pretrained Resnet v1.5
   2) nlp fine-tuning: Runs hugging face nlp finetuning on radiologist notes
   3) inference: Ensembles nlp and vision results to classify a patient as malignant , normal or benign
  
# Disclaimer

This reference implementation shows how to train a model to examine and evaluate a diagnostic theory and the associated performance of Intel technology solutions using very limited, non-diverse datasets to train the model. The model was not developed with any intention of clinical deployment and therefore lacks the requisite breadth and depth of quality information in its underlying datasets, or the scientific rigor necessary to be considered for use in actual diagnostic applications. Accordingly, while the model may serve as a foundation for additional research and development of more robust models, Intel expressly recommends and requests that this model not be used in clinical implementations or as a diagnostic tool.