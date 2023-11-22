## Run Ensemble Inference
`Inference` library is for running ensemble inference on image and data. After the models are trained and saved using the script from load the NLP and vision models using the inference option. This applies a weighted ensemble method to generate a final prediction. To only run inference, set the inference parameter to true in the disease_prediction.yaml file 

## Input Arguments

The 'disease_prediction_container.yaml' file includes the following parameters:

output_dir: specifies the location of the output model and inference results
write: a container parameter that is set to true for container

nlp:
    finetune: runs nlp fine-tuning
    inference: runs nlp inference
   
vision:
    finetune: runs vision fine-tuning
    inference: runs vision inference

