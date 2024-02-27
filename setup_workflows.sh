pip install python-dotenv 

# Clone vision workflows
pip install -r ${PWD}/vision_wf/workflows/disease_prediction/requirements.txt

# Clone hf-finetuning-inference-nlp-workflows
pip install -r ${PWD}/hf_nlp/workflows/hf_finetuning_and_inference_nlp/requirements.txt

pip install accelerate "transformers<4.30.0"
pip install fsspec==2023.9.2
