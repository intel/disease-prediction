git submodule update --init --recursive
pip install docx2txt openpyxl  ipywidgets jupyterlab python-dotenv dataset-librarian

# Clone vision workflows
pip install -r ${PWD}/vision_wf/workflows/disease_prediction/requirements.txt

# Clone hf-finetuning-inference-nlp-workflows
pip install -r ${PWD}/hf_nlp/workflows/hf_finetuning_and_inference_nlp/requirements.txt
