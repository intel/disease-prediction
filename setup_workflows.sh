git submodule update --init --recursive
pip install docx2txt openpyxl  ipywidgets jupyterlab python-dotenv

# Clone vision workflows
pip install -r ${PWD}/vision_wf/workflows/disease_prediction/requirements.txt

# Clone hf-finetuning-inference-nlp-workflows
pip install -r ${PWD}/hf_nlp/workflows/hf_finetuning_and_inference_nlp/requirements.txt

# Clone dataset_api
cd  ${PWD}/intel-models/datasets/dataset_api/
bash setup.sh
# run preprocessing steps for the dataset
python dataset.py -n brca --download --preprocess -d ${PWD}/../../../data/
cd  ${PWD}