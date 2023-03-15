# Clone Transfer Learning Toolkit
# git clone https://github.com/IntelAI/transfer-learning.git vision/tlt_toolkit
git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git vision/tlt_toolkit
cd ${PWD}/vision/tlt_toolkit
pip install --editable .[tensorflow,pytorch]
pip install tensorflow-text==2.10.0
cd ../../
export PYTHONPATH=${PWD}/vision/tlt_toolkit
