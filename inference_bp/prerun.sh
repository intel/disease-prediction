echo "Running Inference Bash script"
mkdir -p /cnvrg/output
cp -r /input/dataset_download/data /cnvrg/
cp -r /input/vision_finetune/output/ /cnvrg/
cp -r /input/nlp_finetune/output/ /cnvrg/
rm -rf /workspace/output
ln -s /cnvrg/output /workspace/
ln -s /cnvrg/data/ /workspace/data
echo "Running Inference Bash script Done"