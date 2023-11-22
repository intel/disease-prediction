## Run Vision Finetuning
`Vision Finetune` library is for running vision finetuning on the images. To fine-tune and inference the CESM images, the reference kit uses Transfer Learning Tool vision based vision workflow  which is optimized for image fine-tuning and inference, along with TensorFlow Hub’s ResNet-50 model, to fine-tune a new convolutional neural network model with the subtracted CESM image dataset. If the user would like to change any of the training args please make changes in the vision_finetune.yaml file. The results of this task is a vision model saved in 'output' directory 


