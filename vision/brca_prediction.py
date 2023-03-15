import numpy as np
import os
import shutil
import tensorflow as tf
import time
from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.types import FrameworkType
from PIL import Image


IMAGE_SIZE = 224

def collect_class_labels(dataset_dir):
    dataset = dataset_factory.load_dataset(dataset_dir=dataset_dir,
                                        use_case='image_classification', 
                                        framework='tensorflow')
    return dataset.class_names

def quantize_model(output_dir, saved_model_dir, model ):
    clean_output_folder(output_dir, 'quantized_models')

    quantization_output_dir = os.path.join(output_dir, 'quantized_models', "vision",
                                       os.path.basename(saved_model_dir))

    
    # Create a tuning workspace directory for INC
    root_folder = os.path.dirname(os.path.abspath(__file__))
    inc_config_file = os.path.join( root_folder, "config.yaml")

    # inc_config_file = 'vision/config.yaml'
    model.quantize(saved_model_dir, quantization_output_dir, inc_config_file)

def clean_output_folder(output_dir, model_name):
    folder_path = os.path.join(output_dir, model_name)
    if os.path.exists(folder_path):
        shutil.rmtree(os.path.join(output_dir, model_name))

def train_vision_wl(dataset_dir, output_dir, batch_size = 32, epochs=5, save_model=True, quantization=True ):
    
    #Clean the output folder first
    clean_output_folder(output_dir, "resnet_v1_50") 
       
    dict_metrics = {}
    ######### Loading the model ####---------------
    tstart = time.time()
    model = model_factory.get_model(model_name="resnet_v1_50", framework=FrameworkType.TENSORFLOW)
    tend = time.time()
    print("\nModal Loading time (s): ", tend - tstart)
    # Load the dataset from the custom dataset path
    #######  Data loading and preprocessing ####-----------
    dataset = dataset_factory.load_dataset(dataset_dir=dataset_dir,
                                        use_case='image_classification', 
                                        framework='tensorflow', shuffle_files=True)

    print("Class names:", str(dataset.class_names))
    
    dataset.preprocess(model.image_size, batch_size=batch_size, add_aug=['hvflip', 'rotate'])
    dataset.shuffle_split(train_pct=.80, val_pct=.20)
    
    #######  Finetuning ####-----------
    tstart = time.time()
    history = model.train(
        dataset, 
        output_dir=output_dir, 
        epochs=epochs,
        seed=10, 
        enable_auto_mixed_precision=True, 
        extra_layers=[1024,512]
    )
    tend = time.time()
    print("\nTotal Vision Finetuning time (s): ", tend - tstart)
    dict_metrics['e2e_training_time'] = tend - tstart

    metrics = model.evaluate(dataset)
    for metric_name, metric_value in zip(model._model.metrics_names, metrics):
        print("{}: {}".format(metric_name, metric_value))
        dict_metrics[metric_name] = metric_value
    
    print('dict_metrics:', dict_metrics)    
    print('Finished Fine-tuning the vision model...')

    if save_model:
        saved_model_dir = model.export(output_dir)
                
    if quantization:
        print('Quantizing the model')
        quantize_model(output_dir, saved_model_dir, model)

    print("Done finetuning the vision model ............")
    
    return(model, history, dict_metrics)



def infer_vision_wl(model, image_location):
    image_shape = (model.image_size, model.image_size)
    image = Image.open(image_location).resize(image_shape)
    # Get the image as a np array and call predict while adding a batch dimension (with np.newaxis) 
    image = np.array(image)/255.0
    result = model.predict(image[np.newaxis, ...], 'probabilities')[0]
    return result

def infer_int8_vision_wl(model, image_location):
    image_shape = (IMAGE_SIZE, IMAGE_SIZE)
    image = Image.open(image_location).resize(image_shape)
    # Get the image as a np array and call predict while adding a batch dimension (with np.newaxis) 
    image = np.array(image)/255.0
    image = image[np.newaxis, ...].astype('float32')
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    #result = model.predict(image[np.newaxis, ...])
    #result=model.predict(image[np.newaxis, ...], 'probabilities')[0]
    output_name = list(infer.structured_outputs.keys())
    result = infer(tf.constant(image))[output_name[0]][0]
    return result
