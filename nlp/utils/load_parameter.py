import argparse
import os

root_folder = os.path.dirname(__file__)

class hls_param:
    image_name = os.path.join(root_folder, "../../data/vision_images/Benign/P300_R_DM_MLO.jpg")
    saved_models_quantization = os.path.join(root_folder, "../../output/saved_models_quantization/")
    saved_models_inference = os.path.join(root_folder, "../../output/saved_models_inference/")
    dataset_path_and_name = os.path.join(root_folder, "../../data/annotation/annotation.csv")
    label_column = "label"
    data_column = "symptoms"
    patient_id_column = "Patient_ID"
    seq_length = 64
    batch_size = 16
    image_data_folder = os.path.join(root_folder, '../../data/train_test_split_images/train/')
    nlp_int8_inference = False

    # training 
    model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"
    data_column = "symptoms"
    logfile = "nlp.log"
    output_dir = "temp_models/"
    epochs = 4
    fine_tune = True
    quantization = False
    quantization_criterion = 0.05
    quantization_max_trial = 50
    bf16 = False

    
def get_test_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_name",
        default=hls_param.image_name, 
        type=str,
        help="Name of the image that will be used to define the subject id.",
    )

    parser.add_argument(
        "--patient_id_column",
        default=hls_param.patient_id_column, 
        type=str,
        help="Name of the image that will be used to define the subject id.",
    )

    parser.add_argument(
        "--data_column",
        type=str,
        default=hls_param.data_column, 
        help="Column name that holds the annotaion data.",
    )

    parser.add_argument(
        "--saved_models_inference",
        default= hls_param.saved_models_inference, 
        type=str,
        help="Directory of the trained model and reqired parameters for inferance.",
    )

    parser.add_argument(
        "--saved_models_quantization",
        default=hls_param.saved_models_quantization, 
        type=str,
        help="Directory of the quantized model and reqired parameters for inferance.",
    )

    parser.add_argument(
        "--label_column",
        default=hls_param.label_column, 
        type=str,
        help="Column name that holds the class label.",
    )

    parser.add_argument(
        "--dataset_path_and_name",
        default=hls_param.dataset_path_and_name, 
        type=str,
        help="Directory of the training/testing data set and name.",
    )

    parser.add_argument(
        "--seq_length",
        default=hls_param.seq_length,
        type=int,
        help="Sequence length that used for training.",
    )

    parser.add_argument(
        "--batch_size", 
        default=hls_param.batch_size, 
        type=int, 
        help="Batch size that used for training."
    )
    
    parser.add_argument(
        "--nlp_int8_inference",
        default= hls_param.quantization, 
        type=bool,
        help="Use INT8 (quantized) model.",
    )

    return parser.parse_args()

def get_train_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default= hls_param.model_name_or_path, 
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--dataset_path_and_name",
        type=str,
        default=hls_param.dataset_path_and_name , 
        help="Directory of the training data set and name.",
    )

    parser.add_argument(
        "--label_column",
        type=str,
        default= hls_param.label_column, 
        help="Column name that holds the class label.",
    )

    parser.add_argument(
        "--data_column",
        type=str,
        default= hls_param.data_column, 
        help="Column name that holds the annotaion data.",
    )

    parser.add_argument(
        "--patient_id_column",
        type=str,
        default= hls_param.patient_id_column, 
        help="Column name that holds the class label.",
    )

    parser.add_argument(
        "--image_data_folder",
        type=str,
        default= hls_param.image_data_folder, 
        help="Directory of the training data set and name.",
    )

    parser.add_argument(
        "--logfile", type=str, 
        default= hls_param.logfile, 
        help="Log file to and location."
    )

    parser.add_argument(
        "--output_dir",
        default= hls_param.output_dir , 
        type=str,
        help="Directory to save trainer outputs.",
    )

    parser.add_argument(
        "--saved_models_inference",
        default= hls_param.saved_models_inference, 
        type=str,
        help="Directory to save the model and reqired parameters for inferance.",
    )

    parser.add_argument(
        "--saved_models_quantization",
        default=hls_param.saved_models_quantization, 
        type=str,
        help="Directory to save the quamtized model and reqired parameters for inferance.",
    )

    parser.add_argument(
        "--seq_length",
        default=hls_param.seq_length , 
        type=int,
        help="Sequence length to use when training.",
    )

    parser.add_argument(
        "--batch_size", 
        default= hls_param.batch_size,  
        type=int, 
        help="Batch size to use when training."
    )

    parser.add_argument(
        "--epochs", 
        default= hls_param.epochs , 
        type=int, 
        help="Number of training epochs."
    )

    parser.add_argument(
        "--fine_tune", 
        default= hls_param.fine_tune, 
        type=bool, 
        help="Fine tuning the model."
    )

    parser.add_argument(
        "--quantization",
        default= hls_param.quantization, 
        type=bool,
        help="Apply quantization to the model.",
    )

    parser.add_argument(
        "--quantization_criterion",
        default=hls_param.quantization_criterion, 
        type=float,
        help="Tolerance for quantization",
    )

    parser.add_argument(
        "--quantization_max_trial",
        default=hls_param.quantization_max_trial, 
        type=int,
        help="Max iteration number for quantization",
    )

    parser.add_argument(
        "--bf16",
        default= hls_param.bf16, 
        action='store_true',
        help="Apply bf16 data type.",
    )

    return parser.parse_args()
