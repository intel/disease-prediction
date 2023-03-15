import argparse
import sys
from os import path

root_folder = path.dirname(path.abspath(__file__))
sys.path.insert(0, path.join(root_folder, "tlt_toolkit") )

from brca_prediction import train_vision_wl

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size", 
        default= 16,  
        type=int, 
        help="Batch size to use when training."
    )

    parser.add_argument(
        "--epochs", 
        default= 5 , 
        type=int, 
        help="Number of training epochs."
    )

    parser.add_argument(
        "--dataset_dir",
        default= '/data/datad/mcetin/brca_multimodal/images/train', 
        type=str,
        help="Directory to save trainer outputs.",
    )

    parser.add_argument(
        "--output_dir",
        default= '/data/datad/mcetin/brca_multimodal/output', 
        type=str,
        help="Directory to save trainer outputs.",
    )

    parser.add_argument(
        "--save_model",
        default= False, 
        action='store_false',
        help="Save model options.",
    )

    parser.add_argument(
        "--quantization",
        default= False, 
        action='store_false',
        help="Apply quantization to the model.",
    )

    param = parser.parse_args()
    
    train_vision_wl(param.dataset_dir, param.output_dir, param.batch_size, param.epochs, param.save_model, param.quantization)
