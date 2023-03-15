import sys
from os import path, listdir

root_folder = path.dirname(path.abspath(__file__))
sys.path.insert(0, path.join(root_folder, "vision/tlt_toolkit"))

import argparse
import tensorflow as tf
import pandas as pd
import time


from vision.brca_prediction import (
    train_vision_wl,
    infer_vision_wl,
    infer_int8_vision_wl,
    collect_class_labels,
)
from tlt.models import model_factory

from nlp.utils.nlp_prediction import create_prediction
from nlp.utils.load_parameter import hls_param
from nlp.utils.nlp_load_model import (
    load_model_param,
    load_model_from_file,
    get_subject_id,
)
from nlp.utils.load_data import get_data
from nlp.nlp_trainer_functions import nlp_trainer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def ensemble_score(nlp, vis):
    nlp_acc = nlp[0]["model_accuracy"]
    vis_acc = vis[0]["model_accuracy"]
    ensb_prognosis = {}
    for i in nlp[0]["prognosis"].keys():
        ensb_prognosis[i] = (
            nlp_acc * nlp[0]["prognosis"][i] + vis_acc * vis[0]["prognosis"][i]
        )
    return max(ensb_prognosis, key=ensb_prognosis.get)


def post_process_ensemble(df_pp, labels, nlp_int8_inference):
    nlp_model, nlp_param = None, None
    df_g = df_pp.groupby("item")
    df_list = []

    for k in df_g.groups.keys():
        df_temp = df_g.get_group(k)
        prognosis = {}

        # Ensemble vision segments to single vision prediction
        for label_name in labels:
            for idx in df_temp.index:
                prognosis[label_name] = (
                    prognosis.get(label_name, 0)
                    + df_pp.loc[idx, "result_vision"][0]["prognosis"][
                        label_name
                    ].numpy()
                )

            prognosis[label_name] = prognosis[label_name] / len(df_temp)

        pred_vision = max(prognosis, key=prognosis.get)

        # NLP prediction
        result_NLP, nlp_model, nlp_param = nlp_prediction(
            df_pp.loc[idx, "file_loc"], nlp_model, nlp_param, nlp_int8_inference
        )
        if len(result_NLP) != 0:
            pred_nlp = max(
                result_NLP[0]["prognosis"], key=result_NLP[0]["prognosis"].get
            )
        else:
            pred_nlp = "None"
            result_NLP = df_pp.loc[idx, "result_vision"]

        # Ensemble sub model prediction
        pred_ensemble = ensemble_score(result_NLP, df_pp.loc[idx, "result_vision"])

        df_list.append(
            [k, df_pp.loc[idx, "truth"], pred_vision, pred_nlp, pred_ensemble]
        )

    return pd.DataFrame(
        df_list, columns=["item", "truth", "pred_vision", "pred_nlp", "pred_ensemble"]
    )


def nlp_prediction(fn, nlp_model, nlp_param, nlp_int8_inference):
    patient_ids = get_subject_id(fn)
    if nlp_model is None:
        param = hls_param
        (
            _,
            saved_model_dir,
            tokenizer,
            reverse_label_map,
            _,
            folder,
            eval_accuracy,
        ) = load_model_param(
            param.image_name,
            param.saved_models_quantization,
            param.saved_models_inference,
            nlp_int8_inference
        )

        _, _, _, _, test_loader = get_data(
            tokenizer,
            folder,
            param.dataset_path_and_name,
            param.image_data_folder,
            param.label_column,
            param.data_column,
            param.patient_id_column,
            param.seq_length,
            param.batch_size,
            patient_id_list=patient_ids,
        )
        nlp_model = load_model_from_file(nlp_int8_inference, saved_model_dir, test_loader)
        nlp_param = {
            "tokenizer": tokenizer,
            "param": param,
            "folder": folder,
            "reverse_label_map": reverse_label_map,
            "eval_accuracy": eval_accuracy,
        }

    else:
        _, _, _, _, test_loader = get_data(
            nlp_param["tokenizer"],
            nlp_param["folder"],
            nlp_param["param"].dataset_path_and_name,
            nlp_param["param"].image_data_folder,
            nlp_param["param"].label_column,
            nlp_param["param"].data_column,
            nlp_param["param"].patient_id_column,
            nlp_param["param"].seq_length,
            nlp_param["param"].batch_size,
            patient_id_list=patient_ids,
        )

    # create prediction
    predictions = create_prediction(
        nlp_model,
        test_loader,
        nlp_param["reverse_label_map"],
        patient_ids,
        nlp_param["eval_accuracy"],
    )

    return predictions, nlp_model, nlp_param


def create_confusion_matrix(df, labels, pred_col):
    cm_res = classification_report(
        df.truth.to_list(), df[pred_col].to_list(), output_dict=True
    )
    # print(cm_res)
    df_cm_res = pd.DataFrame(cm_res).transpose().round(3)

    cm = confusion_matrix(df.truth.to_list(), df[pred_col].to_list(), labels=labels)
    df_cm = pd.DataFrame(cm, columns=labels, index=labels)
    df_cm["Precision"] = None
    df_cm.loc["Recall"] = df_cm_res.loc[labels + ["accuracy"], "recall"].to_list()
    df_cm["Precision"] = df_cm_res.loc[labels + ["accuracy"], "precision"].to_list()

    return df_cm


def run_inference(data_dir, class_labels, vision_int8_inference, nlp_int8_inference):
    # Load the vision model
    tstart = time.time()
    vision_model_dir = path.join(root_folder, "output/resnet_v1_50/1")
    vision_model = model_factory.load_model(
        "resnet_v1_50", vision_model_dir, "tensorflow", "image_classification"
    )

    if vision_int8_inference:
        vision_int8_model = tf.saved_model.load(vision_model_dir)

    tend = time.time()
    print("\n Vision Model Loading time: ", tend - tstart)

    # Check the NLP model
    # if models are not created run the training algorithims first
    nlp_model_folder = path.join(root_folder, "output/saved_models_inference/")
    if not path.exists(nlp_model_folder):
        raise ValueError(
            "Train model is not exist for NLP workload in '"
            + nlp_model_folder
            + "' location. Please train the NLP model first."
        )

    test_dir = path.join(data_dir, "test")
    df_pp = pd.DataFrame(columns=["item", "truth", "result_vision", "file_loc"])

    tstart = time.time()
    for label in listdir(test_dir):
        print("Infering data in folder: ", label)
        fns = listdir(path.join(test_dir, label))
        for fn in fns:
            fn = path.join(path.join(test_dir, label, fn))
            # ------------------------
            # call inference on vision WL
            # ------------------------
            if vision_int8_inference:
                result_vision = infer_int8_vision_wl(vision_int8_model, fn)
            else:
                result_vision = infer_vision_wl(vision_model, fn)

            result_vision = [
                {
                    "model_accuracy": 0.7,
                    "prognosis": dict(zip(class_labels, result_vision)),
                }
            ]

            df_pp.loc[len(df_pp)] = [get_subject_id(fn)[0], label, result_vision, fn]
    
    print("Vision inference time: ", time.time() - tstart)

    # Apply ensemble in different layers ----
    tstart = time.time()
    df_pred = post_process_ensemble(df_pp, class_labels, nlp_int8_inference)
    print("NLP inference time: ", time.time() - tstart)

    # Results -------------------------------
    print("------ Confusion Matrix for Vision model -----")
    print(create_confusion_matrix(df_pred, class_labels, "pred_vision"))

    print("------ Confusion Matrix for NLP model --------")
    print(create_confusion_matrix(df_pred, class_labels, "pred_nlp"))

    print("------ Confusion Matrix for Ensemble --------")
    print(create_confusion_matrix(df_pred, class_labels, "pred_ensemble"))

    print("done")


def wrapper(
    vision_finetune,
    vision_epochs,
    vision_quantization,
    nlp_finetune,
    nlp_bf16,
    nlp_quantization,
    nlp_epochs,
    inference,
    vision_int8_inference,
    nlp_int8_inference,
):
    print("inside wrapper................")
    data_dir = path.join(root_folder, "data/train_test_split_images")
    model_dir = path.join(root_folder, "output")

 
    class_labels = collect_class_labels(path.join(data_dir, "train"))

    # -----------------------------------------------
    # Script to call the finetuning -----------------
    if vision_finetune:
        # Finetune vision WL
        _ = train_vision_wl(
            path.join(data_dir, "train"),
            model_dir,
            epochs=vision_epochs,
            quantization=vision_quantization,
        )

    if nlp_finetune:
        # finetune NLP WL
        nlp_trainer(bf16=nlp_bf16, quantization=nlp_quantization, epochs=nlp_epochs, image_data_folder=path.join(data_dir, "train"))

    if inference:
        run_inference(data_dir, class_labels, vision_int8_inference, nlp_int8_inference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This pipeline perform End-to-End Multi-Modal classification for Breast Cancer."
    )

    # == Vision ==========================================================
    parser.add_argument(
        "--vision_finetune", action="store_true", help="Perform Vision Finetune"
    )

    parser.add_argument(
        "--vision_quantization",
        action="store_true",
        help="Create quantized vision model.",
        default=False,
    )

    parser.add_argument(
        "--vision_epochs", help="Number of training epochs.", default=50
    )

    # == NLP ==============================================================
    parser.add_argument(
        "--nlp_finetune",
        action="store_true",
        help="Perform NLP Finetune",
    )

    parser.add_argument(
        "--nlp_bf16", action="store_true", help="Apply bf16 data type.", default=False
    )

    parser.add_argument(
        "--nlp_quantization",
        action="store_true",
        help="Create quantized NLP model.",
        default=False,
    )

    parser.add_argument("--nlp_epochs", help="Number of training epochs..", default=4)

    # == Inferance ======================================================
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Perform Inference on Vision and NLP pipeline",
    )

    parser.add_argument(
        "--vision_int8_inference",
        action="store_true",
        help="Perform INT8 Inference on Vision",
    )

    parser.add_argument(
        "--nlp_int8_inference",
        action="store_true",
        help="Perform INT8 Inference on NLP",
    )

    params = parser.parse_args()

    print("\n\n----------You Selected to perform the following: -------")
    print("vision_finetune ", "         = ", params.vision_finetune)
    print("vision_epochs ", "           = ", params.vision_epochs)
    print("vision_quantization ", "     = ", params.vision_quantization)
    print("-------------------------------------------------------------")
    print("nlp_finetune","              = ", params.nlp_finetune)
    print("nlp_bf16 ", "                = ", params.nlp_bf16)
    print("nlp_quantization ", "        = ", params.nlp_quantization)
    print("nlp_epochs ", "              = ", params.nlp_epochs)
    print("-------------------------------------------------------------")
    print("inference ", "               = ", params.inference)
    print("vision_int8_inference ", "   = ", params.vision_int8_inference)
    print("nlp_int8_inference ", "      = ", params.nlp_int8_inference)
    print("---------------------------------------------------------\n\n")

    wrapper(
        params.vision_finetune,
        int(params.vision_epochs),
        params.vision_quantization,
        params.nlp_finetune,
        params.nlp_bf16,
        params.nlp_quantization,
        int(params.nlp_epochs),
        params.inference,
        params.vision_int8_inference,
        params.nlp_int8_inference,
    )
