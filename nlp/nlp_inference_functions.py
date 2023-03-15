import os
import json

from utils.nlp_load_model import load_model_param, load_model_from_file

from utils.nlp_prediction import create_prediction

from utils.load_parameter import hls_param as PARAM

from utils.load_data import get_data


def save_predictions_results(predictions):
    predictions_file = os.path.join(
        os.path.dirname(__file__), "../output/predictions_results.json"
    )

    predictions = [str(i) for i in predictions]
    predictions_dict = json.dumps({"predictions": predictions})

    # Writing to json file
    with open(predictions_file, "w") as outfile:
        outfile.write(predictions_dict)

    print("Results are saved to ", predictions_file)


def nlp_inference(
    saved_models_quantization=PARAM.saved_models_quantization,
    saved_models_inference=PARAM.saved_models_inference,
    data_column=PARAM.data_column,
    dataset_path_and_name=PARAM.dataset_path_and_name,
    label_column=PARAM.label_column,
    patient_id_column=PARAM.patient_id_column,
    max_seq_length=PARAM.seq_length,
    batch_size=PARAM.batch_size,
    image_name=PARAM.image_name,
    nlp_int8_inference = PARAM.nlp_int8_inference
):
    # get the model related items need for prediction
    (
        quantization,
        saved_model_dir,
        tokenizer,
        reverse_label_map,
        patient_ids,
        folder,
        eval_accuracy,
    ) = load_model_param(image_name, saved_models_quantization, saved_models_inference, nlp_int8_inference)

    # get test data
    image_data_folder = None

    _, _, _, _, test_loader = get_data(
        tokenizer,
        folder,
        dataset_path_and_name,
        image_data_folder,
        label_column,
        data_column,
        patient_id_column,
        max_seq_length,
        batch_size,
        patient_id_list=patient_ids,
    )

    # load the pre-trained model
    model = load_model_from_file(nlp_int8_inference, saved_model_dir, test_loader)

    # create prediction
    predictions = create_prediction(
        model, test_loader, reverse_label_map, patient_ids, eval_accuracy
    )

    save_predictions_results(predictions)
    