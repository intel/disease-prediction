import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification
from neural_compressor.utils.pytorch import load


def get_subject_id(image_name):
    image_name = image_name.split("/")[-1]
    patient_id = "".join(image_name.split("_")[:2])[1:]
    return [patient_id]


def load_tokenizer(saved_model_dir):
    if not os.path.exists(saved_model_dir):
        Warning("Saved model %s not found!", saved_model_dir)
        return

    return AutoTokenizer.from_pretrained(saved_model_dir)


def get_reverse_mapping(folder):
    json_file = os.path.join(folder, "../../output/nlp_stat.json")
    # Opening JSON file
    dict_json = json.load(open(json_file))
    return (
        dict_json["eval_accuracy"],
        dict_json["reverse_label_map"],
        dict_json["quantization"],
    )


def load_model_from_file(nlp_int8_inference, saved_model_dir, test_loader):
    if nlp_int8_inference:
        config = BertConfig.from_json_file(os.path.join(saved_model_dir, "config.json"))
        model = BertForSequenceClassification(config=config)
        model = load(saved_model_dir, model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)

    # JIT model for faster execution
    batch = next(iter(test_loader))
    token_ids = batch[0]["input_ids"]
    mask = batch[0]["attention_mask"]
    jit_inputs = (token_ids, mask)

    model.eval()
    model = torch.jit.trace(model, jit_inputs, check_trace=False, strict=False)
    model = torch.jit.freeze(model)

    return model


def load_model_param(image_name, saved_models_quantization, saved_models_inference, nlp_int8_inference):
    patient_ids = get_subject_id(image_name)
    folder = os.path.dirname(__file__)
    eval_accuracy, reverse_label_map, quantization = get_reverse_mapping(folder)

    if nlp_int8_inference:
        saved_model_dir = os.path.join(folder, saved_models_quantization)
    else:
        saved_model_dir = os.path.join(folder, saved_models_inference)

    tokenizer = load_tokenizer(saved_model_dir)

    return (
        quantization,
        saved_model_dir,
        tokenizer,
        reverse_label_map,
        patient_ids,
        folder,
        eval_accuracy,
    )
