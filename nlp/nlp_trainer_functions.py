import pathlib
import logging
import torch
import os
import numpy as np
import json
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
)

from intel_extension_for_transformers import (
    metrics,
    objectives,
    QuantizationConfig,
)
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

try:
    from utils.load_data import get_data
    from utils.load_parameter import hls_param as PARAM
except ModuleNotFoundError:
    from nlp.utils.load_data import get_data
    from nlp.utils.load_parameter import hls_param as PARAM

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_DISABLED"] = "true"


def save_nlp_stat(eval_acc, reverse_label_map, quantization):
    stat_dict = json.dumps(
        {
            "eval_accuracy": eval_acc,
            "reverse_label_map": reverse_label_map,
            "quantization": quantization,
        }
    )

    # Writing to json file
    with open(
        os.path.join(os.path.dirname(__file__), "../output/nlp_stat.json"), "w"
    ) as outfile:
        outfile.write(stat_dict)


def save_model(tokenizer, model, target_folder):
    model_folder = os.path.join(os.path.dirname(__file__), target_folder)
    # save model ----------------------
    path = pathlib.Path(model_folder)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)
    return model_folder


def save_quantized_model(model, model_folder, target_folder):
    output_dir = os.path.join(os.path.dirname(__file__), target_folder)
    model.save(output_dir)
    shutil.copytree(
        model_folder,
        output_dir,
        ignore=shutil.ignore_patterns("*.bin*"),
        dirs_exist_ok=True,
    )
    return output_dir


def set_log_file(logfile):
    path = pathlib.Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    logger = logging.getLogger()
    return logger


def get_trainer(
    model,
    epochs,
    batch_size,
    output_folder,
    train_dataset,
    test_dataset,
    tokenizer,
):
    training_args = TrainingArguments(
        output_dir=output_folder,
        do_eval=True,
        do_train=True,
        no_cuda=True,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
    )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = NLPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    return trainer


def get_quantization(quantization_criterion, quantization_max_trial, trainer):
    # Get the metric function
    tune_metric = metrics.Metric(
        name="eval_accuracy",
        greater_is_better=True,
        is_relative=True,
        criterion=quantization_criterion,
        weight_ratio=None,
    )

    objective = objectives.Objective(
        name="performance", greater_is_better=True, weight_ratio=None
    )

    quantization_config = QuantizationConfig(
        # approach="PostTrainingStatic",
        approach="PostTrainingDynamic",
        max_trials=quantization_max_trial,
        metrics=[tune_metric],
        objectives=[objective],
    )

    model = trainer.quantize(quant_config=quantization_config)

    return model


def nlp_trainer(
    model_name_or_path=PARAM.model_name_or_path,
    saved_models_quantization=PARAM.saved_models_quantization,
    saved_models_inference=PARAM.saved_models_inference,
    data_column=PARAM.data_column,
    logfile=PARAM.logfile,
    output_dir=PARAM.output_dir,
    epochs=PARAM.epochs,
    fine_tune=PARAM.fine_tune,
    quantization=PARAM.quantization,
    quantization_criterion=PARAM.quantization_criterion,
    quantization_max_trial=PARAM.quantization_max_trial,
    bf16=PARAM.bf16,
    dataset_path_and_name=PARAM.dataset_path_and_name,
    image_data_folder=PARAM.image_data_folder,
    label_column=PARAM.label_column,
    patient_id_column=PARAM.patient_id_column,
    max_seq_length=PARAM.seq_length,
    batch_size=PARAM.batch_size,
):
    # set logger
    logger = set_log_file(logfile)

    # Load the tokenizer
    logger.info("Loading the  %s tokenizer", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # create data sets
    logger.info("Creating the data sets")
    folder = os.path.dirname(__file__)
    train_dataset, test_dataset, num_labels, reverse_label_map, _ = get_data(
        tokenizer,
        folder,
        dataset_path_and_name,
        image_data_folder,
        label_column,
        data_column,
        patient_id_column,
        max_seq_length,
        batch_size,
    )

    logger.info("Loading %s as a pretrained BERT", model_name_or_path)
    # Load the C BERT sequence classifier
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )

    if bf16:
        import intel_extension_for_pytorch as ipex
        # model = ipex.optimize(model, dtype=torch.bfloat16, level='O0')
        model = ipex.optimize(model, dtype=torch.bfloat16, level="O1")

    output_folder = os.path.join(os.path.dirname(__file__), output_dir)
    trainer = get_trainer(
        model, epochs, batch_size, output_folder, train_dataset, test_dataset, tokenizer
    )

    # fine tuning with transfer learning
    if fine_tune:
        print("Started finetuning the NLP model ......")
        trainer.train()

    # save model ----------------------
    model_folder = save_model(tokenizer, model, saved_models_inference)
    logger.info("Saved model files to %s", model_folder)

    # apply quantization from nlp_toolkit
    if quantization:
        model = get_quantization(
            quantization_criterion, quantization_max_trial, trainer
        )

    results = trainer.evaluate()
    eval_acc = results.get("eval_accuracy")
    print("Finally eval_accuracy Accuracy: {:.5f}".format(eval_acc))

    # save acc score and reverse label map to json file
    save_nlp_stat(eval_acc, reverse_label_map, quantization)

    if quantization:
        model_folder = save_quantized_model(
            model, model_folder, saved_models_quantization
        )
        logger.info("Saved quantized model files to %s", model_folder)

    print(">>> Training is completed .... ")
