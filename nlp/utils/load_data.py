import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import Dataset

import torch

from transformers import PreTrainedTokenizer
from typing import List


class DiseasePrognosisDataset(Dataset):
    """Dataset with symptom strings to predict disease.

    Args:
        symptoms (List[str]): list of symptom strings
        prognosis (List[str]): list of corresponding prognosis
    """

    def __init__(
        self,
        symptoms: List[str],
        prognosis: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 64,
    ):
        self.symptoms = symptoms
        self.prognosis = prognosis
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.symptoms)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.symptoms[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.prognosis is not None:
            return (item, torch.as_tensor(self.prognosis[idx]))
        return item


def prepare_data(data, tokenizer, max_seq_length, label_column, data_column):
    # Padding strategy
    padding = False

    def preprocess_function(examples):
        args = (examples["symptoms"],)
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )
        return result

    df = pd.DataFrame(columns=["label", "symptoms"])
    df["label"] = data[label_column]
    df["symptoms"] = data[data_column]

    return Dataset.from_pandas(df).map(preprocess_function, batched=True)


def get_subject_id(image_name):
    image_name = image_name.split("/")[-1]
    patient_id = "".join(image_name.split("_")[:2])[1:]
    return patient_id


def read_annotation(patient_id, param, annotation_file):
    df = pd.read_csv(annotation_file)
    return " ".join(
        df[df[param.patient_id_column].isin([patient_id])][param.data_column].to_list()
    )


def create_patient_id_list(image_data_folder, folder):
    folder_pth = os.path.join(folder, image_data_folder)
    patient_id_list = []
    for fldr in os.listdir(folder_pth):
        for f in os.listdir(os.path.join(folder_pth, fldr)):
            patient_id_list.append(get_subject_id(f))

    return np.unique(patient_id_list)


def get_val_loader(
    df, data_column, label_column, tokenizer, max_seq_length, batch_size
):
    loader = DiseasePrognosisDataset(
        df[data_column].values,
        df[label_column].values,
        tokenizer,
        max_length=max_seq_length,
    )

    val_loader = torch.utils.data.DataLoader(
        loader, batch_size=batch_size, shuffle=False
    )

    return val_loader


def read_annotation_file(
    folder,
    dataset_path_and_name,
    label_column,
    data_column,
    patient_id,
    patient_id_list,
    image_data_folder,
):
    df_path = os.path.join(folder, dataset_path_and_name)
    df = pd.read_csv(df_path)
    label_map, reverse_label_map = label2map(df, label_column)

    if patient_id_list is not None:
        df = df[df[patient_id].isin(patient_id_list)]
    else:
        image_name_list = []
        for label in os.listdir(image_data_folder):
            image_name_list.extend(os.listdir(os.path.join(image_data_folder, label)))
        df = df[
            df[patient_id].isin(np.unique([get_subject_id(i) for i in image_name_list]))
        ]
 
    return df, label_map, reverse_label_map # df_new


def label2map(df, label_column):
    label_map, reverse_label_map = {}, {}
    for i, v in enumerate(df[label_column].unique().tolist()):
        label_map[v] = i
        reverse_label_map[i] = v

    return label_map, reverse_label_map


def create_train_test_set(df, patient_id, patient_id_list):
    train_label, test_label = train_test_split(
        patient_id_list, test_size=0.33, random_state=42
    )

    df_test = df[df[patient_id].isin(test_label)]
    df_train = df[df[patient_id].isin(train_label)]

    return df_train, df_test


def get_data(
    tokenizer,
    folder,
    dataset_path_and_name,
    image_data_folder,
    label_column,
    data_column,
    patient_id,
    max_seq_length,
    batch_size,
    patient_id_list=None,
):
    df, label_map, reverse_label_map = read_annotation_file(
        folder,
        dataset_path_and_name,
        label_column,
        data_column,
        patient_id,
        patient_id_list,
        image_data_folder,
    )

    # apply label map
    mapped_labels = [label_map[i] for i in df[label_column].to_list() if i is not None]
    if len(mapped_labels) > 0:
        df[label_column] = mapped_labels

    # creating patient list from given location
    if patient_id_list is None:
        patient_id_list = create_patient_id_list(image_data_folder, folder)

        df_train, df_test = create_train_test_set(df, patient_id, patient_id_list)

        train_dataset = prepare_data(
            df_train, tokenizer, max_seq_length, label_column, data_column
        )

        test_dataset = prepare_data(
            df_test, tokenizer, max_seq_length, label_column, data_column
        )

    else:
        train_dataset = None
        df_test = df
        test_dataset = prepare_data(
            df, tokenizer, max_seq_length, label_column, data_column
        )

    val_loader = get_val_loader(
        df_test, data_column, label_column, tokenizer, max_seq_length, batch_size
    )

    return (
        train_dataset,
        test_dataset,
        len(df[label_column].unique()),
        reverse_label_map,
        val_loader,
    )
