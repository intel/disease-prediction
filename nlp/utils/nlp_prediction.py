import torch
import numpy as np
import json
import os


def predict(model, batch) -> torch.Tensor:
    """Predicts the output for the given batch
        using the given PyTorch model.

    Args:
        batch (torch.Tensor): data batch from data loader
            transformers tokenizer

    Returns:
        torch.Tensor: predicted quantities
    """

    return model(
        input_ids=batch[0]["input_ids"],
        attention_mask=batch[0]["attention_mask"],
    )


def create_prediction(
    model, test_loader, reverse_label_map, patient_ids, model_accuracy
):

    predictions = []
    index = 0

    for _, batch in enumerate(test_loader):
        pred_probs = (
            torch.softmax(predict(model, batch)["logits"], axis=1).detach().numpy()
        )

        for i in range(len(pred_probs)):
            probs = {
                reverse_label_map[str(x)]: pred_probs[i, x]
                for x in np.argsort(pred_probs[i, :])[::-1][:5]
            }
            predictions.append(
                {
                    "id": patient_ids[index],
                    "model_accuracy": model_accuracy,
                    "prognosis": probs,
                }
            )
            index += 1
    """
    print("Predictions:")
    for i in predictions:
        print(i)
    """
    return predictions
