from src.dataloaders import create_data_loaders
from src.training import train_model
import pandas as pd
import torch
from src.model import CNNmodel
from src.pred_img_resize import pred_img_resize


def main_training():
    labels = pd.read_csv(
        "E:/Coding_stuffs/Python/notebooks/Medical_image_classification/data/raw/histopathologic-cancer-detection/train_labels.csv"
    )
    labels["file_path"] = (
        "E:/Coding_stuffs/Python/notebooks/Medical_image_classification/data/processed/medical_images_resized/"
        + labels["id"]
        + ".png"
    )

    train_loader, val_loader = create_data_loaders(labels, batch_size=128)

    train_model(train_loader, val_loader, epochs=50, lr=0.001, device="cuda")


def main_preds():
    test_files_list = pred_img_resize()

    test_loader = create_data_loaders(test_files_list, batch_size=128, labelled=False)

    model = CNNmodel()
    model.load_state_dict(
        torch.load(
            "E:/Coding_stuffs/Python/notebooks/Medical_image_classification/models/model.pth"
        )
    )
    model.to("cuda")
    model.eval()

    threshold = 0.85
    predictions = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to("cuda")
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            predictions.extend(preds.flatten().tolist())

    test_files_df = pd.DataFrame({"file_path": test_files_list})
    test_files_df["label"] = predictions
    test_files_df.to_csv(
        "E:/Coding_stuffs/Python/notebooks/Medical_image_classification/data/processed/test_predictions/test_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main_preds()
