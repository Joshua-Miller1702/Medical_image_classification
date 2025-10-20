from src.model import CNNmodel
import torch
from torch import optim
from torch import nn
import tqdm


def train_model(
    train_loader,
    val_loader,
    epochs=10,
    lr=0.001,
    device="cuda",
    checkpoint_path="E:/Coding_stuffs/Python/notebooks/Medical_image_classification/models/model.pth",
):
    # setup
    device = device
    model = CNNmodel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    # train
    for epoch in range(epochs):
        model.train()
        (
            train_loss,
            train_correct,
            train_total,
        ) = 0.0, 0, 0

        for imgs, labels in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"
        ):
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm.tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"
            ):
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(
            f"\nEpoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved new best model with val loss: {best_val_loss:.4f}")

    print("Training complete!")

    return model
