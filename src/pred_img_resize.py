import tqdm
import pandas as pd
import cv2
import os


def pred_img_resize():
    test_images = "E:/Coding_stuffs/Python/notebooks/Medical_image_classification/data/raw/histopathologic-cancer-detection/test/"
    test_files = pd.DataFrame(
        {"file_path": [test_images + f for f in os.listdir(test_images)]}
    )
    resized_file_paths = []

    for idx, row in tqdm.tqdm(test_files.iterrows(), total=len(test_files)):
        orig_path = row["file_path"]
        file_id = os.path.splitext(os.path.basename(orig_path))[0]
        save_path = os.path.join(
            "E:/Coding_stuffs/Python/notebooks/Medical_image_classification/data/processed/test_images_resized",
            f"{file_id}.png",
        )

        if not os.path.exists(save_path):
            img = cv2.imread(orig_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (50, 50))
            cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

        resized_file_paths.append(save_path)

    return resized_file_paths


reszied_file_paths = pred_img_resize()
print(reszied_file_paths[:5])
