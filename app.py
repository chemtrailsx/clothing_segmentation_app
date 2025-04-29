from flask import Flask, render_template, request
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import os

app = Flask(__name__)

# Load model and processor once
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Color map for 19 classes
colors = np.array([
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0]
])

# Clothing class labels
id2label = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper-clothes",
    5: "dress", 6: "coat", 7: "socks", 8: "pants", 9: "jumpsuits", 10: "scarf",
    11: "skirt", 12: "face", 13: "left-arm", 14: "right-arm",
    15: "left-leg", 16: "right-leg", 17: "left-shoe", 18: "right-shoe"
}

# Set of labels to keep (main clothing items only)
clothing_labels = {
    "upper-clothes", "dress", "coat", "socks", "pants", "jumpsuits",
    "scarf", "skirt", "hat", "left-shoe", "right-shoe"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_url = request.form["image_url"]

        # Load and preprocess image
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()

        upsampled_logits = F.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        segmentation_colored = colors[pred_seg]

        # Save the segmentation result
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Original")

        plt.subplot(1, 2, 2)
        plt.imshow(segmentation_colored.astype(np.uint8))
        plt.axis("off")
        plt.title("Clothing Segmentation")

        output_path = os.path.join("static", "output.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        # Clean up old item images
        for file in os.listdir("static"):
            if file.startswith("item_") and file.endswith(".png"):
                os.remove(os.path.join("static", file))

        # Save individual clothing items from real image with transparency
        unique_ids = np.unique(pred_seg)
        present_labels = [label for label in unique_ids if id2label[label] in clothing_labels]

        image_np = np.array(image)
        highlighted_image_paths = []

        for label_id in present_labels:
            label_name = id2label[label_id]
            mask = (pred_seg == label_id)

            # Create RGBA image from original with transparency
            masked_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
            masked_image[..., :3] = image_np
            masked_image[..., 3] = np.where(mask, 255, 0)  # Alpha channel

            item_image = Image.fromarray(masked_image, mode="RGBA")
            single_item_path = os.path.join("static", f"item_{label_id}.png")
            item_image.save(single_item_path)

            highlighted_image_paths.append((label_name, single_item_path))

        return render_template(
            "index.html",
            segmented_image=output_path,
            image_url=image_url,
            highlighted_images=highlighted_image_paths
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True) 
