import os
import clip
import torch
import numpy as np

def get_label_list(input_dir):
    images = [os.path.join(input_dir, file_path) for file_path in os.listdir(input_dir)]
    label_list = []
    for image in images:
        if len(os.path.split(image)[1].split("-")) == 1:
            continue
        else:
            label = os.path.split(image)[1].split("-")[-2]
            if label not in label_list:
                label_list.append(label)
    return label_list

class ClipPipeline():
    def __init__(self, input_dir, device) -> None:
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.labels = get_label_list(input_dir)
        # self.labels = ["Fish", "Rabbit", "Butterfly", "Bird", "Cat", "Dog", "Duck", "Bee", "Owl", "Frog"]

    def forward(self, image):
        img = self.preprocess(image).unsqueeze(0).to(self.device)

        # labels = get_label_list(input_dir)
        txt = clip.tokenize(self.labels).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(txt)
            
            logits_per_image, logits_per_text = self.model(img, txt)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        idx = np.argmax(probs)
        print(f"Predicted label {self.labels[idx]} has the probality of {probs[0][idx]*100}%")
        label = self.labels[idx]
        prob = probs[0][idx]

        return label, prob


