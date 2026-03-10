import torch
import clip
import numpy as np


class CLIPEncoder:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_image(self, image):

        img = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(img)

        return embedding.cpu().numpy()

    def encode_text(self, text):

        tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokens)

        return embedding.cpu().numpy()