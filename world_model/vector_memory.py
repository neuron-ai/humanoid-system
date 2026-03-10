import numpy as np


class VectorMemory:

    def __init__(self):

        self.embeddings = {}

    def add(self, obj_id, embedding):

        self.embeddings[obj_id] = embedding

    def search(self, query_embedding):

        best_id = None
        best_score = -1

        for obj_id, emb in self.embeddings.items():

            score = np.dot(query_embedding, emb)

            if score > best_score:
                best_score = score
                best_id = obj_id

        return best_id, best_score