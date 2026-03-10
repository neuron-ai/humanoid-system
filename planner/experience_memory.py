import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import json


class ExperienceMemory:

    def __init__(
        self,
        db_path="experience_memory.db",
        embedding_model="all-MiniLM-L6-v2"
    ):

        # embedding model
        self.model = SentenceTransformer(embedding_model)

        # sqlite database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self._init_db()

        # FAISS index
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)

        self.id_map = []

        self._load_index()

    def _init_db(self):

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiences(
            id INTEGER PRIMARY KEY,
            instruction TEXT,
            goal TEXT,
            tasks TEXT,
            success INTEGER
        )
        """)

        self.conn.commit()

    def _load_index(self):

        self.cursor.execute("SELECT id, instruction FROM experiences")

        rows = self.cursor.fetchall()

        if not rows:
            return

        texts = [r[1] for r in rows]

        embeddings = self.model.encode(texts)

        self.index.add(np.array(embeddings).astype("float32"))

        self.id_map = [r[0] for r in rows]

    def add_experience(self, instruction, goal, tasks, success):

        tasks_json = json.dumps(tasks)

        self.cursor.execute("""
        INSERT INTO experiences (instruction, goal, tasks, success)
        VALUES (?, ?, ?, ?)
        """, (instruction, goal, tasks_json, int(success)))

        self.conn.commit()

        exp_id = self.cursor.lastrowid

        embedding = self.model.encode([instruction])[0]

        self.index.add(np.array([embedding]).astype("float32"))

        self.id_map.append(exp_id)

    def retrieve(self, instruction, k=3):

        if len(self.id_map) == 0:
            return []

        query = self.model.encode([instruction])

        distances, indices = self.index.search(
            np.array(query).astype("float32"),
            k
        )

        results = []

        for idx in indices[0]:

            exp_id = self.id_map[idx]

            self.cursor.execute("""
            SELECT instruction, goal, tasks, success
            FROM experiences WHERE id=?
            """, (exp_id,))

            row = self.cursor.fetchone()

            if row:

                results.append({
                    "instruction": row[0],
                    "goal": row[1],
                    "tasks": json.loads(row[2]),
                    "success": bool(row[3])
                })

        return results