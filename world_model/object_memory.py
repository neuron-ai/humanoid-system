import time


class ObjectMemory:

    def __init__(self, ttl=5.0):
        self.objects = {}
        self.ttl = ttl

    def update(self, obj):
        obj_id = obj["id"]

        obj["last_seen"] = time.time()

        if obj_id not in self.objects:
            self.objects[obj_id] = obj
        else:
            self.objects[obj_id].update(obj)

    def remove_stale(self):
        now = time.time()

        to_delete = []

        for obj_id, obj in self.objects.items():
            if now - obj["last_seen"] > self.ttl:
                to_delete.append(obj_id)

        for obj_id in to_delete:
            del self.objects[obj_id]

    def get(self, obj_id):
        return self.objects.get(obj_id)

    def all(self):
        return list(self.objects.values())