import json


class ContextBuilder:

    def build(self, world_model):

        objects = world_model.get_objects()

        simplified = []

        for o in objects:

            simplified.append({
                "id": o["id"],
                "label": o["label"],
                "position": o.get("world_coord")
            })

        return json.dumps(simplified)