class SceneGraph:

    def __init__(self):

        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, node_type, data=None):

        self.nodes[node_id] = {
            "type": node_type,
            "data": data or {}
        }

    def add_relation(self, parent, child, relation):

        self.edges.append({
            "parent": parent,
            "child": child,
            "relation": relation
        })

    def get_relations(self, node_id):

        rel = []

        for edge in self.edges:

            if edge["parent"] == node_id or edge["child"] == node_id:
                rel.append(edge)

        return rel

    def clear(self):
        self.edges = []