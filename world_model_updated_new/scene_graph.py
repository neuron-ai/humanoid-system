import logging
import os
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("SceneGraph: neo4j not installed — using in-memory graph fallback")


class SceneGraph:
    """
    Stores spatial relationships between objects as a graph.
    Primary backend: Neo4j (if available and reachable).
    Fallback: in-memory dict graph (no persistence, no extra dependencies).

    Relationships computed:
        NEAR_TO — objects within NEAR_THRESHOLD metres of each other

    Changes vs original:
    - numpy imported at module level (was imported inside _compute_relations)
    - No other functional changes — already well structured
    """

    NEAR_THRESHOLD   = 0.5
    ON_TOP_THRESHOLD = 0.3

    def __init__(self):
        self._driver    = None
        self._neo4j_ok  = False
        self._graph: Dict[str, dict] = {}
        self._relations: List[dict]  = []

        if NEO4J_AVAILABLE:
            self._init_neo4j()

    def _init_neo4j(self) -> None:
        uri  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER",     "neo4j")
        pwd  = os.environ.get("NEO4J_PASSWORD",  "password")
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self._driver.verify_connectivity()
            self._neo4j_ok = True
            logger.info("SceneGraph: Neo4j connected at %s", uri)
        except Exception as e:
            self._neo4j_ok = False
            logger.warning("SceneGraph: Neo4j unavailable (%s) — in-memory fallback", e)

    # ---------------------------------------------------------------- #
    #  Public API
    # ---------------------------------------------------------------- #

    def update(self, objects) -> None:
        """
        Rebuild scene graph from current object set.
        Accepts either dict {obj_id: obj} or list of obj dicts.
        """
        if not objects:
            return
        obj_list = list(objects.values()) if isinstance(objects, dict) else list(objects)
        if self._neo4j_ok:
            self._update_neo4j(obj_list)
        else:
            self._update_memory(obj_list)

    def get_all(self) -> List[dict]:
        """Return all relations as [{from_id, to_id, relation_type, distance}]."""
        if self._neo4j_ok:
            return self._get_relations_neo4j()
        return list(self._relations)

    def get_neighbors(self, obj_id: str) -> List[dict]:
        if self._neo4j_ok:
            return self._get_neighbors_neo4j(obj_id)
        return [r for r in self._relations
                if r.get("from_id") == obj_id or r.get("to_id") == obj_id]

    # ---------------------------------------------------------------- #
    #  Neo4j backend
    # ---------------------------------------------------------------- #

    def _update_neo4j(self, obj_list: list) -> None:
        try:
            with self._driver.session() as session:
                for obj in obj_list:
                    pos = obj.get("position") or [0.0, 0.0, 0.0]
                    session.run(
                        """
                        MERGE (o:Object {id: $id})
                        SET o.label     = $label,
                            o.x         = $x,
                            o.y         = $y,
                            o.z         = $z,
                            o.timestamp = $ts
                        """,
                        id    = str(obj["id"]),
                        label = obj.get("label", "unknown"),
                        x     = float(pos[0]),
                        y     = float(pos[1]),
                        z     = float(pos[2]),
                        ts    = obj.get("timestamp", 0.0),
                    )
                for rel in self._compute_relations(obj_list):
                    session.run(
                        """
                        MATCH (a:Object {id: $from_id}), (b:Object {id: $to_id})
                        MERGE (a)-[r:NEAR_TO]->(b)
                        SET r.distance = $distance
                        """,
                        from_id  = str(rel["from_id"]),
                        to_id    = str(rel["to_id"]),
                        distance = rel["distance"],
                    )
        except Exception as e:
            logger.error("SceneGraph._update_neo4j: failed (%s)", e)

    def _get_relations_neo4j(self) -> List[dict]:
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Object)-[r]->(b:Object)
                    RETURN a.id AS from_id, b.id AS to_id,
                           type(r) AS relation_type, r.distance AS distance
                    """
                )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error("SceneGraph._get_relations_neo4j: failed (%s)", e)
            return []

    def _get_neighbors_neo4j(self, obj_id: str) -> List[dict]:
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Object {id: $id})-[r]-(b:Object)
                    RETURN b.id AS neighbor_id, b.label AS label,
                           type(r) AS relation_type
                    """,
                    id=str(obj_id),
                )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error("SceneGraph._get_neighbors_neo4j: failed (%s)", e)
            return []

    # ---------------------------------------------------------------- #
    #  In-memory fallback
    # ---------------------------------------------------------------- #

    def _update_memory(self, obj_list: list) -> None:
        self._graph     = {str(o["id"]): o for o in obj_list}
        self._relations = self._compute_relations(obj_list)

    # ---------------------------------------------------------------- #
    #  Relation computation (shared by both backends)
    # ---------------------------------------------------------------- #

    def _compute_relations(self, obj_list: list) -> List[dict]:
        relations = []
        for i, a in enumerate(obj_list):
            if not a.get("position"):
                continue
            pos_a = np.array(a["position"][:3])
            for j, b in enumerate(obj_list):
                if i >= j or not b.get("position"):
                    continue
                dist = float(np.linalg.norm(pos_a - np.array(b["position"][:3])))
                if dist < self.NEAR_THRESHOLD:
                    relations.append({
                        "from_id":       str(a["id"]),
                        "to_id":         str(b["id"]),
                        "relation_type": "NEAR_TO",
                        "distance":      round(dist, 3),
                    })
        return relations
