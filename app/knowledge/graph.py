"""
Course knowledge graph (Graph RAG), stored in Postgres:

  kg_nodes:  unit / topic entities per course
  kg_edges:  topic --part_of--> unit,  topic --prerequisite_of--> topic
  pyq_questions: each past exam question linked to a topic node (year, marks)

Built from syllabus and PYQ documents at ingestion (one LLM call each); queried at answer
time to add structure the vector search can't see: where a topic sits in the syllabus,
what it depends on, and how often/when it was examined.
"""
import logging
import re
from collections import defaultdict
from sqlalchemy.orm import Session
from app.agents import prompts
from app.agents.llm_setup import chat, parse_json
from app.db.models import KGEdge, KGNode, PYQQuestion
from app.retrieval.hybrid_retriever import content_terms

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 12000
MAX_NAME = 120


def node_key(name: str) -> str:
    name = re.sub(r"^(unit|module|chapter|part)\s*[-:\divx]*\s*[:.\-–]?\s*", "", name.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _stem(token: str) -> str:
    """Tiny plural stemmer so "deadlock" links to "Deadlocks"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _terms(key: str) -> set[str]:
    return {_stem(t) for t in key.split()}


def _clean(name) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip()[:MAX_NAME]


def clear_document(db: Session, document_id: int) -> None:
    db.query(PYQQuestion).filter(PYQQuestion.document_id == document_id).delete(synchronize_session=False)
    db.query(KGEdge).filter(KGEdge.document_id == document_id).delete(synchronize_session=False)
    db.query(KGNode).filter(KGNode.document_id == document_id).delete(synchronize_session=False)


class _NodeIndex:
    """Find-or-create nodes by normalized key within a course."""

    def __init__(self, db: Session, course_id: int, document_id: int):
        self.db, self.course_id, self.document_id = db, course_id, document_id
        self.by_key = {(n.type, n.key): n for n in db.query(KGNode).filter(KGNode.course_id == course_id)}

    def get(self, type_: str, name: str) -> KGNode | None:
        name = _clean(name)
        key = node_key(name)
        if not key:
            return None
        node = self.by_key.get((type_, key))
        if node is None:
            node = KGNode(course_id=self.course_id, document_id=self.document_id, type=type_, name=name, key=key)
            self.db.add(node)
            self.db.flush()
            self.by_key[(type_, key)] = node
        return node

    def match_topic(self, name: str) -> KGNode:
        """Link to an existing topic when names overlap strongly, else create one."""
        key = node_key(name)
        if ("topic", key) in self.by_key:
            return self.by_key[("topic", key)]
        terms = _terms(key)
        best, best_score = None, 0.0
        for (type_, k), node in self.by_key.items():
            if type_ != "topic":
                continue
            other = _terms(k)
            score = len(terms & other) / len(terms | other) if terms | other else 0
            if score > best_score:
                best, best_score = node, score
        if best is not None and best_score >= 0.6:
            return best
        return self.get("topic", name)


def build_from_syllabus(db: Session, course_id: int, document_id: int, text: str) -> int:
    data = parse_json(chat(
        [{"role": "system", "content": prompts.KG_SYLLABUS.system},
         {"role": "user", "content": prompts.KG_SYLLABUS.user.format(input=text[:MAX_INPUT_CHARS])}],
        purpose="kg_syllabus", json_mode=True, max_tokens=2000, temperature=0,
        prompt_version=prompts.KG_SYLLABUS.id,
    ))
    if not data or not isinstance(data.get("units"), list):
        raise ValueError("syllabus extraction returned no units")

    index = _NodeIndex(db, course_id, document_id)
    pending_prereqs = []
    count = 0
    for unit in data["units"][:40]:
        if not isinstance(unit, dict):
            continue
        unit_node = index.get("unit", unit.get("name"))
        for topic in (unit.get("topics") or [])[:60]:
            if isinstance(topic, str):
                topic = {"name": topic}
            if not isinstance(topic, dict):
                continue
            topic_node = index.get("topic", topic.get("name"))
            if topic_node is None:
                continue
            count += 1
            if unit_node is not None:
                db.add(KGEdge(course_id=course_id, document_id=document_id,
                              src_id=topic_node.id, dst_id=unit_node.id, type="part_of"))
            for pre in (topic.get("prerequisites") or [])[:10]:
                pending_prereqs.append((pre, topic_node))
    for pre_name, topic_node in pending_prereqs:
        pre = index.by_key.get(("topic", node_key(_clean(pre_name))))
        if pre is not None and pre.id != topic_node.id:
            db.add(KGEdge(course_id=course_id, document_id=document_id,
                          src_id=pre.id, dst_id=topic_node.id, type="prerequisite_of"))
    return count


def build_from_pyq(db: Session, course_id: int, document_id: int, text: str) -> int:
    index = _NodeIndex(db, course_id, document_id)
    known = ", ".join(sorted(n.name for (t, _), n in index.by_key.items() if t == "topic"))[:2000] or "(none yet)"
    data = parse_json(chat(
        [{"role": "system", "content": prompts.KG_PYQ.system.replace("{known_topics}", known)},
         {"role": "user", "content": prompts.KG_PYQ.user.format(input=text[:MAX_INPUT_CHARS])}],
        purpose="kg_pyq", json_mode=True, max_tokens=3000, temperature=0,
        prompt_version=prompts.KG_PYQ.id,
    ))
    if not data or not isinstance(data.get("questions"), list):
        raise ValueError("PYQ extraction returned no questions")

    count = 0
    for q in data["questions"][:300]:
        if not isinstance(q, dict) or not _clean(q.get("question")):
            continue
        topics = [t for t in (q.get("topics") or []) if _clean(t)] or ["General"]
        topic_node = index.match_topic(_clean(topics[0]))
        db.add(PYQQuestion(
            question_text=str(q["question"])[:2000],
            year=_int_or_none(q.get("year"), 1950, 2100),
            marks=_int_or_none(q.get("marks"), 0, 200),
            topic=topic_node.name,
            topic_node_id=topic_node.id,
            course_id=course_id,
            document_id=document_id,
        ))
        count += 1
    return count


def _int_or_none(value, lo: int, hi: int) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None


# ── Queries ─────────────────────────────────────────────────────────────

def pyq_stats(db: Session, course_id: int, limit: int = 15) -> list[dict]:
    """Topics ranked by how often they were examined (then most recent year)."""
    stats: dict[int, dict] = {}
    rows = db.query(PYQQuestion, KGNode).join(KGNode, KGNode.id == PYQQuestion.topic_node_id) \
        .filter(PYQQuestion.course_id == course_id).all()
    for q, node in rows:
        s = stats.setdefault(node.id, {"topic_id": node.id, "topic": node.name, "count": 0,
                                       "years": set(), "marks": 0, "questions": []})
        s["count"] += 1
        if q.year:
            s["years"].add(q.year)
        s["marks"] += q.marks or 0
        if len(s["questions"]) < 3:
            s["questions"].append({"text": q.question_text, "year": q.year, "marks": q.marks})
    ranked = sorted(stats.values(), key=lambda s: (-s["count"], -max(s["years"], default=0)))
    for s in ranked:
        s["years"] = sorted(s["years"], reverse=True)
    return ranked[:limit]


def link_topics(db: Session, course_id: int, query: str, limit: int = 3) -> list[KGNode]:
    """Entity linking: topics whose names overlap the query's content words."""
    q_terms = {_stem(t) for t in content_terms(query)}
    if not q_terms:
        return []
    scored = []
    for node in db.query(KGNode).filter(KGNode.course_id == course_id, KGNode.type == "topic"):
        terms = {_stem(t) for t in content_terms(node.name)}
        if not terms:
            continue
        overlap = len(q_terms & terms) / len(terms)
        if overlap >= 0.5:
            scored.append((overlap, len(terms), node))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    return [n for _, _, n in scored[:limit]]


def neighborhood(db: Session, course_id: int, nodes: list[KGNode]) -> list[dict]:
    """1-hop expansion: unit membership, prerequisites, dependents and PYQ history per topic."""
    if not nodes:
        return []
    ids = [n.id for n in nodes]
    edges = db.query(KGEdge).filter(KGEdge.course_id == course_id,
                                    (KGEdge.src_id.in_(ids)) | (KGEdge.dst_id.in_(ids))).all()
    other_ids = {e.src_id for e in edges} | {e.dst_id for e in edges}
    names = {n.id: n.name for n in db.query(KGNode).filter(KGNode.id.in_(other_ids | set(ids)))}
    pyqs = defaultdict(list)
    for q in db.query(PYQQuestion).filter(PYQQuestion.topic_node_id.in_(ids)):
        pyqs[q.topic_node_id].append(q)

    facts = []
    for node in nodes:
        fact = {"topic": node.name, "unit": None, "prerequisites": [], "leads_to": [], "pyq_count": 0,
                "pyq_years": [], "pyq_examples": []}
        for e in edges:
            if e.type == "part_of" and e.src_id == node.id and e.dst_id in names:
                fact["unit"] = names[e.dst_id]
            elif e.type == "prerequisite_of" and e.dst_id == node.id and e.src_id in names:
                fact["prerequisites"].append(names[e.src_id])
            elif e.type == "prerequisite_of" and e.src_id == node.id and e.dst_id in names:
                fact["leads_to"].append(names[e.dst_id])
        qs = pyqs.get(node.id, [])
        fact["pyq_count"] = len(qs)
        fact["pyq_years"] = sorted({q.year for q in qs if q.year}, reverse=True)
        fact["pyq_examples"] = [q.question_text[:200] for q in qs[:2]]
        facts.append(fact)
    return facts


def format_facts(facts: list[dict]) -> str:
    lines = []
    for f in facts:
        parts = [f"- **{f['topic']}**"]
        if f["unit"]:
            parts.append(f"in {f['unit']}")
        if f["prerequisites"]:
            parts.append("requires " + ", ".join(f["prerequisites"]))
        if f["leads_to"]:
            parts.append("needed for " + ", ".join(f["leads_to"]))
        if f["pyq_count"]:
            years = f" ({', '.join(map(str, f['pyq_years']))})" if f["pyq_years"] else ""
            parts.append(f"asked {f['pyq_count']}x in past papers{years}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def topic_map(db: Session, course_id: int) -> dict:
    nodes = db.query(KGNode).filter(KGNode.course_id == course_id).all()
    if not nodes:
        return {"units": [], "unassigned": [], "most_asked": []}
    stats = {s["topic_id"]: s for s in pyq_stats(db, course_id, limit=10_000)}
    by_id = {n.id: n for n in nodes}
    edges = db.query(KGEdge).filter(KGEdge.course_id == course_id).all()
    unit_of, prereqs = {}, defaultdict(list)
    for e in edges:
        if e.type == "part_of":
            unit_of[e.src_id] = e.dst_id
        elif e.type == "prerequisite_of" and e.src_id in by_id:
            prereqs[e.dst_id].append(by_id[e.src_id].name)

    def topic_entry(n):
        s = stats.get(n.id, {})
        return {"id": n.id, "name": n.name, "pyq_count": s.get("count", 0), "years": s.get("years", []),
                "prerequisites": prereqs.get(n.id, [])}

    units = [{"id": u.id, "name": u.name,
              "topics": [topic_entry(t) for t in nodes if t.type == "topic" and unit_of.get(t.id) == u.id]}
             for u in nodes if u.type == "unit"]
    unassigned = [topic_entry(t) for t in nodes if t.type == "topic" and t.id not in unit_of]
    most_asked = [{"topic": s["topic"], "count": s["count"], "years": s["years"], "questions": s["questions"]}
                  for s in pyq_stats(db, course_id)]
    return {"units": units, "unassigned": unassigned, "most_asked": most_asked}
