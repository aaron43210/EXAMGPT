from sqlalchemy import Boolean, Column, Integer, String, Text, ForeignKey, DateTime, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func, false as sa_false
from .database import Base
import enum


class UserRole(str, enum.Enum):
    student = "student"
    instructor = "instructor"
    admin = "admin"


class DocumentStatus(str, enum.Enum):
    uploaded = "uploaded"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class DocumentType(str, enum.Enum):
    syllabus = "syllabus"
    notes = "notes"
    pyq = "pyq"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    password_hash = Column(String)
    role = Column(String, default=UserRole.student.value)
    # Bumped on logout so every previously issued token stops working
    token_version = Column(Integer, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    courses = relationship("Course", back_populates="owner")


class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="courses")
    documents = relationship("Document", back_populates="course")
    study_plans = relationship("StudyPlan", back_populates="course")
    chats = relationship("ChatHistory", back_populates="course")


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    filename = Column(String)
    doc_type = Column(String, default=DocumentType.notes.value)
    status = Column(String, default=DocumentStatus.uploaded.value)
    file_url = Column(String, nullable=True)  # legacy, unused
    content_text = Column(Text, nullable=True)  # extracted text, used by study plans
    error_message = Column(String, nullable=True)
    kg_status = Column(String, nullable=True)  # knowledge-graph build: ready | skipped | failed
    chunk_count = Column(Integer, nullable=False, default=0, server_default="0")
    course_id = Column(Integer, ForeignKey("courses.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    course = relationship("Course", back_populates="documents")


class Chunk(Base):
    """A retrievable piece of a document. Stored in the DB so it survives restarts."""
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), index=True, nullable=False)
    doc_type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)  # list[float], normalized
    # On Postgres with pgvector, an extra `embedding_vec vector(384)` column + HNSW index
    # is managed by the migration and queried with raw SQL (see hybrid_retriever).


class KGNode(Base):
    """Knowledge-graph entity: a syllabus unit, topic or concept within one course."""
    __tablename__ = "kg_nodes"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=True)
    type = Column(String, nullable=False)  # unit | topic | concept
    name = Column(String, nullable=False)
    key = Column(String, nullable=False, index=True)  # normalized name for de-duplication
    description = Column(Text, nullable=True)


class KGEdge(Base):
    __tablename__ = "kg_edges"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=True)
    src_id = Column(Integer, ForeignKey("kg_nodes.id", ondelete="CASCADE"), index=True, nullable=False)
    dst_id = Column(Integer, ForeignKey("kg_nodes.id", ondelete="CASCADE"), index=True, nullable=False)
    type = Column(String, nullable=False)  # part_of | prerequisite_of | related_to


class PYQQuestion(Base):
    __tablename__ = "pyq_questions"
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text)
    year = Column(Integer)
    topic = Column(String)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    marks = Column(Integer, nullable=True)
    topic_node_id = Column(Integer, ForeignKey("kg_nodes.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StudyPlan(Base):
    __tablename__ = "study_plans"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    course = relationship("Course", back_populates="study_plans")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question_id = Column(Integer, ForeignKey("pyq_questions.id"), nullable=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)
    question_text = Column(Text, nullable=True)
    user_answer = Column(Text)
    feedback = Column(Text)
    score = Column(Integer)
    max_score = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    query = Column(Text)
    response = Column(Text)
    citations_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    course = relationship("Course", back_populates="chats")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chat_history.id"), nullable=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    node_timings_json = Column(JSON, nullable=True)  # trace: [{"name", "ms", "tokens", "model"}]
    retry_count = Column(Integer, default=0)
    total_latency_ms = Column(Float, nullable=True)
    status = Column(String, default="running")
    intent = Column(String, nullable=True)
    model = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    retrieval_confidence = Column(Float, nullable=True)
    groundedness = Column(Float, nullable=True)  # 0-1, from the online judge (sampled)
    guardrail_flags = Column(JSON, nullable=True)
    cache_hit = Column(Boolean, nullable=False, default=False, server_default=sa_false())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AnswerCache(Base):
    """Semantic cache of grounded answers, per course. Invalidated when course content changes."""
    __tablename__ = "answer_cache"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), index=True, nullable=False)
    intent = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    query_key = Column(String, nullable=False, index=True)  # normalized text for exact matches
    embedding = Column(JSON, nullable=True)
    answer = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    prompt_version = Column(String, nullable=True)
    hits = Column(Integer, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    """Thumbs up/down on an answer — the online eval signal."""
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chat_history.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), index=True, nullable=False)
    rating = Column(Integer, nullable=False)  # 1 or -1
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
