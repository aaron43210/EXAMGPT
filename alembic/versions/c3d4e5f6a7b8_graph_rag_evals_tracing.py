"""knowledge graph, feedback, answer cache, tracing columns, page numbers, optional pgvector

Revision ID: c3d4e5f6a7b8
Revises: b7c1e2f3a4d5
Create Date: 2026-10-18 20:00:00.000000

"""
from typing import Sequence, Union

import logging
from alembic import op
import sqlalchemy as sa

logger = logging.getLogger("alembic.runtime.migration")


revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b7c1e2f3a4d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _pgvector_available() -> bool:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return False
    return bind.execute(sa.text(
        "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")).first() is not None


def upgrade() -> None:
    op.add_column('documents', sa.Column('kg_status', sa.String(), nullable=True))
    op.add_column('chunks', sa.Column('page', sa.Integer(), nullable=True))

    op.create_table(
        'kg_nodes',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('course_id', sa.Integer(), sa.ForeignKey('courses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=True),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
    )
    op.create_index('ix_kg_nodes_course_id', 'kg_nodes', ['course_id'])
    op.create_index('ix_kg_nodes_document_id', 'kg_nodes', ['document_id'])
    op.create_index('ix_kg_nodes_key', 'kg_nodes', ['key'])

    op.create_table(
        'kg_edges',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('course_id', sa.Integer(), sa.ForeignKey('courses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=True),
        sa.Column('src_id', sa.Integer(), sa.ForeignKey('kg_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dst_id', sa.Integer(), sa.ForeignKey('kg_nodes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
    )
    for col in ('course_id', 'document_id', 'src_id', 'dst_id'):
        op.create_index(f'ix_kg_edges_{col}', 'kg_edges', [col])

    with op.batch_alter_table('pyq_questions') as b:
        b.add_column(sa.Column('marks', sa.Integer(), nullable=True))
        b.add_column(sa.Column('topic_node_id', sa.Integer(), nullable=True))
        b.create_foreign_key('fk_pyq_topic_node', 'kg_nodes', ['topic_node_id'], ['id'], ondelete='SET NULL')

    with op.batch_alter_table('evaluation_results') as b:
        b.add_column(sa.Column('course_id', sa.Integer(), nullable=True))
        b.add_column(sa.Column('question_text', sa.Text(), nullable=True))
        b.add_column(sa.Column('max_score', sa.Integer(), nullable=True))
        b.create_foreign_key('fk_eval_course', 'courses', ['course_id'], ['id'])

    for name, type_ in (('intent', sa.String()), ('model', sa.String()), ('prompt_version', sa.String()),
                        ('prompt_tokens', sa.Integer()), ('completion_tokens', sa.Integer()),
                        ('retrieval_confidence', sa.Float()), ('groundedness', sa.Float()),
                        ('guardrail_flags', sa.JSON())):
        op.add_column('pipeline_runs', sa.Column(name, type_, nullable=True))
    op.add_column('pipeline_runs', sa.Column('cache_hit', sa.Boolean(), nullable=False, server_default=sa.false()))

    op.create_table(
        'answer_cache',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('course_id', sa.Integer(), sa.ForeignKey('courses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('intent', sa.String(), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('query_key', sa.String(), nullable=False),
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('citations', sa.JSON(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('prompt_version', sa.String(), nullable=True),
        sa.Column('hits', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_answer_cache_course_id', 'answer_cache', ['course_id'])
    op.create_index('ix_answer_cache_query_key', 'answer_cache', ['query_key'])

    op.create_table(
        'feedback',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('chat_id', sa.Integer(), sa.ForeignKey('chat_history.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('course_id', sa.Integer(), sa.ForeignKey('courses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_feedback_chat_id', 'feedback', ['chat_id'])
    op.create_index('ix_feedback_course_id', 'feedback', ['course_id'])

    # Vector index: Supabase ships pgvector. Elsewhere retrieval falls back to in-process cosine.
    # Runs in a savepoint so a pgvector problem can never block a deploy.
    if _pgvector_available():
        bind = op.get_bind()
        try:
            with bind.begin_nested():
                bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
                bind.execute(sa.text("ALTER TABLE chunks ADD COLUMN embedding_vec vector(384)"))
                bind.execute(sa.text("UPDATE chunks SET embedding_vec = CAST(embedding::text AS vector) "
                                     "WHERE embedding IS NOT NULL AND json_array_length(embedding) = 384"))
                bind.execute(sa.text("CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
                                     "USING hnsw (embedding_vec vector_cosine_ops)"))
        except Exception as e:  # noqa: BLE001
            logger.warning("pgvector setup skipped, falling back to in-process vectors: %s", e)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
        op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_vec")
    op.drop_table('feedback')
    op.drop_table('answer_cache')
    op.drop_column('pipeline_runs', 'cache_hit')
    for name in ('intent', 'model', 'prompt_version', 'prompt_tokens', 'completion_tokens',
                 'retrieval_confidence', 'groundedness', 'guardrail_flags'):
        op.drop_column('pipeline_runs', name)
    with op.batch_alter_table('evaluation_results') as b:
        b.drop_constraint('fk_eval_course', type_='foreignkey')
        b.drop_column('max_score')
        b.drop_column('question_text')
        b.drop_column('course_id')
    with op.batch_alter_table('pyq_questions') as b:
        b.drop_constraint('fk_pyq_topic_node', type_='foreignkey')
        b.drop_column('topic_node_id')
        b.drop_column('marks')
    op.drop_table('kg_edges')
    op.drop_table('kg_nodes')
    op.drop_column('chunks', 'page')
    op.drop_column('documents', 'kg_status')
