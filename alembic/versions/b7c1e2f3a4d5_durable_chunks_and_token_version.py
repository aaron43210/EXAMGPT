"""durable chunks table, token_version, document text

Revision ID: b7c1e2f3a4d5
Revises: 4a5d0be3e5d1
Create Date: 2026-10-18 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7c1e2f3a4d5'
down_revision: Union[str, Sequence[str], None] = '4a5d0be3e5d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('token_version', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('documents', sa.Column('content_text', sa.Text(), nullable=True))
    op.add_column('documents', sa.Column('error_message', sa.String(), nullable=True))
    op.add_column('documents', sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'))
    op.create_table(
        'chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('doc_type', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_chunks_document_id', 'chunks', ['document_id'])
    op.create_index('ix_chunks_course_id', 'chunks', ['course_id'])
    # Documents indexed before this migration lived only in server memory and
    # are gone; mark them failed so users know to re-upload.
    op.execute("UPDATE documents SET status = 'failed', error_message = 'Please re-upload this file' "
               "WHERE content_text IS NULL")


def downgrade() -> None:
    op.drop_index('ix_chunks_course_id', table_name='chunks')
    op.drop_index('ix_chunks_document_id', table_name='chunks')
    op.drop_table('chunks')
    op.drop_column('documents', 'chunk_count')
    op.drop_column('documents', 'error_message')
    op.drop_column('documents', 'content_text')
    op.drop_column('users', 'token_version')
