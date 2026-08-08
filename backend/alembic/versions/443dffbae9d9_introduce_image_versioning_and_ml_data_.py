"""introduce image versioning, content-based dedup, and ML data models

Revision ID: 443dffbae9d9
Revises: a0254d40c3c2
Create Date: 2026-08-01 18:56:54.151277

NOTE: this migration intentionally DROPS all existing tables/data before
recreating the schema. It is only safe to run against dev/local
databases. Do not run against production without a separate,
data-preserving migration plan (add nullable column -> backfill ->
set NOT NULL).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '443dffbae9d9'
down_revision: Union[str, Sequence[str], None] = 'a0254d40c3c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("DROP TABLE IF EXISTS assets CASCADE")
    op.execute("DROP TABLE IF EXISTS image_edit_history CASCADE")
    op.execute("DROP TABLE IF EXISTS ml_jobs CASCADE")
    op.execute("DROP TABLE IF EXISTS segmentation_masks CASCADE")
    op.execute("DROP TABLE IF EXISTS detections CASCADE")
    op.execute("DROP TABLE IF EXISTS image_versions CASCADE")
    op.execute("DROP TABLE IF EXISTS image_contents CASCADE")
    op.execute("DROP TABLE IF EXISTS images CASCADE")
    op.execute("DROP TABLE IF EXISTS users CASCADE")

    op.execute("DROP TYPE IF EXISTS image_status")
    op.execute("DROP TYPE IF EXISTS ml_task_type")
    op.execute("DROP TYPE IF EXISTS job_status")
    op.execute("DROP TYPE IF EXISTS segmentation_mode")
    op.execute("DROP TYPE IF EXISTS edit_operation")
    op.execute("DROP TYPE IF EXISTS engine_type")

    # ### enum types ###
    image_status_enum = postgresql.ENUM(
        'uploaded', 'processing', 'ready', 'failed', 'deleted',
        name='image_status', create_type=False
    )
    ml_task_type_enum = postgresql.ENUM(
        'detection',
        'segmentation', 'segmentation_prompt', 'segmentation_polygon', 'segmentation_hybrid',
        'sam_remove_object', 'sam_replace_object', 'diffusion',
        'remove_object', 'remove_multiple_objects', 'replace_object',
        'extract_object',
        name='ml_task_type', create_type=False
    )
    job_status_enum = postgresql.ENUM(
        'pending', 'running', 'success', 'failed',
        name='job_status', create_type=False
    )
    segmentation_mode_enum = postgresql.ENUM(
        'sam', 'hybrid', 'polygon',
        name='segmentation_mode', create_type=False
    )
    edit_operation_enum = postgresql.ENUM(
        'detect', 'segment', 'remove', 'replace',
        name='edit_operation', create_type=False
    )
    engine_type_enum = postgresql.ENUM(
        'yolo', 'sam', 'lama', 'diffusion',
        name='engine_type', create_type=False
    )

    bind = op.get_bind()
    image_status_enum.create(bind, checkfirst=True)
    ml_task_type_enum.create(bind, checkfirst=True)
    job_status_enum.create(bind, checkfirst=True)
    segmentation_mode_enum.create(bind, checkfirst=True)
    edit_operation_enum.create(bind, checkfirst=True)
    engine_type_enum.create(bind, checkfirst=True)

    # ### users ###
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    op.create_table(
        'images',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        sa.Column('mime_type', sa.String(length=50), nullable=False),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('cache_key', sa.String(length=100), nullable=True),
        sa.Column('current_version_id', sa.Integer(), nullable=True),
        sa.Column('status', image_status_enum, nullable=False, server_default='uploaded'),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_images_user_id'), 'images', ['user_id'], unique=False)
    op.create_index(op.f('ix_images_current_version_id'), 'images', ['current_version_id'], unique=False)

    op.create_table(
        'image_contents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_image_contents_content_hash'), 'image_contents', ['content_hash'], unique=True)

    # ### image_versions ###
    op.create_table(
        'image_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('parent_version_id', sa.Integer(), nullable=True),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('version_number', sa.Integer(), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['image_id'], ['images.id']),
        sa.ForeignKeyConstraint(['parent_version_id'], ['image_versions.id']),
        sa.ForeignKeyConstraint(['content_id'], ['image_contents.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('image_id', 'version_number', name='uq_image_version_number'),
    )
    op.create_index(op.f('ix_image_versions_image_id'), 'image_versions', ['image_id'], unique=False)
    op.create_index(op.f('ix_image_versions_parent_version_id'), 'image_versions', ['parent_version_id'], unique=False)
    op.create_index(op.f('ix_image_versions_content_id'), 'image_versions', ['content_id'], unique=False)

    op.create_foreign_key(
        'fk_images_current_version_id_image_versions',
        'images', 'image_versions',
        ['current_version_id'], ['id'],
    )

    op.create_table(
        'detections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('bbox_id', sa.Integer(), nullable=False),
        sa.Column('x1', sa.Integer(), nullable=False),
        sa.Column('y1', sa.Integer(), nullable=False),
        sa.Column('x2', sa.Integer(), nullable=False),
        sa.Column('y2', sa.Integer(), nullable=False),
        sa.Column('detected_class', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_version', sa.String(length=30), nullable=False),
        sa.Column('inference_time_ms', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['content_id'], ['image_contents.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('content_id', 'bbox_id', name='uq_detection_content_bbox'),
    )
    op.create_index(op.f('ix_detections_content_id'), 'detections', ['content_id'], unique=False)

    op.create_table(
        'segmentation_masks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('mask_id', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('mask_storage_path', sa.String(length=500), nullable=False),
        sa.Column('preview_storage_path', sa.String(length=500), nullable=False),
        sa.Column('x1', sa.Integer(), nullable=False),
        sa.Column('y1', sa.Integer(), nullable=False),
        sa.Column('x2', sa.Integer(), nullable=False),
        sa.Column('y2', sa.Integer(), nullable=False),
        sa.Column('area', sa.Float(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('segmentation_mode', segmentation_mode_enum, nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('inference_time_ms', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['content_id'], ['image_contents.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('content_id', 'mask_id', name='uq_segmentation_content_mask'),
    )
    op.create_index(op.f('ix_segmentation_masks_content_id'), 'segmentation_masks', ['content_id'], unique=False)

    op.create_table(
        'ml_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('image_version_id', sa.Integer(), nullable=False),
        sa.Column('task_type', ml_task_type_enum, nullable=False),
        sa.Column('status', job_status_enum, nullable=False, server_default='pending'),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['content_id'], ['image_contents.id']),
        sa.ForeignKeyConstraint(['image_version_id'], ['image_versions.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_ml_jobs_content_id'), 'ml_jobs', ['content_id'], unique=False)
    op.create_index(op.f('ix_ml_jobs_image_version_id'), 'ml_jobs', ['image_version_id'], unique=False)

    op.create_table(
        'image_edit_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_version_id', sa.Integer(), nullable=False),
        sa.Column('operation', edit_operation_enum, nullable=False),
        sa.Column('engine', engine_type_enum, nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['image_version_id'], ['image_versions.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_image_edit_history_image_version_id'), 'image_edit_history', ['image_version_id'], unique=False)

    op.create_table(
        'assets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('public_id', sa.String(length=32), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('source_image_version_id', sa.Integer(), nullable=True),
        sa.Column('source_segmentation_mask_id', sa.Integer(), nullable=True),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        sa.Column('thumbnail_path', sa.String(length=500), nullable=True),
        sa.Column('content_type', sa.String(length=50), nullable=False, server_default='image/png'),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('area_pixels', sa.Integer(), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('label', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['source_image_version_id'], ['image_versions.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['source_segmentation_mask_id'], ['segmentation_masks.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_assets_public_id'), 'assets', ['public_id'], unique=True)
    op.create_index(op.f('ix_assets_user_id'), 'assets', ['user_id'], unique=False)
    op.create_index(op.f('ix_assets_source_image_version_id'), 'assets', ['source_image_version_id'], unique=False)
    op.create_index(op.f('ix_assets_source_segmentation_mask_id'), 'assets', ['source_segmentation_mask_id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### drop new schema ###
    op.drop_index(op.f('ix_assets_source_segmentation_mask_id'), table_name='assets')
    op.drop_index(op.f('ix_assets_source_image_version_id'), table_name='assets')
    op.drop_index(op.f('ix_assets_user_id'), table_name='assets')
    op.drop_index(op.f('ix_assets_public_id'), table_name='assets')
    op.drop_table('assets')

    op.drop_index(op.f('ix_image_edit_history_image_version_id'), table_name='image_edit_history')
    op.drop_table('image_edit_history')

    op.drop_index(op.f('ix_ml_jobs_image_version_id'), table_name='ml_jobs')
    op.drop_index(op.f('ix_ml_jobs_content_id'), table_name='ml_jobs')
    op.drop_table('ml_jobs')

    op.drop_index(op.f('ix_segmentation_masks_content_id'), table_name='segmentation_masks')
    op.drop_table('segmentation_masks')

    op.drop_index(op.f('ix_detections_content_id'), table_name='detections')
    op.drop_table('detections')

    op.drop_constraint('fk_images_current_version_id_image_versions', 'images', type_='foreignkey')

    op.drop_index(op.f('ix_image_versions_content_id'), table_name='image_versions')
    op.drop_index(op.f('ix_image_versions_parent_version_id'), table_name='image_versions')
    op.drop_index(op.f('ix_image_versions_image_id'), table_name='image_versions')
    op.drop_table('image_versions')

    op.drop_index(op.f('ix_image_contents_content_hash'), table_name='image_contents')
    op.drop_table('image_contents')

    op.drop_index(op.f('ix_images_current_version_id'), table_name='images')
    op.drop_index(op.f('ix_images_user_id'), table_name='images')
    op.drop_table('images')

    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

    # ### drop enum types ###
    bind = op.get_bind()
    postgresql.ENUM(name='engine_type').drop(bind, checkfirst=True)
    postgresql.ENUM(name='edit_operation').drop(bind, checkfirst=True)
    postgresql.ENUM(name='segmentation_mode').drop(bind, checkfirst=True)
    postgresql.ENUM(name='job_status').drop(bind, checkfirst=True)
    postgresql.ENUM(name='ml_task_type').drop(bind, checkfirst=True)
    postgresql.ENUM(name='image_status').drop(bind, checkfirst=True)