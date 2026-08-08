# Місце: app/repository/assets_repo.py

from typing import Optional, List

from sqlalchemy import select, func

from app.db.models.assets import Asset
from app.repository.base_repo import BaseRepository

MAX_ASSETS_PER_USER = 200


class AssetRepository(BaseRepository):

    async def create(self, asset: Asset) -> Asset:
        async with self.session_factory() as db:
            db.add(asset)
            await db.commit()
            await db.refresh(asset)
            return asset

    async def get_by_public_id(self, user_id: int, public_id: str) -> Optional[Asset]:
        """Scoped by user_id too — a public_id belonging to another user
        should look like it doesn't exist, not leak a 403 vs 404 distinction."""
        async with self.session_factory() as db:
            result = await db.execute(
                select(Asset).where(Asset.public_id == public_id, Asset.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def get_overflow(self, user_id: int, max_assets: int = MAX_ASSETS_PER_USER) -> List[Asset]:
        """
        Returns the oldest assets over the cap WITHOUT deleting them —
        caller is responsible for cleaning up S3 first, then calling
        delete_many().
        """
        async with self.session_factory() as db:
            count_result = await db.execute(
                select(func.count()).select_from(Asset).where(Asset.user_id == user_id)
            )
            count = count_result.scalar_one()
            overflow = count - max_assets
            if overflow <= 0:
                return []

            result = await db.execute(
                select(Asset)
                .where(Asset.user_id == user_id)
                .order_by(Asset.created_at.asc())
                .limit(overflow)
            )
            return result.scalars().all()  # type: ignore

    async def list_by_user(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Asset]:
        async with self.session_factory() as db:
            result = await db.execute(
                select(Asset)
                .where(Asset.user_id == user_id)
                .order_by(Asset.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()  # type: ignore

    async def rename(self, asset: Asset, label: str) -> Asset:
        """`asset` comes from a previous short-lived session — merge it in
        rather than mutating+committing a detached instance."""
        async with self.session_factory() as db:
            merged = await db.merge(asset)
            merged.label = label
            await db.commit()
            await db.refresh(merged)
            return merged

    async def delete_many(self, assets: List[Asset]) -> None:
        async with self.session_factory() as db:
            for asset in assets:
                merged = await db.merge(asset)
                await db.delete(merged)
            await db.commit()

    async def delete(self, asset: Asset) -> None:
        async with self.session_factory() as db:
            merged = await db.merge(asset)
            await db.delete(merged)
            await db.commit()