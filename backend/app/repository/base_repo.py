from typing import Callable, AsyncContextManager
from sqlalchemy.ext.asyncio import AsyncSession

SessionFactory = Callable[[], AsyncContextManager[AsyncSession]]


class BaseRepository:

    def __init__(self, session_factory: SessionFactory):
        self.session_factory = session_factory