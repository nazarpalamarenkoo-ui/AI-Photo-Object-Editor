import pytest


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_image_with_mock(mock_redis_cache, mock_image_bytes):
    cache = mock_redis_cache
    
    # Cache image
    key = await cache.cache_image(123, mock_image_bytes, "processed")
    
    assert key == "image:123:processed"
    
    # Get cached
    cached = await cache.get_cached_image(123, "processed")
    
    assert cached == mock_image_bytes


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_cache_detections_with_mock(mock_redis_cache):
    cache = mock_redis_cache
    
    detections = [
        {'x1': 10, 'y1': 10, 'x2': 100, 'y2': 100, 'class': 'person'}
    ]
    
    # Cache detections
    key = await cache.cache_detections(123, detections)
    
    # Get cached
    cached = await cache.get_cached_detections(123)
    
    assert cached == detections


@pytest.mark.integration
@pytest.mark.storage
@pytest.mark.asyncio
async def test_invalidate_image_with_mock(mock_redis_cache, mock_image_bytes):
    cache = mock_redis_cache
    
    # Cache multiple versions
    await cache.cache_image(123, mock_image_bytes, "processed")
    await cache.cache_image(123, mock_image_bytes, "thumbnail")
    await cache.cache_detections(123, [])
    
    # Invalidate all
    await cache.invalidate_image(123)
    
    # Verify deleted
    assert await cache.get_cached_image(123, "processed") is None
    assert await cache.get_cached_image(123, "thumbnail") is None
    assert await cache.get_cached_detections(123) is None