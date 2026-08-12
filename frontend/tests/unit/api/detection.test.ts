import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Detection, DetectionStats } from '@/types/Index'

vi.mock('@/api/clients', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn()
  }
}))

import apiClient from '@/api/clients'
import { detectionsApi } from '@/api/detections'

const mockedClient = vi.mocked(apiClient, true)

const fakeDetection: Detection = {
  id: 1,
  content_id: 42,
  bbox_id: 7,
  x1: 10,
  y1: 20,
  x2: 100,
  y2: 200,
  detected_class: 'car',
  confidence: 0.92,
  is_active: true,
  model_name: 'yolo',
  model_version: 'v8',
  inference_time_ms: 15,
  created_at: '2026-01-01T00:00:00Z'
}

const fakeDetections: Detection[] = [
  fakeDetection,
  {
    id: 2,
    content_id: 42,
    bbox_id: 8,
    x1: 30,
    y1: 40,
    x2: 120,
    y2: 220,
    detected_class: 'person',
    confidence: 0.87,
    is_active: true,
    model_name: 'yolo',
    model_version: 'v8',
    inference_time_ms: 15,
    created_at: '2026-01-01T00:00:00Z'
  }
]

const fakeStats: DetectionStats = {
  total_detections: 2,
  classes: ['car', 'person'],
  avg_confidence: 0.895,
  min_confidence: 0.87,
  max_confidence: 0.92
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('detectionsApi: getByImage', () => {
  it('defaults active_only to true and leaves version_id undefined when not passed', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetections })

    const result = await detectionsApi.getByImage(42)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42', {
      params: { version_id: undefined, active_only: true }
    })
    expect(result).toEqual(fakeDetections)
  })

  it('passes an explicit version_id through to the request params', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetections })

    await detectionsApi.getByImage(42, 3)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42', {
      params: { version_id: 3, active_only: true }
    })
  })

  it('passes active_only=false through to the request params', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetections })

    await detectionsApi.getByImage(42, undefined, false)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42', {
      params: { version_id: undefined, active_only: false }
    })
  })

  it('supports version_id and active_only=false together', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetections })

    await detectionsApi.getByImage(42, 3, false)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42', {
      params: { version_id: 3, active_only: false }
    })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.get.mockRejectedValue(new Error('image not found'))

    await expect(detectionsApi.getByImage(999)).rejects.toThrow('image not found')
  })
})

describe('detectionsApi: getByBboxId', () => {
  it('gets a single detection by image and bbox id with no version_id', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetection })

    const result = await detectionsApi.getByBboxId(42, 7)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42/bbox/7', {
      params: { version_id: undefined }
    })
    expect(result).toEqual(fakeDetection)
  })

  it('passes an explicit version_id through to the request params', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetection })

    await detectionsApi.getByBboxId(42, 7, 5)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42/bbox/7', {
      params: { version_id: 5 }
    })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.get.mockRejectedValue(new Error('bbox not found'))

    await expect(detectionsApi.getByBboxId(42, 999)).rejects.toThrow('bbox not found')
  })
})

describe('detectionsApi: getStats', () => {
  it('gets detection stats for an image', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeStats })

    const result = await detectionsApi.getStats(42)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42/stats')
    expect(result).toEqual(fakeStats)
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.get.mockRejectedValue(new Error('image not found'))

    await expect(detectionsApi.getStats(999)).rejects.toThrow('image not found')
  })
})

describe('detectionsApi: deleteByImage', () => {
  it('deletes detections for an image and returns the deleted count', async () => {
    mockedClient.delete.mockResolvedValue({ data: { deleted: 2 } })

    const result = await detectionsApi.deleteByImage(42)

    expect(mockedClient.delete).toHaveBeenCalledWith('/detections/images/42')
    expect(result).toEqual({ deleted: 2 })
  })

  it('returns deleted: 0 when there was nothing to delete', async () => {
    mockedClient.delete.mockResolvedValue({ data: { deleted: 0 } })

    const result = await detectionsApi.deleteByImage(42)

    expect(result).toEqual({ deleted: 0 })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.delete.mockRejectedValue(new Error('image not found'))

    await expect(detectionsApi.deleteByImage(999)).rejects.toThrow('image not found')
  })
})