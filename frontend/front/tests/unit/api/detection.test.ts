import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Detection, DetectionStats } from '@/types/Index'

// detectionsApi calls .get/.delete directly on the default-exported
// apiClient instance, so we mock the whole module with stubbed methods.
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
  image_id: 42,
  bbox_id: 7,
  x1: 10,
  y1: 20,
  x2: 100,
  y2: 200,
  detected_class: 'car',
  confidence: 0.92
}

const fakeDetections: Detection[] = [
  fakeDetection,
  {
    id: 2,
    image_id: 42,
    bbox_id: 8,
    x1: 30,
    y1: 40,
    x2: 120,
    y2: 220,
    detected_class: 'person',
    confidence: 0.87
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
  it('gets detections for an image with useCache defaulting to true', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetections })

    const result = await detectionsApi.getByImage(42)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42', {
      params: { useCache: true }
    })
    expect(result).toEqual(fakeDetections)
  })

  it('passes useCache=false through to the request params', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetections })

    await detectionsApi.getByImage(42, false)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42', {
      params: { useCache: false }
    })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.get.mockRejectedValue(new Error('image not found'))

    await expect(detectionsApi.getByImage(999)).rejects.toThrow('image not found')
  })
})

describe('detectionsApi: getByBboxId', () => {
  it('gets a single detection by image and bbox id', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeDetection })

    const result = await detectionsApi.getByBboxId(42, 7)

    expect(mockedClient.get).toHaveBeenCalledWith('/detections/images/42/bbox/7')
    expect(result).toEqual(fakeDetection)
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

  it('propagates the error when the request fails', async () => {
    mockedClient.delete.mockRejectedValue(new Error('image not found'))

    await expect(detectionsApi.deleteByImage(999)).rejects.toThrow('image not found')
  })
})