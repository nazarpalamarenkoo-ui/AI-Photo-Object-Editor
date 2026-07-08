import { describe, it, expect, vi, beforeEach } from 'vitest'
import type {
  SegmentResponse,
  ExtractResponse,
  PasteResponse,
  MLResultResponse,
  LdmConfig,
  DetectResponse,
  Asset,
} from '@/types/Index'

vi.mock('@/api/clients', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
  },
}))

import apiClient from '@/api/clients'
import { mlApi, PRESETS } from '@/api/ml'

const mockedClient = vi.mocked(apiClient, true)

const qualityLdm: LdmConfig = { ldm_steps: 25, ldm_sampler: 'plms', hd_strategy: 'CROP' }
const fastLdm: LdmConfig = { ldm_steps: 10, ldm_sampler: 'plms', hd_strategy: 'CROP' }

const fakeSegmentResponse: SegmentResponse = {
  segments: [
    {
      mask_id: 0,
      bbox_id: 0,
      bbox: { x1: 10, y1: 10, x2: 100, y2: 200 },
      area: 9000,
      stability_score: 0.95,
    },
  ],
  metrics: { num_segments: 1, avg_stability: 0.95 },
  image_size: [640, 480],
  timestamp: '2026-01-01T00:00:00Z',
}

const fakeExtractResponse: ExtractResponse = {
  asset_id: 'asset-123',
  extracted_url: 's3://bucket/extracted.png',
  presigned_url: 'https://storage.example.com/extracted.png?sig=abc',
  object_size: [90, 190],
  area_pixels: 12000,
  cropped_bbox: { x1: 2, y1: 2, x2: 92, y2: 192 },
  timestamp: '2026-01-01T00:00:00Z',
}

const fakePasteResponse: PasteResponse = {
  result_url: 's3://bucket/pasted.jpg',
  presigned_url: 'https://storage.example.com/pasted.jpg?sig=xyz',
  paste_bbox: { x1: 50, y1: 60, x2: 140, y2: 250 },
  object_size: [90, 190],
  timestamp: '2026-01-01T00:00:00Z',
}

const fakeMLResult: MLResultResponse = {
  result_url: 's3://bucket/result.jpg',
  presigned_url: 'https://storage.example.com/result.jpg?sig=abc',
  metrics: { processing_time_ms: 250.5 },
  timestamp: '2026-01-01T00:00:00Z',
}

const fakeDetectResponse: DetectResponse = {
  detections: [
    {
      id: 1,
      image_id: 1,
      bbox_id: 0,
      x1: 10,
      y1: 10,
      x2: 100,
      y2: 100,
      detected_class: 'cat',
      confidence: 0.9,
    },
  ],
  image_size: [640, 480],
  metrics: { count: 1 },
  timestamp: '2026-01-01T00:00:00Z',
}

const fakeAsset: Asset = {
  asset_id: 'asset-123',
  source_image_id: 1,
  object_size: [90, 190],
  area_pixels: 12000,
  label: 'cat cutout',
  s3_url: 's3://bucket/asset.png',
  created_at: '2026-01-01T00:00:00Z',
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('mlApi: PRESETS', () => {
  it('exposes fast and quality presets', () => {
    expect(PRESETS.fast).toEqual(fastLdm)
    expect(PRESETS.quality).toEqual(qualityLdm)
  })
})

describe('mlApi: detectObjects', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeDetectResponse })

    await mlApi.detectObjects(1)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/detect', {})
  })

  it('passes custom detect params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeDetectResponse })

    await mlApi.detectObjects(1, { conf_threshold: 0.5, classes: ['cat', 'dog'] })

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/detect', {
      conf_threshold: 0.5,
      classes: ['cat', 'dog'],
    })
  })

  it('returns DetectResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeDetectResponse })

    const result = await mlApi.detectObjects(1)

    expect(result).toEqual(fakeDetectResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('detect failed'))

    await expect(mlApi.detectObjects(1)).rejects.toThrow('detect failed')
  })
})

describe('mlApi: getSupportedClasses', () => {
  it('gets from correct url', async () => {
    mockedClient.get.mockResolvedValue({ data: ['cat', 'dog'] })

    await mlApi.getSupportedClasses()

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/classes')
  })

  it('returns list of classes', async () => {
    mockedClient.get.mockResolvedValue({ data: ['cat', 'dog'] })

    const result = await mlApi.getSupportedClasses()

    expect(result).toEqual(['cat', 'dog'])
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('classes failed'))

    await expect(mlApi.getSupportedClasses()).rejects.toThrow('classes failed')
  })
})

describe('mlApi: removeObject', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeObject(1, 2)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/remove/2', {
      expand_mask_pixels: 5,
      use_edge_blending: false,
      ldm: qualityLdm,
    })
  })

  it('passes custom params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeObject(1, 2, 15, true, fastLdm)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/remove/2', {
      expand_mask_pixels: 15,
      use_edge_blending: true,
      ldm: fastLdm,
    })
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.removeObject(1, 2)

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('remove failed'))

    await expect(mlApi.removeObject(1, 2)).rejects.toThrow('remove failed')
  })
})

describe('mlApi: removeMultipleObjects', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeMultipleObjects(1, [2, 3, 4])

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/remove-multiple', {
      bbox_ids: [2, 3, 4],
      expand_mask_pixels: 5,
      use_edge_blending: false,
      ldm: qualityLdm,
    })
  })

  it('passes custom params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeMultipleObjects(1, [2, 3], 10, true, fastLdm)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/remove-multiple', {
      bbox_ids: [2, 3],
      expand_mask_pixels: 10,
      use_edge_blending: true,
      ldm: fastLdm,
    })
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.removeMultipleObjects(1, [2, 3])

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('remove multiple failed'))

    await expect(mlApi.removeMultipleObjects(1, [2, 3])).rejects.toThrow('remove multiple failed')
  })
})

describe('mlApi: replaceObject', () => {
  const fakeFile = new File(['img'], 'replacement.png', { type: 'image/png' })

  it('posts multipart/form-data to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.replaceObject(1, 2, fakeFile)

    const [url, body] = mockedClient.post.mock.calls[0]
    expect(url).toBe('/ml/images/1/replace/2')
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('replacement_file')).toBe(fakeFile)
  })

  it('sends default query params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.replaceObject(1, 2, fakeFile)

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).params).toEqual({
      expand_mask_pixels: 0,
      use_color_matching: false,
      use_edge_blending: false,
      color_match_method: 'mean_std',
      ldm_steps: 25,
      ldm_sampler: 'plms',
      hd_strategy: 'CROP',
    })
  })

  it('passes custom options', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.replaceObject(1, 2, fakeFile, {
      expandMaskPixels: 12,
      useColorMatching: true,
      useEdgeBlending: true,
      colorMatchMethod: 'histogram',
      ldm: fastLdm,
    })

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).params.expand_mask_pixels).toBe(12)
    expect((config as any).params.use_color_matching).toBe(true)
    expect((config as any).params.use_edge_blending).toBe(true)
    expect((config as any).params.color_match_method).toBe('histogram')
    expect((config as any).params.ldm_steps).toBe(10)
  })

  it('sets multipart/form-data header', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.replaceObject(1, 2, fakeFile)

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).headers['Content-Type']).toBe('multipart/form-data')
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.replaceObject(1, 2, fakeFile)

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('replace failed'))

    await expect(mlApi.replaceObject(1, 2, fakeFile)).rejects.toThrow('replace failed')
  })
})

describe('mlApi: segmentObjects', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    await mlApi.segmentObjects(1)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment',
      { min_area: 500, max_segments: 50 }
    )
  })

  it('passes custom minArea and maxSegments', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    await mlApi.segmentObjects(1, 1000, 10)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment',
      { min_area: 1000, max_segments: 10 }
    )
  })

  it('returns SegmentResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    const result = await mlApi.segmentObjects(1)

    expect(result).toEqual(fakeSegmentResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('segmentation failed'))

    await expect(mlApi.segmentObjects(1)).rejects.toThrow('segmentation failed')
  })
})

describe('mlApi: segmentWithPrompt', () => {
  it('posts point_coords and point_labels', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    await mlApi.segmentWithPrompt(1, {
      pointCoords: [[120, 340]],
      pointLabels: [1],
    })

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment/prompt',
      {
        point_coords: [[120, 340]],
        point_labels: [1],
        bbox: null,
        multimask_output: false,
      }
    )
  })

  it('posts bbox prompt', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    await mlApi.segmentWithPrompt(1, {
      bbox: { x1: 10, y1: 10, x2: 100, y2: 100 },
    })

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).bbox).toEqual({ x1: 10, y1: 10, x2: 100, y2: 100 })
    expect((body as any).point_coords).toBeNull()
  })

  it('passes multimask_output when set', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    await mlApi.segmentWithPrompt(1, {
      pointCoords: [[1, 1]],
      pointLabels: [1],
      multimask_output: true,
    })

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).multimask_output).toBe(true)
  })

  it('returns SegmentResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    const result = await mlApi.segmentWithPrompt(1, { pointCoords: [[1, 1]], pointLabels: [1] })

    expect(result).toEqual(fakeSegmentResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('prompt segmentation failed'))

    await expect(
      mlApi.segmentWithPrompt(1, { pointCoords: [[1, 1]], pointLabels: [1] })
    ).rejects.toThrow('prompt segmentation failed')
  })
})

describe('mlApi: segmentByPolygon', () => {
  const points: [number, number][] = [
    [10, 10],
    [100, 10],
    [100, 100],
    [10, 100],
  ]

  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    await mlApi.segmentByPolygon(1, { points })

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/segment/polygon', {
      points,
      smooth: true,
      smoothing_factor: 0,
      feather_px: 0,
    })
  })

  it('passes custom smoothing and feathering params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    await mlApi.segmentByPolygon(1, {
      points,
      smooth: false,
      smoothingFactor: 0.5,
      featherPx: 3,
    })

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/segment/polygon', {
      points,
      smooth: false,
      smoothing_factor: 0.5,
      feather_px: 3,
    })
  })

  it('returns SegmentResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSegmentResponse })

    const result = await mlApi.segmentByPolygon(1, { points })

    expect(result).toEqual(fakeSegmentResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('polygon segmentation failed'))

    await expect(mlApi.segmentByPolygon(1, { points })).rejects.toThrow('polygon segmentation failed')
  })
})

describe('mlApi: samRemoveObject', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samRemoveObject(1, 3)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment/3/remove',
      {
        expand_mask_pixels: 12,
        use_edge_blending: false,
        ldm: qualityLdm,
      }
    )
  })

  it('passes custom expandMaskPixels, useEdgeBlending and ldm', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samRemoveObject(1, 3, 20, false, fastLdm)

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).expand_mask_pixels).toBe(20)
    expect((body as any).use_edge_blending).toBe(false)
    expect((body as any).ldm).toEqual(fastLdm)
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.samRemoveObject(1, 3)

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('sam remove failed'))

    await expect(mlApi.samRemoveObject(1, 3)).rejects.toThrow('sam remove failed')
  })
})

describe('mlApi: samReplaceObject', () => {
  const fakeFile = new File(['img'], 'replacement.png', { type: 'image/png' })

  it('posts multipart/form-data to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samReplaceObject(1, 3, fakeFile)

    const [url, body] = mockedClient.post.mock.calls[0]
    expect(url).toBe('/ml/images/1/segment/3/replace')
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('replacement_file')).toBe(fakeFile)
  })

  it('sends default query params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samReplaceObject(1, 3, fakeFile)

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).params).toEqual({
      expand_mask_pixels: 8,
      use_color_matching: false,
      use_edge_blending: false,
      color_match_method: 'color_transfer',
      ldm_steps: 25,
      ldm_sampler: 'plms',
      hd_strategy: 'CROP',
    })
  })

  it('passes custom options', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samReplaceObject(1, 3, fakeFile, {
      expandMaskPixels: 15,
      useColorMatching: false,
      useEdgeBlending: true,
      colorMatchMethod: 'histogram',
      ldm: fastLdm,
    })

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).params.expand_mask_pixels).toBe(15)
    expect((config as any).params.use_color_matching).toBe(false)
    expect((config as any).params.use_edge_blending).toBe(true)
    expect((config as any).params.color_match_method).toBe('histogram')
    expect((config as any).params.ldm_steps).toBe(10)
  })

  it('sets multipart/form-data header', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samReplaceObject(1, 3, fakeFile)

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).headers['Content-Type']).toBe('multipart/form-data')
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.samReplaceObject(1, 3, fakeFile)

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('sam replace failed'))

    await expect(mlApi.samReplaceObject(1, 3, fakeFile)).rejects.toThrow('sam replace failed')
  })
})

describe('mlApi: samReplaceObjectWithAsset', () => {
  it('posts to correct url with undefined body and default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samReplaceObjectWithAsset(1, 3, 'asset-123')

    const [url, body, config] = mockedClient.post.mock.calls[0]
    expect(url).toBe('/ml/images/1/segment/3/replace')
    expect(body).toBeUndefined()
    expect((config as any).params).toEqual({
      asset_id: 'asset-123',
      expand_mask_pixels: 8,
      use_color_matching: false,
      use_edge_blending: false,
      color_match_method: 'color_transfer',
      ldm_steps: 25,
      ldm_sampler: 'plms',
      hd_strategy: 'CROP',
    })
  })

  it('passes custom options', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samReplaceObjectWithAsset(1, 3, 'asset-123', {
      expandMaskPixels: 20,
      useColorMatching: true,
      useEdgeBlending: true,
      colorMatchMethod: 'mean_std',
      ldm: fastLdm,
    })

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).params.expand_mask_pixels).toBe(20)
    expect((config as any).params.use_color_matching).toBe(true)
    expect((config as any).params.use_edge_blending).toBe(true)
    expect((config as any).params.color_match_method).toBe('mean_std')
    expect((config as any).params.ldm_steps).toBe(10)
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.samReplaceObjectWithAsset(1, 3, 'asset-123')

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('replace with asset failed'))

    await expect(
      mlApi.samReplaceObjectWithAsset(1, 3, 'asset-123')
    ).rejects.toThrow('replace with asset failed')
  })
})

describe('mlApi: extractObject', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeExtractResponse })

    await mlApi.extractObject(1, 4)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment/4/extract',
      { padding_pixels: 8, label: undefined, persist_to_s3: false }
    )
  })

  it('passes custom paddingPixels, label and persistToS3', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeExtractResponse })

    await mlApi.extractObject(1, 4, { paddingPixels: 20, label: 'my cat', persistToS3: true })

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment/4/extract',
      { padding_pixels: 20, label: 'my cat', persist_to_s3: true }
    )
  })

  it('returns ExtractResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeExtractResponse })

    const result = await mlApi.extractObject(1, 4)

    expect(result).toEqual(fakeExtractResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('extract failed'))

    await expect(mlApi.extractObject(1, 4)).rejects.toThrow('extract failed')
  })
})

describe('mlApi: pasteExtractedObject', () => {
  const targetBbox = { x1: 50, y1: 60, x2: 140, y2: 250 }

  it('posts to correct url with default options', async () => {
    mockedClient.post.mockResolvedValue({ data: fakePasteResponse })

    await mlApi.pasteExtractedObject(1, {
      extractedUrl: 's3://bucket/extracted.png',
      targetBbox,
    })

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/paste',
      {
        asset_id: undefined,
        extracted_url: 's3://bucket/extracted.png',
        target_bbox: targetBbox,
        scale: 1.0,
        use_color_matching: false,
        use_edge_blending: false,
        color_match_method: 'color_transfer',
      }
    )
  })

  it('posts with assetId instead of extractedUrl', async () => {
    mockedClient.post.mockResolvedValue({ data: fakePasteResponse })

    await mlApi.pasteExtractedObject(1, {
      assetId: 'asset-123',
      targetBbox,
    })

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).asset_id).toBe('asset-123')
    expect((body as any).extracted_url).toBeUndefined()
  })

  it('passes custom scale and matching options', async () => {
    mockedClient.post.mockResolvedValue({ data: fakePasteResponse })

    await mlApi.pasteExtractedObject(1, {
      extractedUrl: 's3://bucket/extracted.png',
      targetBbox,
      scale: 1.5,
      useColorMatching: true,
      useEdgeBlending: true,
      colorMatchMethod: 'histogram',
    })

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).scale).toBe(1.5)
    expect((body as any).use_color_matching).toBe(true)
    expect((body as any).use_edge_blending).toBe(true)
    expect((body as any).color_match_method).toBe('histogram')
  })

  it('returns PasteResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakePasteResponse })

    const result = await mlApi.pasteExtractedObject(1, {
      extractedUrl: 's3://bucket/extracted.png',
      targetBbox,
    })

    expect(result).toEqual(fakePasteResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('paste failed'))

    await expect(
      mlApi.pasteExtractedObject(1, { extractedUrl: 's3://x.png', targetBbox })
    ).rejects.toThrow('paste failed')
  })
})

describe('mlApi: listAssets', () => {
  it('gets from correct url with default pagination', async () => {
    mockedClient.get.mockResolvedValue({ data: [fakeAsset] })

    await mlApi.listAssets()

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/assets', {
      params: { limit: 50, offset: 0 },
    })
  })

  it('passes custom limit and offset', async () => {
    mockedClient.get.mockResolvedValue({ data: [fakeAsset] })

    await mlApi.listAssets(10, 20)

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/assets', {
      params: { limit: 10, offset: 20 },
    })
  })

  it('returns list of assets', async () => {
    mockedClient.get.mockResolvedValue({ data: [fakeAsset] })

    const result = await mlApi.listAssets()

    expect(result).toEqual([fakeAsset])
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('list assets failed'))

    await expect(mlApi.listAssets()).rejects.toThrow('list assets failed')
  })
})

describe('mlApi: getAssetThumbnailBlob', () => {
  it('gets from correct url with blob responseType', async () => {
    const fakeBlob = new Blob(['thumb'])
    mockedClient.get.mockResolvedValue({ data: fakeBlob })

    await mlApi.getAssetThumbnailBlob('asset-123')

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/assets/asset-123/thumbnail', {
      responseType: 'blob',
    })
  })

  it('returns a Blob', async () => {
    const fakeBlob = new Blob(['thumb'])
    mockedClient.get.mockResolvedValue({ data: fakeBlob })

    const result = await mlApi.getAssetThumbnailBlob('asset-123')

    expect(result).toBe(fakeBlob)
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('thumbnail failed'))

    await expect(mlApi.getAssetThumbnailBlob('asset-123')).rejects.toThrow('thumbnail failed')
  })
})

describe('mlApi: getAssetImageBlob', () => {
  it('gets from correct url with blob responseType', async () => {
    const fakeBlob = new Blob(['image'])
    mockedClient.get.mockResolvedValue({ data: fakeBlob })

    await mlApi.getAssetImageBlob('asset-123')

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/assets/asset-123/image', {
      responseType: 'blob',
    })
  })

  it('returns a Blob', async () => {
    const fakeBlob = new Blob(['image'])
    mockedClient.get.mockResolvedValue({ data: fakeBlob })

    const result = await mlApi.getAssetImageBlob('asset-123')

    expect(result).toBe(fakeBlob)
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('image blob failed'))

    await expect(mlApi.getAssetImageBlob('asset-123')).rejects.toThrow('image blob failed')
  })
})

describe('mlApi: renameAsset', () => {
  it('patches to correct url with label', async () => {
    mockedClient.patch.mockResolvedValue({ data: fakeAsset })

    await mlApi.renameAsset('asset-123', 'new label')

    expect(mockedClient.patch).toHaveBeenCalledWith('/ml/assets/asset-123', {
      label: 'new label',
    })
  })

  it('returns updated Asset', async () => {
    mockedClient.patch.mockResolvedValue({ data: fakeAsset })

    const result = await mlApi.renameAsset('asset-123', 'new label')

    expect(result).toEqual(fakeAsset)
  })

  it('propagates error', async () => {
    mockedClient.patch.mockRejectedValue(new Error('rename failed'))

    await expect(mlApi.renameAsset('asset-123', 'new label')).rejects.toThrow('rename failed')
  })
})

describe('mlApi: deleteAsset', () => {
  it('deletes from correct url', async () => {
    mockedClient.delete.mockResolvedValue({ data: undefined })

    await mlApi.deleteAsset('asset-123')

    expect(mockedClient.delete).toHaveBeenCalledWith('/ml/assets/asset-123')
  })

  it('propagates error', async () => {
    mockedClient.delete.mockRejectedValue(new Error('delete failed'))

    await expect(mlApi.deleteAsset('asset-123')).rejects.toThrow('delete failed')
  })
})

describe('mlApi: saveResult', () => {
  const fakeImage = {
    id: 1,
    filename: 'photo.jpg',
    storage_path: '/path/photo.jpg',
    status: 'saved',
    uploaded_at: '2026-01-01T00:00:00Z',
    user_id: 1,
  }

  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeImage })

    await mlApi.saveResult(1)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/save')
  })

  it('returns the saved Image', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeImage })

    const result = await mlApi.saveResult(1)

    expect(result).toEqual(fakeImage)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('save failed'))

    await expect(mlApi.saveResult(1)).rejects.toThrow('save failed')
  })
})

describe('mlApi: undo', () => {
  const fakeUndoResponse = {
    presigned_url: 'https://storage.example.com/undo.jpg',
    label: 'remove',
    history: ['remove', 'replace'],
  }

  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeUndoResponse })

    await mlApi.undo(1)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/undo')
  })

  it('returns undo response', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeUndoResponse })

    const result = await mlApi.undo(1)

    expect(result).toEqual(fakeUndoResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('undo failed'))

    await expect(mlApi.undo(1)).rejects.toThrow('undo failed')
  })
})

describe('mlApi: redo', () => {
  const fakeRedoResponse = {
    presigned_url: 'https://storage.example.com/redo.jpg',
    label: 'replace',
    history: ['remove', 'replace'],
  }

  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeRedoResponse })

    await mlApi.redo(1)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/redo')
  })

  it('returns redo response', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeRedoResponse })

    const result = await mlApi.redo(1)

    expect(result).toEqual(fakeRedoResponse)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('redo failed'))

    await expect(mlApi.redo(1)).rejects.toThrow('redo failed')
  })
})

describe('mlApi: getHistory', () => {
  it('gets from correct url', async () => {
    mockedClient.get.mockResolvedValue({ data: { history: ['remove', 'replace'] } })

    await mlApi.getHistory(1)

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/images/1/history')
  })

  it('returns history response', async () => {
    mockedClient.get.mockResolvedValue({ data: { history: ['remove', 'replace'] } })

    const result = await mlApi.getHistory(1)

    expect(result).toEqual({ history: ['remove', 'replace'] })
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('history failed'))

    await expect(mlApi.getHistory(1)).rejects.toThrow('history failed')
  })
})

describe('mlApi: resetState', () => {
  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: undefined })

    await mlApi.resetState(1)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/reset')
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('reset failed'))

    await expect(mlApi.resetState(1)).rejects.toThrow('reset failed')
  })
})