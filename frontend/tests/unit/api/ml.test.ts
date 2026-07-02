import { describe, it, expect, vi, beforeEach } from 'vitest'
import type {
  SegmentResponse,
  ExtractResponse,
  PasteResponse,
  MLResultResponse,
  LdmConfig,
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

beforeEach(() => {
  vi.clearAllMocks()
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
        bbox: undefined,
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
    expect((body as any).point_coords).toBeUndefined()
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

describe('mlApi: samRemoveObject', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.samRemoveObject(1, 3)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment/3/remove',
      {
        expand_mask_pixels: 12,
        use_edge_blending: true,
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
      use_color_matching: true,
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

describe('mlApi: extractObject', () => {
  it('posts to correct url with default padding', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeExtractResponse })

    await mlApi.extractObject(1, 4)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment/4/extract',
      { padding_pixels: 8 }
    )
  })

  it('passes custom paddingPixels', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeExtractResponse })

    await mlApi.extractObject(1, 4, 20)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/segment/4/extract',
      { padding_pixels: 20 }
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
        extracted_url: 's3://bucket/extracted.png',
        target_bbox: targetBbox,
        scale: 1.0,
        use_color_matching: true,
        use_edge_blending: true,
        color_match_method: 'color_transfer',
      }
    )
  })

  it('passes custom scale and matching options', async () => {
    mockedClient.post.mockResolvedValue({ data: fakePasteResponse })

    await mlApi.pasteExtractedObject(1, {
      extractedUrl: 's3://bucket/extracted.png',
      targetBbox,
      scale: 1.5,
      useColorMatching: false,
      useEdgeBlending: false,
      colorMatchMethod: 'histogram',
    })

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).scale).toBe(1.5)
    expect((body as any).use_color_matching).toBe(false)
    expect((body as any).use_edge_blending).toBe(false)
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