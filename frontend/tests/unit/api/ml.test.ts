import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { MLResultResponse, LdmConfig, Image } from '@/types/Index'

vi.mock('@/api/clients', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn()
  }
}))

import apiClient from '@/api/clients'
import { mlApi, PRESETS } from '@/api/ml'

const mockedClient = vi.mocked(apiClient, true)

const fakeMLResult: MLResultResponse = {
  result_url: 's3://bucket/result.jpg',
  presigned_url: 'https://storage.example.com/result.jpg?sig=abc',
  metrics: { processing_time_ms: 250.5, mask_size_pixels: 10000 },
  timestamp: '2026-01-01T00:00:00Z'
}

const fakeSavedImage: Image = {
  id: 99,
  filename: 'edited_photo.jpg',
  storage_path: 'saved/1/99/result.jpg',
  status: 'processed',
  uploaded_at: '2026-01-01T00:00:00Z',
  user_id: 1
}

const fakeUndoRedo = {
  presigned_url: 'https://storage.example.com/undo.jpg?sig=xyz',
  label: 'remove bbox_id=5',
  history: ['remove bbox_id=5', 'replace bbox_id=2']
}

const qualityLdm: LdmConfig = { ldm_steps: 25, ldm_sampler: 'plms', hd_strategy: 'CROP' }
const fastLdm: LdmConfig = { ldm_steps: 10, ldm_sampler: 'plms', hd_strategy: 'CROP' }

beforeEach(() => {
  vi.clearAllMocks()
})


describe('PRESETS', () => {
  it('has fast preset with 10 steps', () => {
    expect(PRESETS.fast.ldm_steps).toBe(10)
    expect(PRESETS.fast.ldm_sampler).toBe('plms')
    expect(PRESETS.fast.hd_strategy).toBe('CROP')
  })

  it('has quality preset with 25 steps', () => {
    expect(PRESETS.quality.ldm_steps).toBe(25)
    expect(PRESETS.quality.ldm_sampler).toBe('plms')
    expect(PRESETS.quality.hd_strategy).toBe('CROP')
  })
})


describe('mlApi: detectObjects', () => {
  it('posts to correct url with params', async () => {
    mockedClient.post.mockResolvedValue({ data: { detections: [], metrics: {} } })

    await mlApi.detectObjects(42, { conf_threshold: 0.7 })

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/42/detect',
      { conf_threshold: 0.7 }
    )
  })

  it('uses empty params by default', async () => {
    mockedClient.post.mockResolvedValue({ data: { detections: [] } })

    await mlApi.detectObjects(1)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/1/detect', {})
  })

  it('returns detection result', async () => {
    const fakeDetect = { detections: [{ bbox_id: 0, detected_class: 'person' }], metrics: {} }
    mockedClient.post.mockResolvedValue({ data: fakeDetect })

    const result = await mlApi.detectObjects(1)

    expect(result).toEqual(fakeDetect)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('detection failed'))

    await expect(mlApi.detectObjects(1)).rejects.toThrow('detection failed')
  })
})


describe('mlApi: removeObject', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeObject(1, 5)

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/remove/5',
      {
        expand_mask_pixels: 5,
        use_edge_blending: false,
        ldm: qualityLdm
      }
    )
  })

  it('passes custom expandMaskPixels and useEdgeBlending', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeObject(2, 3, 10, true)

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).expand_mask_pixels).toBe(10)
    expect((body as any).use_edge_blending).toBe(true)
  })

  it('passes custom ldm config', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeObject(1, 0, 5, false, fastLdm)

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).ldm).toEqual(fastLdm)
  })

  it('uses quality preset as default ldm', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeObject(1, 0)

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).ldm).toEqual(qualityLdm)
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.removeObject(1, 0)

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('remove failed'))

    await expect(mlApi.removeObject(1, 0)).rejects.toThrow('remove failed')
  })
})


describe('mlApi: replaceObject', () => {
  const fakeFile = new File(['img'], 'replacement.png', { type: 'image/png' })

  it('posts multipart/form-data to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.replaceObject(1, 2, fakeFile)

    expect(mockedClient.post).toHaveBeenCalledTimes(1)
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
      color_match_method: 'color_transfer',
      ldm_steps: 25,
      ldm_sampler: 'plms',
      hd_strategy: 'CROP'
    })
  })

  it('passes custom ldm params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.replaceObject(1, 2, fakeFile, {
      ldmSteps: 10,
      ldmSampler: 'ddim',
      hdStrategy: 'RESIZE'
    })

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).params.ldm_steps).toBe(10)
    expect((config as any).params.ldm_sampler).toBe('ddim')
    expect((config as any).params.hd_strategy).toBe('RESIZE')
  })

  it('passes custom color matching options', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.replaceObject(1, 2, fakeFile, {
      useColorMatching: true,
      colorMatchMethod: 'histogram',
      useEdgeBlending: true,
      expandMaskPixels: 15
    })

    const [, , config] = mockedClient.post.mock.calls[0]
    expect((config as any).params.use_color_matching).toBe(true)
    expect((config as any).params.color_match_method).toBe('histogram')
    expect((config as any).params.use_edge_blending).toBe(true)
    expect((config as any).params.expand_mask_pixels).toBe(15)
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


describe('mlApi: removeMultipleObjects', () => {
  it('posts to correct url with default params', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeMultipleObjects(1, [0, 1, 2])

    expect(mockedClient.post).toHaveBeenCalledWith(
      '/ml/images/1/remove-multiple',
      {
        bbox_ids: [0, 1, 2],
        expand_mask_pixels: 5,
        use_edge_blending: false,
        ldm: qualityLdm
      }
    )
  })

  it('passes custom expandMaskPixels', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeMultipleObjects(1, [0, 1], 20, true)

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).expand_mask_pixels).toBe(20)
    expect((body as any).use_edge_blending).toBe(true)
  })

  it('passes custom ldm config', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeMultipleObjects(1, [0], 5, false, fastLdm)

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).ldm).toEqual(fastLdm)
  })

  it('uses quality preset as default ldm', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    await mlApi.removeMultipleObjects(1, [0, 1])

    const [, body] = mockedClient.post.mock.calls[0]
    expect((body as any).ldm).toEqual(qualityLdm)
  })

  it('returns MLResultResponse', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeMLResult })

    const result = await mlApi.removeMultipleObjects(1, [0])

    expect(result).toEqual(fakeMLResult)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('remove multiple failed'))

    await expect(mlApi.removeMultipleObjects(1, [0])).rejects.toThrow('remove multiple failed')
  })
})


describe('mlApi: getSupportedClasses', () => {
  it('gets classes from correct url', async () => {
    mockedClient.get.mockResolvedValue({ data: ['person', 'car', 'dog'] })

    const result = await mlApi.getSupportedClasses()

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/classes')
    expect(result).toEqual(['person', 'car', 'dog'])
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('server error'))

    await expect(mlApi.getSupportedClasses()).rejects.toThrow('server error')
  })
})


describe('mlApi: saveResult', () => {
  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeSavedImage })

    const result = await mlApi.saveResult(42)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/42/save')
    expect(result).toEqual(fakeSavedImage)
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('nothing to save'))

    await expect(mlApi.saveResult(1)).rejects.toThrow('nothing to save')
  })
})


describe('mlApi: undo', () => {
  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeUndoRedo })

    const result = await mlApi.undo(5)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/5/undo')
    expect(result).toEqual(fakeUndoRedo)
  })

  it('returns presigned_url, label and history', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeUndoRedo })

    const result = await mlApi.undo(1)

    expect(result.presigned_url).toBe(fakeUndoRedo.presigned_url)
    expect(result.label).toBe('remove bbox_id=5')
    expect(result.history).toHaveLength(2)
  })

  it('propagates error when nothing to undo', async () => {
    mockedClient.post.mockRejectedValue(new Error('Nothing to undo'))

    await expect(mlApi.undo(1)).rejects.toThrow('Nothing to undo')
  })
})


describe('mlApi: redo', () => {
  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeUndoRedo })

    const result = await mlApi.redo(7)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/7/redo')
    expect(result).toEqual(fakeUndoRedo)
  })

  it('returns presigned_url, label and history', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeUndoRedo })

    const result = await mlApi.redo(1)

    expect(result.presigned_url).toBeDefined()
    expect(Array.isArray(result.history)).toBe(true)
  })

  it('propagates error when nothing to redo', async () => {
    mockedClient.post.mockRejectedValue(new Error('Nothing to redo'))

    await expect(mlApi.redo(1)).rejects.toThrow('Nothing to redo')
  })
})


describe('mlApi: getHistory', () => {
  it('gets history from correct url', async () => {
    mockedClient.get.mockResolvedValue({ data: { history: ['remove bbox_id=1', 'replace bbox_id=2'] } })

    const result = await mlApi.getHistory(3)

    expect(mockedClient.get).toHaveBeenCalledWith('/ml/images/3/history')
    expect(result.history).toHaveLength(2)
    expect(result.history[0]).toBe('remove bbox_id=1')
  })

  it('returns empty history when no operations', async () => {
    mockedClient.get.mockResolvedValue({ data: { history: [] } })

    const result = await mlApi.getHistory(1)

    expect(result.history).toEqual([])
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('image not found'))

    await expect(mlApi.getHistory(999)).rejects.toThrow('image not found')
  })
})


describe('mlApi: resetState', () => {
  it('posts to correct url', async () => {
    mockedClient.post.mockResolvedValue({ data: { detail: 'State reset to original image' } })

    await mlApi.resetState(10)

    expect(mockedClient.post).toHaveBeenCalledWith('/ml/images/10/reset')
  })

  it('returns void', async () => {
    mockedClient.post.mockResolvedValue({ data: undefined })

    const result = await mlApi.resetState(1)

    expect(result).toBeUndefined()
  })

  it('propagates error', async () => {
    mockedClient.post.mockRejectedValue(new Error('image not found'))

    await expect(mlApi.resetState(999)).rejects.toThrow('image not found')
  })
})