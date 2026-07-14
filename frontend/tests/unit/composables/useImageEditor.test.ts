import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Image } from '@/types/Index'

vi.mock('@/api/images', () => ({
  imagesApi: {
    getById: vi.fn(),
    getPresignedUrl: vi.fn()
  }
}))

vi.mock('@/api/ml', () => ({
  mlApi: {
    getCurrentState: vi.fn()
  }
}))

vi.mock('vue', async () => {
  const actual = await vi.importActual<typeof import('vue')>('vue')
  return { ...actual, onMounted: (fn: () => void) => fn() }
})

import { imagesApi } from '@/api/images'
import { mlApi } from '@/api/ml'
import { useImageEditor } from '@/composables/useImageEditor'

const mockedImagesApi = vi.mocked(imagesApi, true)
const mockedMlApi = vi.mocked(mlApi, true)

const fakeImage: Image = {
  id: 5,
  filename: 'shot.jpg',
  storage_path: 'uploads/1/5/shot.jpg',
  status: 'ready',
  uploaded_at: '2026-01-01T00:00:00Z',
  user_id: 1
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('useImageEditor: onMounted', () => {
  it('fetches image, original url and current ml state on mount', async () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/original.jpg', expires_in: 3600 })
    mockedMlApi.getCurrentState.mockResolvedValue({ presigned_url: 'https://cdn.example.com/edited.jpg', is_edited: true })

    const { image, imageUrl, originalImageUrl, isEdited } = useImageEditor(5)

    await vi.waitUntil(() => image.value !== null)

    expect(mockedImagesApi.getById).toHaveBeenCalledWith(5)
    expect(mockedImagesApi.getPresignedUrl).toHaveBeenCalledWith(5, 3600)
    expect(mockedMlApi.getCurrentState).toHaveBeenCalledWith(5)

    expect(image.value).toEqual(fakeImage)
    expect(originalImageUrl.value).toBe('https://cdn.example.com/original.jpg')
    expect(imageUrl.value).toBe('https://cdn.example.com/edited.jpg')
    expect(isEdited.value).toBe(true)
  })

  it('sets isEdited to false when ml state reports no edits', async () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/original.jpg', expires_in: 3600 })
    mockedMlApi.getCurrentState.mockResolvedValue({ presigned_url: 'https://cdn.example.com/original.jpg', is_edited: false })

    const { isEdited } = useImageEditor(5)

    await vi.waitUntil(() => isEdited.value !== null)

    expect(isEdited.value).toBe(false)
  })

  it('sets loading to false after success', async () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/original.jpg', expires_in: 3600 })
    mockedMlApi.getCurrentState.mockResolvedValue({ presigned_url: 'https://cdn.example.com/edited.jpg', is_edited: false })

    const { loading } = useImageEditor(5)

    await vi.waitUntil(() => loading.value === false)

    expect(loading.value).toBe(false)
  })

  it('sets loading to false even when request fails', async () => {
    mockedImagesApi.getById.mockRejectedValue(new Error('not found'))

    const { loading } = useImageEditor(5)

    await vi.waitUntil(() => loading.value === false)

    expect(loading.value).toBe(false)
  })

  it('leaves image, imageUrl and originalImageUrl empty when request fails', async () => {
    mockedImagesApi.getById.mockRejectedValue(new Error('not found'))

    const { image, imageUrl, originalImageUrl, isEdited } = useImageEditor(5)

    await vi.waitUntil(() => image.value === null)

    expect(image.value).toBeNull()
    expect(imageUrl.value).toBe('')
    expect(originalImageUrl.value).toBe('')
    expect(isEdited.value).toBe(false)
  })

  it('leaves imageUrl and isEdited unset when ml state request fails after image is fetched', async () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/original.jpg', expires_in: 3600 })
    mockedMlApi.getCurrentState.mockRejectedValue(new Error('ml service unavailable'))

    const { image, originalImageUrl, imageUrl, loading } = useImageEditor(5)

    await vi.waitUntil(() => loading.value === false)

    expect(image.value).toEqual(fakeImage)
    expect(originalImageUrl.value).toBe('https://cdn.example.com/original.jpg')
    expect(imageUrl.value).toBe('')
  })
})

describe('useImageEditor: initial state', () => {
  it('starts with empty detections, naturalSize, originalImageUrl and isEdited false', () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: '', expires_in: 3600 })
    mockedMlApi.getCurrentState.mockResolvedValue({ presigned_url: '', is_edited: false })

    const { detections, naturalSize, imageLoaded, originalImageUrl, isEdited } = useImageEditor(5)

    expect(detections.value).toEqual([])
    expect(naturalSize.value).toEqual({ w: 0, h: 0 })
    expect(imageLoaded.value).toBe(false)
    expect(originalImageUrl.value).toBe('')
    expect(isEdited.value).toBe(false)
  })
})

describe('useImageEditor: onImageLoad', () => {
  it('sets naturalSize and imageLoaded to true', () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: '', expires_in: 3600 })
    mockedMlApi.getCurrentState.mockResolvedValue({ presigned_url: '', is_edited: false })

    const { onImageLoad, naturalSize, imageLoaded } = useImageEditor(5)

    onImageLoad({ w: 1920, h: 1080 })

    expect(naturalSize.value).toEqual({ w: 1920, h: 1080 })
    expect(imageLoaded.value).toBe(true)
  })

  it('updates naturalSize when called again with new size', () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: '', expires_in: 3600 })
    mockedMlApi.getCurrentState.mockResolvedValue({ presigned_url: '', is_edited: false })

    const { onImageLoad, naturalSize } = useImageEditor(5)

    onImageLoad({ w: 800, h: 600 })
    onImageLoad({ w: 1280, h: 720 })

    expect(naturalSize.value).toEqual({ w: 1280, h: 720 })
  })
})