import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Image} from '@/types/Index'

vi.mock('@/api/images', () => ({
  imagesApi: {
    getById: vi.fn(),
    getPresignedUrl: vi.fn()
  }
}))

vi.mock('vue', async () => {
  const actual = await vi.importActual<typeof import('vue')>('vue')
  return { ...actual, onMounted: (fn: () => void) => fn() }
})

import { imagesApi } from '@/api/images'
import { useImageEditor } from '@/composables/useImageEditor'

const mockedImagesApi = vi.mocked(imagesApi, true)

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
  it('fetches image and presigned url on mount', async () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/shot.jpg', expires_in: 3600 })

    const { image, imageUrl } = useImageEditor(5)

    await vi.waitUntil(() => image.value !== null)

    expect(mockedImagesApi.getById).toHaveBeenCalledWith(5)
    expect(mockedImagesApi.getPresignedUrl).toHaveBeenCalledWith(5, 3600)
    expect(image.value).toEqual(fakeImage)
    expect(imageUrl.value).toBe('https://cdn.example.com/shot.jpg')
  })

  it('sets loading to false after success', async () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/shot.jpg', expires_in: 3600 })

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

  it('leaves image and imageUrl empty when request fails', async () => {
    mockedImagesApi.getById.mockRejectedValue(new Error('not found'))

    const { image, imageUrl } = useImageEditor(5)

    await vi.waitUntil(() => image.value === null)

    expect(image.value).toBeNull()
    expect(imageUrl.value).toBe('')
  })
})


describe('useImageEditor: initial state', () => {
  it('starts with empty detections and naturalSize', () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: '', expires_in: 3600 })

    const { detections, naturalSize, imageLoaded } = useImageEditor(5)

    expect(detections.value).toEqual([])
    expect(naturalSize.value).toEqual({ w: 0, h: 0 })
    expect(imageLoaded.value).toBe(false)
  })
})


describe('useImageEditor: onImageLoad', () => {
  it('sets naturalSize and imageLoaded to true', () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: '', expires_in: 3600 })

    const { onImageLoad, naturalSize, imageLoaded } = useImageEditor(5)

    onImageLoad({ w: 1920, h: 1080 })

    expect(naturalSize.value).toEqual({ w: 1920, h: 1080 })
    expect(imageLoaded.value).toBe(true)
  })

  it('updates naturalSize when called again with new size', () => {
    mockedImagesApi.getById.mockResolvedValue(fakeImage)
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: '', expires_in: 3600 })

    const { onImageLoad, naturalSize } = useImageEditor(5)

    onImageLoad({ w: 800, h: 600 })
    onImageLoad({ w: 1280, h: 720 })

    expect(naturalSize.value).toEqual({ w: 1280, h: 720 })
  })
})