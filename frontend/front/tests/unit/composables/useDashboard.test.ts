import { describe, it, expect, vi, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import type { Image } from '@/types/Index'

vi.mock('@/api/images', () => ({
  imagesApi: {
    getAll: vi.fn(),
    getPresignedUrl: vi.fn(),
    upload: vi.fn(),
    delete: vi.fn()
  }
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() })
}))

import { imagesApi } from '@/api/images'
import { useDashboard } from '@/composables/useDashboard'

const mockedImagesApi = vi.mocked(imagesApi, true)

const fakeImage: Image = {
  id: 1,
  filename: 'photo.jpg',
  storage_path: 'uploads/1/photo.jpg',
  status: 'ready',
  uploaded_at: '2026-06-01T00:00:00Z',
  user_id: 7
}

const fakeImage2: Image = {
  id: 2,
  filename: 'photo2.jpg',
  storage_path: 'uploads/1/photo2.jpg',
  status: 'ready',
  uploaded_at: '2026-06-02T00:00:00Z',
  user_id: 7
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.clearAllMocks()
})


describe('useDashboard: fetchImages', () => {
  it('loads images and their presigned urls', async () => {
    mockedImagesApi.getAll.mockResolvedValue([fakeImage, fakeImage2])
    mockedImagesApi.getPresignedUrl
      .mockResolvedValueOnce({ url: 'https://cdn.example.com/1.jpg', expires_in: 3600 })
      .mockResolvedValueOnce({ url: 'https://cdn.example.com/2.jpg', expires_in: 3600 })

    const { images, imageUrls, loading, fetchImages } = useDashboard()

    const promise = fetchImages()
    expect(loading.value).toBe(true)
    await promise

    expect(images.value).toEqual([fakeImage, fakeImage2])
    expect(imageUrls.value[1]).toBe('https://cdn.example.com/1.jpg')
    expect(imageUrls.value[2]).toBe('https://cdn.example.com/2.jpg')
    expect(loading.value).toBe(false)
  })

  it('sets loading to false even when getAll fails', async () => {
    mockedImagesApi.getAll.mockRejectedValue(new Error('server error'))

    const { loading, fetchImages } = useDashboard()
    await fetchImages()

    expect(loading.value).toBe(false)
  })

  it('still loads other images if one presigned url request fails', async () => {
    mockedImagesApi.getAll.mockResolvedValue([fakeImage, fakeImage2])
    mockedImagesApi.getPresignedUrl
      .mockRejectedValueOnce(new Error('forbidden'))
      .mockResolvedValueOnce({ url: 'https://cdn.example.com/2.jpg', expires_in: 3600 })

    const { images, imageUrls, fetchImages } = useDashboard()
    await fetchImages()

    expect(images.value).toHaveLength(2)
    expect(imageUrls.value[1]).toBeUndefined()
    expect(imageUrls.value[2]).toBe('https://cdn.example.com/2.jpg')
  })

  it('requests presigned url with 3600s expiration for each image', async () => {
    mockedImagesApi.getAll.mockResolvedValue([fakeImage])
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/1.jpg', expires_in: 3600 })

    const { fetchImages } = useDashboard()
    await fetchImages()

    expect(mockedImagesApi.getPresignedUrl).toHaveBeenCalledWith(fakeImage.id, 3600)
  })
})


describe('useDashboard: handleFileChange', () => {
  function makeFileEvent(file: File | null): Event {
    const input = document.createElement('input')
    Object.defineProperty(input, 'files', {
      value: file ? { 0: file, length: 1, item: () => file } : null
    })
    return { target: input } as unknown as Event
  }

  it('uploads the file, prepends to images list and redirects', async () => {
    const pushMock = vi.fn()
    vi.doMock('vue-router', () => ({ useRouter: () => ({ push: pushMock }) }))

    mockedImagesApi.upload.mockResolvedValue(fakeImage)

    const { images, uploading, handleFileChange } = useDashboard()
    const file = new File(['img'], 'photo.jpg', { type: 'image/jpeg' })

    const promise = handleFileChange(makeFileEvent(file))
    expect(uploading.value).toBe(true)
    await promise

    expect(mockedImagesApi.upload).toHaveBeenCalledWith(file)
    expect(images.value[0]).toEqual(fakeImage)
    expect(uploading.value).toBe(false)
  })

  it('does nothing when no file is selected', async () => {
    const { handleFileChange } = useDashboard()
    await handleFileChange(makeFileEvent(null))

    expect(mockedImagesApi.upload).not.toHaveBeenCalled()
  })

  it('sets uploading to false even when upload fails', async () => {
    mockedImagesApi.upload.mockRejectedValue(new Error('too large'))

    const { uploading, handleFileChange } = useDashboard()
    const file = new File(['x'], 'big.jpg', { type: 'image/jpeg' })
    await handleFileChange(makeFileEvent(file))

    expect(uploading.value).toBe(false)
  })
})


describe('useDashboard: handleDelete', () => {
  it('removes the image from the list and clears its url', async () => {
    mockedImagesApi.getAll.mockResolvedValue([fakeImage, fakeImage2])
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/1.jpg', expires_in: 3600 })
    mockedImagesApi.delete.mockResolvedValue(undefined)

    const { images, imageUrls, fetchImages, handleDelete } = useDashboard()
    await fetchImages()
    await handleDelete(fakeImage.id)

    expect(mockedImagesApi.delete).toHaveBeenCalledWith(fakeImage.id)
    expect(images.value.find(img => img.id === fakeImage.id)).toBeUndefined()
    expect(imageUrls.value[fakeImage.id]).toBeUndefined()
  })

  it('does not throw and leaves list intact when delete fails', async () => {
    mockedImagesApi.getAll.mockResolvedValue([fakeImage])
    mockedImagesApi.getPresignedUrl.mockResolvedValue({ url: 'https://cdn.example.com/1.jpg', expires_in: 3600 })
    mockedImagesApi.delete.mockRejectedValue(new Error('not found'))

    const { images, fetchImages, handleDelete } = useDashboard()
    await fetchImages()
    await handleDelete(fakeImage.id)

    expect(images.value).toHaveLength(1)
  })
})


describe('useDashboard: formatDate', () => {
  it('formats ISO date to readable string', () => {
    const { formatDate } = useDashboard()
    const result = formatDate('2026-06-01T00:00:00Z')
    expect(result).toMatch(/Jun/)
    expect(result).toMatch(/2026/)
  })

  it('formats different months correctly', () => {
    const { formatDate } = useDashboard()
    expect(formatDate('2026-01-15T00:00:00Z')).toMatch(/Jan/)
    expect(formatDate('2026-12-31T00:00:00Z')).toMatch(/Dec/)
  })
})