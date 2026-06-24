import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import type { Image, PresignedUrlResponse } from '@/types/Index'

// imagesApi calls .get/.post/.delete directly on the default-exported
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
import { imagesApi } from '@/api/images'

const mockedClient = vi.mocked(apiClient, true)

const fakeImage: Image = {
  id: 1,
  filename: 'photo.jpg',
  storage_path: 'uploads/1/photo.jpg',
  status: 'ready',
  uploaded_at: '2026-01-01T00:00:00Z',
  user_id: 7
}

const fakeImages: Image[] = [
  fakeImage,
  {
    id: 2,
    filename: 'photo2.jpg',
    storage_path: 'uploads/1/photo2.jpg',
    status: 'processing',
    uploaded_at: '2026-01-02T00:00:00Z',
    user_id: 7
  }
]

const fakePresignedUrl: PresignedUrlResponse = {
  url: 'https://storage.example.com/photo.jpg?sig=abc',
  expires_in: 3600
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('imagesApi: upload', () => {
  it('posts the file as multipart/form-data to /images/upload', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeImage })
    const file = new File(['file-contents'], 'photo.jpg', { type: 'image/jpeg' })

    const result = await imagesApi.upload(file)

    expect(mockedClient.post).toHaveBeenCalledTimes(1)
    const [url, body, config] = mockedClient.post.mock.calls[0]
    expect(url).toBe('/images/upload')
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('file')).toBe(file)
    expect(config).toEqual({ headers: { 'Content-Type': 'multipart/form-data' } })
    expect(result).toEqual(fakeImage)
  })

  it('propagates the error when the upload fails', async () => {
    mockedClient.post.mockRejectedValue(new Error('file too large'))
    const file = new File(['x'], 'big.jpg', { type: 'image/jpeg' })

    await expect(imagesApi.upload(file)).rejects.toThrow('file too large')
  })
})

describe('imagesApi: getAll', () => {
  it('gets all images without pagination params', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeImages })

    const result = await imagesApi.getAll()

    expect(mockedClient.get).toHaveBeenCalledWith('/images/', {
      params: { limit: undefined, offset: undefined }
    })
    expect(result).toEqual(fakeImages)
  })

  it('passes limit and offset through to the request params', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeImages })

    await imagesApi.getAll(10, 20)

    expect(mockedClient.get).toHaveBeenCalledWith('/images/', {
      params: { limit: 10, offset: 20 }
    })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.get.mockRejectedValue(new Error('server error'))

    await expect(imagesApi.getAll()).rejects.toThrow('server error')
  })
})

describe('imagesApi: getById', () => {
  it('gets a single image by id', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeImage })

    const result = await imagesApi.getById(1)

    expect(mockedClient.get).toHaveBeenCalledWith('/images/1')
    expect(result).toEqual(fakeImage)
  })

  it('propagates the error when the image is not found', async () => {
    mockedClient.get.mockRejectedValue(new Error('image not found'))

    await expect(imagesApi.getById(999)).rejects.toThrow('image not found')
  })
})

describe('imagesApi: getPresignedUrl', () => {
  it('gets a presigned url with the given expiration', async () => {
    mockedClient.get.mockResolvedValue({ data: fakePresignedUrl })

    const result = await imagesApi.getPresignedUrl(1, 3600)

    expect(mockedClient.get).toHaveBeenCalledWith('/images/1/url', {
      params: { expiration: 3600 }
    })
    expect(result).toEqual(fakePresignedUrl)
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.get.mockRejectedValue(new Error('image not found'))

    await expect(imagesApi.getPresignedUrl(999, 3600)).rejects.toThrow('image not found')
  })
})

describe('imagesApi: delete', () => {
  it('deletes the image by id', async () => {
    mockedClient.delete.mockResolvedValue({ data: undefined })

    await imagesApi.delete(1)

    expect(mockedClient.delete).toHaveBeenCalledWith('/images/1')
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.delete.mockRejectedValue(new Error('image not found'))

    await expect(imagesApi.delete(999)).rejects.toThrow('image not found')
  })
})

describe('imagesApi: download', () => {
  let createObjectURLSpy: ReturnType<typeof vi.fn>
  let revokeObjectURLSpy: ReturnType<typeof vi.fn>
  let clickSpy: () => void
  let createElementSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    createObjectURLSpy = vi.fn().mockReturnValue('blob:fake-url')
    revokeObjectURLSpy = vi.fn()
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: createObjectURLSpy,
      revokeObjectURL: revokeObjectURLSpy
    })

    clickSpy = vi.fn() as () => void
    const realCreateElement = document.createElement.bind(document)
    createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = realCreateElement(tag)
      if (tag === 'a') {
        el.click = clickSpy
      }
      return el
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    createElementSpy.mockRestore()
  })

  it('downloads the image as a blob and triggers a click on an anchor', async () => {
    const fakeBlob = new Blob(['binary-data'], { type: 'image/jpeg' })
    mockedClient.get.mockResolvedValue({ data: fakeBlob })

    await imagesApi.download(1)

    expect(mockedClient.get).toHaveBeenCalledWith('/images/1/download', {
      responseType: 'blob'
    })
    expect(createObjectURLSpy).toHaveBeenCalledWith(fakeBlob)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:fake-url')
  })

  it('sets the anchor download attribute based on the image id', async () => {
    const fakeBlob = new Blob(['binary-data'], { type: 'image/jpeg' })
    mockedClient.get.mockResolvedValue({ data: fakeBlob })

    let capturedAnchor: HTMLAnchorElement | null = null
    createElementSpy.mockRestore()
    const realCreateElement = document.createElement.bind(document)
    createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = realCreateElement(tag)
      if (tag === 'a') {
        el.click = clickSpy
        capturedAnchor = el as HTMLAnchorElement
      }
      return el
    })

    await imagesApi.download(42)

    expect(capturedAnchor!.download).toBe('image_42.jpg')
    expect(capturedAnchor!.href).toContain('blob:fake-url')
  })

  it('propagates the error when the download request fails', async () => {
    mockedClient.get.mockRejectedValue(new Error('image not found'))

    await expect(imagesApi.download(999)).rejects.toThrow('image not found')
    expect(createObjectURLSpy).not.toHaveBeenCalled()
  })
})