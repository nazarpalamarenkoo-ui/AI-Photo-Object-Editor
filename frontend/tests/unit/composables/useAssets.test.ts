import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'
import type { Asset } from '@/types/Index'

vi.mock('@/api/ml', () => ({
  mlApi: {
    listAssets: vi.fn(),
    getAssetThumbnailBlob: vi.fn(),
    getAssetImageBlob: vi.fn(),
    extractObject: vi.fn(),
    pasteExtractedObject: vi.fn(),
    renameAsset: vi.fn(),
    deleteAsset: vi.fn(),
    getHistory: vi.fn(),
  }
}))

globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock-url')
globalThis.URL.revokeObjectURL = vi.fn()

import { mlApi } from '@/api/ml'
import { useAssets } from '@/composables/useAssets'

const mockedMlApi = vi.mocked(mlApi, true)

const makeAsset = (id: string, label = `Asset ${id}`): Asset => ({
  public_id: id,
  label,
  created_at: '2026-01-01T00:00:00Z',
})

const makeExtractResult = (assetId: string) => ({
  asset_id: assetId,
  presigned_url: `https://cdn.example.com/${assetId}.png`,
})

const makePasteResult = (presigned_url: string) => ({
  presigned_url,
  result_url: 'https://cdn.example.com/result.jpg',
  metrics: {},
  timestamp: '2026-01-01T00:00:00Z',
})

beforeEach(() => {
  vi.clearAllMocks()
  // loadThumb silently succeeds by default
  mockedMlApi.getAssetThumbnailBlob.mockResolvedValue(new Blob(['thumb']))
})


describe('useAssets: initial state', () => {
  it('starts with correct defaults', () => {
    const { extracting, pasting, mlError, selectedAssetId, extractedPreviewUrl } =
      useAssets(1, ref(''), ref([]))

    expect(extracting.value).toBe(false)
    expect(pasting.value).toBe(false)
    expect(mlError.value).toBe('')
    expect(selectedAssetId.value).toBeNull()
    expect(extractedPreviewUrl.value).toBeNull()
  })

  it('assets list starts empty with assetsHasMore true', () => {
    const { assets, assetsLoading, assetsError, assetsHasMore, thumbUrls, deletingId } =
      useAssets(1, ref(''), ref([]))

    expect(assets.value).toEqual([])
    expect(assetsLoading.value).toBe(false)
    expect(assetsError.value).toBe('')
    expect(assetsHasMore.value).toBe(true)
    expect(thumbUrls.value).toEqual({})
    expect(deletingId.value).toBeNull()
  })
})


describe('useAssets: fetchAssets', () => {
  it('loads a page and populates assets', async () => {
    const page = [makeAsset('a'), makeAsset('b')]
    mockedMlApi.listAssets.mockResolvedValue(page)

    const { fetchAssets, assets } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(mockedMlApi.listAssets).toHaveBeenCalledWith(30, 0)
    expect(assets.value).toEqual(page)
  })

  it('sets assetsHasMore to false when page is shorter than PAGE_SIZE', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('a')])

    const { fetchAssets, assetsHasMore } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(assetsHasMore.value).toBe(false)
  })

  it('sets assetsHasMore to true when page equals PAGE_SIZE (30)', async () => {
    const fullPage = Array.from({ length: 30 }, (_, i) => makeAsset(String(i)))
    mockedMlApi.listAssets.mockResolvedValue(fullPage)

    const { fetchAssets, assetsHasMore } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(assetsHasMore.value).toBe(true)
  })

  it('resets assets on reset=true (default)', async () => {
    const first = [makeAsset('a')]
    const second = [makeAsset('b')]
    mockedMlApi.listAssets.mockResolvedValueOnce(first).mockResolvedValueOnce(second)

    const { fetchAssets, assets } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    await fetchAssets(true)

    expect(assets.value).toEqual(second)
  })

  it('appends assets on reset=false', async () => {
    const first = [makeAsset('a')]
    const second = [makeAsset('b')]
    mockedMlApi.listAssets.mockResolvedValueOnce(first).mockResolvedValueOnce(second)

    const { fetchAssets, assets } = useAssets(1, ref(''), ref([]))
    await fetchAssets(true)
    await fetchAssets(false)

    expect(assets.value).toEqual([makeAsset('a'), makeAsset('b')])
  })

  it('uses current assets length as offset when appending', async () => {
    mockedMlApi.listAssets
      .mockResolvedValueOnce([makeAsset('a')])
      .mockResolvedValueOnce([makeAsset('b')])

    const { fetchAssets } = useAssets(1, ref(''), ref([]))
    await fetchAssets(true)
    await fetchAssets(false)

    expect(mockedMlApi.listAssets).toHaveBeenNthCalledWith(2, 30, 1)
  })

  it('sets assetsLoading to true during the call and false after', async () => {
    let resolve: (v: any) => void
    mockedMlApi.listAssets.mockReturnValue(new Promise(r => { resolve = r }))

    const { fetchAssets, assetsLoading } = useAssets(1, ref(''), ref([]))
    const promise = fetchAssets()
    expect(assetsLoading.value).toBe(true)

    resolve!([])
    await promise
    expect(assetsLoading.value).toBe(false)
  })

  it('does not start a second fetch while one is in progress', async () => {
    let resolve: (v: any) => void
    mockedMlApi.listAssets.mockReturnValue(new Promise(r => { resolve = r }))

    const { fetchAssets } = useAssets(1, ref(''), ref([]))
    const p1 = fetchAssets()
    fetchAssets() // should be ignored
    resolve!([])
    await p1

    expect(mockedMlApi.listAssets).toHaveBeenCalledTimes(1)
  })

  it('sets assetsError when listAssets fails', async () => {
    mockedMlApi.listAssets.mockRejectedValue({
      response: { data: { detail: 'Load failed on server' } }
    })

    const { fetchAssets, assetsError } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(assetsError.value).toBe('Load failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.listAssets.mockRejectedValue(new Error('fail'))

    const { fetchAssets, assetsError } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(assetsError.value).toBe('Failed to load asset library')
  })

  it('sets assetsLoading to false after failure', async () => {
    mockedMlApi.listAssets.mockRejectedValue(new Error('fail'))

    const { fetchAssets, assetsLoading } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(assetsLoading.value).toBe(false)
  })

  it('loads thumbnails for each asset in the page', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('x'), makeAsset('y')])

    const { fetchAssets } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(mockedMlApi.getAssetThumbnailBlob).toHaveBeenCalledWith('x')
    expect(mockedMlApi.getAssetThumbnailBlob).toHaveBeenCalledWith('y')
  })

  it('stores blob object URL in thumbUrls', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('x')])

    const { fetchAssets, thumbUrls } = useAssets(1, ref(''), ref([]))
    await fetchAssets()

    expect(thumbUrls.value['x']).toBe('blob:mock-url')
  })

  it('clears assetsError before each fetch', async () => {
    mockedMlApi.listAssets.mockRejectedValueOnce(new Error('fail'))

    const { fetchAssets, assetsError } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    expect(assetsError.value).toBe('Failed to load asset library')

    mockedMlApi.listAssets.mockResolvedValueOnce([])
    await fetchAssets()
    expect(assetsError.value).toBe('')
  })
})


describe('useAssets: handleExtract', () => {
  it('calls extractObject with imageId, maskId and params', async () => {
    mockedMlApi.extractObject.mockResolvedValue(makeExtractResult('asset-1'))
    mockedMlApi.getAssetImageBlob.mockResolvedValue(new Blob(['img']))
    mockedMlApi.listAssets.mockResolvedValue([])

    const { handleExtract } = useAssets(9, ref(''), ref([]))
    await handleExtract(3, { paddingPixels: 10, label: 'car' })

    expect(mockedMlApi.extractObject).toHaveBeenCalledWith(9, 3, { paddingPixels: 10, label: 'car' })
  })

  it('calls extractObject with empty params by default', async () => {
    mockedMlApi.extractObject.mockResolvedValue(makeExtractResult('asset-1'))
    mockedMlApi.getAssetImageBlob.mockResolvedValue(new Blob(['img']))
    mockedMlApi.listAssets.mockResolvedValue([])

    const { handleExtract } = useAssets(9, ref(''), ref([]))
    await handleExtract(3)

    expect(mockedMlApi.extractObject).toHaveBeenCalledWith(9, 3, {})
  })

  it('sets selectedAssetId from result', async () => {
    mockedMlApi.extractObject.mockResolvedValue(makeExtractResult('asset-1'))
    mockedMlApi.getAssetImageBlob.mockResolvedValue(new Blob(['img']))
    mockedMlApi.listAssets.mockResolvedValue([])

    const { handleExtract, selectedAssetId } = useAssets(9, ref(''), ref([]))
    await handleExtract(1)

    expect(selectedAssetId.value).toBe('asset-1')
  })

  it('sets extractedPreviewUrl from blob', async () => {
    mockedMlApi.extractObject.mockResolvedValue(makeExtractResult('asset-1'))
    mockedMlApi.getAssetImageBlob.mockResolvedValue(new Blob(['img']))
    mockedMlApi.listAssets.mockResolvedValue([])

    const { handleExtract, extractedPreviewUrl } = useAssets(9, ref(''), ref([]))
    await handleExtract(1)

    expect(extractedPreviewUrl.value).toBe('blob:mock-url')
  })

  it('sets extracting to true during and false after success', async () => {
    let resolve: (v: any) => void
    mockedMlApi.extractObject.mockReturnValue(new Promise(r => { resolve = r }))

    const { handleExtract, extracting } = useAssets(9, ref(''), ref([]))
    const promise = handleExtract(1)
    expect(extracting.value).toBe(true)

    mockedMlApi.getAssetImageBlob.mockResolvedValue(new Blob(['img']))
    mockedMlApi.listAssets.mockResolvedValue([])
    resolve!(makeExtractResult('asset-1'))
    await promise
    expect(extracting.value).toBe(false)
  })

  it('returns the extract result', async () => {
    const result = makeExtractResult('asset-1')
    mockedMlApi.extractObject.mockResolvedValue(result)
    mockedMlApi.getAssetImageBlob.mockResolvedValue(new Blob(['img']))
    mockedMlApi.listAssets.mockResolvedValue([])

    const { handleExtract } = useAssets(9, ref(''), ref([]))
    const ret = await handleExtract(1)

    expect(ret).toEqual(result)
  })

  it('sets mlError when extractObject fails', async () => {
    mockedMlApi.extractObject.mockRejectedValue({
      response: { data: { detail: 'Extract failed on server' } }
    })

    const { handleExtract, mlError } = useAssets(9, ref(''), ref([]))
    await handleExtract(1)

    expect(mlError.value).toBe('Extract failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.extractObject.mockRejectedValue(new Error('fail'))

    const { handleExtract, mlError } = useAssets(9, ref(''), ref([]))
    await handleExtract(1)

    expect(mlError.value).toBe('Extract failed')
  })

  it('sets extracting to false after failure', async () => {
    mockedMlApi.extractObject.mockRejectedValue(new Error('fail'))

    const { handleExtract, extracting } = useAssets(9, ref(''), ref([]))
    await handleExtract(1)

    expect(extracting.value).toBe(false)
  })

  it('refreshes asset list after extract', async () => {
    mockedMlApi.extractObject.mockResolvedValue(makeExtractResult('asset-1'))
    mockedMlApi.getAssetImageBlob.mockResolvedValue(new Blob(['img']))
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('asset-1')])

    const { handleExtract, assets } = useAssets(9, ref(''), ref([]))
    await handleExtract(1)

    expect(mockedMlApi.listAssets).toHaveBeenCalled()
    expect(assets.value).toEqual([makeAsset('asset-1')])
  })
})

describe('useAssets: handlePaste', () => {
  it('does nothing if no asset is selected', async () => {
    const { handlePaste } = useAssets(9, ref(''), ref([]))
    await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 100, y2: 100 } })
    expect(mockedMlApi.pasteExtractedObject).not.toHaveBeenCalled()
  })

  it('calls pasteExtractedObject with selectedAssetId and params', async () => {
    mockedMlApi.pasteExtractedObject.mockResolvedValue(makePasteResult('https://cdn.example.com/pasted.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { selectedAssetId, handlePaste } = useAssets(9, ref(''), ref([]))
    selectedAssetId.value = 'asset-1'

    await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 100, y2: 100 }, scale: 1.5 })

    expect(mockedMlApi.pasteExtractedObject).toHaveBeenCalledWith(9, {
      assetId: 'asset-1',
      targetBbox: { x1: 0, y1: 0, x2: 100, y2: 100 },
      scale: 1.5,
    })
  })

  it('updates currentImageUrl from result', async () => {
    mockedMlApi.pasteExtractedObject.mockResolvedValue(makePasteResult('https://cdn.example.com/pasted.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const currentImageUrl = ref('')
    const { selectedAssetId, handlePaste } = useAssets(9, currentImageUrl, ref([]))
    selectedAssetId.value = 'asset-1'

    await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })

    expect(currentImageUrl.value).toBe('https://cdn.example.com/pasted.jpg')
  })

  it('updates history after paste', async () => {
    mockedMlApi.pasteExtractedObject.mockResolvedValue(makePasteResult('https://cdn.example.com/pasted.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: ['step1'] })

    const history = ref<string[]>([])
    const { selectedAssetId, handlePaste } = useAssets(9, ref(''), history)
    selectedAssetId.value = 'asset-1'

    await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })

    expect(mockedMlApi.getHistory).toHaveBeenCalledWith(9)
    expect(history.value).toEqual(['step1'])
  })

  it('sets pasting to true during and false after success', async () => {
    let resolve: (v: any) => void
    mockedMlApi.pasteExtractedObject.mockReturnValue(new Promise(r => { resolve = r }))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { selectedAssetId, handlePaste, pasting } = useAssets(9, ref(''), ref([]))
    selectedAssetId.value = 'asset-1'

    const promise = handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })
    expect(pasting.value).toBe(true)

    resolve!(makePasteResult('https://cdn.example.com/pasted.jpg'))
    await promise
    expect(pasting.value).toBe(false)
  })

  it('returns the paste result', async () => {
    const result = makePasteResult('https://cdn.example.com/pasted.jpg')
    mockedMlApi.pasteExtractedObject.mockResolvedValue(result)
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { selectedAssetId, handlePaste } = useAssets(9, ref(''), ref([]))
    selectedAssetId.value = 'asset-1'

    const ret = await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })
    expect(ret).toEqual(result)
  })

  it('sets mlError when pasteExtractedObject fails', async () => {
    mockedMlApi.pasteExtractedObject.mockRejectedValue({
      response: { data: { detail: 'Paste failed on server' } }
    })

    const { selectedAssetId, handlePaste, mlError } = useAssets(9, ref(''), ref([]))
    selectedAssetId.value = 'asset-1'

    await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })

    expect(mlError.value).toBe('Paste failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.pasteExtractedObject.mockRejectedValue(new Error('fail'))

    const { selectedAssetId, handlePaste, mlError } = useAssets(9, ref(''), ref([]))
    selectedAssetId.value = 'asset-1'

    await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })

    expect(mlError.value).toBe('Paste failed')
  })

  it('sets pasting to false after failure', async () => {
    mockedMlApi.pasteExtractedObject.mockRejectedValue(new Error('fail'))

    const { selectedAssetId, handlePaste, pasting } = useAssets(9, ref(''), ref([]))
    selectedAssetId.value = 'asset-1'

    await handlePaste({ targetBbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })

    expect(pasting.value).toBe(false)
  })
})


describe('useAssets: selectFromLibrary', () => {
  it('selects an asset from the library', () => {
    const { selectFromLibrary, selectedAssetId } = useAssets(1, ref(''), ref([]))

    selectFromLibrary(makeAsset('asset-1'))

    expect(selectedAssetId.value).toBe('asset-1')
  })

  it('deselects when the same asset is selected again', () => {
    const { selectFromLibrary, selectedAssetId } = useAssets(1, ref(''), ref([]))

    selectFromLibrary(makeAsset('asset-1'))
    selectFromLibrary(makeAsset('asset-1'))

    expect(selectedAssetId.value).toBeNull()
  })

  it('switches selection to a different asset', () => {
    const { selectFromLibrary, selectedAssetId } = useAssets(1, ref(''), ref([]))

    selectFromLibrary(makeAsset('asset-1'))
    selectFromLibrary(makeAsset('asset-2'))

    expect(selectedAssetId.value).toBe('asset-2')
  })

  it('clears extractedPreviewUrl when selecting from library', () => {
    const { selectFromLibrary, extractedPreviewUrl } = useAssets(1, ref(''), ref([]))
    extractedPreviewUrl.value = 'blob:something'

    selectFromLibrary(makeAsset('asset-1'))

    expect(extractedPreviewUrl.value).toBeNull()
  })
})


describe('useAssets: clearExtracted', () => {
  it('clears selectedAssetId and extractedPreviewUrl', () => {
    const { selectedAssetId, extractedPreviewUrl, clearExtracted } = useAssets(1, ref(''), ref([]))
    selectedAssetId.value = 'asset-1'
    extractedPreviewUrl.value = 'blob:something'

    clearExtracted()

    expect(selectedAssetId.value).toBeNull()
    expect(extractedPreviewUrl.value).toBeNull()
  })
})


describe('useAssets: renameAsset', () => {
  it('calls renameAsset api with assetId and label', async () => {
    mockedMlApi.renameAsset.mockResolvedValue(makeAsset('asset-1', 'New Name'))

    const { renameAsset } = useAssets(1, ref(''), ref([]))
    await renameAsset('asset-1', 'New Name')

    expect(mockedMlApi.renameAsset).toHaveBeenCalledWith('asset-1', 'New Name')
  })

  it('updates the asset in the assets list', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('asset-1', 'Old Name')])
    mockedMlApi.renameAsset.mockResolvedValue(makeAsset('asset-1', 'New Name'))

    const { fetchAssets, renameAsset, assets } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    await renameAsset('asset-1', 'New Name')

    expect(assets.value[0].label).toBe('New Name')
  })

  it('sets assetsError when renameAsset fails', async () => {
    mockedMlApi.renameAsset.mockRejectedValue({
      response: { data: { detail: 'Rename failed on server' } }
    })

    const { renameAsset, assetsError } = useAssets(1, ref(''), ref([]))
    await renameAsset('asset-1', 'New Name')

    expect(assetsError.value).toBe('Rename failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.renameAsset.mockRejectedValue(new Error('fail'))

    const { renameAsset, assetsError } = useAssets(1, ref(''), ref([]))
    await renameAsset('asset-1', 'New Name')

    expect(assetsError.value).toBe('Rename failed')
  })
})


describe('useAssets: deleteAsset', () => {
  it('calls deleteAsset api with assetId', async () => {
    mockedMlApi.deleteAsset.mockResolvedValue(undefined)
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('asset-1')])

    const { fetchAssets, deleteAsset } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    await deleteAsset('asset-1')

    expect(mockedMlApi.deleteAsset).toHaveBeenCalledWith('asset-1')
  })

  it('removes the asset from the assets list', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('asset-1'), makeAsset('asset-2')])
    mockedMlApi.deleteAsset.mockResolvedValue(undefined)

    const { fetchAssets, deleteAsset, assets } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    await deleteAsset('asset-1')

    expect(assets.value).toHaveLength(1)
    expect(assets.value[0].public_id).toBe('asset-2')
  })

  it('clears selectedAssetId when the selected asset is deleted', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('asset-1')])
    mockedMlApi.deleteAsset.mockResolvedValue(undefined)

    const { fetchAssets, deleteAsset, selectedAssetId } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    selectedAssetId.value = 'asset-1'

    await deleteAsset('asset-1')

    expect(selectedAssetId.value).toBeNull()
  })

  it('clears extractedPreviewUrl when the selected asset is deleted', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('asset-1')])
    mockedMlApi.deleteAsset.mockResolvedValue(undefined)

    const { fetchAssets, deleteAsset, selectedAssetId, extractedPreviewUrl } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    selectedAssetId.value = 'asset-1'
    extractedPreviewUrl.value = 'blob:something'

    await deleteAsset('asset-1')

    expect(extractedPreviewUrl.value).toBeNull()
  })

  it('does not clear selectedAssetId when a different asset is deleted', async () => {
    mockedMlApi.listAssets.mockResolvedValue([makeAsset('asset-1'), makeAsset('asset-2')])
    mockedMlApi.deleteAsset.mockResolvedValue(undefined)

    const { fetchAssets, deleteAsset, selectedAssetId } = useAssets(1, ref(''), ref([]))
    await fetchAssets()
    selectedAssetId.value = 'asset-1'

    await deleteAsset('asset-2')

    expect(selectedAssetId.value).toBe('asset-1')
  })

  it('sets deletingId during the call and clears it after', async () => {
    let resolve: (v: any) => void
    mockedMlApi.deleteAsset.mockReturnValue(new Promise(r => { resolve = r }))

    const { deleteAsset, deletingId } = useAssets(1, ref(''), ref([]))
    const promise = deleteAsset('asset-1')
    expect(deletingId.value).toBe('asset-1')

    resolve!(undefined)
    await promise
    expect(deletingId.value).toBeNull()
  })

  it('sets assetsError when deleteAsset fails', async () => {
    mockedMlApi.deleteAsset.mockRejectedValue({
      response: { data: { detail: 'Delete failed on server' } }
    })

    const { deleteAsset, assetsError } = useAssets(1, ref(''), ref([]))
    await deleteAsset('asset-1')

    expect(assetsError.value).toBe('Delete failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.deleteAsset.mockRejectedValue(new Error('fail'))

    const { deleteAsset, assetsError } = useAssets(1, ref(''), ref([]))
    await deleteAsset('asset-1')

    expect(assetsError.value).toBe('Delete failed')
  })

  it('clears deletingId after failure', async () => {
    mockedMlApi.deleteAsset.mockRejectedValue(new Error('fail'))

    const { deleteAsset, deletingId } = useAssets(1, ref(''), ref([]))
    await deleteAsset('asset-1')

    expect(deletingId.value).toBeNull()
  })
})