import { ref } from 'vue'
import { mlApi } from '@/api/ml'
import type { Asset, Bbox, ColorMatchMethod } from '@/types/Index'

const PAGE_SIZE = 30

export function useAssets(
  imageId: number,
  currentImageUrl: ReturnType<typeof ref<string>>,
  history: ReturnType<typeof ref<string[]>>,
) {
  const extracting = ref(false)
  const pasting = ref(false)
  const mlError = ref('')

  const selectedAssetId = ref<string | null>(null)
  const extractedPreviewUrl = ref<string | null>(null)

  const assets = ref<Asset[]>([])
  const assetsLoading = ref(false)
  const assetsError = ref('')
  const assetsHasMore = ref(true)
  const thumbUrls = ref<Record<string, string>>({})
  const deletingId = ref<string | null>(null)

  function revokeThumb(assetId: string) {
    const url = thumbUrls.value[assetId]
    if (url) {
      URL.revokeObjectURL(url)
      delete thumbUrls.value[assetId]
    }
  }

  async function loadThumb(assetId: string) {
    if (thumbUrls.value[assetId]) return
    try {
      const blob = await mlApi.getAssetThumbnailBlob(assetId)
      thumbUrls.value[assetId] = URL.createObjectURL(blob)
    } catch {
    }
  }

  async function fetchAssets(reset = true) {
    if (assetsLoading.value) return
    assetsLoading.value = true
    assetsError.value = ''
    try {
      const offset = reset ? 0 : assets.value.length
      const page = await mlApi.listAssets(PAGE_SIZE, offset)

      assets.value = reset ? page : [...assets.value, ...page]
      assetsHasMore.value = page.length === PAGE_SIZE

      await Promise.all(page.map(a => loadThumb(a.asset_id)))
    } catch (e: any) {
      assetsError.value = e.response?.data?.detail ?? 'Failed to load asset library'
    } finally {
      assetsLoading.value = false
    }
  }


  async function handleExtract(
    maskId: number,
    params: { paddingPixels?: number; label?: string; persistToS3?: boolean } = {}
  ) {
    extracting.value = true
    mlError.value = ''
    try {
      const result = await mlApi.extractObject(imageId, maskId, params)

      selectedAssetId.value = result.asset_id
      extractedPreviewUrl.value = result.presigned_url

      await fetchAssets(true)
      await loadThumb(result.asset_id)

      return result
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Extract failed'
    } finally {
      extracting.value = false
    }
  }


  async function handlePaste(params: {
    targetBbox: Bbox
    scale?: number
    useColorMatching?: boolean
    useEdgeBlending?: boolean
    colorMatchMethod?: ColorMatchMethod
  }) {
    if (!selectedAssetId.value) return
    pasting.value = true
    mlError.value = ''
    try {
      const result = await mlApi.pasteExtractedObject(imageId, {
        assetId: selectedAssetId.value,
        ...params,
      })
      currentImageUrl.value = result.presigned_url
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
      return result
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Paste failed'
    } finally {
      pasting.value = false
    }
  }


  function selectFromLibrary(asset: Asset) {
    selectedAssetId.value = selectedAssetId.value === asset.asset_id ? null : asset.asset_id
    extractedPreviewUrl.value = null
  }

  async function renameAsset(assetId: string, label: string) {
    mlError.value = ''
    try {
      const updated = await mlApi.renameAsset(assetId, label)
      const idx = assets.value.findIndex(a => a.asset_id === assetId)
      if (idx !== -1) assets.value[idx] = updated
    } catch (e: any) {
      assetsError.value = e.response?.data?.detail ?? 'Rename failed'
    }
  }

  async function deleteAsset(assetId: string) {
    deletingId.value = assetId
    assetsError.value = ''
    try {
      await mlApi.deleteAsset(assetId)
      assets.value = assets.value.filter(a => a.asset_id !== assetId)
      revokeThumb(assetId)
      if (selectedAssetId.value === assetId) {
        selectedAssetId.value = null
        extractedPreviewUrl.value = null
      }
    } catch (e: any) {
      assetsError.value = e.response?.data?.detail ?? 'Delete failed'
    } finally {
      deletingId.value = null
    }
  }

  function clearExtracted() {
    selectedAssetId.value = null
    extractedPreviewUrl.value = null
  }

  return {
    extracting,
    pasting,
    mlError,
    selectedAssetId,
    extractedPreviewUrl,
    handleExtract,
    handlePaste,
    clearExtracted,
    selectFromLibrary,

    assets,
    assetsLoading,
    assetsError,
    assetsHasMore,
    thumbUrls,
    deletingId,
    fetchAssets,
    renameAsset,
    deleteAsset,
  }
}