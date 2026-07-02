import { ref } from 'vue'
import { mlApi } from '@/api/ml'
import type { Bbox, ColorMatchMethod } from '@/types/Index'

export function useAssets(
  imageId: number,
  currentImageUrl: ReturnType<typeof ref<string>>,
  history: ReturnType<typeof ref<string[]>>,
) {
  const extracting = ref(false)
  const pasting = ref(false)
  const mlError = ref('')
  const extractedUrl = ref<string | null>(null)
  const extractedPreviewUrl = ref<string | null>(null)

  async function handleExtract(maskId: number, paddingPixels = 8) {
    extracting.value = true
    mlError.value = ''
    try {
      const result = await mlApi.extractObject(imageId, maskId, paddingPixels)
      extractedUrl.value = result.extracted_url
      extractedPreviewUrl.value = result.presigned_url
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
    if (!extractedUrl.value) return
    pasting.value = true
    mlError.value = ''
    try {
      const result = await mlApi.pasteExtractedObject(imageId, {
        extractedUrl: extractedUrl.value,
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

  function clearExtracted() {
    extractedUrl.value = null
    extractedPreviewUrl.value = null
  }

  return {
    extracting,
    pasting,
    mlError,
    extractedUrl,
    extractedPreviewUrl,
    handleExtract,
    handlePaste,
    clearExtracted,
  }
}