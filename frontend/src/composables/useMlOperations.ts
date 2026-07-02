import { ref, computed } from 'vue'
import { mlApi, PRESETS } from '@/api/ml'
import type { Detection, LdmConfig, Image } from '@/types/Index'

export function useMlOperations(
  imageId: number,
  detections: ReturnType<typeof ref<Detection[]>>,
  selectedBboxIds: ReturnType<typeof ref<number[]>>
) {
  const mlLoading = ref(false)
  const mlError = ref('')
  const currentImageUrl = ref('')
  const replacementFile = ref<File | null>(null)
  const useEdgeBlending = ref(true)
  const saveLoading = ref(false)
  const saveError = ref('')
  const savedSuccess = ref(false)
  const history = ref<string[]>([])
  const canUndo = computed(() => history.value.length > 0)

  async function handleSave(id: number, onSaved?: (img: Image) => void) {
    saveLoading.value = true
    saveError.value = ''
    savedSuccess.value = false
    try {
      const saved = await mlApi.saveResult(id)
      savedSuccess.value = true
      onSaved?.(saved)
    } catch (e: any) {
      saveError.value = e?.response?.data?.detail ?? 'Failed to save'
    } finally {
      saveLoading.value = false
    }
  }

  async function handleRemove(bboxIds: number[], ldm: LdmConfig = PRESETS.quality) {
    mlLoading.value = true
    mlError.value = ''
    try {
      let result
      if (bboxIds.length === 1) {
        result = await mlApi.removeObject(imageId, bboxIds[0], 5, useEdgeBlending.value, ldm)
      } else {
        result = await mlApi.removeMultipleObjects(imageId, bboxIds, 5, useEdgeBlending.value, ldm)
      }
      currentImageUrl.value = result.presigned_url
      detections.value = detections.value!.filter(d => !bboxIds.includes(d.bbox_id))
      selectedBboxIds.value = []
      savedSuccess.value = false
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Remove failed'
    } finally {
      mlLoading.value = false
    }
  }

  async function handleReplace(bboxIds: number[], ldm: LdmConfig = PRESETS.quality) {
    if (!replacementFile.value || bboxIds.length !== 1) return
    mlLoading.value = true
    mlError.value = ''
    try {
      const result = await mlApi.replaceObject(
        imageId,
        bboxIds[0],
        replacementFile.value,
        {
          useEdgeBlending: useEdgeBlending.value,
          ldm,
        }
      )
      currentImageUrl.value = result.presigned_url
      detections.value = detections.value!.filter(d => d.bbox_id !== bboxIds[0])
      selectedBboxIds.value = []
      replacementFile.value = null
      savedSuccess.value = false
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Replace failed'
    } finally {
      mlLoading.value = false
    }
  }

  async function handleUndo() {
    mlLoading.value = true
    mlError.value = ''
    try {
      const result = await mlApi.undo(imageId)
      currentImageUrl.value = result.presigned_url
      history.value = result.history
      savedSuccess.value = false
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Undo failed'
    } finally {
      mlLoading.value = false
    }
  }

  async function handleRedo() {
    mlLoading.value = true
    mlError.value = ''
    try {
      const result = await mlApi.redo(imageId)
      currentImageUrl.value = result.presigned_url
      history.value = result.history
      savedSuccess.value = false
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Redo failed'
    } finally {
      mlLoading.value = false
    }
  }
  
  async function handleReset(originalUrl: string) {
    mlLoading.value = true
    mlError.value = ''
    try {
      await mlApi.resetState(imageId)
      currentImageUrl.value = originalUrl
      detections.value = []
      selectedBboxIds.value = []
      history.value = []
      savedSuccess.value = false
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Reset failed'
    } finally {
      mlLoading.value = false
    }
  }

  async function fetchHistory() {
    try {
      const result = await mlApi.getHistory(imageId)
      history.value = result.history
    } catch {}
  }

  function onReplacementSelect(event: Event) {
    const input = event.target as HTMLInputElement
    replacementFile.value = input.files?.[0] ?? null
  }

  return {
    mlLoading,
    mlError,
    currentImageUrl,
    replacementFile,
    useEdgeBlending,
    handleRemove,
    handleReplace,
    onReplacementSelect,
    saveLoading,
    saveError,
    savedSuccess,
    handleSave,
    history,
    canUndo,
    handleUndo,
    handleRedo,
    handleReset,
    fetchHistory,
  }
}