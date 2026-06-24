import { ref } from 'vue'
import { detectionsApi } from '@/api/detections'
import { mlApi } from '@/api/ml'
import type { Detection } from '@/types/Index'

export function useDetections(
  imageId: number,
  detections: ReturnType<typeof ref<Detection[]>>,
) {
  const selectedBboxIds = ref<number[]>([])
  const detecting = ref(false)
  const confThreshold = ref(0.5)
  const mlError = ref('')

  async function handleDetect() {
    detecting.value = true
    mlError.value = ''
    selectedBboxIds.value = []
    try {
      const result = await mlApi.detectObjects(imageId, { conf_threshold: confThreshold.value })
      detections.value = result.detections
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Detection failed'
    } finally {
      detecting.value = false
    }
  }

  async function handleClearDetections(onClear?: () => void) {
    await detectionsApi.deleteByImage(imageId)
    detections.value = []
    selectedBboxIds.value = []
    onClear?.()
  }

  function toggleSelection(bboxId: number) {
    const idx = selectedBboxIds.value.indexOf(bboxId)
    if (idx === -1) selectedBboxIds.value.push(bboxId)
    else selectedBboxIds.value.splice(idx, 1)
  }

  return {
    selectedBboxIds,
    detecting,
    confThreshold,
    mlError,
    handleDetect,
    handleClearDetections,
    toggleSelection,
  }
}