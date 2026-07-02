import { ref, computed } from 'vue'
import { mlApi, PRESETS } from '@/api/ml'
import type { SegmentInfo, LdmConfig, Bbox, RegionItem } from '@/types/Index'

export function useSegmentation(
  imageId: number,
  currentImageUrl: ReturnType<typeof ref<string>>,
  history: ReturnType<typeof ref<string[]>>,
) {
  const segments = ref<SegmentInfo[]>([])
  const selectedMaskId = ref<number | null>(null)
  const segmenting = ref(false)
  const mlError = ref('')
  const useEdgeBlending = ref(true)
  const replacementFile = ref<File | null>(null)

  const regions = computed<RegionItem[]>(() =>
    segments.value.map(s => ({
      id: s.mask_id,
      bbox: s.bbox,
      label: `Object #${s.mask_id}`,
    }))
  )

  async function handleSegment(minArea = 500, maxSegments = 50) {
    segmenting.value = true
    mlError.value = ''
    selectedMaskId.value = null
    try {
      const result = await mlApi.segmentObjects(imageId, minArea, maxSegments)
      segments.value = result.segments
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Segmentation failed'
    } finally {
      segmenting.value = false
    }
  }

  async function handleSegmentWithPrompt(params: {
    pointCoords?: [number, number][]
    pointLabels?: number[]
    bbox?: Bbox
  }) {
    segmenting.value = true
    mlError.value = ''
    try {
      const result = await mlApi.segmentWithPrompt(imageId, params)
      // мержимо, бо prompt-сегментація зазвичай додає один результат
      segments.value = [...segments.value, ...result.segments]
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Prompted segmentation failed'
    } finally {
      segmenting.value = false
    }
  }

  function toggleMaskSelection(maskId: number) {
    selectedMaskId.value = selectedMaskId.value === maskId ? null : maskId
  }

  async function handleSamRemove(ldm: LdmConfig = PRESETS.quality) {
    if (selectedMaskId.value === null) return
    mlError.value = ''
    try {
      const result = await mlApi.samRemoveObject(imageId, selectedMaskId.value, 12, useEdgeBlending.value, ldm)
      currentImageUrl.value = result.presigned_url
      segments.value = segments.value.filter(s => s.mask_id !== selectedMaskId.value)
      selectedMaskId.value = null
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'SAM remove failed'
    }
  }

  async function handleSamReplace(ldm: LdmConfig = PRESETS.quality) {
    if (selectedMaskId.value === null || !replacementFile.value) return
    mlError.value = ''
    try {
      const result = await mlApi.samReplaceObject(imageId, selectedMaskId.value, replacementFile.value, {
        useEdgeBlending: useEdgeBlending.value,
        ldm,
      })
      currentImageUrl.value = result.presigned_url
      segments.value = segments.value.filter(s => s.mask_id !== selectedMaskId.value)
      selectedMaskId.value = null
      replacementFile.value = null
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'SAM replace failed'
    }
  }

  function onReplacementSelect(event: Event) {
    const input = event.target as HTMLInputElement
    replacementFile.value = input.files?.[0] ?? null
  }

  function clearSegments() {
    segments.value = []
    selectedMaskId.value = null
  }

  return {
    segments, regions, selectedMaskId, segmenting, mlError,
    useEdgeBlending, replacementFile,
    handleSegment, handleSegmentWithPrompt,
    toggleMaskSelection, handleSamRemove, handleSamReplace,
    onReplacementSelect, clearSegments,
  }
}