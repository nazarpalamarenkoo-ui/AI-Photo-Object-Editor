import { ref, computed } from 'vue'
import { mlApi, PRESETS } from '@/api/ml'
import type { SegmentInfo, LdmConfig, Bbox, RegionItem, PolygonPoint, SegmentHybridParams, DiffusionReplaceOptions } from '@/types/Index'

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
  
  const promptBbox = ref<Bbox | null>(null)
  const polygonPoints = ref<PolygonPoint[]>([])
  const polygonShapes = ref<Record<number, PolygonPoint[]>>({})
  const canRunPolygon = computed(() => polygonPoints.value.length >= 3)
  const canRunBbox = computed(() => promptBbox.value !== null)

  function addPolygonPoint(x: number, y: number) {
    polygonPoints.value.push({ x: Math.round(x), y: Math.round(y) })
  }

  function removeLastPolygonPoint() {
    polygonPoints.value.pop()
  }

  function clearPolygon() {
    polygonPoints.value = []
  }

  function clearBbox() {
    promptBbox.value = null
  }

  function setPromptBbox(bbox: Bbox) {
    promptBbox.value = {
      x1: Math.round(bbox.x1),
      y1: Math.round(bbox.y1),
      x2: Math.round(bbox.x2),
      y2: Math.round(bbox.y2),
    }
  }

  const regions = computed<RegionItem[]>(() =>
  segments.value.map(s => ({
    id: s.mask_id,
    bbox: s.bbox,
    label: `Object #${s.mask_id}`,
    mask_url: s.mask_url,
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

  async function handleSegmentHybrid(params?: SegmentHybridParams) {
    segmenting.value = true
    mlError.value = ''
    selectedMaskId.value = null
    try {
      const result = await mlApi.segmentHybrid(imageId, params)
      segments.value = result.segments
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Hybrid segmentation failed'
    } finally {
      segmenting.value = false
    }
  }

  async function handleSegmentWithPrompt(params?: { bbox?: Bbox }) {
    const bbox = params?.bbox ?? promptBbox.value ?? undefined

    if (!bbox) {
      mlError.value = 'Draw a bbox first'
      return
    }

    segmenting.value = true
    mlError.value = ''
    try {
      const result = await mlApi.segmentWithPrompt(imageId, { bbox, multimask_output: false })
      segments.value = [...segments.value, ...result.segments]
      clearBbox()
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Prompted segmentation failed'
    } finally {
      segmenting.value = false
    }
  }
  async function handleSegmentByPolygon(params?: {
    smooth?: boolean
    smoothingFactor?: number
    featherPx?: number
  }) {
    if (polygonPoints.value.length < 3) {
      mlError.value = 'Add at least 3 points to form a polygon'
      return
    }

    segmenting.value = true
    mlError.value = ''
    try {
      const result = await mlApi.segmentByPolygon(imageId, {
        points: polygonPoints.value.map(p => [p.x, p.y] as [number, number]),
        smooth: params?.smooth,
        smoothingFactor: params?.smoothingFactor,
        featherPx: params?.featherPx,
      })
      for (const seg of result.segments) {
        polygonShapes.value[seg.mask_id] = [...polygonPoints.value]
      }
      segments.value = [...segments.value, ...result.segments]
      clearPolygon()
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Polygon segmentation failed'
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

  async function handleSamReplaceWithAsset(assetId: string, ldm: LdmConfig = PRESETS.quality) {
    if (selectedMaskId.value === null || !assetId) return
    mlError.value = ''
    try {
      const result = await mlApi.samReplaceObjectWithAsset(imageId, selectedMaskId.value, assetId, {
        useEdgeBlending: useEdgeBlending.value,
        ldm,
      })
      currentImageUrl.value = result.presigned_url
      segments.value = segments.value.filter(s => s.mask_id !== selectedMaskId.value)
      selectedMaskId.value = null
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'SAM replace with asset failed'
    }
  }
  async function handleSamReplaceDiffusion(options: DiffusionReplaceOptions = {}) {
    if (selectedMaskId.value === null || !replacementFile.value) return
    const seg = segments.value.find(s => s.mask_id === selectedMaskId.value)
    if (!seg?.mask_url) {
      mlError.value = 'Mask preview unavailable, cannot run diffusion replace'
      return
    }
    mlError.value = ''
    try {
      const result = await mlApi.samReplaceObjectDiffusion(imageId, seg.mask_url, seg.bbox, {
        referenceFile: replacementFile.value,
        ...options,
      })
      currentImageUrl.value = result.presigned_url
      segments.value = segments.value.filter(s => s.mask_id !== selectedMaskId.value)
      selectedMaskId.value = null
      replacementFile.value = null
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Diffusion replace failed'
    }
  }

  async function handleSamReplaceWithAssetDiffusion(assetId: string, options: DiffusionReplaceOptions = {}) {
    if (selectedMaskId.value === null || !assetId) return
    const seg = segments.value.find(s => s.mask_id === selectedMaskId.value)
    if (!seg?.mask_url) {
      mlError.value = 'Mask preview unavailable, cannot run diffusion replace'
      return
    }
    mlError.value = ''
    try {
      const result = await mlApi.samReplaceObjectDiffusion(imageId, seg.mask_url, seg.bbox, {
        assetId,
        ...options,
      })
      currentImageUrl.value = result.presigned_url
      segments.value = segments.value.filter(s => s.mask_id !== selectedMaskId.value)
      selectedMaskId.value = null
      const h = await mlApi.getHistory(imageId)
      history.value = h.history
    } catch (e: any) {
      mlError.value = e.response?.data?.detail ?? 'Diffusion replace with asset failed'
    }
  }

  function onReplacementSelect(event: Event) {
    const input = event.target as HTMLInputElement
    replacementFile.value = input.files?.[0] ?? null
  }

  function clearSegments() {
    segments.value = []
    selectedMaskId.value = null
    clearBbox()
    clearPolygon()
  }

  return {
    segments, regions, selectedMaskId, segmenting, mlError,
    useEdgeBlending, replacementFile,
    handleSegment, handleSegmentHybrid, handleSegmentWithPrompt, handleSegmentByPolygon,
    toggleMaskSelection, handleSamRemove, handleSamReplace, handleSamReplaceWithAsset,
    handleSamReplaceDiffusion, handleSamReplaceWithAssetDiffusion,
    onReplacementSelect, clearSegments, polygonShapes,
    promptBbox, canRunBbox, setPromptBbox, clearBbox,
    polygonPoints, canRunPolygon, addPolygonPoint, removeLastPolygonPoint, clearPolygon,
  }
}