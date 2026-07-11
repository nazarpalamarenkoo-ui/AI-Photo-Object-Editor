import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'
import type { MLResultResponse, SegmentInfo } from '@/types/Index'

vi.mock('@/api/ml', () => ({
  PRESETS: {
    quality: { ldm_steps: 20, ldm_sampler: 'plms', hd_strategy: 'RESIZE' }
  },
  mlApi: {
    segmentObjects: vi.fn(),
    segmentHybrid: vi.fn(),          
    segmentWithPrompt: vi.fn(),
    segmentByPolygon: vi.fn(),
    samRemoveObject: vi.fn(),
    samReplaceObject: vi.fn(),
    samReplaceObjectWithAsset: vi.fn(),
    getHistory: vi.fn()
  }
}))

import { mlApi, PRESETS } from '@/api/ml'
import { useSegmentation } from '@/composables/useSegmentation'

const mockedMlApi = vi.mocked(mlApi, true)

const makeSegment = (maskId: number, bboxId = maskId): SegmentInfo => ({
  mask_id: maskId,
  bbox_id: bboxId,
  bbox: { x1: 0, y1: 0, x2: 50, y2: 50 },
  area: 2500,
  stability_score: 0.9
})

const makeSegmentResult = (segments: SegmentInfo[]) => ({
  segments,
  metrics: {},
  image_size: [800, 600] as [number, number],
  timestamp: '2026-01-01T00:00:00Z'
})

const makeMLResult = (presigned_url: string): MLResultResponse => ({
  result_url: 'https://cdn.example.com/result.jpg',
  presigned_url,
  metrics: {},
  timestamp: '2026-01-01T00:00:00Z',
})

beforeEach(() => {
  vi.clearAllMocks()
})

describe('useSegmentation: initial state', () => {
  it('starts with empty segments, no selection, not segmenting, no error', () => {
    const { segments, selectedMaskId, segmenting, mlError } =
      useSegmentation(1, ref(''), ref([]))

    expect(segments.value).toEqual([])
    expect(selectedMaskId.value).toBeNull()
    expect(segmenting.value).toBe(false)
    expect(mlError.value).toBe('')
  })

  it('useEdgeBlending defaults to true and replacementFile defaults to null', () => {
    const { useEdgeBlending, replacementFile } = useSegmentation(1, ref(''), ref([]))

    expect(useEdgeBlending.value).toBe(true)
    expect(replacementFile.value).toBeNull()
  })

  it('regions is empty when segments is empty', () => {
    const { regions } = useSegmentation(1, ref(''), ref([]))
    expect(regions.value).toEqual([])
  })

  it('prompt and polygon state start empty', () => {
    const { promptMode, promptLabel, promptPoints, promptBbox, polygonPoints, polygonShapes, canRunPrompt, canRunPolygon } =
      useSegmentation(1, ref(''), ref([]))

    expect(promptMode.value).toBeNull()
    expect(promptLabel.value).toBe(1)
    expect(promptPoints.value).toEqual([])
    expect(promptBbox.value).toBeNull()
    expect(polygonPoints.value).toEqual([])
    expect(polygonShapes.value).toEqual({})
    expect(canRunPrompt.value).toBe(false)
    expect(canRunPolygon.value).toBe(false)
  })
})

describe('useSegmentation: regions', () => {
  it('maps segments to region items with generated labels', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1), makeSegment(2)]))

    const { handleSegment, regions } = useSegmentation(1, ref(''), ref([]))
    await handleSegment()

    expect(regions.value).toEqual([
      { id: 1, bbox: { x1: 0, y1: 0, x2: 50, y2: 50 }, label: 'Object #1' },
      { id: 2, bbox: { x1: 0, y1: 0, x2: 50, y2: 50 }, label: 'Object #2' }
    ])
  })

  it('updates reactively when segments change', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(5)]))

    const { handleSegment, regions } = useSegmentation(1, ref(''), ref([]))
    await handleSegment()

    expect(regions.value).toHaveLength(1)
    expect(regions.value[0].label).toBe('Object #5')
  })
})

describe('useSegmentation: handleSegment', () => {
  it('calls segmentObjects with provided minArea and maxSegments', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([]))

    const { handleSegment } = useSegmentation(9, ref(''), ref([]))
    await handleSegment(300, 20)

    expect(mockedMlApi.segmentObjects).toHaveBeenCalledWith(9, 300, 20)
  })

  it('uses default minArea and maxSegments when not provided', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([]))

    const { handleSegment } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()

    expect(mockedMlApi.segmentObjects).toHaveBeenCalledWith(9, 500, 50)
  })

  it('sets segments from result', async () => {
    const seg = makeSegment(1)
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([seg]))

    const { handleSegment, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()

    expect(segments.value).toEqual([seg])
  })

  it('resets selectedMaskId to null', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { handleSegment, toggleMaskSelection, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)
    expect(selectedMaskId.value).toBe(3)

    await handleSegment()

    expect(selectedMaskId.value).toBeNull()
  })

  it('sets segmenting to true during the call and false after success', async () => {
    let resolvePromise: (value: any) => void
    mockedMlApi.segmentObjects.mockReturnValue(
      new Promise(resolve => { resolvePromise = resolve })
    )

    const { handleSegment, segmenting } = useSegmentation(9, ref(''), ref([]))
    const promise = handleSegment()

    expect(segmenting.value).toBe(true)

    resolvePromise!(makeSegmentResult([]))
    await promise

    expect(segmenting.value).toBe(false)
  })

  it('sets mlError when segmentObjects fails', async () => {
    mockedMlApi.segmentObjects.mockRejectedValue({
      response: { data: { detail: 'Segmentation failed on server' } }
    })

    const { handleSegment, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()

    expect(mlError.value).toBe('Segmentation failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.segmentObjects.mockRejectedValue(new Error('fail'))

    const { handleSegment, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()

    expect(mlError.value).toBe('Segmentation failed')
  })

  it('sets segmenting to false after failure', async () => {
    mockedMlApi.segmentObjects.mockRejectedValue(new Error('fail'))

    const { handleSegment, segmenting } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()

    expect(segmenting.value).toBe(false)
  })

  it('clears previous mlError on new call', async () => {
    mockedMlApi.segmentObjects.mockRejectedValueOnce({
      response: { data: { detail: 'first error' } }
    })

    const { handleSegment, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    expect(mlError.value).toBe('first error')

    mockedMlApi.segmentObjects.mockResolvedValueOnce(makeSegmentResult([]))
    await handleSegment()

    expect(mlError.value).toBe('')
  })
})

describe('useSegmentation: prompt point / bbox management', () => {
  it('addPromptPoint pushes a rounded point with the current promptLabel', () => {
    const { addPromptPoint, promptPoints, promptLabel } = useSegmentation(1, ref(''), ref([]))

    promptLabel.value = 0
    addPromptPoint(10.6, 20.2)

    expect(promptPoints.value).toEqual([{ x: 11, y: 20, label: 0 }])
  })

  it('addPromptPoint accumulates multiple points', () => {
    const { addPromptPoint, promptPoints } = useSegmentation(1, ref(''), ref([]))

    addPromptPoint(1, 1)
    addPromptPoint(2, 2)

    expect(promptPoints.value).toHaveLength(2)
  })

  it('removeLastPromptPoint pops the last point', () => {
    const { addPromptPoint, removeLastPromptPoint, promptPoints } = useSegmentation(1, ref(''), ref([]))

    addPromptPoint(1, 1)
    addPromptPoint(2, 2)
    removeLastPromptPoint()

    expect(promptPoints.value).toEqual([{ x: 1, y: 1, label: 1 }])
  })

  it('setPromptBbox rounds and stores the bbox', () => {
    const { setPromptBbox, promptBbox } = useSegmentation(1, ref(''), ref([]))

    setPromptBbox({ x1: 1.4, y1: 2.6, x2: 10.5, y2: 20.5 })

    expect(promptBbox.value).toEqual({ x1: 1, y1: 3, x2: 11, y2: 21 })
  })

  it('clearPrompt resets promptPoints, promptBbox and polygonPoints', () => {
    const { addPromptPoint, setPromptBbox, addPolygonPoint, clearPrompt, promptPoints, promptBbox, polygonPoints } =
      useSegmentation(1, ref(''), ref([]))

    addPromptPoint(1, 1)
    setPromptBbox({ x1: 0, y1: 0, x2: 10, y2: 10 })
    addPolygonPoint(1, 1)

    clearPrompt()

    expect(promptPoints.value).toEqual([])
    expect(promptBbox.value).toBeNull()
    expect(polygonPoints.value).toEqual([])
  })

  it('setPromptMode sets the mode and clears prompt state', () => {
    const { addPromptPoint, setPromptMode, promptMode, promptPoints } = useSegmentation(1, ref(''), ref([]))

    addPromptPoint(1, 1)
    setPromptMode('points')

    expect(promptMode.value).toBe('points')
    expect(promptPoints.value).toEqual([])
  })

  it('canRunPrompt is true when there are prompt points', () => {
    const { addPromptPoint, canRunPrompt } = useSegmentation(1, ref(''), ref([]))

    addPromptPoint(1, 1)

    expect(canRunPrompt.value).toBe(true)
  })

  it('canRunPrompt is true when a bbox is set', () => {
    const { setPromptBbox, canRunPrompt } = useSegmentation(1, ref(''), ref([]))

    setPromptBbox({ x1: 0, y1: 0, x2: 10, y2: 10 })

    expect(canRunPrompt.value).toBe(true)
  })

  it('canRunPrompt is false with no points and no bbox', () => {
    const { canRunPrompt } = useSegmentation(1, ref(''), ref([]))

    expect(canRunPrompt.value).toBe(false)
  })
})

describe('useSegmentation: polygon point management', () => {
  it('addPolygonPoint pushes a rounded point', () => {
    const { addPolygonPoint, polygonPoints } = useSegmentation(1, ref(''), ref([]))

    addPolygonPoint(10.6, 20.4)

    expect(polygonPoints.value).toEqual([{ x: 11, y: 20 }])
  })

  it('removeLastPolygonPoint pops the last point', () => {
    const { addPolygonPoint, removeLastPolygonPoint, polygonPoints } = useSegmentation(1, ref(''), ref([]))

    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    removeLastPolygonPoint()

    expect(polygonPoints.value).toEqual([{ x: 1, y: 1 }])
  })

  it('clearPolygon only resets polygonPoints', () => {
    const { addPolygonPoint, addPromptPoint, clearPolygon, polygonPoints, promptPoints } =
      useSegmentation(1, ref(''), ref([]))

    addPolygonPoint(1, 1)
    addPromptPoint(2, 2)

    clearPolygon()

    expect(polygonPoints.value).toEqual([])
    expect(promptPoints.value).toHaveLength(1)
  })

  it('canRunPolygon is false with fewer than 3 points', () => {
    const { addPolygonPoint, canRunPolygon } = useSegmentation(1, ref(''), ref([]))

    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)

    expect(canRunPolygon.value).toBe(false)
  })

  it('canRunPolygon is true with 3 or more points', () => {
    const { addPolygonPoint, canRunPolygon } = useSegmentation(1, ref(''), ref([]))

    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    expect(canRunPolygon.value).toBe(true)
  })
})

describe('useSegmentation: handleSegmentWithPrompt', () => {
  it('calls segmentWithPrompt with given params and multimask_output false', async () => {
    mockedMlApi.segmentWithPrompt.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { handleSegmentWithPrompt } = useSegmentation(9, ref(''), ref([]))
    const params = { pointCoords: [[10, 20]] as [number, number][], pointLabels: [1] }

    await handleSegmentWithPrompt(params)

    expect(mockedMlApi.segmentWithPrompt).toHaveBeenCalledWith(9, {
      pointCoords: [[10, 20]],
      pointLabels: [1],
      bbox: undefined,
      multimask_output: false
    })
  })

  it('falls back to promptPoints/promptBbox state when no params are given', async () => {
    mockedMlApi.segmentWithPrompt.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { addPromptPoint, handleSegmentWithPrompt } = useSegmentation(9, ref(''), ref([]))
    addPromptPoint(5, 5)

    await handleSegmentWithPrompt()

    expect(mockedMlApi.segmentWithPrompt).toHaveBeenCalledWith(9, {
      pointCoords: [[5, 5]],
      pointLabels: [1],
      bbox: undefined,
      multimask_output: false
    })
  })

  it('falls back to promptBbox state when no params are given', async () => {
    mockedMlApi.segmentWithPrompt.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { setPromptBbox, handleSegmentWithPrompt } = useSegmentation(9, ref(''), ref([]))
    setPromptBbox({ x1: 0, y1: 0, x2: 10, y2: 10 })

    await handleSegmentWithPrompt()

    expect(mockedMlApi.segmentWithPrompt).toHaveBeenCalledWith(9, {
      pointCoords: undefined,
      pointLabels: undefined,
      bbox: { x1: 0, y1: 0, x2: 10, y2: 10 },
      multimask_output: false
    })
  })

  it('sets an error and does not call the api when neither points nor bbox are available', async () => {
    const { handleSegmentWithPrompt, mlError } = useSegmentation(9, ref(''), ref([]))

    await handleSegmentWithPrompt()

    expect(mlError.value).toBe('Add one point or bbox')
    expect(mockedMlApi.segmentWithPrompt).not.toHaveBeenCalled()
  })

  it('merges new segments with existing ones', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1)]))
    mockedMlApi.segmentWithPrompt.mockResolvedValue(makeSegmentResult([makeSegment(2)]))

    const { handleSegment, handleSegmentWithPrompt, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    await handleSegmentWithPrompt({ bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })

    expect(segments.value).toHaveLength(2)
    expect(segments.value.map(s => s.mask_id)).toEqual([1, 2])
  })

  it('does not reset selectedMaskId', async () => {
    mockedMlApi.segmentWithPrompt.mockResolvedValue(makeSegmentResult([makeSegment(2)]))

    const { handleSegmentWithPrompt, toggleMaskSelection, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(7)

    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(selectedMaskId.value).toBe(7)
  })

  it('clears prompt state and prompt mode on success', async () => {
    mockedMlApi.segmentWithPrompt.mockResolvedValue(makeSegmentResult([makeSegment(2)]))

    const { addPromptPoint, setPromptMode, handleSegmentWithPrompt, promptPoints, promptMode } =
      useSegmentation(9, ref(''), ref([]))
    setPromptMode('points')
    addPromptPoint(1, 1)

    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(promptPoints.value).toEqual([])
    expect(promptMode.value).toBeNull()
  })

  it('sets segmenting to true during and false after success', async () => {
    let resolvePromise: (value: any) => void
    mockedMlApi.segmentWithPrompt.mockReturnValue(
      new Promise(resolve => { resolvePromise = resolve })
    )

    const { handleSegmentWithPrompt, segmenting } = useSegmentation(9, ref(''), ref([]))
    const promise = handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(segmenting.value).toBe(true)

    resolvePromise!(makeSegmentResult([]))
    await promise

    expect(segmenting.value).toBe(false)
  })

  it('sets mlError when segmentWithPrompt fails', async () => {
    mockedMlApi.segmentWithPrompt.mockRejectedValue({
      response: { data: { detail: 'Prompted segmentation failed on server' } }
    })

    const { handleSegmentWithPrompt, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(mlError.value).toBe('Prompted segmentation failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.segmentWithPrompt.mockRejectedValue(new Error('fail'))

    const { handleSegmentWithPrompt, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(mlError.value).toBe('Prompted segmentation failed')
  })

  it('sets segmenting to false after failure', async () => {
    mockedMlApi.segmentWithPrompt.mockRejectedValue(new Error('fail'))

    const { handleSegmentWithPrompt, segmenting } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(segmenting.value).toBe(false)
  })

  it('does not clear prompt state on failure', async () => {
    mockedMlApi.segmentWithPrompt.mockRejectedValue(new Error('fail'))

    const { addPromptPoint, handleSegmentWithPrompt, promptPoints } = useSegmentation(9, ref(''), ref([]))
    addPromptPoint(1, 1)

    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(promptPoints.value).toHaveLength(1)
  })
})

describe('useSegmentation: handleSegmentByPolygon', () => {
  it('sets an error and does not call the api when fewer than 3 points exist', async () => {
    const { addPolygonPoint, handleSegmentByPolygon, mlError } = useSegmentation(9, ref(''), ref([]))
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)

    await handleSegmentByPolygon()

    expect(mlError.value).toBe('Add at least 3 points to form a polygon')
    expect(mockedMlApi.segmentByPolygon).not.toHaveBeenCalled()
  })

  it('calls segmentByPolygon with mapped points and given options', async () => {
    mockedMlApi.segmentByPolygon.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { addPolygonPoint, handleSegmentByPolygon } = useSegmentation(9, ref(''), ref([]))
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    await handleSegmentByPolygon({ smooth: false, smoothingFactor: 0.5, featherPx: 2 })

    expect(mockedMlApi.segmentByPolygon).toHaveBeenCalledWith(9, {
      points: [[1, 1], [2, 2], [3, 3]],
      smooth: false,
      smoothingFactor: 0.5,
      featherPx: 2
    })
  })

  it('appends new segments to existing ones', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1)]))
    mockedMlApi.segmentByPolygon.mockResolvedValue(makeSegmentResult([makeSegment(2)]))

    const { handleSegment, addPolygonPoint, handleSegmentByPolygon, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    await handleSegmentByPolygon()

    expect(segments.value).toHaveLength(2)
    expect(segments.value.map(s => s.mask_id)).toEqual([1, 2])
  })

  it('records the polygon shape used for each returned segment', async () => {
    mockedMlApi.segmentByPolygon.mockResolvedValue(makeSegmentResult([makeSegment(7)]))

    const { addPolygonPoint, handleSegmentByPolygon, polygonShapes } = useSegmentation(9, ref(''), ref([]))
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    await handleSegmentByPolygon()

    expect(polygonShapes.value[7]).toEqual([{ x: 1, y: 1 }, { x: 2, y: 2 }, { x: 3, y: 3 }])
  })

  it('clears polygon points and prompt mode on success', async () => {
    mockedMlApi.segmentByPolygon.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { addPolygonPoint, setPromptMode, handleSegmentByPolygon, polygonPoints, promptMode } =
      useSegmentation(9, ref(''), ref([]))
    setPromptMode('polygon')
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    await handleSegmentByPolygon()

    expect(polygonPoints.value).toEqual([])
    expect(promptMode.value).toBeNull()
  })

  it('sets segmenting to true during and false after success', async () => {
    let resolvePromise: (value: any) => void
    mockedMlApi.segmentByPolygon.mockReturnValue(
      new Promise(resolve => { resolvePromise = resolve })
    )

    const { addPolygonPoint, handleSegmentByPolygon, segmenting } = useSegmentation(9, ref(''), ref([]))
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    const promise = handleSegmentByPolygon()
    expect(segmenting.value).toBe(true)

    resolvePromise!(makeSegmentResult([]))
    await promise

    expect(segmenting.value).toBe(false)
  })

  it('sets mlError when segmentByPolygon fails', async () => {
    mockedMlApi.segmentByPolygon.mockRejectedValue({
      response: { data: { detail: 'Polygon segmentation failed on server' } }
    })

    const { addPolygonPoint, handleSegmentByPolygon, mlError } = useSegmentation(9, ref(''), ref([]))
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    await handleSegmentByPolygon()

    expect(mlError.value).toBe('Polygon segmentation failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.segmentByPolygon.mockRejectedValue(new Error('fail'))

    const { addPolygonPoint, handleSegmentByPolygon, mlError } = useSegmentation(9, ref(''), ref([]))
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    await handleSegmentByPolygon()

    expect(mlError.value).toBe('Polygon segmentation failed')
  })

  it('sets segmenting to false and keeps polygon points after failure', async () => {
    mockedMlApi.segmentByPolygon.mockRejectedValue(new Error('fail'))

    const { addPolygonPoint, handleSegmentByPolygon, segmenting, polygonPoints } =
      useSegmentation(9, ref(''), ref([]))
    addPolygonPoint(1, 1)
    addPolygonPoint(2, 2)
    addPolygonPoint(3, 3)

    await handleSegmentByPolygon()

    expect(segmenting.value).toBe(false)
    expect(polygonPoints.value).toHaveLength(3)
  })
})

describe('useSegmentation: toggleMaskSelection', () => {
  it('selects a mask id when none is selected', () => {
    const { toggleMaskSelection, selectedMaskId } = useSegmentation(1, ref(''), ref([]))

    toggleMaskSelection(4)

    expect(selectedMaskId.value).toBe(4)
  })

  it('deselects the mask id when the same id is toggled again', () => {
    const { toggleMaskSelection, selectedMaskId } = useSegmentation(1, ref(''), ref([]))

    toggleMaskSelection(4)
    toggleMaskSelection(4)

    expect(selectedMaskId.value).toBeNull()
  })

  it('switches selection when a different mask id is toggled', () => {
    const { toggleMaskSelection, selectedMaskId } = useSegmentation(1, ref(''), ref([]))

    toggleMaskSelection(4)
    toggleMaskSelection(9)

    expect(selectedMaskId.value).toBe(9)
  })
})

describe('useSegmentation: handleSamRemove', () => {
  it('does nothing if no mask is selected', async () => {
    const { handleSamRemove } = useSegmentation(9, ref(''), ref([]))

    await handleSamRemove()

    expect(mockedMlApi.samRemoveObject).not.toHaveBeenCalled()
  })

  it('calls samRemoveObject with default preset and expandMaskPixels of 12', async () => {
    mockedMlApi.samRemoveObject.mockResolvedValue(makeMLResult('https://cdn.example.com/removed.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { toggleMaskSelection, handleSamRemove } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamRemove()

    expect(mockedMlApi.samRemoveObject).toHaveBeenCalledWith(9, 3, 12, true, PRESETS.quality)
  })

  it('calls samRemoveObject with a custom ldm config when provided', async () => {
    mockedMlApi.samRemoveObject.mockResolvedValue(makeMLResult('https://cdn.example.com/removed.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const customLdm = { ldm_steps: 5, ldm_sampler: 'ddim' as const, hd_strategy: 'ORIGINAL' as const }
    const { toggleMaskSelection, handleSamRemove } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamRemove(customLdm)

    expect(mockedMlApi.samRemoveObject).toHaveBeenCalledWith(9, 3, 12, true, customLdm)
  })

  it('updates currentImageUrl from result', async () => {
    mockedMlApi.samRemoveObject.mockResolvedValue(makeMLResult('https://cdn.example.com/removed.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const currentImageUrl = ref('')
    const { toggleMaskSelection, handleSamRemove } = useSegmentation(9, currentImageUrl, ref([]))
    toggleMaskSelection(3)

    await handleSamRemove()

    expect(currentImageUrl.value).toBe('https://cdn.example.com/removed.jpg')
  })

  it('removes the segment matching selectedMaskId from segments', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1), makeSegment(2)]))
    mockedMlApi.samRemoveObject.mockResolvedValue(makeMLResult('https://cdn.example.com/removed.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { handleSegment, toggleMaskSelection, handleSamRemove, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    toggleMaskSelection(1)

    await handleSamRemove()

    expect(segments.value).toHaveLength(1)
    expect(segments.value[0].mask_id).toBe(2)
  })

  it('clears selectedMaskId after removal', async () => {
    mockedMlApi.samRemoveObject.mockResolvedValue(makeMLResult('https://cdn.example.com/removed.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { toggleMaskSelection, handleSamRemove, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamRemove()

    expect(selectedMaskId.value).toBeNull()
  })

  it('updates history after removal', async () => {
    mockedMlApi.samRemoveObject.mockResolvedValue(makeMLResult('https://cdn.example.com/removed.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: ['step1', 'step2'] })

    const history = ref<string[]>([])
    const { toggleMaskSelection, handleSamRemove } = useSegmentation(9, ref(''), history)
    toggleMaskSelection(3)

    await handleSamRemove()

    expect(mockedMlApi.getHistory).toHaveBeenCalledWith(9)
    expect(history.value).toEqual(['step1', 'step2'])
  })

  it('sets mlError when samRemoveObject fails', async () => {
    mockedMlApi.samRemoveObject.mockRejectedValue({
      response: { data: { detail: 'SAM remove failed on server' } }
    })

    const { toggleMaskSelection, handleSamRemove, mlError } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamRemove()

    expect(mlError.value).toBe('SAM remove failed on server')
  })

  it('keeps selectedMaskId when samRemoveObject fails', async () => {
    mockedMlApi.samRemoveObject.mockRejectedValue(new Error('fail'))

    const { toggleMaskSelection, handleSamRemove, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamRemove()

    expect(selectedMaskId.value).toBe(3)
  })
})

describe('useSegmentation: handleSamReplace', () => {
  const makeFile = () => new File(['img'], 'replacement.jpg', { type: 'image/jpeg' })

  it('does nothing if no mask is selected', async () => {
    const { replacementFile, handleSamReplace } = useSegmentation(9, ref(''), ref([]))
    replacementFile.value = makeFile()

    await handleSamReplace()

    expect(mockedMlApi.samReplaceObject).not.toHaveBeenCalled()
  })

  it('does nothing if no replacementFile is set', async () => {
    const { toggleMaskSelection, handleSamReplace } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplace()

    expect(mockedMlApi.samReplaceObject).not.toHaveBeenCalled()
  })

  it('calls samReplaceObject with correct arguments and default ldm preset', async () => {
    const file = makeFile()
    mockedMlApi.samReplaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { toggleMaskSelection, replacementFile, handleSamReplace } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)
    replacementFile.value = file

    await handleSamReplace()

    expect(mockedMlApi.samReplaceObject).toHaveBeenCalledWith(9, 3, file, {
      useEdgeBlending: true,
      ldm: PRESETS.quality
    })
  })

  it('calls samReplaceObject with a custom ldm config when provided', async () => {
    const file = makeFile()
    mockedMlApi.samReplaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const customLdm = { ldm_steps: 5, ldm_sampler: 'ddim' as const, hd_strategy: 'ORIGINAL' as const }
    const { toggleMaskSelection, replacementFile, handleSamReplace } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)
    replacementFile.value = file

    await handleSamReplace(customLdm)

    expect(mockedMlApi.samReplaceObject).toHaveBeenCalledWith(9, 3, file, {
      useEdgeBlending: true,
      ldm: customLdm
    })
  })

  it('updates currentImageUrl after replacement', async () => {
    mockedMlApi.samReplaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const currentImageUrl = ref('')
    const { toggleMaskSelection, replacementFile, handleSamReplace } = useSegmentation(9, currentImageUrl, ref([]))
    toggleMaskSelection(3)
    replacementFile.value = makeFile()

    await handleSamReplace()

    expect(currentImageUrl.value).toBe('https://cdn.example.com/replaced.jpg')
  })

  it('removes the replaced segment from segments', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1), makeSegment(2)]))
    mockedMlApi.samReplaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { handleSegment, toggleMaskSelection, replacementFile, handleSamReplace, segments } =
      useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    toggleMaskSelection(1)
    replacementFile.value = makeFile()

    await handleSamReplace()

    expect(segments.value).toHaveLength(1)
    expect(segments.value[0].mask_id).toBe(2)
  })

  it('clears selectedMaskId and replacementFile after success', async () => {
    mockedMlApi.samReplaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { toggleMaskSelection, replacementFile, handleSamReplace, selectedMaskId } =
      useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)
    replacementFile.value = makeFile()

    await handleSamReplace()

    expect(selectedMaskId.value).toBeNull()
    expect(replacementFile.value).toBeNull()
  })

  it('updates history after replacement', async () => {
    mockedMlApi.samReplaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: ['step1'] })

    const history = ref<string[]>([])
    const { toggleMaskSelection, replacementFile, handleSamReplace } = useSegmentation(9, ref(''), history)
    toggleMaskSelection(3)
    replacementFile.value = makeFile()

    await handleSamReplace()

    expect(mockedMlApi.getHistory).toHaveBeenCalledWith(9)
    expect(history.value).toEqual(['step1'])
  })

  it('sets mlError when samReplaceObject fails', async () => {
    mockedMlApi.samReplaceObject.mockRejectedValue({
      response: { data: { detail: 'SAM replace failed on server' } }
    })

    const { toggleMaskSelection, replacementFile, handleSamReplace, mlError } =
      useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)
    replacementFile.value = makeFile()

    await handleSamReplace()

    expect(mlError.value).toBe('SAM replace failed on server')
  })

  it('keeps selectedMaskId and replacementFile when samReplaceObject fails', async () => {
    mockedMlApi.samReplaceObject.mockRejectedValue(new Error('fail'))

    const file = makeFile()
    const { toggleMaskSelection, replacementFile, handleSamReplace, selectedMaskId } =
      useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)
    replacementFile.value = file

    await handleSamReplace()

    expect(selectedMaskId.value).toBe(3)
    expect(replacementFile.value).toBe(file)
  })
})

describe('useSegmentation: handleSegmentHybrid', () => {
  it('calls segmentHybrid with the provided params', async () => {
    mockedMlApi.segmentHybrid.mockResolvedValue(makeSegmentResult([]))

    const { handleSegmentHybrid } = useSegmentation(9, ref(''), ref([]))
    const params = {
      yoloConfThreshold: 0.5,
      yoloClasses: ['cat', 'dog'],
      fallbackMinArea: 1000,
      fallbackMaxSegments: 10,
      overlapIouThresh: 0.6,
    }

    await handleSegmentHybrid(params)

    expect(mockedMlApi.segmentHybrid).toHaveBeenCalledWith(9, params)
  })

  it('calls segmentHybrid with undefined params when none are provided', async () => {
    mockedMlApi.segmentHybrid.mockResolvedValue(makeSegmentResult([]))

    const { handleSegmentHybrid } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentHybrid()

    expect(mockedMlApi.segmentHybrid).toHaveBeenCalledWith(9, undefined)
  })

  it('sets segments from result', async () => {
    const seg = makeSegment(1)
    mockedMlApi.segmentHybrid.mockResolvedValue(makeSegmentResult([seg]))

    const { handleSegmentHybrid, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentHybrid()

    expect(segments.value).toEqual([seg])
  })

  it('replaces existing segments rather than appending', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1)]))
    mockedMlApi.segmentHybrid.mockResolvedValue(makeSegmentResult([makeSegment(2)]))

    const { handleSegment, handleSegmentHybrid, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    await handleSegmentHybrid()

    expect(segments.value).toEqual([makeSegment(2)])
  })

  it('resets selectedMaskId to null', async () => {
    mockedMlApi.segmentHybrid.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { handleSegmentHybrid, toggleMaskSelection, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)
    expect(selectedMaskId.value).toBe(3)

    await handleSegmentHybrid()

    expect(selectedMaskId.value).toBeNull()
  })

  it('sets segmenting to true during the call and false after success', async () => {
    let resolvePromise: (value: any) => void
    mockedMlApi.segmentHybrid.mockReturnValue(
      new Promise(resolve => { resolvePromise = resolve })
    )

    const { handleSegmentHybrid, segmenting } = useSegmentation(9, ref(''), ref([]))
    const promise = handleSegmentHybrid()

    expect(segmenting.value).toBe(true)

    resolvePromise!(makeSegmentResult([]))
    await promise

    expect(segmenting.value).toBe(false)
  })

  it('sets mlError when segmentHybrid fails', async () => {
    mockedMlApi.segmentHybrid.mockRejectedValue({
      response: { data: { detail: 'Hybrid segmentation failed on server' } }
    })

    const { handleSegmentHybrid, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentHybrid()

    expect(mlError.value).toBe('Hybrid segmentation failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.segmentHybrid.mockRejectedValue(new Error('fail'))

    const { handleSegmentHybrid, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentHybrid()

    expect(mlError.value).toBe('Hybrid segmentation failed')
  })

  it('sets segmenting to false after failure', async () => {
    mockedMlApi.segmentHybrid.mockRejectedValue(new Error('fail'))

    const { handleSegmentHybrid, segmenting } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentHybrid()

    expect(segmenting.value).toBe(false)
  })

  it('clears previous mlError on new call', async () => {
    mockedMlApi.segmentHybrid.mockRejectedValueOnce({
      response: { data: { detail: 'first error' } }
    })

    const { handleSegmentHybrid, mlError } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentHybrid()
    expect(mlError.value).toBe('first error')

    mockedMlApi.segmentHybrid.mockResolvedValueOnce(makeSegmentResult([]))
    await handleSegmentHybrid()

    expect(mlError.value).toBe('')
  })

  it('does not clear segments on failure', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1)]))
    mockedMlApi.segmentHybrid.mockRejectedValue(new Error('fail'))

    const { handleSegment, handleSegmentHybrid, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    await handleSegmentHybrid()

    expect(segments.value).toEqual([makeSegment(1)])
  })
})
describe('useSegmentation: handleSamReplaceWithAsset', () => {
  it('does nothing if no mask is selected', async () => {
    const { handleSamReplaceWithAsset } = useSegmentation(9, ref(''), ref([]))

    await handleSamReplaceWithAsset('asset-1')

    expect(mockedMlApi.samReplaceObjectWithAsset).not.toHaveBeenCalled()
  })

  it('does nothing if assetId is empty', async () => {
    const { toggleMaskSelection, handleSamReplaceWithAsset } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('')

    expect(mockedMlApi.samReplaceObjectWithAsset).not.toHaveBeenCalled()
  })

  it('calls samReplaceObjectWithAsset with correct arguments and default ldm preset', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockResolvedValue(makeMLResult('https://cdn.example.com/asset-replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { toggleMaskSelection, handleSamReplaceWithAsset } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1')

    expect(mockedMlApi.samReplaceObjectWithAsset).toHaveBeenCalledWith(9, 3, 'asset-1', {
      useEdgeBlending: true,
      ldm: PRESETS.quality
    })
  })

  it('calls samReplaceObjectWithAsset with a custom ldm config when provided', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockResolvedValue(makeMLResult('https://cdn.example.com/asset-replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const customLdm = { ldm_steps: 5, ldm_sampler: 'ddim' as const, hd_strategy: 'ORIGINAL' as const }
    const { toggleMaskSelection, handleSamReplaceWithAsset } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1', customLdm)

    expect(mockedMlApi.samReplaceObjectWithAsset).toHaveBeenCalledWith(9, 3, 'asset-1', {
      useEdgeBlending: true,
      ldm: customLdm
    })
  })

  it('updates currentImageUrl after replacement', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockResolvedValue(makeMLResult('https://cdn.example.com/asset-replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const currentImageUrl = ref('')
    const { toggleMaskSelection, handleSamReplaceWithAsset } = useSegmentation(9, currentImageUrl, ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1')

    expect(currentImageUrl.value).toBe('https://cdn.example.com/asset-replaced.jpg')
  })

  it('removes the replaced segment from segments', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1), makeSegment(2)]))
    mockedMlApi.samReplaceObjectWithAsset.mockResolvedValue(makeMLResult('https://cdn.example.com/asset-replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { handleSegment, toggleMaskSelection, handleSamReplaceWithAsset, segments } =
      useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    toggleMaskSelection(1)

    await handleSamReplaceWithAsset('asset-1')

    expect(segments.value).toHaveLength(1)
    expect(segments.value[0].mask_id).toBe(2)
  })

  it('clears selectedMaskId after success', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockResolvedValue(makeMLResult('https://cdn.example.com/asset-replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const { toggleMaskSelection, handleSamReplaceWithAsset, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1')

    expect(selectedMaskId.value).toBeNull()
  })

  it('updates history after replacement', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockResolvedValue(makeMLResult('https://cdn.example.com/asset-replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: ['step1'] })

    const history = ref<string[]>([])
    const { toggleMaskSelection, handleSamReplaceWithAsset } = useSegmentation(9, ref(''), history)
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1')

    expect(mockedMlApi.getHistory).toHaveBeenCalledWith(9)
    expect(history.value).toEqual(['step1'])
  })

  it('sets mlError when samReplaceObjectWithAsset fails', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockRejectedValue({
      response: { data: { detail: 'SAM replace with asset failed on server' } }
    })

    const { toggleMaskSelection, handleSamReplaceWithAsset, mlError } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1')

    expect(mlError.value).toBe('SAM replace with asset failed on server')
  })

  it('falls back to default error message when server gives none', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockRejectedValue(new Error('fail'))

    const { toggleMaskSelection, handleSamReplaceWithAsset, mlError } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1')

    expect(mlError.value).toBe('SAM replace with asset failed')
  })

  it('keeps selectedMaskId when samReplaceObjectWithAsset fails', async () => {
    mockedMlApi.samReplaceObjectWithAsset.mockRejectedValue(new Error('fail'))

    const { toggleMaskSelection, handleSamReplaceWithAsset, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(3)

    await handleSamReplaceWithAsset('asset-1')

    expect(selectedMaskId.value).toBe(3)
  })
})

describe('useSegmentation: onReplacementSelect', () => {
  it('sets replacementFile from input event', () => {
    const { onReplacementSelect, replacementFile } = useSegmentation(9, ref(''), ref([]))

    const file = new File(['img'], 'photo.jpg', { type: 'image/jpeg' })
    const input = document.createElement('input')
    input.type = 'file'
    Object.defineProperty(input, 'files', { value: [file] })

    onReplacementSelect({ target: input } as unknown as Event)

    expect(replacementFile.value).toBe(file)
  })

  it('sets replacementFile to null when no file selected', () => {
    const { onReplacementSelect, replacementFile } = useSegmentation(9, ref(''), ref([]))

    const input = document.createElement('input')
    input.type = 'file'
    Object.defineProperty(input, 'files', { value: [] })

    onReplacementSelect({ target: input } as unknown as Event)

    expect(replacementFile.value).toBeNull()
  })
})

describe('useSegmentation: clearSegments', () => {
  it('clears segments and selectedMaskId', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1), makeSegment(2)]))

    const { handleSegment, toggleMaskSelection, clearSegments, segments, selectedMaskId } =
      useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    toggleMaskSelection(1)

    clearSegments()

    expect(segments.value).toEqual([])
    expect(selectedMaskId.value).toBeNull()
  })

  it('empties regions as a side effect', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue(makeSegmentResult([makeSegment(1)]))

    const { handleSegment, clearSegments, regions } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    expect(regions.value).toHaveLength(1)

    clearSegments()

    expect(regions.value).toEqual([])
  })

  it('also clears prompt state and prompt mode', async () => {
    const { addPromptPoint, setPromptMode, clearSegments, promptPoints, promptMode } =
      useSegmentation(9, ref(''), ref([]))
    setPromptMode('points')
    addPromptPoint(1, 1)

    clearSegments()

    expect(promptPoints.value).toEqual([])
    expect(promptMode.value).toBeNull()
  })
})