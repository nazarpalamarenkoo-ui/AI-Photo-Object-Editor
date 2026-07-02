import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'
import type { SegmentInfo } from '@/types/Index'

vi.mock('@/api/ml', () => ({
  PRESETS: {
    quality: { ldm_steps: 20, ldm_sampler: 'plms', hd_strategy: 'RESIZE' }
  },
  mlApi: {
    segmentObjects: vi.fn(),
    segmentWithPrompt: vi.fn(),
    samRemoveObject: vi.fn(),
    samReplaceObject: vi.fn(),
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

const makeMLResult = (presigned_url: string) => ({ presigned_url })

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
})

describe('useSegmentation: regions', () => {
  it('maps segments to region items with generated labels', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(1), makeSegment(2)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment, regions } = useSegmentation(1, ref(''), ref([]))
    await handleSegment()

    expect(regions.value).toEqual([
      { id: 1, bbox: { x1: 0, y1: 0, x2: 50, y2: 50 }, label: 'Object #1' },
      { id: 2, bbox: { x1: 0, y1: 0, x2: 50, y2: 50 }, label: 'Object #2' }
    ])
  })

  it('updates reactively when segments change', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(5)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment, regions } = useSegmentation(1, ref(''), ref([]))
    await handleSegment()

    expect(regions.value).toHaveLength(1)
    expect(regions.value[0].label).toBe('Object #5')
  })
})

describe('useSegmentation: handleSegment', () => {
  it('calls segmentObjects with provided minArea and maxSegments', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment } = useSegmentation(9, ref(''), ref([]))
    await handleSegment(300, 20)

    expect(mockedMlApi.segmentObjects).toHaveBeenCalledWith(9, 300, 20)
  })

  it('uses default minArea and maxSegments when not provided', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()

    expect(mockedMlApi.segmentObjects).toHaveBeenCalledWith(9, 500, 50)
  })

  it('sets segments from result', async () => {
    const seg = makeSegment(1)
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [seg],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()

    expect(segments.value).toEqual([seg])
  })

  it('resets selectedMaskId to null', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(1)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

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

    resolvePromise!({ segments: [], metrics: {}, image_size: [800, 600], timestamp: '2026-01-01T00:00:00Z' })
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

    mockedMlApi.segmentObjects.mockResolvedValueOnce({
      segments: [], metrics: {}, image_size: [800, 600], timestamp: '2026-01-01T00:00:00Z'
    })
    await handleSegment()

    expect(mlError.value).toBe('')
  })
})

describe('useSegmentation: handleSegmentWithPrompt', () => {
  it('calls segmentWithPrompt with given params', async () => {
    mockedMlApi.segmentWithPrompt.mockResolvedValue({
      segments: [makeSegment(1)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegmentWithPrompt } = useSegmentation(9, ref(''), ref([]))
    const params = { pointCoords: [[10, 20]] as [number, number][], pointLabels: [1] }

    await handleSegmentWithPrompt(params)

    expect(mockedMlApi.segmentWithPrompt).toHaveBeenCalledWith(9, params)
  })

  it('merges new segments with existing ones', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(1)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })
    mockedMlApi.segmentWithPrompt.mockResolvedValue({
      segments: [makeSegment(2)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment, handleSegmentWithPrompt, segments } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    await handleSegmentWithPrompt({ bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } })

    expect(segments.value).toHaveLength(2)
    expect(segments.value.map(s => s.mask_id)).toEqual([1, 2])
  })

  it('does not reset selectedMaskId', async () => {
    mockedMlApi.segmentWithPrompt.mockResolvedValue({
      segments: [makeSegment(2)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegmentWithPrompt, toggleMaskSelection, selectedMaskId } = useSegmentation(9, ref(''), ref([]))
    toggleMaskSelection(7)

    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(selectedMaskId.value).toBe(7)
  })

  it('sets segmenting to true during and false after success', async () => {
    let resolvePromise: (value: any) => void
    mockedMlApi.segmentWithPrompt.mockReturnValue(
      new Promise(resolve => { resolvePromise = resolve })
    )

    const { handleSegmentWithPrompt, segmenting } = useSegmentation(9, ref(''), ref([]))
    const promise = handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(segmenting.value).toBe(true)

    resolvePromise!({ segments: [], metrics: {}, image_size: [800, 600], timestamp: '2026-01-01T00:00:00Z' })
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

  it('sets segmenting to false after failure', async () => {
    mockedMlApi.segmentWithPrompt.mockRejectedValue(new Error('fail'))

    const { handleSegmentWithPrompt, segmenting } = useSegmentation(9, ref(''), ref([]))
    await handleSegmentWithPrompt({ pointCoords: [[1, 1]], pointLabels: [1] })

    expect(segmenting.value).toBe(false)
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
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(1), makeSegment(2)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })
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
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(1), makeSegment(2)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })
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
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(1), makeSegment(2)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment, toggleMaskSelection, clearSegments, segments, selectedMaskId } =
      useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    toggleMaskSelection(1)

    clearSegments()

    expect(segments.value).toEqual([])
    expect(selectedMaskId.value).toBeNull()
  })

  it('empties regions as a side effect', async () => {
    mockedMlApi.segmentObjects.mockResolvedValue({
      segments: [makeSegment(1)],
      metrics: {},
      image_size: [800, 600],
      timestamp: '2026-01-01T00:00:00Z'
    })

    const { handleSegment, clearSegments, regions } = useSegmentation(9, ref(''), ref([]))
    await handleSegment()
    expect(regions.value).toHaveLength(1)

    clearSegments()

    expect(regions.value).toEqual([])
  })
})