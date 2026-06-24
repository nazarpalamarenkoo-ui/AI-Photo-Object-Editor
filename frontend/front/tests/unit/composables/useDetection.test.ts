import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'
import type { Detection } from '@/types/Index'

vi.mock('@/api/detections', () => ({
  detectionsApi: {
    deleteByImage: vi.fn()
  }
}))

vi.mock('@/api/ml', () => ({
  mlApi: {
    detectObjects: vi.fn()
  }
}))

import { detectionsApi } from '@/api/detections'
import { mlApi } from '@/api/ml'
import { useDetections } from '@/composables/useDetections'

const mockedDetectionsApi = vi.mocked(detectionsApi, true)
const mockedMlApi = vi.mocked(mlApi, true)

const fakeDetections: Detection[] = [
  { id: 1, image_id: 10, bbox_id: 0, x1: 0, y1: 0, x2: 100, y2: 100, detected_class: 'person', confidence: 0.9 },
  { id: 2, image_id: 10, bbox_id: 1, x1: 50, y1: 50, x2: 150, y2: 150, detected_class: 'car', confidence: 0.75 }
]

beforeEach(() => {
  vi.clearAllMocks()
})


describe('useDetections: initial state', () => {
  it('starts with empty selection, no error, not detecting', () => {
    const detections = ref<Detection[]>([])
    const { selectedBboxIds, detecting, mlError, confThreshold } = useDetections(10, detections)

    expect(selectedBboxIds.value).toEqual([])
    expect(detecting.value).toBe(false)
    expect(mlError.value).toBe('')
    expect(confThreshold.value).toBe(0.5)
  })
})


describe('useDetections: handleDetect', () => {
  it('calls detectObjects with imageId and current confThreshold', async () => {
    mockedMlApi.detectObjects.mockResolvedValue({ detections: fakeDetections, image_size: [640, 480], metrics: {}, timestamp: '' })
    const detections = ref<Detection[]>([])
    const { handleDetect, confThreshold } = useDetections(10, detections)

    confThreshold.value = 0.7
    await handleDetect()

    expect(mockedMlApi.detectObjects).toHaveBeenCalledWith(10, { conf_threshold: 0.7 })
  })

  it('sets detections from result', async () => {
    mockedMlApi.detectObjects.mockResolvedValue({ detections: fakeDetections, image_size: [640, 480], metrics: {}, timestamp: '' })
    const detections = ref<Detection[]>([])
    const { handleDetect } = useDetections(10, detections)

    await handleDetect()

    expect(detections.value).toEqual(fakeDetections)
  })

  it('clears selectedBboxIds and mlError before detecting', async () => {
    mockedMlApi.detectObjects.mockResolvedValue({ detections: [], image_size: [640, 480], metrics: {}, timestamp: '' })
    const detections = ref<Detection[]>([])
    const { handleDetect, selectedBboxIds, mlError, toggleSelection } = useDetections(10, detections)

    toggleSelection(0)
    mlError.value = 'old error'

    await handleDetect()

    expect(selectedBboxIds.value).toEqual([])
    expect(mlError.value).toBe('')
  })

  it('sets detecting to true during request and false after', async () => {
    let resolveFn!: () => void
    mockedMlApi.detectObjects.mockReturnValue(new Promise(resolve => { resolveFn = () => resolve({ detections: [], image_size: [640, 480], metrics: {}, timestamp: '' }) }))
    const detections = ref<Detection[]>([])
    const { handleDetect, detecting } = useDetections(10, detections)

    const promise = handleDetect()
    expect(detecting.value).toBe(true)
    resolveFn()
    await promise
    expect(detecting.value).toBe(false)
  })

  it('sets mlError from response detail on failure', async () => {
    mockedMlApi.detectObjects.mockRejectedValue({ response: { data: { detail: 'Model not loaded' } } })
    const detections = ref<Detection[]>([])
    const { handleDetect, mlError } = useDetections(10, detections)

    await handleDetect()

    expect(mlError.value).toBe('Model not loaded')
  })

  it('sets fallback mlError when response detail is missing', async () => {
    mockedMlApi.detectObjects.mockRejectedValue(new Error('network error'))
    const detections = ref<Detection[]>([])
    const { handleDetect, mlError } = useDetections(10, detections)

    await handleDetect()

    expect(mlError.value).toBe('Detection failed')
  })

  it('sets detecting to false even when request fails', async () => {
    mockedMlApi.detectObjects.mockRejectedValue(new Error('fail'))
    const detections = ref<Detection[]>([])
    const { handleDetect, detecting } = useDetections(10, detections)

    await handleDetect()

    expect(detecting.value).toBe(false)
  })
})


describe('useDetections: handleClearDetections', () => {
  it('calls deleteByImage with imageId', async () => {
    mockedDetectionsApi.deleteByImage.mockResolvedValue(undefined as never)
    const detections = ref<Detection[]>([...fakeDetections])
    const { handleClearDetections } = useDetections(10, detections)

    await handleClearDetections()

    expect(mockedDetectionsApi.deleteByImage).toHaveBeenCalledWith(10)
  })

  it('clears detections and selectedBboxIds', async () => {
    mockedDetectionsApi.deleteByImage.mockResolvedValue(undefined as never)
    const detections = ref<Detection[]>([...fakeDetections])
    const { handleClearDetections, selectedBboxIds, toggleSelection } = useDetections(10, detections)

    toggleSelection(0)
    await handleClearDetections()

    expect(detections.value).toEqual([])
    expect(selectedBboxIds.value).toEqual([])
  })

  it('calls onClear callback if provided', async () => {
    mockedDetectionsApi.deleteByImage.mockResolvedValue(undefined as never)
    const detections = ref<Detection[]>([])
    const { handleClearDetections } = useDetections(10, detections)
    const onClear = vi.fn()

    await handleClearDetections(onClear)

    expect(onClear).toHaveBeenCalledTimes(1)
  })

  it('does not throw when onClear is not provided', async () => {
    mockedDetectionsApi.deleteByImage.mockResolvedValue(undefined as never)
    const detections = ref<Detection[]>([])
    const { handleClearDetections } = useDetections(10, detections)

    await expect(handleClearDetections()).resolves.toBeUndefined()
  })
})


describe('useDetections: toggleSelection', () => {
  it('adds bboxId when not selected', () => {
    const detections = ref<Detection[]>([])
    const { toggleSelection, selectedBboxIds } = useDetections(10, detections)

    toggleSelection(3)

    expect(selectedBboxIds.value).toContain(3)
  })

  it('removes bboxId when already selected', () => {
    const detections = ref<Detection[]>([])
    const { toggleSelection, selectedBboxIds } = useDetections(10, detections)

    toggleSelection(3)
    toggleSelection(3)

    expect(selectedBboxIds.value).not.toContain(3)
  })

  it('handles multiple different bboxIds independently', () => {
    const detections = ref<Detection[]>([])
    const { toggleSelection, selectedBboxIds } = useDetections(10, detections)

    toggleSelection(0)
    toggleSelection(1)
    toggleSelection(2)
    toggleSelection(1)

    expect(selectedBboxIds.value).toEqual([0, 2])
  })
})