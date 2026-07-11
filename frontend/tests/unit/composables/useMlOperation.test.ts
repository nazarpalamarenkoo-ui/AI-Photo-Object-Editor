import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'
import type { Detection, Image, MLResultResponse } from '@/types/Index'

vi.mock('@/api/ml', () => ({
  PRESETS: {
    quality: { ldm_steps: 20, ldm_sampler: 'plms', hd_strategy: 'RESIZE' }
  },
  mlApi: {
    saveResult: vi.fn(),
    removeObject: vi.fn(),
    removeMultipleObjects: vi.fn(),
    replaceObject: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    resetState: vi.fn(),
    getHistory: vi.fn()
  }
}))

import { mlApi, PRESETS } from '@/api/ml'
import { useMlOperations } from '@/composables/useMlOperations'

const mockedMlApi = vi.mocked(mlApi, true)

const fakeImage: Image = {
  id: 5,
  filename: 'shot.jpg',
  storage_path: 'uploads/1/5/shot.jpg',
  status: 'ready',
  uploaded_at: '2026-01-01T00:00:00Z',
  user_id: 1
}

const makeDetection = (bboxId: number): Detection => ({
  id: bboxId,
  image_id: 7,
  bbox_id: bboxId,
  x1: 0,
  y1: 0,
  x2: 100,
  y2: 100,
  detected_class: 'person',
  confidence: 0.95,
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

describe('useMlOperations: initial state', () => {
  it('starts with empty loading and error states', () => {
    const detections = ref<Detection[]>([])
    const selectedBboxIds = ref<number[]>([])

    const { mlLoading, mlError, saveLoading, saveError, savedSuccess } =
      useMlOperations(1, detections, selectedBboxIds)

    expect(mlLoading.value).toBe(false)
    expect(mlError.value).toBe('')
    expect(saveLoading.value).toBe(false)
    expect(saveError.value).toBe('')
    expect(savedSuccess.value).toBe(false)
  })

  it('starts with empty currentImageUrl, history and replacementFile', () => {
    const detections = ref<Detection[]>([])
    const selectedBboxIds = ref<number[]>([])

    const { currentImageUrl, history, replacementFile, canUndo } =
      useMlOperations(1, detections, selectedBboxIds)

    expect(currentImageUrl.value).toBe('')
    expect(history.value).toEqual([])
    expect(replacementFile.value).toBeNull()
    expect(canUndo.value).toBe(false)
  })

  it('useEdgeBlending defaults to true', () => {
    const { useEdgeBlending } = useMlOperations(1, ref([]), ref([]))
    expect(useEdgeBlending.value).toBe(true)
  })
})

describe('useMlOperations: canUndo', () => {
  it('is true when history has entries', async () => {
    mockedMlApi.removeObject.mockResolvedValue(makeMLResult('https://cdn.example.com/result.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: ['step1'] })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { canUndo, handleRemove } = useMlOperations(1, detections, selectedBboxIds)

    await handleRemove([1])

    expect(canUndo.value).toBe(true)
  })

  it('is false after reset clears history', async () => {
    mockedMlApi.resetState.mockResolvedValue(undefined)

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([])
    const { canUndo, handleReset } = useMlOperations(1, detections, selectedBboxIds)

    await handleReset('https://cdn.example.com/original.jpg')

    expect(canUndo.value).toBe(false)
  })
})

describe('useMlOperations: handleSave', () => {
  it('calls saveResult with correct id and sets savedSuccess', async () => {
    mockedMlApi.saveResult.mockResolvedValue(fakeImage)

    const { handleSave, savedSuccess } = useMlOperations(1, ref([]), ref([]))

    await handleSave(5)

    expect(mockedMlApi.saveResult).toHaveBeenCalledWith(5)
    expect(savedSuccess.value).toBe(true)
  })

  it('calls onSaved callback with returned image', async () => {
    mockedMlApi.saveResult.mockResolvedValue(fakeImage)
    const onSaved = vi.fn()

    const { handleSave } = useMlOperations(1, ref([]), ref([]))

    await handleSave(5, onSaved)

    expect(onSaved).toHaveBeenCalledWith(fakeImage)
  })

  it('sets saveError when saveResult fails', async () => {
    mockedMlApi.saveResult.mockRejectedValue({
      response: { data: { detail: 'Save failed on server' } }
    })

    const { handleSave, saveError } = useMlOperations(1, ref([]), ref([]))

    await handleSave(5)

    expect(saveError.value).toBe('Save failed on server')
    expect(mockedMlApi.saveResult).toHaveBeenCalledWith(5)
  })

  it('sets saveLoading to false after success', async () => {
    mockedMlApi.saveResult.mockResolvedValue(fakeImage)

    const { handleSave, saveLoading } = useMlOperations(1, ref([]), ref([]))

    await handleSave(5)

    expect(saveLoading.value).toBe(false)
  })

  it('sets saveLoading to false after failure', async () => {
    mockedMlApi.saveResult.mockRejectedValue(new Error('fail'))

    const { handleSave, saveLoading } = useMlOperations(1, ref([]), ref([]))

    await handleSave(5)

    expect(saveLoading.value).toBe(false)
  })
})

describe('useMlOperations: handleRemove', () => {
  it('calls removeObject for a single bbox id', async () => {
    mockedMlApi.removeObject.mockResolvedValue(makeMLResult('https://cdn.example.com/result.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleRemove } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([1])

    expect(mockedMlApi.removeObject).toHaveBeenCalledWith(7, 1, 5, true, PRESETS.quality)
    expect(mockedMlApi.removeMultipleObjects).not.toHaveBeenCalled()
  })

  it('calls removeMultipleObjects for multiple bbox ids', async () => {
    mockedMlApi.removeMultipleObjects.mockResolvedValue(makeMLResult('https://cdn.example.com/result.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1), makeDetection(2)])
    const selectedBboxIds = ref<number[]>([1, 2])
    const { handleRemove } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([1, 2])

    expect(mockedMlApi.removeMultipleObjects).toHaveBeenCalledWith(7, [1, 2], 5, true, PRESETS.quality)
    expect(mockedMlApi.removeObject).not.toHaveBeenCalled()
  })

  it('updates currentImageUrl from result', async () => {
    mockedMlApi.removeObject.mockResolvedValue(makeMLResult('https://cdn.example.com/result.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(3)])
    const selectedBboxIds = ref<number[]>([3])
    const { handleRemove, currentImageUrl } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([3])

    expect(currentImageUrl.value).toBe('https://cdn.example.com/result.jpg')
  })

  it('removes processed detections from the list', async () => {
    mockedMlApi.removeObject.mockResolvedValue(makeMLResult('https://cdn.example.com/result.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1), makeDetection(2)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleRemove } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([1])

    expect(detections.value).toHaveLength(1)
    expect(detections.value[0].bbox_id).toBe(2)
  })

  it('clears selectedBboxIds after removal', async () => {
    mockedMlApi.removeObject.mockResolvedValue(makeMLResult('https://cdn.example.com/result.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleRemove } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([1])

    expect(selectedBboxIds.value).toEqual([])
  })

  it('updates history after removal', async () => {
    mockedMlApi.removeObject.mockResolvedValue(makeMLResult('https://cdn.example.com/result.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: ['step1', 'step2'] })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleRemove, history } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([1])

    expect(history.value).toEqual(['step1', 'step2'])
  })

  it('sets mlError when removeObject fails', async () => {
    mockedMlApi.removeObject.mockRejectedValue({
      response: { data: { detail: 'Removal failed on server' } }
    })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleRemove, mlError } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([1])

    expect(mlError.value).toBe('Removal failed on server')
  })

  it('sets mlLoading to false after failure', async () => {
    mockedMlApi.removeObject.mockRejectedValue(new Error('fail'))

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleRemove, mlLoading } = useMlOperations(7, detections, selectedBboxIds)

    await handleRemove([1])

    expect(mlLoading.value).toBe(false)
  })
})

describe('useMlOperations: handleReplace', () => {
  const makeFile = () => new File(['img'], 'replacement.jpg', { type: 'image/jpeg' })

  it('does nothing if no replacementFile is set', async () => {
    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReplace } = useMlOperations(7, detections, selectedBboxIds)

    await handleReplace([1])

    expect(mockedMlApi.replaceObject).not.toHaveBeenCalled()
  })

  it('does nothing if more than one bbox is passed', async () => {
    const detections = ref<Detection[]>([makeDetection(1), makeDetection(2)])
    const selectedBboxIds = ref<number[]>([1, 2])
    const { handleReplace, replacementFile } = useMlOperations(7, detections, selectedBboxIds)

    replacementFile.value = makeFile()

    await handleReplace([1, 2])

    expect(mockedMlApi.replaceObject).not.toHaveBeenCalled()
  })

  it('calls replaceObject with correct arguments', async () => {
    const file = makeFile()
    mockedMlApi.replaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReplace, replacementFile } = useMlOperations(7, detections, selectedBboxIds)

    replacementFile.value = file

    await handleReplace([1])

    expect(mockedMlApi.replaceObject).toHaveBeenCalledWith(
      7,
      1,
      file,
      {
        useEdgeBlending: true,
        ldm: PRESETS.quality
      }
    )
  })

  it('updates currentImageUrl after replacement', async () => {
    mockedMlApi.replaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReplace, replacementFile, currentImageUrl } = useMlOperations(7, detections, selectedBboxIds)

    replacementFile.value = makeFile()
    await handleReplace([1])

    expect(currentImageUrl.value).toBe('https://cdn.example.com/replaced.jpg')
  })

  it('removes the replaced detection from the list', async () => {
    mockedMlApi.replaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1), makeDetection(2)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReplace, replacementFile } = useMlOperations(7, detections, selectedBboxIds)

    replacementFile.value = makeFile()
    await handleReplace([1])

    expect(detections.value).toHaveLength(1)
    expect(detections.value[0].bbox_id).toBe(2)
  })

  it('clears replacementFile and selectedBboxIds after success', async () => {
    mockedMlApi.replaceObject.mockResolvedValue(makeMLResult('https://cdn.example.com/replaced.jpg'))
    mockedMlApi.getHistory.mockResolvedValue({ history: [] })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReplace, replacementFile } = useMlOperations(7, detections, selectedBboxIds)

    replacementFile.value = makeFile()
    await handleReplace([1])

    expect(replacementFile.value).toBeNull()
    expect(selectedBboxIds.value).toEqual([])
  })

  it('sets mlError when replaceObject fails', async () => {
    mockedMlApi.replaceObject.mockRejectedValue({
      response: { data: { detail: 'Replace failed on server' } }
    })

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReplace, replacementFile, mlError } = useMlOperations(7, detections, selectedBboxIds)

    replacementFile.value = makeFile()
    await handleReplace([1])

    expect(mlError.value).toBe('Replace failed on server')
  })
})

describe('useMlOperations: handleUndo', () => {
  it('calls undo API and updates currentImageUrl and history', async () => {
    mockedMlApi.undo.mockResolvedValue({
      presigned_url: 'https://cdn.example.com/undone.jpg',
      label: 'remove bbox_id=1',
      history: ['step1']
    })

    const { handleUndo, currentImageUrl, history } = useMlOperations(7, ref([]), ref([]))

    await handleUndo()

    expect(mockedMlApi.undo).toHaveBeenCalledWith(7)
    expect(currentImageUrl.value).toBe('https://cdn.example.com/undone.jpg')
    expect(history.value).toEqual(['step1'])
  })

  it('sets mlError when undo fails', async () => {
    mockedMlApi.undo.mockRejectedValue({
      response: { data: { detail: 'Undo failed on server' } }
    })

    const { handleUndo, mlError } = useMlOperations(7, ref([]), ref([]))

    await handleUndo()

    expect(mlError.value).toBe('Undo failed on server')
  })

  it('sets mlLoading to false after undo', async () => {
    mockedMlApi.undo.mockResolvedValue({
      presigned_url: 'https://cdn.example.com/undone.jpg',
      label: 'remove bbox_id=1',
      history: []
    })

    const { handleUndo, mlLoading } = useMlOperations(7, ref([]), ref([]))

    await handleUndo()

    expect(mlLoading.value).toBe(false)
  })

  it('resets savedSuccess after undo', async () => {
    mockedMlApi.undo.mockResolvedValue({
      presigned_url: 'https://cdn.example.com/undone.jpg',
      label: 'remove bbox_id=1',
      history: []
    })
    mockedMlApi.saveResult.mockResolvedValue(fakeImage)

    const { handleSave, handleUndo, savedSuccess } = useMlOperations(7, ref([]), ref([]))

    await handleSave(7)
    expect(savedSuccess.value).toBe(true)
    await handleUndo()
    expect(savedSuccess.value).toBe(false)
  })
})

describe('useMlOperations: handleRedo', () => {
  it('calls redo API and updates currentImageUrl and history', async () => {
    mockedMlApi.redo.mockResolvedValue({
      presigned_url: 'https://cdn.example.com/redone.jpg',
      label: 'remove bbox_id=1',
      history: ['step1', 'step2']
    })

    const { handleRedo, currentImageUrl, history } = useMlOperations(7, ref([]), ref([]))

    await handleRedo()

    expect(mockedMlApi.redo).toHaveBeenCalledWith(7)
    expect(currentImageUrl.value).toBe('https://cdn.example.com/redone.jpg')
    expect(history.value).toEqual(['step1', 'step2'])
  })

  it('sets mlError when redo fails', async () => {
    mockedMlApi.redo.mockRejectedValue({
      response: { data: { detail: 'Redo failed on server' } }
    })

    const { handleRedo, mlError } = useMlOperations(7, ref([]), ref([]))

    await handleRedo()

    expect(mlError.value).toBe('Redo failed on server')
  })

  it('sets mlLoading to false after redo', async () => {
    mockedMlApi.redo.mockResolvedValue({
      presigned_url: 'https://cdn.example.com/redone.jpg',
      label: 'remove bbox_id=1',
      history: []
    })

    const { handleRedo, mlLoading } = useMlOperations(7, ref([]), ref([]))

    await handleRedo()

    expect(mlLoading.value).toBe(false)
  })
})

describe('useMlOperations: handleReset', () => {
  it('calls resetState and restores original url', async () => {
    mockedMlApi.resetState.mockResolvedValue(undefined)

    const detections = ref<Detection[]>([makeDetection(1)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReset, currentImageUrl } = useMlOperations(7, detections, selectedBboxIds)

    await handleReset('https://cdn.example.com/original.jpg')

    expect(mockedMlApi.resetState).toHaveBeenCalledWith(7)
    expect(currentImageUrl.value).toBe('https://cdn.example.com/original.jpg')
  })

  it('clears detections, selectedBboxIds and history after reset', async () => {
    mockedMlApi.resetState.mockResolvedValue(undefined)

    const detections = ref<Detection[]>([makeDetection(1), makeDetection(2)])
    const selectedBboxIds = ref<number[]>([1])
    const { handleReset, history } = useMlOperations(7, detections, selectedBboxIds)

    await handleReset('https://cdn.example.com/original.jpg')

    expect(detections.value).toEqual([])
    expect(selectedBboxIds.value).toEqual([])
    expect(history.value).toEqual([])
  })

  it('sets mlError when resetState fails', async () => {
    mockedMlApi.resetState.mockRejectedValue({
      response: { data: { detail: 'Reset failed on server' } }
    })

    const { handleReset, mlError } = useMlOperations(7, ref([]), ref([]))

    await handleReset('https://cdn.example.com/original.jpg')

    expect(mlError.value).toBe('Reset failed on server')
  })

  it('sets mlLoading to false after reset', async () => {
    mockedMlApi.resetState.mockResolvedValue(undefined)

    const { handleReset, mlLoading } = useMlOperations(7, ref([]), ref([]))

    await handleReset('https://cdn.example.com/original.jpg')

    expect(mlLoading.value).toBe(false)
  })
})

describe('useMlOperations: fetchHistory', () => {
  it('fetches and sets history', async () => {
    mockedMlApi.getHistory.mockResolvedValue({ history: ['step1', 'step2', 'step3'] })

    const { fetchHistory, history } = useMlOperations(7, ref([]), ref([]))

    await fetchHistory()

    expect(mockedMlApi.getHistory).toHaveBeenCalledWith(7)
    expect(history.value).toEqual(['step1', 'step2', 'step3'])
  })

  it('silently ignores errors', async () => {
    mockedMlApi.getHistory.mockRejectedValue(new Error('network error'))

    const { fetchHistory, history } = useMlOperations(7, ref([]), ref([]))

    await expect(fetchHistory()).resolves.toBeUndefined()
    expect(history.value).toEqual([])
  })
})

describe('useMlOperations: onReplacementSelect', () => {
  it('sets replacementFile from input event', () => {
    const { onReplacementSelect, replacementFile } = useMlOperations(7, ref([]), ref([]))

    const file = new File(['img'], 'photo.jpg', { type: 'image/jpeg' })
    const input = document.createElement('input')
    input.type = 'file'
    Object.defineProperty(input, 'files', { value: [file] })

    onReplacementSelect({ target: input } as unknown as Event)

    expect(replacementFile.value).toBe(file)
  })

  it('sets replacementFile to null when no file selected', () => {
    const { onReplacementSelect, replacementFile } = useMlOperations(7, ref([]), ref([]))

    const input = document.createElement('input')
    input.type = 'file'
    Object.defineProperty(input, 'files', { value: [] })

    onReplacementSelect({ target: input } as unknown as Event)

    expect(replacementFile.value).toBeNull()
  })
})