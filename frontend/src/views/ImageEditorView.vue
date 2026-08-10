<template>
  <div class="editor-page" :class="{ dark: isDark }">

    <EditorNavbar
      :username="auth.user?.username ?? ''"
      :isDark="isDark"
      @back="router.push({ name: 'dashboard' })"
      @toggle-theme="toggleTheme"
    />

    <div v-if="loading" class="center-full">
      <div class="spinner"/>
    </div>

    <div v-else-if="image" class="editor-body">

      <EditorToolbar
        v-model:activeTool="activeTool"
        v-model:modelConfig="modelPreset"
        v-model:mode="mode"
        v-model:useHybrid="useHybridSegment"
        v-model:replaceEngine="replaceEngine"
        v-model:diffusionPrompt="diffusionPrompt"
        :zoom="zoom"
        :fitZoom="fitZoom"
        :canUndo="canUndo"
        :mlLoading="mlLoading"
        :busy="detecting || segmenting || mlLoading"
        @zoom="zoom = $event"
        @undo="handleUndo"
        @redo="handleRedo"
        @reset="onReset"
      />

      <EditorCanvas
        :image="image"
        :imageUrl="currentImageUrl"
        :imageLoaded="imageLoaded"
        :naturalSize="naturalSize"
        :zoom="zoom"
        :regions="regions"
        :selectedIds="selectedIds"
        :running="mode === 'yolo' ? detecting : segmenting"
        :mode="mode"
        :confThreshold="confThreshold"
        :activeTool="activeTool"
        :polygonPoints="polygonPoints"
        :promptBbox="promptBbox"
        @set-bbox="b => setPromptBbox(b)"
        @add-polygon-point="p => addPolygonPoint(p.x, p.y)"
        @image-load="onImageLoad"
        @fit-zoom="onFitZoom"
        @toggle-selection="onToggleSelection"
        @run="onRun"
        @clear="onClearRegions"
        @update:confThreshold="confThreshold = $event"
      />

      <EditorSidebar
        :mode="mode"
        :activeTool="activeTool"
        :regions="regions"
        :selectedIds="selectedIds"
        v-model:useEdgeBlending="activeUseEdgeBlending"
        :mlLoading="mlLoading"
        :replacementFile="activeReplacementFile"
        :selectedAssetId="selectedAssetId"
        :resultUrl="currentImageUrl"
        :mlError="combinedError"
        :saveLoading="saveLoading"
        :savedSuccess="savedSuccess"
        :history="history"
        :assets="assets"
        :assetThumbUrls="thumbUrls"
        :assetsLoading="assetsLoading"
        :assetsError="assetsError"
        :assetsHasMore="assetsHasMore"
        :deletingAssetId="deletingId"
        :polygonPoints="polygonPoints"
        :canRunPolygon="canRunPolygon"
        :promptBbox="promptBbox"
        @clear-bbox="clearBbox"
        @run-prompt-segment="() => handleSegmentWithPrompt()"
        @toggle-selection="onToggleSelection"
        @sam-replace-asset="onSamReplaceAsset"
        @remove="handleRemove(selectedBboxIds, modelPreset)"
        @replace="handleReplace(selectedBboxIds, modelPreset)"
        @sam-remove="handleSamRemove(modelPreset)"
        @sam-replace="onSamReplace"
        @run-polygon-segment="() => handleSegmentByPolygon()"
        @clear-polygon="clearPolygon"
        @extract="onExtract"
        @paste="onPaste"
        @select-asset="selectFromLibrary"
        @rename-asset="renameAsset"
        @delete-asset="deleteAsset"
        @load-more-assets="fetchAssets(false)"
        @replacement-select="onReplacementSelectActive"
        @clear-error="clearAllErrors"
        @save="handleSave(imageId)"
      />

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, computed, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

import { useImageEditor } from '@/composables/useImageEditor'
import { useDetections } from '@/composables/useDetections'
import { useMlOperations } from '@/composables/useMlOperations'
import { useSegmentation } from '@/composables/useSegmentation'
import { useAssets } from '@/composables/useAssets'
import { PRESETS } from '@/api/ml'
import type { LdmConfig, EditingMode, RegionItem, ReplaceEngine } from '@/types/Index'

import EditorNavbar from '@/components/EditorNavbar.vue'
import EditorToolbar from '@/components/EditorToolbar.vue'
import EditorCanvas from '@/components/EditorCanvas.vue'
import EditorSidebar from '@/components/EditorSidebar.vue'



const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const imageId = Number(route.params.id)
const isDark = ref(localStorage.getItem('theme') !== 'light')

function toggleTheme() {
  isDark.value = !isDark.value
  localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
}

async function onSamReplaceAsset() {
  if (!selectedAssetId.value) return
  if (replaceEngine.value === 'diffusion') {
    await handleSamReplaceWithAssetDiffusion(selectedAssetId.value, { prompt: diffusionPrompt.value })
  } else {
    await handleSamReplaceWithAsset(selectedAssetId.value, modelPreset.value)
  }
  clearExtracted()
}

function onSamReplace() {
  if (replaceEngine.value === 'diffusion') {
    handleSamReplaceDiffusion({ prompt: diffusionPrompt.value })
  } else {
    handleSamReplace(modelPreset.value)
  }
}

const activeTool = ref('')
const zoom = ref(1)

const fitZoom = ref(1)

function onFitZoom(value: number) {
  fitZoom.value = value
  zoom.value = value
}
const modelPreset = ref<LdmConfig>(PRESETS.quality)
const mode = ref<EditingMode>('yolo')
const useHybridSegment = ref(false)
const replaceEngine = ref<ReplaceEngine>('lama')
const diffusionPrompt = ref('')

const {
  image, imageUrl, originalImageUrl, loading,
  imageLoaded, naturalSize,
  detections, onImageLoad,
} = useImageEditor(imageId)

const {
  selectedBboxIds, detecting, confThreshold,
  mlError: detectError,
  handleDetect, handleClearDetections, toggleSelection,
} = useDetections(imageId, detections)

const {
  mlLoading, mlError, currentImageUrl,
  replacementFile, useEdgeBlending,
  handleRemove, handleReplace, onReplacementSelect,
  saveLoading, savedSuccess, handleSave,
  history, canUndo, handleUndo, handleRedo, handleReset, fetchHistory,
} = useMlOperations(imageId, detections, selectedBboxIds)

const {
  segments, selectedMaskId, segmenting,
  mlError: samError, useEdgeBlending: samUseEdgeBlending, replacementFile: samReplacementFile,
  handleSegment, handleSegmentHybrid, handleSegmentWithPrompt, handleSegmentByPolygon, toggleMaskSelection,
  handleSamRemove, handleSamReplace, handleSamReplaceWithAsset,
  handleSamReplaceDiffusion, handleSamReplaceWithAssetDiffusion,
  onReplacementSelect: onSamReplacementSelect, clearSegments,
  promptBbox, setPromptBbox, clearBbox,
  polygonPoints, addPolygonPoint, clearPolygon, canRunPolygon,
  polygonShapes,
} = useSegmentation(imageId, currentImageUrl, history)

const {
  mlError: assetOpError, selectedAssetId,
  handleExtract, handlePaste, clearExtracted, selectFromLibrary,
  assets, assetsLoading, assetsError, assetsHasMore, thumbUrls,
  deletingId, fetchAssets, renameAsset, deleteAsset,
} = useAssets(imageId, currentImageUrl, history)


const regions = computed<RegionItem[]>(() =>
  mode.value === 'yolo'
    ? (detections.value ?? []).map(d => ({
        id: d.bbox_id, bbox: { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 },
        label: d.detected_class, confidence: d.confidence,
      }))
    : segments.value.map(s => ({
        id: s.mask_id,
        bbox: s.bbox,
        label: `Object #${s.mask_id}`,
        points: polygonShapes.value[s.mask_id],
        mask_url: s.mask_url,
      }))
)

const selectedIds = computed<number[]>(() =>
  mode.value === 'yolo'
    ? selectedBboxIds.value
    : selectedMaskId.value !== null ? [selectedMaskId.value] : []
)

const activeUseEdgeBlending = computed({
  get: () => (mode.value === 'yolo' ? useEdgeBlending.value : samUseEdgeBlending.value),
  set: (v: boolean) => {
    if (mode.value === 'yolo') useEdgeBlending.value = v
    else samUseEdgeBlending.value = v
  },
})

const activeReplacementFile = computed(() =>
  mode.value === 'yolo' ? replacementFile.value : samReplacementFile.value
)

function onReplacementSelectActive(event: Event) {
  if (mode.value === 'yolo') onReplacementSelect(event)
  else onSamReplacementSelect(event)
}

function onToggleSelection(id: number) {
  if (mode.value === 'yolo') toggleSelection(id)
  else toggleMaskSelection(id)
}

function onRun() {
  if (mode.value === 'yolo') handleDetect()
  else if (useHybridSegment.value) handleSegmentHybrid()
  else handleSegment()
}

async function onClearRegions() {
  if (mode.value === 'yolo') await handleClearDetections()
  else clearSegments()
}

function onReset() {
  handleReset(originalImageUrl.value)
  clearSegments()
  clearExtracted()
  clearBbox()
  clearPolygon()
}

async function onExtract() {
  if (selectedMaskId.value === null) return
  await handleExtract(selectedMaskId.value)
}

async function onPaste() {
  const seg = selectedMaskId.value !== null
    ? segments.value.find(s => s.mask_id === selectedMaskId.value)
    : null
  const targetBbox = seg ? seg.bbox : { x1: 0, y1: 0, x2: naturalSize.value.w, y2: naturalSize.value.h }
  await handlePaste({ targetBbox })
}

watch(mode, () => {
  selectedBboxIds.value = []
  selectedMaskId.value = null
  clearExtracted()
  clearBbox()
  clearPolygon()
  replaceEngine.value = 'lama'
  diffusionPrompt.value = ''
})

const combinedError = computed(() =>
  mlError.value || detectError.value || samError.value || assetOpError.value
)

function clearAllErrors() {
  mlError.value = ''
  detectError.value = ''
  samError.value = ''
  assetOpError.value = ''
}

watch(imageUrl, (url) => {
  if (url && !currentImageUrl.value) currentImageUrl.value = url
}, { immediate: true })

onMounted(() => {
  fetchHistory()
  fetchAssets(true)
  function onKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.key === 'z') { e.preventDefault(); handleUndo() }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); handleRedo() }
  }
  document.addEventListener('keydown', onKeydown)
  onUnmounted(() => document.removeEventListener('keydown', onKeydown))
})
</script>
<style scoped>
@import '@/styles/views/editorvue.css';
</style>