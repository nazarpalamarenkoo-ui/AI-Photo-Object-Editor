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
        :zoom="zoom"
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
        @image-load="onImageLoad"
        @toggle-selection="onToggleSelection"
        @run="onRun"
        @clear="onClearRegions"
        @update:confThreshold="confThreshold = $event"
      />

      <EditorSidebar
        :mode="mode"
        :regions="regions"
        :selectedIds="selectedIds"
        v-model:useEdgeBlending="activeUseEdgeBlending"
        :mlLoading="mlLoading"
        :replacementFile="activeReplacementFile"
        :extractedUrl="extractedUrl"
        :resultUrl="currentImageUrl"
        :mlError="combinedError"
        :saveLoading="saveLoading"
        :savedSuccess="savedSuccess"
        :history="history"
        @toggle-selection="onToggleSelection"
        @remove="handleRemove(selectedBboxIds, modelPreset)"
        @replace="handleReplace(selectedBboxIds, modelPreset)"
        @sam-remove="handleSamRemove(modelPreset)"
        @sam-replace="handleSamReplace(modelPreset)"
        @extract="onExtract"
        @paste="onPaste"
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
import type { LdmConfig, EditingMode, RegionItem } from '@/types/Index'

import EditorNavbar from '@/components/EditorNavbar.vue'
import EditorToolbar from '@/components/EditorToolbar.vue'
import EditorCanvas from '@/components/EditorCanvas.vue'
import EditorSidebar from '@/components/EditorSidebar.vue'

import '@/styles/views/editorvue.css'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const imageId = Number(route.params.id)
const isDark = ref(localStorage.getItem('theme') !== 'light')

function toggleTheme() {
  isDark.value = !isDark.value
  localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
}

const activeTool = ref('select')
const zoom = ref(1)
const modelPreset = ref<LdmConfig>(PRESETS.quality)
const mode = ref<EditingMode>('yolo')

const {
  image, imageUrl, loading,
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
  segments, regions: samRegions, selectedMaskId, segmenting,
  mlError: samError, useEdgeBlending: samUseEdgeBlending, replacementFile: samReplacementFile,
  handleSegment, toggleMaskSelection, handleSamRemove, handleSamReplace,
  onReplacementSelect: onSamReplacementSelect, clearSegments,
} = useSegmentation(imageId, currentImageUrl, history)

const {
  mlError: assetError, extractedUrl,
  handleExtract, handlePaste, clearExtracted,
} = useAssets(imageId, currentImageUrl, history)


const regions = computed<RegionItem[]>(() =>
  mode.value === 'yolo'
    ? (detections.value ?? []).map(d => ({
        id: d.bbox_id, bbox: { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 },
        label: d.detected_class, confidence: d.confidence,
      }))
    : samRegions.value
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
  else handleSegment()
}

async function onClearRegions() {
  if (mode.value === 'yolo') await handleClearDetections()
  else clearSegments()
}

function onReset() {
  handleReset(imageUrl.value)
  clearSegments()
  clearExtracted()
}

async function onExtract() {
  if (selectedMaskId.value === null) return
  await handleExtract(selectedMaskId.value)
}

async function onPaste() {
  if (selectedMaskId.value === null) return
  const seg = segments.value.find(s => s.mask_id === selectedMaskId.value)
  if (!seg) return
  await handlePaste({ targetBbox: seg.bbox })
}

watch(mode, () => {
  selectedBboxIds.value = []
  selectedMaskId.value = null
  clearExtracted()
})

const combinedError = computed(() =>
  mlError.value || detectError.value || samError.value || assetError.value
)

function clearAllErrors() {
  mlError.value = ''
  detectError.value = ''
  samError.value = ''
  assetError.value = ''
}

watch(imageUrl, (url) => {
  if (url && !currentImageUrl.value) currentImageUrl.value = url
}, { immediate: true })

onMounted(() => {
  fetchHistory()
  function onKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.key === 'z') { e.preventDefault(); handleUndo() }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); handleRedo() }
  }
  document.addEventListener('keydown', onKeydown)
  onUnmounted(() => document.removeEventListener('keydown', onKeydown))
})
</script>