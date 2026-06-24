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
        :zoom="zoom"
        :canUndo="canUndo"
        :mlLoading="mlLoading"
        @zoom="zoom = $event"
        @update:preset="modelPreset = $event"
        @undo="handleUndo"
        @redo="handleRedo"
        @reset="handleReset(imageUrl)"
      />

      <EditorCanvas
        :image="image"
        :imageUrl="currentImageUrl"
        :imageLoaded="imageLoaded"
        :naturalSize="naturalSize"
        :zoom="zoom"
        :detections="detections"
        :selectedBboxIds="selectedBboxIds"
        :detecting="detecting"
        :confThreshold="confThreshold"
        @image-load="onImageLoad"
        @toggle-selection="toggleSelection"
        @detect="handleDetect"
        @clear="handleClearDetections()"
        @update:confThreshold="confThreshold = $event"
      />

      <EditorSidebar
        :detections="detections"
        :selectedBboxIds="selectedBboxIds"
        v-model:useEdgeBlending="useEdgeBlending"
        :mlLoading="mlLoading"
        :replacementFile="replacementFile"
        :resultUrl="currentImageUrl"
        :mlError="combinedError"
        :saveLoading="saveLoading"
        :savedSuccess="savedSuccess"
        :history="history"
        @toggle-selection="toggleSelection"
        @remove="handleRemove(selectedBboxIds, modelPreset)"
        @replace="handleReplace(selectedBboxIds, modelPreset)"
        @replacement-select="onReplacementSelect"
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
import { PRESETS } from '@/api/ml'
import type { LdmConfig } from '@/types/Index'

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

const combinedError = computed(() => mlError.value || detectError.value)

function clearAllErrors() {
  mlError.value = ''
  detectError.value = ''
}

watch(imageUrl, (url) => {
  if (url && !currentImageUrl.value) {
    currentImageUrl.value = url
  }
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