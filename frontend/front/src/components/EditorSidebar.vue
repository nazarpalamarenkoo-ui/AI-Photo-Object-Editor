<template>
  <aside class="sidebar">

    <div class="sidebar-section" v-if="history.length > 0">
      <div class="section-header">History</div>
      <div class="history-list">
        <div
          v-for="(label, idx) in history"
          :key="idx"
          :class="['history-item', { current: idx === 0 }]"
        >
          <span class="history-dot"/>
          <span class="history-label">{{ label }}</span>
        </div>
      </div>
    </div>

    <div class="sidebar-section" v-if="detections.length > 0">
      <div class="section-header">
        Detected segments
        <span class="badge">{{ detections.length }}</span>
      </div>
      <div class="detection-list">
        <div
          v-for="det in detections"
          :key="det.bbox_id"
          :class="['det-item', { selected: selectedBboxIds.includes(det.bbox_id) }]"
          @click="$emit('toggle-selection', det.bbox_id)"
        >
          <span
            class="det-dot"
            :style="{ background: selectedBboxIds.includes(det.bbox_id) ? '#ff6b6b' : '#b3f000' }"
          />
          <span class="det-class">{{ det.detected_class }}</span>
          <span class="det-conf">{{ (det.confidence * 100).toFixed(0) }}%</span>
        </div>
      </div>
      <div class="selection-hint" v-if="selectedBboxIds.length > 0">
        {{ selectedBboxIds.length }} selected
      </div>
    </div>

    <div class="sidebar-section" v-if="selectedBboxIds.length > 0">
      <div class="section-header">Operations</div>

      <div class="toggle-row">
        <span class="toggle-label">Edge feather</span>
        <button
          :class="['toggle-btn', { on: useEdgeBlending }]"
          @click="$emit('update:useEdgeBlending', !useEdgeBlending)"
        >{{ useEdgeBlending ? 'On' : 'Off' }}</button>
      </div>

      <div class="action-stack">
        <button class="action-btn danger" :disabled="mlLoading" @click="$emit('remove')">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
          </svg>
          {{ mlLoading ? 'Processing…' : 'Remove selected' }}
        </button>

        <template v-if="selectedBboxIds.length === 1">
          <div class="replace-area">
            <label class="replace-upload">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              {{ replacementFile ? replacementFile.name : 'Choose replacement image' }}
              <input type="file" accept="image/*" style="display:none" @change="$emit('replacement-select', $event)"/>
            </label>
          </div>
          <button
            v-if="replacementFile"
            class="action-btn accent"
            :disabled="mlLoading"
            @click="$emit('replace')"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <polyline points="1 4 1 10 7 10"/>
              <path d="M3.51 15a9 9 0 1 0 .49-4"/>
            </svg>
            Swap object
          </button>
        </template>
      </div>
    </div>

    <div class="sidebar-section" v-if="resultUrl">
      <div class="section-header">Result</div>
      <img :src="resultUrl" alt="Result" class="result-preview"/>

      <button class="download-btn" @click="handleDownload">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download
      </button>

      <button
        class="action-btn save-btn"
        :disabled="saveLoading || savedSuccess"
        @click="$emit('save')"
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
          <polyline points="17 21 17 13 7 13 7 21"/>
          <polyline points="7 3 7 8 15 8"/>
        </svg>
        {{ savedSuccess ? 'Saved ✓' : saveLoading ? 'Saving…' : 'Save to workspace' }}
      </button>
    </div>

    <div class="sidebar-section" v-if="mlError">
      <div class="error-msg">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        {{ mlError }}
        <button @click="$emit('clear-error')" class="err-close">✕</button>
      </div>
    </div>

  </aside>
</template>

<script setup lang="ts">
import type { Detection } from '@/types/Index'

const props = defineProps<{
  detections: Detection[]
  selectedBboxIds: number[]
  useEdgeBlending: boolean
  mlLoading: boolean
  replacementFile: File | null
  resultUrl: string
  mlError: string
  saveLoading: boolean
  savedSuccess: boolean
  history: string[]
}>()

defineEmits<{
  'toggle-selection': [bboxId: number]
  'update:useEdgeBlending': [value: boolean]
  remove: []
  replace: []
  'replacement-select': [event: Event]
  'clear-error': []
  'save': []
}>()

async function handleDownload() {
  if (!props.resultUrl) return
  try {
    const res = await fetch(props.resultUrl)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'result.jpg'
    a.click()
    URL.revokeObjectURL(url)
  } catch {
    window.open(props.resultUrl, '_blank')
  }
}
</script>

<style scoped>
@import '@/styles/components/editorsidebar.css';
</style>