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

    <div class="sidebar-section" v-if="regions.length > 0">
      <div class="section-header">
        {{ mode === 'yolo' ? 'Detected objects' : 'Segments' }}
        <span class="badge">{{ regions.length }}</span>
      </div>
      <div class="detection-list">
        <div
          v-for="r in regions"
          :key="r.id"
          :class="['det-item', { selected: selectedIds.includes(r.id) }]"
          @click="$emit('toggle-selection', r.id)"
        >
          <span
            class="det-dot"
            :style="{ background: selectedIds.includes(r.id) ? '#ff6b6b' : '#b3f000' }"
          />
          <span class="det-class">{{ r.label }}</span>
          <span class="det-conf" v-if="r.confidence != null">{{ (r.confidence * 100).toFixed(0) }}%</span>
        </div>
      </div>
      <div class="selection-hint" v-if="selectedIds.length > 0">
        {{ selectedIds.length }} selected
      </div>
    </div>

    <div class="sidebar-section" v-if="mode === 'yolo' && selectedIds.length > 0">
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

        <template v-if="selectedIds.length === 1">
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

    <div class="sidebar-section" v-if="mode === 'sam' && selectedIds.length === 1">
      <div class="section-header">Operations</div>

      <div class="toggle-row">
        <span class="toggle-label">Edge feather</span>
        <button
          :class="['toggle-btn', { on: useEdgeBlending }]"
          @click="$emit('update:useEdgeBlending', !useEdgeBlending)"
        >{{ useEdgeBlending ? 'On' : 'Off' }}</button>
      </div>

      <div class="action-stack">
        <button class="action-btn danger" :disabled="mlLoading" @click="$emit('sam-remove')">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
          </svg>
          {{ mlLoading ? 'Processing…' : 'Remove object' }}
        </button>
        <button
          v-if="selectedAssetId"
          class="action-btn accent"
          :disabled="mlLoading"
          @click="$emit('sam-replace-asset')"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="1 4 1 10 7 10"/>
            <path d="M3.51 15a9 9 0 1 0 .49-4"/>
          </svg>
          Replace with asset
        </button>
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
          @click="$emit('sam-replace')"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="1 4 1 10 7 10"/>
            <path d="M3.51 15a9 9 0 1 0 .49-4"/>
          </svg>
          Swap object
        </button>

        <div class="action-divider" />

        <button class="action-btn ghost" :disabled="mlLoading" @click="$emit('extract')">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <path d="M9 9h6v6H9z"/>
          </svg>
          Extract object
        </button>

        <button
          v-if="selectedAssetId"
          class="action-btn accent"
          :disabled="mlLoading"
          @click="$emit('paste')"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
            <rect x="8" y="2" width="8" height="4" rx="1"/>
          </svg>
          Paste back
        </button>
      </div>
    </div>

  <div class="sidebar-section" v-if="mode === 'sam'">
    <div class="section-header">Prompt segmentation</div>

    <div class="tool-group mode-switch" style="margin-bottom: 8px;">
      <button
        :class="['tool-btn', { active: promptMode === 'points' }]"
        @click="$emit('update:promptMode', promptMode === 'points' ? null : 'points')"
      >Points</button>
      <button
        :class="['tool-btn', { active: promptMode === 'box' }]"
        @click="$emit('update:promptMode', promptMode === 'box' ? null : 'box')"
      >Box</button>
      <button
        :class="['tool-btn', { active: promptMode === 'polygon' }]"
        @click="$emit('update:promptMode', promptMode === 'polygon' ? null : 'polygon')"
      >Polygon</button>
    </div>

    <div v-if="promptMode === 'points'" class="toggle-row">
        <span class="toggle-label">Point type</span>
        <button
          :class="['toggle-btn', { on: promptLabel === 1 }]"
          @click="$emit('update:promptLabel', 1)"
        >Foreground</button>
        <button
          :class="['toggle-btn', { on: promptLabel === 0 }]"
          @click="$emit('update:promptLabel', 0)"
        >Background</button>
      </div>

      <div v-if="promptMode === 'points' && promptPoints.length" class="selection-hint">
        {{ promptPoints.length }} точок обрано
      </div>
      <div v-if="promptMode === 'box' && promptBbox" class="selection-hint">
        Bbox обрано
      </div>
      <div v-if="promptMode === 'polygon' && promptPolygonPoints.length" class="selection-hint">
        {{ promptPolygonPoints.length }} точок полігону{{ canRunPolygon ? '' : ' (потрібно мінімум 3)' }}
      </div>

    <div class="action-stack" v-if="promptMode === 'points' || promptMode === 'box'">
      <button
        class="action-btn ghost"
        :disabled="promptMode === null || (!promptPoints.length && !promptBbox)"
        @click="$emit('clear-prompt')"
      >Очистити prompt</button>

      <button
        class="action-btn accent"
        :disabled="mlLoading || (!promptPoints.length && !promptBbox)"
        @click="$emit('run-prompt-segment')"
      >{{ mlLoading ? 'Processing…' : 'Segment by prompt' }}</button>
    </div>

    <div class="action-stack" v-if="promptMode === 'polygon'">
      <button
        class="action-btn ghost"
        :disabled="!promptPolygonPoints.length"
        @click="$emit('clear-polygon')"
      >Очистити полігон</button>

      <button
        class="action-btn accent"
        :disabled="mlLoading || !canRunPolygon"
        @click="$emit('run-polygon-segment')"
      >{{ mlLoading ? 'Processing…' : 'Run polygon' }}</button>
    </div>
  </div>
    <div class="sidebar-section">
      <AssetLibrary
        :assets="assets"
        :thumbUrls="assetThumbUrls"
        :selectedAssetId="selectedAssetId"
        :assetsLoading="assetsLoading"
        :assetsError="assetsError"
        :assetsHasMore="assetsHasMore"
        :deletingId="deletingAssetId"
        @select="asset => $emit('select-asset', asset)"
        @rename="(assetId, label) => $emit('rename-asset', assetId, label)"
        @delete="assetId => $emit('delete-asset', assetId)"
        @load-more="$emit('load-more-assets')"
      />
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
import type { RegionItem, EditingMode, Asset, Bbox, PromptMode, PolygonPoint } from '@/types/Index'
import AssetLibrary from '@/components/AssetsLibrary.vue'

const props = defineProps<{
  mode: EditingMode
  regions: RegionItem[]
  selectedIds: number[]
  useEdgeBlending: boolean
  mlLoading: boolean
  replacementFile: File | null

  selectedAssetId: string | null
  resultUrl: string
  mlError: string
  saveLoading: boolean
  savedSuccess: boolean
  history: string[]
  assets: Asset[]
  assetThumbUrls: Record<string, string>
  assetsLoading: boolean
  assetsError: string
  assetsHasMore: boolean
  deletingAssetId: string | null
  promptMode: PromptMode
  promptLabel: 0 | 1
  promptPoints: { x: number; y: number; label: 0 | 1 }[]
  promptBbox: Bbox | null
  promptPolygonPoints: PolygonPoint[]
  canRunPolygon: boolean
}>()

defineEmits<{
  'toggle-selection': [id: number]
  'update:useEdgeBlending': [value: boolean]
  remove: []
  replace: []
  'sam-remove': []
  'sam-replace': []
  'sam-replace-asset': []
  'update:promptMode': [value: PromptMode]
  'update:promptLabel': [value: 0 | 1]
  'clear-prompt': []
  'run-prompt-segment': []
  'run-polygon-segment': []
  'clear-polygon': []
  extract: []
  paste: []
  'replacement-select': [event: Event]
  'clear-error': []
  save: []
  'select-asset': [asset: Asset]
  'rename-asset': [assetId: string, label: string]
  'delete-asset': [assetId: string]
  'load-more-assets': []
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