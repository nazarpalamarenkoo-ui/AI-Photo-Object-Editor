<template>
  <div class="image-panel">
    <div class="canvas-wrap">
      <div class="canvas-inner" :style="{ transform: `scale(${zoom})`, transformOrigin: 'top left' }">
        <img
          :src="imageUrl"
          :alt="image?.filename"
          class="editor-image"
          ref="imageRef"
          @load="onLoad"
        />
        <svg
          v-if="imageLoaded"
          class="bbox-overlay"
          :class="{ 'prompt-cursor': promptMode !== null }"
          :viewBox="`0 0 ${naturalSize.w} ${naturalSize.h}`"
          @click="onSvgClick"
          @contextmenu="onSvgRightClick"
          @mousedown="onSvgMouseDown"
          @mousemove="onSvgMouseMove"
          @mouseup="onSvgMouseUp"
          @mouseleave="onSvgMouseUp"
        >
          <template v-for="r in regions" :key="r.id">
            <polygon
              v-if="r.points && r.points.length"
              :points="r.points.map(p => `${p.x},${p.y}`).join(' ')"
              :class="['bbox-rect', { selected: selectedIds.includes(r.id) }]"
              @click.stop="promptMode === null && $emit('toggle-selection', r.id)"
            />
            <rect
              v-else
              :x="r.bbox.x1" :y="r.bbox.y1"
              :width="r.bbox.x2 - r.bbox.x1"
              :height="r.bbox.y2 - r.bbox.y1"
              :class="['bbox-rect', { selected: selectedIds.includes(r.id) }]"
              @click.stop="promptMode === null && $emit('toggle-selection', r.id)"
            />
          </template>

          <text
            v-for="r in regions"
            :key="`lbl-${r.id}`"
            :x="r.bbox.x1 + 5"
            :y="r.bbox.y1 - 5"
            class="bbox-label"
          >{{ r.label }}{{ r.confidence != null ? ` · ${(r.confidence * 100).toFixed(0)}%` : '' }}</text>

          <g v-if="promptMode === 'points'">
            <circle
              v-for="(p, idx) in promptPoints"
              :key="`pt-${idx}`"
              :cx="p.x" :cy="p.y" r="6"
              :class="['prompt-point', p.label === 1 ? 'fg' : 'bg']"
            />
          </g>
          <g v-if="promptMode === 'polygon'">
            <polyline
              v-if="promptPolygonPoints.length > 1"
              :points="promptPolygonPoints.map(p => `${p.x},${p.y}`).join(' ')"
              class="prompt-polygon-path"
            />
            <line
              v-if="promptPolygonPoints.length > 2"
              :x1="promptPolygonPoints[promptPolygonPoints.length - 1].x"
              :y1="promptPolygonPoints[promptPolygonPoints.length - 1].y"
              :x2="promptPolygonPoints[0].x"
              :y2="promptPolygonPoints[0].y"
              class="prompt-polygon-close"
            />
            <circle
              v-for="(p, idx) in promptPolygonPoints"
              :key="`poly-${idx}`"
              :cx="p.x" :cy="p.y" r="5"
              class="prompt-polygon-point"
            />
          </g>
          <rect
            v-if="promptMode === 'box' && (promptBbox || drawingBbox)"
            :x="(promptBbox ?? drawingBbox)!.x1"
            :y="(promptBbox ?? drawingBbox)!.y1"
            :width="(promptBbox ?? drawingBbox)!.x2 - (promptBbox ?? drawingBbox)!.x1"
            :height="(promptBbox ?? drawingBbox)!.y2 - (promptBbox ?? drawingBbox)!.y1"
            class="prompt-bbox"
          />
        </svg>
      </div>
    </div>

    <div class="canvas-toolbar">
      <button class="canvas-action-btn primary" :disabled="running" @click="$emit('run')">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        {{ running ? runningLabel : runLabel }}
      </button>

      <button
        v-if="regions.length > 0"
        class="canvas-action-btn ghost"
        @click="$emit('clear')"
      >
        Clear
      </button>

      <div class="conf-control" v-if="mode === 'yolo'">
        <span class="conf-label">Threshold</span>
        <input
          type="range"
          class="conf-slider"
          :value="confThreshold"
          @input="$emit('update:confThreshold', Number(($event.target as HTMLInputElement).value))"
          :min="0.1" :max="0.9" :step="0.05"
        />
        <span class="conf-value">{{ confThreshold }}</span>
      </div>

      <div class="prompt-hint" v-if="promptMode === 'points'">
        left click — foreground · right click / Alt+click — background
      </div>
      <div class="prompt-hint" v-else-if="promptMode === 'box'">
        left click and drag to draw a box
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Image, RegionItem, EditingMode, Bbox, PromptPoint, PromptMode, PolygonPoint } from '@/types/Index'

const props = withDefaults(defineProps<{
  image: Image | null
  imageUrl: string
  imageLoaded: boolean
  naturalSize: { w: number; h: number }
  zoom: number
  regions: RegionItem[]
  selectedIds: number[]
  running: boolean
  mode: EditingMode
  confThreshold: number
  promptMode?: PromptMode
  promptPoints?: PromptPoint[]
  promptPolygonPoints?: PolygonPoint[]
  promptBbox?: Bbox | null
}>(), {
  regions: () => [],
  selectedIds: () => [],
  promptMode: null,
  promptPoints: () => [],
  promptPolygonPoints: () => [],
  promptBbox: null,
})

const emit = defineEmits<{
  'image-load': [size: { w: number; h: number }]
  'toggle-selection': [id: number]
  run: []
  clear: []
  'add-polygon-point': [point: { x: number; y: number }]
  'update:confThreshold': [value: number]
  'add-point': [point: { x: number; y: number; label: 0 | 1 }]
  'set-bbox': [bbox: Bbox]
}>()

const runLabel = computed(() => (props.mode === 'yolo' ? 'Run detection' : 'Run segmentation'))
const runningLabel = computed(() => (props.mode === 'yolo' ? 'Detecting…' : 'Segmenting…'))

const imageRef = ref<HTMLImageElement | null>(null)

function onLoad() {
  if (imageRef.value) {
    emit('image-load', {
      w: imageRef.value.naturalWidth,
      h: imageRef.value.naturalHeight,
    })
  }
}

function svgPoint(e: MouseEvent): { x: number; y: number } | null {
  const svg = (e.currentTarget as SVGSVGElement)
  const rect = svg.getBoundingClientRect()
  const scaleX = props.naturalSize.w / rect.width
  const scaleY = props.naturalSize.h / rect.height
  const x = (e.clientX - rect.left) * scaleX
  const y = (e.clientY - rect.top) * scaleY
  if (x < 0 || y < 0 || x > props.naturalSize.w || y > props.naturalSize.h) return null
  return { x, y }
}

function onSvgClick(e: MouseEvent) {
  if (props.promptMode === 'polygon') {
    const p = svgPoint(e)
    if (!p) return
    emit('add-polygon-point', { x: p.x, y: p.y })
    return
  }
  if (props.promptMode !== 'points') return
  const p = svgPoint(e)
  if (!p) return
  const label = e.altKey ? 0 : 1
  emit('add-point', { x: p.x, y: p.y, label })
}

function onSvgRightClick(e: MouseEvent) {
  if (props.promptMode !== 'points') return
  e.preventDefault()
  const p = svgPoint(e)
  if (!p) return
  emit('add-point', { x: p.x, y: p.y, label: 0 })
}

const drawingBbox = ref<Bbox | null>(null)
const dragStart = ref<{ x: number; y: number } | null>(null)

function onSvgMouseDown(e: MouseEvent) {
  if (props.promptMode !== 'box') return
  const p = svgPoint(e)
  if (!p) return
  dragStart.value = p
  drawingBbox.value = { x1: p.x, y1: p.y, x2: p.x, y2: p.y }
}

function onSvgMouseMove(e: MouseEvent) {
  if (props.promptMode !== 'box' || !dragStart.value) return
  const p = svgPoint(e)
  if (!p) return
  drawingBbox.value = {
    x1: Math.min(dragStart.value.x, p.x),
    y1: Math.min(dragStart.value.y, p.y),
    x2: Math.max(dragStart.value.x, p.x),
    y2: Math.max(dragStart.value.y, p.y),
  }
}

function onSvgMouseUp() {
  if (props.promptMode !== 'box' || !dragStart.value || !drawingBbox.value) {
    dragStart.value = null
    return
  }
  const box = drawingBbox.value
  dragStart.value = null
  drawingBbox.value = null
  if (box.x2 - box.x1 > 3 && box.y2 - box.y1 > 3) {
    emit('set-bbox', box)
  }
}
</script>

<style scoped>
@import '@/styles/components/editorcanvas.css';
</style>