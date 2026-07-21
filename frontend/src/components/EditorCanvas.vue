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
          <defs>
            <mask
              v-for="r in regions"
              :key="`mask-def-${r.id}`"
              :id="`region-mask-${r.id}`"
              maskUnits="userSpaceOnUse"
              :x="0" :y="0" :width="naturalSize.w" :height="naturalSize.h"
            >
              <image
                v-if="r.mask_url"
                :href="r.mask_url"
                x="0" y="0"
                :width="naturalSize.w" :height="naturalSize.h"
                preserveAspectRatio="none"
              />
            </mask>
          </defs>

          <template v-for="r in regions" :key="r.id">
            <rect
              v-if="r.mask_url"
              :x="0" :y="0"
              :width="naturalSize.w" :height="naturalSize.h"
              :mask="`url(#region-mask-${r.id})`"
              :fill="regionColor(r.id)"
              :class="['region-mask', { selected: selectedIds.includes(r.id) }]"
              pointer-events="none"
            />
            <polygon
              v-else-if="r.points && r.points.length"
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
            <rect
              v-if="r.mask_url"
              :x="r.bbox.x1" :y="r.bbox.y1"
              :width="r.bbox.x2 - r.bbox.x1"
              :height="r.bbox.y2 - r.bbox.y1"
              :class="['bbox-outline', { selected: selectedIds.includes(r.id) }]"
              pointer-events="none"
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
import { ref, computed, watch } from 'vue'
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

const REGION_PALETTE = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
  '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3',
]
function regionColor(id: number): string {
  return REGION_PALETTE[((id % REGION_PALETTE.length) + REGION_PALETTE.length) % REGION_PALETTE.length]
}

const imageRef = ref<HTMLImageElement | null>(null)

const maskCanvasCache = new Map<number, ImageData>()

async function loadMaskImageData(r: RegionItem): Promise<ImageData | null> {
  if (!r.mask_url) return null
  if (maskCanvasCache.has(r.id)) return maskCanvasCache.get(r.id)!

  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = r.mask_url
  try {
    await new Promise<void>((res, rej) => {
      img.onload = () => res()
      img.onerror = () => rej(new Error(`failed to load mask for region ${r.id}`))
    })
  } catch (err) {
    console.warn('[EditorCanvas] mask image failed to load (CORS?)', r.id, r.mask_url, err)
    return null
  }

  const canvas = document.createElement('canvas')
  canvas.width = props.naturalSize.w
  canvas.height = props.naturalSize.h
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(img, 0, 0, props.naturalSize.w, props.naturalSize.h)

  let data: ImageData
  try {
    data = ctx.getImageData(0, 0, props.naturalSize.w, props.naturalSize.h)
  } catch (err) {
    console.warn('[EditorCanvas] canvas tainted reading mask (missing CORS header?)', r.id, r.mask_url, err)
    return null
  }
  maskCanvasCache.set(r.id, data)
  return data
}

watch(
  () => props.regions,
  (regions) => {
    maskCanvasCache.clear()
    regions.forEach(r => { if (r.mask_url) loadMaskImageData(r) })
  },
  { immediate: true, deep: false },
)

function hitTestMaskedRegion(px: number, py: number): number | null {
  for (let i = props.regions.length - 1; i >= 0; i--) {
    const r = props.regions[i]
    if (!r.mask_url) continue
    const data = maskCanvasCache.get(r.id)
    if (!data) continue
    if (px < 0 || py < 0 || px >= data.width || py >= data.height) continue
    const idx = (py * data.width + px) * 4
    const red = data.data[idx]
    const green = data.data[idx + 1]
    const blue = data.data[idx + 2]
    const alpha = data.data[idx + 3]
    const luminance = (0.2125 * red + 0.7154 * green + 0.0721 * blue) / 255
    const visibility = luminance * (alpha / 255)
    if (visibility > 0.15) return r.id
  }
  return null
}

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
  if (props.promptMode === 'points') {
    const p = svgPoint(e)
    if (!p) return
    const label = e.altKey ? 0 : 1
    emit('add-point', { x: p.x, y: p.y, label })
    return
  }
  if (props.promptMode !== null) return

  const p = svgPoint(e)
  if (!p) return
  const id = hitTestMaskedRegion(Math.floor(p.x), Math.floor(p.y))
  if (id !== null) emit('toggle-selection', id)
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