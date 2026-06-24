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
          :viewBox="`0 0 ${naturalSize.w} ${naturalSize.h}`"
        >
          <rect
            v-for="det in detections"
            :key="det.bbox_id"
            :x="det.x1" :y="det.y1"
            :width="det.x2 - det.x1"
            :height="det.y2 - det.y1"
            :class="['bbox-rect', { selected: selectedBboxIds.includes(det.bbox_id) }]"
            @click="$emit('toggle-selection', det.bbox_id)"
          />
          <text
            v-for="det in detections"
            :key="`lbl-${det.bbox_id}`"
            :x="det.x1 + 5"
            :y="det.y1 - 5"
            class="bbox-label"
          >{{ det.detected_class }} · {{ (det.confidence * 100).toFixed(0) }}%</text>
        </svg>
      </div>
    </div>

    <div class="canvas-toolbar">
      <button class="canvas-action-btn primary" :disabled="detecting" @click="$emit('detect')">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        {{ detecting ? 'Detecting…' : 'Run detection' }}
      </button>
      <button
        v-if="detections.length > 0"
        class="canvas-action-btn ghost"
        @click="$emit('clear')"
      >
        Clear
      </button>
      <div class="conf-control">
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
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { Image, Detection } from '../types/Index'

defineProps<{
  image: Image | null
  imageUrl: string
  imageLoaded: boolean
  naturalSize: { w: number; h: number }
  zoom: number
  detections: Detection[]
  selectedBboxIds: number[]
  detecting: boolean
  confThreshold: number
}>()

const emit = defineEmits<{
  'image-load': [size: { w: number; h: number }]
  'toggle-selection': [bboxId: number]
  detect: []
  clear: []
  'update:confThreshold': [value: number]
}>()

const imageRef = ref<HTMLImageElement | null>(null)

function onLoad() {
  if (imageRef.value) {
    emit('image-load', {
      w: imageRef.value.naturalWidth,
      h: imageRef.value.naturalHeight,
    })
  }
}
</script>

<style scoped>
@import '@/styles/components/editorcanvas.css';
</style>