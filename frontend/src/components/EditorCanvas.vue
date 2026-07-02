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
            v-for="r in regions"
            :key="r.id"
            :x="r.bbox.x1" :y="r.bbox.y1"
            :width="r.bbox.x2 - r.bbox.x1"
            :height="r.bbox.y2 - r.bbox.y1"
            :class="['bbox-rect', { selected: selectedIds.includes(r.id) }]"
            @click="$emit('toggle-selection', r.id)"
          />
          <text
            v-for="r in regions"
            :key="`lbl-${r.id}`"
            :x="r.bbox.x1 + 5"
            :y="r.bbox.y1 - 5"
            class="bbox-label"
          >{{ r.label }}{{ r.confidence != null ? ` · ${(r.confidence * 100).toFixed(0)}%` : '' }}</text>
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
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Image, RegionItem, EditingMode } from '@/types/Index'

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
}>(), {
  regions: () => [],
  selectedIds: () => [],
})

const emit = defineEmits<{
  'image-load': [size: { w: number; h: number }]
  'toggle-selection': [id: number]
  run: []
  clear: []
  'update:confThreshold': [value: number]
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
</script>

<style scoped>
@import '@/styles/components/editorcanvas.css';
</style>