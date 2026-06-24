<template>
  <main class="content">

    <div class="topbar">
      <div class="topbar-left">
        <h1 class="page-title">Workspace</h1>
        <span class="image-count" v-if="images.length > 0">{{ images.length }} images</span>
      </div>
      <div class="topbar-right">
        <label class="upload-btn" :class="{ disabled: uploading }">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          {{ uploading ? 'Uploading…' : 'Upload image' }}
          <input
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/webp"
            style="display:none"
            :disabled="uploading"
            @change="$emit('file-change', $event)"
          />
        </label>
        <span class="upload-hint">JPG · PNG · WEBP · max 10MB</span>
      </div>
    </div>

    <div v-if="loading" class="center">
      <div class="spinner"/>
    </div>

    <div v-else-if="images.length === 0" class="empty-state">
      <div class="empty-icon-wrap">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
      </div>
      <p class="empty-title">No images yet</p>
      <p class="empty-sub">Upload your first image to start detecting objects</p>
    </div>

    <div v-else class="image-grid">
      <div
        v-for="image in images"
        :key="image.id"
        class="image-card"
        @click="$emit('open', image.id)"
      >
        <div class="image-thumb">
          <img v-if="imageUrls[image.id]" :src="imageUrls[image.id]" :alt="image.filename"/>
          <div v-else class="thumb-placeholder">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
          </div>
          <div class="card-overlay">
            <span class="open-label">Open editor →</span>
          </div>
        </div>
        <div class="image-info">
          <span class="image-name">{{ image.filename }}</span>
          <span class="image-date">{{ formatDate(image.uploaded_at) }}</span>
        </div>
        <button class="delete-btn" @click.stop="$emit('delete', image.id)" title="Delete">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
            <path d="M10 11v6M14 11v6"/>
            <path d="M9 6V4h6v2"/>
          </svg>
        </button>
      </div>
    </div>

  </main>
</template>

<script setup lang="ts">
import type { Image } from '../types/Index'

defineProps<{
  images: Image[]
  imageUrls: Record<number, string>
  loading: boolean
  uploading: boolean
  formatDate: (dateStr: string) => string
}>()

defineEmits<{
  'file-change': [event: Event]
  open: [imageId: number]
  delete: [imageId: number]
}>()
</script>

<style scoped>
@import '@/styles/components/dashboardgrid.css';
</style>