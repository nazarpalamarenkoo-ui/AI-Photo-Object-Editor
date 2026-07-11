<template>
  <div class="asset-library">
    <div class="section-header">
      Asset library
      <span class="badge" v-if="assets.length">{{ assets.length }}</span>
    </div>

    <div v-if="assetsError" class="lib-error">
      {{ assetsError }}
    </div>

    <div v-if="!assets.length && !assetsLoading" class="lib-empty">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <path d="M9 9h6v6H9z"/>
      </svg>
      <p>No extracted objects yet</p>
      <span>Cut something out in SAM mode and it'll land here.</span>
    </div>

    <div v-else class="lib-grid">
      <div
        v-for="asset in assets"
        :key="asset.asset_id"
        :class="['lib-card', { active: asset.asset_id === selectedAssetId, busy: deletingId === asset.asset_id }]"
        @click="$emit('select', asset)"
      >
        <div class="lib-thumb">
          <img
            v-if="thumbUrls[asset.asset_id]"
            :src="thumbUrls[asset.asset_id]"
            :alt="asset.label ?? 'Extracted object'"
          />
          <div v-else class="lib-thumb-placeholder">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <path d="M9 9h6v6H9z"/>
            </svg>
          </div>
          <span v-if="asset.asset_id === selectedAssetId" class="lib-check">✓</span>
        </div>

        <div class="lib-meta">
          <template v-if="renamingDraftId === asset.asset_id">
            <input
              ref="renameInputRef"
              class="lib-rename-input"
              v-model="renameDraft"
              @click.stop
              @keydown.enter="commitRename(asset.asset_id)"
              @keydown.esc="renamingDraftId = null"
              @blur="commitRename(asset.asset_id)"
            />
          </template>
          <template v-else>
            <span class="lib-label" :title="asset.label ?? undefined" @click.stop="startRename(asset)">
              {{ asset.label || `Object #${asset.asset_id.slice(0, 6)}` }}
            </span>
          </template>
          <span class="lib-sub">{{ formatSize(asset.object_size) }}</span>
        </div>

        <button
          class="lib-delete"
          title="Delete asset"
          :disabled="deletingId === asset.asset_id"
          @click.stop="$emit('delete', asset.asset_id)"
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
          </svg>
        </button>
      </div>
    </div>

    <button
      v-if="assetsHasMore && assets.length > 0"
      class="lib-load-more"
      :disabled="assetsLoading"
      @click="$emit('load-more')"
    >
      {{ assetsLoading ? 'Loading…' : 'Load more' }}
    </button>
    <div v-else-if="assetsLoading" class="lib-loading">Loading…</div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'
import type { Asset } from '@/types/Index'

defineProps<{
  assets: Asset[]
  thumbUrls: Record<string, string>
  selectedAssetId: string | null
  assetsLoading: boolean
  assetsError: string
  assetsHasMore: boolean
  deletingId: string | null
}>()

const emit = defineEmits<{
  select: [asset: Asset]
  rename: [assetId: string, label: string]
  delete: [assetId: string]
  'load-more': []
}>()

const renamingDraftId = ref<string | null>(null)
const renameDraft = ref('')
const renameInputRef = ref<HTMLInputElement[] | HTMLInputElement | null>(null)

async function startRename(asset: Asset) {
  renamingDraftId.value = asset.asset_id
  renameDraft.value = asset.label ?? ''
  await nextTick()
  const el = Array.isArray(renameInputRef.value) ? renameInputRef.value[0] : renameInputRef.value
  el?.focus()
}

function commitRename(assetId: string) {
  if (renamingDraftId.value !== assetId) return
  const label = renameDraft.value.trim()
  renamingDraftId.value = null
  if (label) emit('rename', assetId, label)
}

function formatSize(size: unknown): string {
  if (Array.isArray(size) && size.length === 2) {
    return `${Math.round(size[0])}×${Math.round(size[1])}`
  }
  return ''
}
</script>

<style scoped>
@import '@/styles/components/assetlibrary.css';
</style>