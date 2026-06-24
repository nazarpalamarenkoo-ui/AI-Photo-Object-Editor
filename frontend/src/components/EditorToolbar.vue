```vue
<template>
  <aside class="toolbar">
    <div class="tool-group">
      <button
        v-for="tool in tools"
        :key="tool.id"
        :class="['tool-btn', { active: activeTool === tool.id }]"
        @click="$emit('update:activeTool', tool.id)"
        :title="`${tool.label} [${tool.shortcut}]`"
      >
        <component :is="tool.icon" />
        <span class="tool-shortcut">{{ tool.shortcut }}</span>
      </button>
    </div>

    <div class="tool-divider" />

    <div class="tool-group">
      <button
        class="tool-btn"
        title="Zoom in"
        @click="$emit('zoom', Math.min(zoom + 0.25, 4))"
      >
        <ZoomInIcon />
      </button>

      <button
        class="tool-btn zoom-label"
        title="Reset zoom"
        @click="$emit('zoom', 1)"
      >
        {{ Math.round(zoom * 100) }}%
      </button>

      <button
        class="tool-btn"
        title="Zoom out"
        @click="$emit('zoom', Math.max(zoom - 0.25, 0.25))"
      >
        <ZoomOutIcon />
      </button>
    </div>

    <div class="tool-divider" />

    <div class="tool-group">
      <button
        class="tool-btn"
        title="Undo [Ctrl+Z]"
        :disabled="!canUndo || mlLoading"
        @click="$emit('undo')"
      >
        <UndoIcon />
      </button>

      <button
        class="tool-btn"
        title="Redo [Ctrl+Y]"
        :disabled="mlLoading"
        @click="$emit('redo')"
      >
        <RedoIcon />
      </button>

      <button
        class="tool-btn"
        title="Reset to original"
        :disabled="mlLoading"
        @click="$emit('reset')"
      >
        <ResetIcon />
      </button>
    </div>

    <div class="tool-divider" />

    <div class="tool-group model-settings-group" ref="settingsRef">
      <button
        :class="['tool-btn', { active: settingsOpen }]"
        title="Model settings"
        @click="settingsOpen = !settingsOpen"
      >
        <SettingsIcon />
      </button>

      <div v-if="settingsOpen" class="settings-dropdown">
        <div class="settings-title">
          Model Settings
        </div>

        <div class="settings-field">
          <label>Preset</label>

          <select
            v-model="preset"
            class="settings-select"
            @change="onPresetChange"
          >
            <option value="fast">Fast</option>
            <option value="quality">High Quality</option>
            <option value="custom">Custom</option>
          </select>
        </div>

        <template v-if="preset === 'custom'">
          <div class="settings-field">
            <label>Steps</label>

            <input
              v-model.number="localConfig.ldm_steps"
              type="number"
              min="5"
              max="50"
              class="settings-input"
            />
          </div>

          <div class="settings-field">
            <label>Sampler</label>

            <select
              v-model="localConfig.ldm_sampler"
              class="settings-select"
            >
              <option value="plms">PLMS</option>
              <option value="ddim">DDIM</option>
            </select>
          </div>

          <div class="settings-field">
            <label>HD Strategy</label>

            <select
              v-model="localConfig.hd_strategy"
              class="settings-select"
            >
              <option value="CROP">Crop</option>
              <option value="RESIZE">Resize</option>
              <option value="ORIGINAL">Original</option>
            </select>
          </div>
        </template>

        <button
          class="apply-settings-btn"
          @click="applySettings"
        >
          Apply Settings
        </button>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import {
  ZoomInIcon,
  ZoomOutIcon,
  UndoIcon,
  RedoIcon,
  ResetIcon,
  SettingsIcon,
  tools
} from '../composables/useEditorIcons'

import type { LdmConfig } from '@/types/Index'
import { PRESETS } from '@/api/ml'

const props = defineProps<{
  activeTool: string
  zoom: number
  canUndo: boolean
  mlLoading: boolean
  modelConfig: LdmConfig
}>()

const emit = defineEmits<{
  'update:activeTool': [value: string]
  zoom: [value: number]
  undo: []
  redo: []
  reset: []
  'update:modelConfig': [value: LdmConfig]
}>()

const settingsOpen = ref(false)
const settingsRef = ref<HTMLElement | null>(null)

const preset = ref<'fast' | 'quality' | 'custom'>('quality')

const localConfig = ref<LdmConfig>({
  ...props.modelConfig
})

watch(
  () => props.modelConfig,
  (value) => {
    localConfig.value = { ...value }
  },
  { deep: true }
)

function onPresetChange() {
  if (preset.value === 'fast') {
    localConfig.value = { ...PRESETS.fast }
  }

  if (preset.value === 'quality') {
    localConfig.value = { ...PRESETS.quality }
  }
}

function applySettings() {
  emit('update:modelConfig', {
    ...localConfig.value
  })

  settingsOpen.value = false
}

function onClickOutside(e: MouseEvent) {
  if (
    settingsRef.value &&
    !settingsRef.value.contains(e.target as Node)
  ) {
    settingsOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('mousedown', onClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('mousedown', onClickOutside)
})
</script>

<style scoped>
@import '@/styles/components/editortoolbar.css';
</style>

