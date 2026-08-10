<template>
  <aside class="toolbar">
    <div class="tool-group">
      <button
        v-for="tool in tools"
        :key="tool.id"
        :class="['tool-btn', { active: activeTool === tool.id }]"
        @click="$emit('update:activeTool', activeTool === tool.id ? '' : tool.id)"
        :title="`${tool.label} [${tool.shortcut}]`"
      >
        <component :is="tool.icon" />
        <span class="tool-shortcut">{{ tool.shortcut }}</span>
      </button>
    </div>

    <div class="tool-divider" />

    <div class="tool-group mode-switch">
      <button
        :class="['tool-btn', { active: mode === 'yolo' }]"
        :disabled="busy"
        title="Detection mode (YOLO)"
        @click="$emit('update:mode', 'yolo')"
      >
        YOLO
      </button>
      <button
        :class="['tool-btn', { active: mode === 'sam' }]"
        :disabled="busy"
        title="Segmentation mode (SAM)"
        @click="$emit('update:mode', 'sam')"
      >
        SAM
      </button>
    </div>

    <div v-if="mode === 'sam'" class="tool-group">
      <button
        :class="['tool-btn', 'hybrid-btn', { active: useHybrid }]"
        :disabled="busy"
        title="Hybrid segmentation: YOLO detects common objects first, SAM2 batch-segments them in one encoder pass, sparse SAM2 auto-pass catches the rest — faster than full auto segmentation"
        @click="$emit('update:useHybrid', !useHybrid)"
      >
        Hybrid
      </button>
    </div>

    <div class="tool-divider" />

    <div v-if="mode === 'sam'" class="tool-group mode-switch">
      <button
        :class="['tool-btn', { active: replaceEngine === 'lama' }]"
        :disabled="busy"
        title="Remove/replace via LaMa inpainting + compositing"
        @click="$emit('update:replaceEngine', 'lama')"
      >
        LaMa
      </button>
      <button
        :class="['tool-btn', { active: replaceEngine === 'diffusion' }]"
        :disabled="busy"
        title="Replace via diffusion (SD-inpaint + IP-Adapter) — blends into scene lighting/perspective"
        @click="$emit('update:replaceEngine', 'diffusion')"
      >
        Diffusion
      </button>
    </div>

    <div v-if="mode === 'sam' && replaceEngine === 'diffusion'" class="tool-group prompt-group" ref="promptRef">
      <button
        :class="['tool-btn', 'prompt-btn', { active: promptOpen || !!diffusionPrompt }]"
        :disabled="busy"
        title="Diffusion prompt"
        @click="togglePrompt"
      >
        Prompt
        <span v-if="diffusionPrompt" class="prompt-dot" />
      </button>

      <div v-if="promptOpen" class="settings-dropdown prompt-dropdown">
        <div class="settings-title">Diffusion Prompt</div>

        <div class="settings-field prompt-field">
          <label>Description</label>
          <textarea
            class="settings-input diffusion-prompt-input"
            v-model="localPrompt"
            :disabled="busy"
            rows="4"
            placeholder="Describe the replacement, e.g. 'a red vintage armchair, soft studio light'"
            title="Prompt describing the desired diffusion result"
            @keydown.enter.exact.prevent="applyPrompt"
          />
        </div>

        <button class="apply-settings-btn" @click="applyPrompt">
          Apply
        </button>
      </div>
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
        title="Fit to screen"
        @click="$emit('zoom', fitZoom)"
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
        :disabled="!canUndo || mlLoading || busy"
        @click="$emit('undo')"
      >
        <UndoIcon />
      </button>

      <button
        class="tool-btn"
        title="Redo [Ctrl+Y]"
        :disabled="mlLoading || busy"
        @click="$emit('redo')"
      >
        <RedoIcon />
      </button>

      <button
        class="tool-btn"
        title="Reset to original"
        :disabled="mlLoading || busy"
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

import type { LdmConfig, EditingMode, ReplaceEngine } from '@/types/Index'
import { PRESETS } from '@/api/ml'

const props = withDefaults(defineProps<{
  activeTool: string
  zoom: number
  fitZoom?: number
  canUndo: boolean
  mlLoading: boolean
  modelConfig: LdmConfig
  mode: EditingMode
  busy: boolean
  useHybrid: boolean
  replaceEngine: ReplaceEngine
  diffusionPrompt: string
}>(), {
  fitZoom: 1,
})

const emit = defineEmits<{
  'update:activeTool': [value: string]
  zoom: [value: number]
  undo: []
  redo: []
  reset: []
  'update:modelConfig': [value: LdmConfig]
  'update:mode': [value: EditingMode]
  'update:useHybrid': [value: boolean]
  'update:replaceEngine': [value: ReplaceEngine]
  'update:diffusionPrompt': [value: string]
}>()

const settingsOpen = ref(false)
const settingsRef = ref<HTMLElement | null>(null)

const promptOpen = ref(false)
const promptRef = ref<HTMLElement | null>(null)
const localPrompt = ref(props.diffusionPrompt)

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

watch(
  () => props.diffusionPrompt,
  (value) => {
    localPrompt.value = value
  }
)

function togglePrompt() {
  if (!promptOpen.value) {
    localPrompt.value = props.diffusionPrompt
  }
  promptOpen.value = !promptOpen.value
  settingsOpen.value = false
}

function applyPrompt() {
  emit('update:diffusionPrompt', localPrompt.value)
  promptOpen.value = false
}

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

watch(settingsOpen, (open) => {
  if (open) promptOpen.value = false
})

function onClickOutside(e: MouseEvent) {
  if (
    settingsRef.value &&
    !settingsRef.value.contains(e.target as Node)
  ) {
    settingsOpen.value = false
  }

  if (
    promptRef.value &&
    !promptRef.value.contains(e.target as Node)
  ) {
    promptOpen.value = false
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