<template>
  <header class="navbar">
    <div class="navbar-left">
      <div class="logo-mark">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <rect x="1" y="1" width="8" height="8" rx="1.5" fill="currentColor" opacity="0.9"/>
          <rect x="11" y="1" width="8" height="8" rx="1.5" fill="currentColor" opacity="0.5"/>
          <rect x="1" y="11" width="8" height="8" rx="1.5" fill="currentColor" opacity="0.5"/>
          <rect x="11" y="11" width="8" height="8" rx="1.5" fill="currentColor" opacity="0.9"/>
        </svg>
      </div>
      <span class="navbar-logo">ImageEditor</span>
      <span class="version-badge">v1.0</span>
    </div>

    <div class="navbar-right">
      <button class="theme-toggle" @click="$emit('toggle-theme')" :title="isDark ? 'Light mode' : 'Dark mode'">
        <svg v-if="isDark" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      </button>
      <div class="nav-divider"/>
      <span class="navbar-user">{{ username }}</span>
      <button class="nav-icon-btn" @click="$emit('profile')" title="Profile">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
          <circle cx="12" cy="7" r="4"/>
        </svg>
      </button>
      <button class="nav-icon-btn logout" @click="$emit('logout')" title="Sign out">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
          <polyline points="16 17 21 12 16 7"/>
          <line x1="21" y1="12" x2="9" y2="12"/>
        </svg>
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
defineProps<{
  username: string
  isDark: boolean
}>()

defineEmits<{
  'toggle-theme': []
  profile: []
  logout: []
}>()
</script>

<style scoped>
@import '@/styles/components/dashboardnavbar.css';
</style>