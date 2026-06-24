<template>
  <div class="dashboard" :class="{ dark: isDark }">

    <DashboardNavbar
      :username="auth.user?.username ?? ''"
      :isDark="isDark"
      @toggle-theme="toggleTheme"
      @profile="router.push({ name: 'profile' })"
      @logout="handleLogout"
    />

    <DashboardGrid
      :images="images"
      :imageUrls="imageUrls"
      :loading="loading"
      :uploading="uploading"
      :formatDate="formatDate"
      @file-change="handleFileChange"
      @open="router.push({ name: 'image-editor', params: { id: $event } })"
      @delete="handleDelete"
    />

  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useDashboard } from '@/composables/useDashboard'

import DashboardNavbar from '@/components/DashboardNavbar.vue'
import DashboardGrid from '@/components/DashboardGrid.vue'

import '@/styles/views/dashboardvue.css'

const router = useRouter()
const auth = useAuthStore()

const isDark = ref(localStorage.getItem('theme') !== 'light')

function toggleTheme() {
  isDark.value = !isDark.value
  localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
}

function handleLogout() {
  auth.logout()
  router.push({ name: 'login' })
}

const {
  images,
  imageUrls,
  loading,
  uploading,
  handleFileChange,
  handleDelete,
  formatDate,
} = useDashboard()
</script>