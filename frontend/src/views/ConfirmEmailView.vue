<template>
  <div class="auth-page">
    <div class="auth-card">

      <div v-if="status === 'loading'" class="status-block">
        <ProgressSpinner />
        <p>Confirming your email...</p>
      </div>

      <div v-else-if="status === 'success'" class="status-block">
        <i class="pi pi-check-circle status-icon success" />
        <h2>Email confirmed!</h2>
        <p>Your account is ready. Redirecting...</p>
      </div>

      <div v-else-if="status === 'error'" class="status-block">
        <i class="pi pi-times-circle status-icon error" />
        <h2>Confirmation failed</h2>
        <p class="error-text">{{ errorMessage }}</p>
        <div class="action-btns">
          <Button label="Back to login" @click="router.push({ name: 'login' })" />
          <Button label="Register again" text @click="router.push({ name: 'register' })" />
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import ProgressSpinner from 'primevue/progressspinner'
import Button from 'primevue/button'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

type Status = 'loading' | 'success' | 'error'

const status = ref<Status>('loading')
const errorMessage = ref('')

onMounted(async () => {
  const token = route.params.token as string

  if (!token) {
    status.value = 'error'
    errorMessage.value = 'Invalid confirmation link'
    return
  }

  try {
    await auth.confirmEmail(token)
    status.value = 'success'
    setTimeout(() => router.push({ name: 'dashboard' }), 1500)
  } catch (e: any) {
    status.value = 'error'
    errorMessage.value = e.response?.data?.detail ?? 'Link is invalid or has expired'
  }
})
</script>

<style>
@import '../styles/views/registervue.css';
</style>