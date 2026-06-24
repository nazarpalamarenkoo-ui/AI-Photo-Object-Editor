<template>
  <div class="auth-page">
    <div class="auth-card">

      <div v-if="status === 'success'" class="email-sent">
        <i class="pi pi-envelope email-icon" />
        <h2>Check your inbox</h2>
        <p>If that email exists, a reset link has been sent.</p>
        <Button label="Back to login" text @click="router.push({ name: 'login' })" />
      </div>

      <template v-else>
        <h1 class="auth-title">Forgot password</h1>
        <p class="auth-subtitle">Enter your email and we'll send you a reset link</p>

        <form class="auth-form" @submit.prevent="submit">
          <div class="field">
            <label for="email">Email</label>
            <InputText
              id="email"
              v-model="email"
              type="email"
              placeholder="you@example.com"
              :disabled="status === 'loading'"
              @keyup.enter="submit"
            />
          </div>

          <small v-if="errorMessage" class="error-msg">{{ errorMessage }}</small>

          <Button
            type="submit"
            label="Send reset link"
            :loading="status === 'loading'"
            class="submit-btn"
          />
        </form>

        <p class="auth-footer">
          <RouterLink to="/login">Back to login</RouterLink>
        </p>
      </template>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import InputText from 'primevue/inputtext'
import Button from 'primevue/button'
import { authApi } from '@/api/auth'

const router = useRouter()

type Status = 'idle' | 'loading' | 'success' | 'error'

const email = ref('')
const status = ref<Status>('idle')
const errorMessage = ref('')

async function submit() {
  errorMessage.value = ''

  if (!email.value.trim()) {
    errorMessage.value = 'Please enter your email'
    return
  }

  status.value = 'loading'

  try {
    await authApi.recoverPassword(email.value.trim())
    status.value = 'success'
  } catch (e: any) {
    status.value = 'error'
    errorMessage.value = e.response?.data?.detail ?? 'Something went wrong. Please try again.'
  }
}
</script>

<style scoped>
@import '../styles/views/registervue.css';
</style>