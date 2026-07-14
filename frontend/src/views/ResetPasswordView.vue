<template>
  <div class="auth-page">
    <div class="auth-card">

      <div v-if="status === 'loading'" class="status-block">
        <ProgressSpinner />
        <p>Resetting your password...</p>
      </div>

      <div v-else-if="status === 'success'" class="status-block">
        <i class="pi pi-check-circle status-icon success" />
        <h2>Password updated!</h2>
        <p>You can now log in with your new password. Redirecting...</p>
      </div>

      <div v-else-if="status === 'invalid-token'" class="status-block">
        <i class="pi pi-times-circle status-icon error" />
        <h2>Link expired</h2>
        <p class="error-text">This reset link is invalid or has already been used.</p>
        <div class="action-btns">
          <Button label="Request new link" @click="router.push({ name: 'forgot-password' })" />
          <Button label="Back to login" text @click="router.push({ name: 'login' })" />
        </div>
      </div>

      <template v-else>
        <h2>Set new password</h2>

        <div class="field">
          <label for="password">New password</label>
          <Password
            id="password"
            v-model="newPassword"
            placeholder="At least 8 characters"
            :feedback="true"
            toggle-mask
            :disabled="status === 'submitting'"
          />
        </div>

        <div class="field">
          <label for="confirm">Confirm password</label>
          <Password
            id="confirm"
            v-model="confirmPassword"
            placeholder="Repeat your password"
            :feedback="false"
            toggle-mask
            :disabled="status === 'submitting'"
            @keyup.enter="submit"
          />
        </div>

        <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>

        <Button
          label="Reset password"
          :loading="status === 'submitting'"
          @click="submit"
        />
      </template>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import ProgressSpinner from 'primevue/progressspinner'
import Password from 'primevue/password'
import Button from 'primevue/button'
import { authApi } from '@/api/auth'

const router = useRouter()
const route = useRoute()

type Status = 'idle' | 'loading' | 'submitting' | 'success' | 'invalid-token' | 'error'

const token = ref('')
const newPassword = ref('')
const confirmPassword = ref('')
const status = ref<Status>('idle')
const errorMessage = ref('')

onMounted(() => {
  const t = route.params.token as string

  if (!t) {
    status.value = 'invalid-token'
    return
  }

  token.value = t
})

async function submit() {
  errorMessage.value = ''

  if (!newPassword.value) {
    errorMessage.value = 'Please enter a new password'
    return
  }

  if (newPassword.value.length < 8) {
    errorMessage.value = 'Password must be at least 8 characters'
    return
  }

  if (newPassword.value !== confirmPassword.value) {
    errorMessage.value = 'Passwords do not match'
    return
  }

  status.value = 'submitting'

  try {
    await authApi.resetPassword(token.value, newPassword.value)
    status.value = 'success'
    setTimeout(() => router.push({ name: 'login' }), 1500)
  } catch (e: any) {
    const detail = e.response?.data?.detail
    if (e.response?.status === 400 || e.response?.status === 404) {
      status.value = 'invalid-token'
    } else {
      status.value = 'error'
      errorMessage.value = detail ?? 'Something went wrong. Please try again.'
    }
  }
}
</script>

<style>
@import '@/styles/views/registervue.css';
</style>