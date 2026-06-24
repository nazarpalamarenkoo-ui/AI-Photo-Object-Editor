<template>
  <div class="auth-page">
    <div class="auth-card">
      <h1 class="auth-title">Welcome back</h1>
      <p class="auth-subtitle">Sign in to your account</p>

      <form class="auth-form" @submit.prevent="handleLogin">
        <div class="field">
          <label for="email">Email</label>
          <InputText
            id="email"
            v-model="form.email"
            type="email"
            placeholder="you@example.com"
            :class="{ 'p-invalid': errors.email }"
            autocomplete="email"
          />
          <small class="error-msg">{{ errors.email }}</small>
        </div>

        <div class="field">
          <label for="password">Password</label>
          <Password
            id="password"
            v-model="form.password"
            placeholder="Your password"
            :feedback="false"
            toggleMask
            :class="{ 'p-invalid': errors.password }"
            autocomplete="current-password"
          />
          <small class="error-msg">{{ errors.password }}</small>
        </div>

        <div class="forgot-link">
          <RouterLink to="/forgot-password">Forgot password?</RouterLink>
        </div>

        <Message v-if="serverError" severity="error" :closable="false">
          {{ serverError }}
        </Message>

        <Button
          type="submit"
          label="Sign in"
          :loading="loading"
          class="submit-btn"
        />
      </form>

      <p class="auth-footer">
        Don't have an account?
        <RouterLink to="/register">Sign up</RouterLink>
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import InputText from 'primevue/inputtext'
import Password from 'primevue/password'
import Button from 'primevue/button'
import Message from 'primevue/message'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const auth = useAuthStore()

const form = reactive({
  email: '',
  password: ''
})

const errors = reactive({
  email: '',
  password: ''
})

const loading = ref(false)
const serverError = ref('')

function validate(): boolean {
  errors.email = ''
  errors.password = ''

  if (!form.email) {
    errors.email = 'Email is required'
  } else if (!/\S+@\S+\.\S+/.test(form.email)) {
    errors.email = 'Enter a valid email'
  }

  if (!form.password) {
    errors.password = 'Password is required'
  }

  return !errors.email && !errors.password
}

async function handleLogin() {
  if (!validate()) return

  loading.value = true
  serverError.value = ''

  try {
    await auth.login({ email: form.email, password: form.password })
    router.push({ name: 'dashboard' })
  } catch (e: any) {
    serverError.value = e.response?.data?.detail ?? 'Something went wrong'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
@import '../styles/views/registervue.css';
@import '../styles/views/loginvue.css';
</style>