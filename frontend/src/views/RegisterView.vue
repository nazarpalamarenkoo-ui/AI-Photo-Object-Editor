<template>
  <div class="auth-page">
    <div class="auth-card">

      <div v-if="emailSent" class="email-sent">
        <i class="pi pi-envelope email-icon" />
        <h2>Check your email</h2>
        <p>We sent a confirmation link to <strong>{{ form.email }}</strong></p>
        <p class="hint">Click the link in the email to activate your account.</p>
        <Button label="Back to login" text @click="router.push({ name: 'login' })" />
      </div>

      <template v-else>
        <h1 class="auth-title">Create account</h1>
        <p class="auth-subtitle">Start using the image editor</p>

        <form class="auth-form" @submit.prevent="handleRegister">
          <div class="field">
            <label for="username">Username</label>
            <InputText
              id="username"
              v-model="form.username"
              placeholder="yourname"
              :class="{ 'p-invalid': errors.username }"
              autocomplete="username"
            />
            <small class="error-msg">{{ errors.username }}</small>
          </div>

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
              placeholder="Min 8 characters"
              toggleMask
              :class="{ 'p-invalid': errors.password }"
              autocomplete="new-password"
            />
            <small class="error-msg">{{ errors.password }}</small>
          </div>

          <div class="field">
            <label for="confirm">Confirm password</label>
            <Password
              id="confirm"
              v-model="form.confirm"
              placeholder="Repeat password"
              :feedback="false"
              toggleMask
              :class="{ 'p-invalid': errors.confirm }"
              autocomplete="new-password"
            />
            <small class="error-msg">{{ errors.confirm }}</small>
          </div>

          <Message v-if="serverError" severity="error" :closable="false">
            {{ serverError }}
          </Message>

          <Button
            type="submit"
            label="Create account"
            :loading="loading"
            class="submit-btn"
          />
        </form>

        <p class="auth-footer">
          Already have an account?
          <RouterLink to="/login">Sign in</RouterLink>
        </p>
      </template>

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
  username: '',
  email: '',
  password: '',
  confirm: ''
})

const errors = reactive({
  username: '',
  email: '',
  password: '',
  confirm: ''
})

const loading = ref(false)
const serverError = ref('')
const emailSent = ref(false)

function validate(): boolean {
  errors.username = ''
  errors.email = ''
  errors.password = ''
  errors.confirm = ''

  if (!form.username || form.username.length < 3) {
    errors.username = 'Username must be at least 3 characters'
  }

  if (!form.email) {
    errors.email = 'Email is required'
  } else if (!/\S+@\S+\.\S+/.test(form.email)) {
    errors.email = 'Enter a valid email'
  }

  if (!form.password || form.password.length < 8) {
    errors.password = 'Password must be at least 8 characters'
  } else if (!/[A-Z]/.test(form.password)) {
    errors.password = 'Must contain at least one uppercase letter'
  } else if (!/[0-9]/.test(form.password)) {
    errors.password = 'Must contain at least one number'
  }

  if (form.password !== form.confirm) {
    errors.confirm = 'Passwords do not match'
  }

  return !errors.username && !errors.email && !errors.password && !errors.confirm
}

async function handleRegister() {
  if (!validate()) return

  loading.value = true
  serverError.value = ''

  try {
    await auth.signup({
      username: form.username,
      email: form.email,
      password: form.password
    })
    emailSent.value = true
  } catch (e: any) {
    serverError.value = e.response?.data?.detail ?? 'Something went wrong'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
@import '../styles/views/registervue.css';
</style>