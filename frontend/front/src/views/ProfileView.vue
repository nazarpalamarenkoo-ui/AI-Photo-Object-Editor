<template>
  <div class="dashboard">

    <header class="navbar">
      <div class="navbar-left">
        <Button icon="pi pi-arrow-left" text rounded @click="router.push({ name: 'dashboard' })" />
        <span class="navbar-logo">ImageEditor</span>
      </div>
      <div class="navbar-right">
        <span class="navbar-user">{{ auth.user?.username }}</span>
        <Button icon="pi pi-sign-out" text rounded severity="secondary" @click="handleLogout" />
      </div>
    </header>

    <main class="content">
      <div class="profile-wrapper">

        <div class="card">
          <h2 class="card-title">Profile</h2>

          <form @submit.prevent="handleUpdate" class="profile-form">
            <div class="field">
              <label>Username</label>
              <InputText v-model="profileForm.username" :class="{ 'p-invalid': profileErrors.username }" />
              <small class="error-msg">{{ profileErrors.username }}</small>
            </div>

            <div class="field">
              <label>Email</label>
              <InputText v-model="profileForm.email" type="email" :class="{ 'p-invalid': profileErrors.email }" />
              <small class="error-msg">{{ profileErrors.email }}</small>
            </div>

            <Message v-if="profileSuccess" severity="success" :closable="false">
              Profile updated successfully
            </Message>
            <Message v-if="profileError" severity="error" :closable="false">
              {{ profileError }}
            </Message>

            <Button type="submit" label="Save changes" :loading="profileLoading" />
          </form>
        </div>

        <div class="card">
          <h2 class="card-title">Change password</h2>

          <form @submit.prevent="handleChangePassword" class="profile-form">
            <div class="field">
              <label>Current password</label>
              <Password
                v-model="passwordForm.old_password"
                :feedback="false"
                toggleMask
                :class="{ 'p-invalid': passwordErrors.old_password }"
              />
              <small class="error-msg">{{ passwordErrors.old_password }}</small>
            </div>

            <div class="field">
              <label>New password</label>
              <Password
                v-model="passwordForm.new_password"
                toggleMask
                :class="{ 'p-invalid': passwordErrors.new_password }"
              />
              <small class="error-msg">{{ passwordErrors.new_password }}</small>
            </div>

            <div class="field">
              <label>Confirm new password</label>
              <Password
                v-model="passwordForm.confirm"
                :feedback="false"
                toggleMask
                :class="{ 'p-invalid': passwordErrors.confirm }"
              />
              <small class="error-msg">{{ passwordErrors.confirm }}</small>
            </div>

            <Message v-if="passwordSuccess" severity="success" :closable="false">
              Password changed successfully
            </Message>
            <Message v-if="passwordError" severity="error" :closable="false">
              {{ passwordError }}
            </Message>

            <Button type="submit" label="Change password" :loading="passwordLoading" />
          </form>
        </div>

        <div class="card danger-card">
          <h2 class="card-title danger-title">Danger zone</h2>
          <p class="danger-text">
            Deleting your account will permanently remove all your images and detections.
          </p>
          <Button
            label="Delete account"
            severity="danger"
            outlined
            @click="confirmDeleteVisible = true"
          />
        </div>

      </div>
    </main>

    <Dialog
      v-model:visible="confirmDeleteVisible"
      header="Delete account"
      :modal="true"
      :style="{ width: '400px' }"
    >
      <p>Are you sure you want to delete your account? This action cannot be undone.</p>
      <template #footer>
        <Button label="Cancel" text @click="confirmDeleteVisible = false" />
        <Button label="Delete" severity="danger" :loading="deleteLoading" @click="handleDelete" />
      </template>
    </Dialog>

  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import InputText from 'primevue/inputtext'
import Password from 'primevue/password'
import Button from 'primevue/button'
import Message from 'primevue/message'
import Dialog from 'primevue/dialog'
import { useAuthStore } from '../stores/auth'
import { userApi } from '../api/user'

const router = useRouter()
const auth = useAuthStore()

const profileForm = reactive({ username: '', email: '' })
const profileErrors = reactive({ username: '', email: '' })
const profileLoading = ref(false)
const profileSuccess = ref(false)
const profileError = ref('')

onMounted(() => {
  if (auth.user) {
    profileForm.username = auth.user.username
    profileForm.email = auth.user.email
  }
})

async function handleUpdate() {
  profileErrors.username = ''
  profileErrors.email = ''
  profileSuccess.value = false
  profileError.value = ''

  if (!profileForm.username || profileForm.username.length < 3) {
    profileErrors.username = 'Username must be at least 3 characters'
    return
  }

  profileLoading.value = true
  try {
    const updated = await userApi.updateMe({
      username: profileForm.username,
      email: profileForm.email
    })
    auth.user = updated
    profileSuccess.value = true
  } catch (e: any) {
    profileError.value = e.response?.data?.detail ?? 'Something went wrong'
  } finally {
    profileLoading.value = false
  }
}

const passwordForm = reactive({ old_password: '', new_password: '', confirm: '' })
const passwordErrors = reactive({ old_password: '', new_password: '', confirm: '' })
const passwordLoading = ref(false)
const passwordSuccess = ref(false)
const passwordError = ref('')

async function handleChangePassword() {
  passwordErrors.old_password = ''
  passwordErrors.new_password = ''
  passwordErrors.confirm = ''
  passwordSuccess.value = false
  passwordError.value = ''

  if (!passwordForm.old_password) {
    passwordErrors.old_password = 'Required'
    return
  }
  if (!passwordForm.new_password || passwordForm.new_password.length < 8) {
    passwordErrors.new_password = 'Min 8 characters'
    return
  }
  if (!/[A-Z]/.test(passwordForm.new_password)) {
    passwordErrors.new_password = 'Must contain uppercase letter'
    return
  }
  if (!/[0-9]/.test(passwordForm.new_password)) {
    passwordErrors.new_password = 'Must contain a number'
    return
  }
  if (passwordForm.new_password !== passwordForm.confirm) {
    passwordErrors.confirm = 'Passwords do not match'
    return
  }

  passwordLoading.value = true
  try {
    await userApi.changePassword({
      old_password: passwordForm.old_password,
      new_password: passwordForm.new_password
    })
    passwordSuccess.value = true
    passwordForm.old_password = ''
    passwordForm.new_password = ''
    passwordForm.confirm = ''
  } catch (e: any) {
    passwordError.value = e.response?.data?.detail ?? 'Something went wrong'
  } finally {
    passwordLoading.value = false
  }
}

const confirmDeleteVisible = ref(false)
const deleteLoading = ref(false)

async function handleDelete() {
  deleteLoading.value = true
  try {
    await userApi.deleteMe()
    auth.logout()
    router.push({ name: 'login' })
  } catch (e) {
    console.error('Delete failed', e)
  } finally {
    deleteLoading.value = false
  }
}

function handleLogout() {
  auth.logout()
  router.push({ name: 'login' })
}
</script>

<style scoped>
@import '../styles/views/profilevue.css';
</style>