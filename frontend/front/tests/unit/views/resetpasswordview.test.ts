import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'

function makeResetPasswordFlow(
  resetPassword: (token: string, newPassword: string) => Promise<{ detail: string }>
) {
  type Status = 'idle' | 'submitting' | 'success' | 'invalid-token' | 'error'

  const token = ref('')
  const newPassword = ref('')
  const confirmPassword = ref('')
  const status = ref<Status>('idle')
  const errorMessage = ref('')
  const redirectedTo = ref<string | null>(null)

  function init(routeToken: string | undefined) {
    if (!routeToken) {
      status.value = 'invalid-token'
      return
    }
    token.value = routeToken
  }

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
      await resetPassword(token.value, newPassword.value)
      status.value = 'success'
      redirectedTo.value = 'login'
    } catch (e: any) {
      if (e.response?.status === 400 || e.response?.status === 404) {
        status.value = 'invalid-token'
      } else {
        status.value = 'error'
        errorMessage.value = e.response?.data?.detail ?? 'Something went wrong. Please try again.'
      }
    }
  }

  return { token, newPassword, confirmPassword, status, errorMessage, redirectedTo, init, submit }
}

describe('ResetPasswordView', () => {
  let resetPassword: ReturnType<typeof vi.fn> & ((token: string, newPassword: string) => Promise<{ detail: string }>)

  beforeEach(() => {
    resetPassword = vi.fn() as ReturnType<typeof vi.fn> & ((token: string, newPassword: string) => Promise<{ detail: string }>)
  })

  describe('init (onMounted)', () => {
    it('sets invalid-token when token is missing', () => {
      const { status, init } = makeResetPasswordFlow(resetPassword)
      init(undefined)

      expect(status.value).toBe('invalid-token')
    })

    it('sets invalid-token when token is empty string', () => {
      const { status, init } = makeResetPasswordFlow(resetPassword)
      init('')

      expect(status.value).toBe('invalid-token')
    })

    it('stores token when present', () => {
      const { token, status, init } = makeResetPasswordFlow(resetPassword)
      init('reset-token-xyz')

      expect(token.value).toBe('reset-token-xyz')
      expect(status.value).toBe('idle')
    })
  })

  describe('validation', () => {
    it('sets error when password is empty', async () => {
      const { errorMessage, submit } = makeResetPasswordFlow(resetPassword)
      await submit()

      expect(errorMessage.value).toBe('Please enter a new password')
      expect(resetPassword).not.toHaveBeenCalled()
    })

    it('sets error when password is shorter than 8 characters', async () => {
      const { newPassword, errorMessage, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'abc123'
      await submit()

      expect(errorMessage.value).toBe('Password must be at least 8 characters')
      expect(resetPassword).not.toHaveBeenCalled()
    })

    it('sets error when passwords do not match', async () => {
      const { newPassword, confirmPassword, errorMessage, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'password123'
      confirmPassword.value = 'different123'
      await submit()

      expect(errorMessage.value).toBe('Passwords do not match')
      expect(resetPassword).not.toHaveBeenCalled()
    })

    it('status stays idle on validation failure', async () => {
      const { newPassword, confirmPassword, status, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'password123'
      confirmPassword.value = 'different123'
      await submit()

      expect(status.value).toBe('idle')
    })
  })

  describe('success', () => {
    it('calls API with token and new password', async () => {
      resetPassword.mockResolvedValue({ detail: 'ok' })
      const { token, newPassword, confirmPassword, submit } = makeResetPasswordFlow(resetPassword)
      token.value = 'reset-token-xyz'
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(resetPassword).toHaveBeenCalledWith('reset-token-xyz', 'newpassword1')
      expect(resetPassword).toHaveBeenCalledOnce()
    })

    it('sets status to success', async () => {
      resetPassword.mockResolvedValue({ detail: 'ok' })
      const { newPassword, confirmPassword, status, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(status.value).toBe('success')
    })

    it('schedules redirect to login on success', async () => {
      resetPassword.mockResolvedValue({ detail: 'ok' })
      const { newPassword, confirmPassword, redirectedTo, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(redirectedTo.value).toBe('login')
    })
  })

  describe('error', () => {
    it('sets invalid-token on 400 response', async () => {
      resetPassword.mockRejectedValue({ response: { status: 400, data: { detail: 'Token expired' } } })
      const { newPassword, confirmPassword, status, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(status.value).toBe('invalid-token')
    })

    it('sets invalid-token on 404 response', async () => {
      resetPassword.mockRejectedValue({ response: { status: 404, data: { detail: 'Not found' } } })
      const { newPassword, confirmPassword, status, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(status.value).toBe('invalid-token')
    })

    it('sets error status on unexpected failure', async () => {
      resetPassword.mockRejectedValue(new Error('Network error'))
      const { newPassword, confirmPassword, status, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(status.value).toBe('error')
    })

    it('shows API detail on unexpected failure', async () => {
      resetPassword.mockRejectedValue({ response: { status: 500, data: { detail: 'Server error' } } })
      const { newPassword, confirmPassword, errorMessage, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(errorMessage.value).toBe('Server error')
    })

    it('shows fallback message when no detail in response', async () => {
      resetPassword.mockRejectedValue(new Error('Network error'))
      const { newPassword, confirmPassword, errorMessage, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(errorMessage.value).toBe('Something went wrong. Please try again.')
    })

    it('does not redirect on error', async () => {
      resetPassword.mockRejectedValue(new Error('fail'))
      const { newPassword, confirmPassword, redirectedTo, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'
      await submit()

      expect(redirectedTo.value).toBeNull()
    })

    it('clears error message before each submit', async () => {
      resetPassword.mockRejectedValueOnce(new Error('fail')).mockResolvedValueOnce({ detail: 'ok' })
      const { newPassword, confirmPassword, errorMessage, submit } = makeResetPasswordFlow(resetPassword)
      newPassword.value = 'newpassword1'
      confirmPassword.value = 'newpassword1'

      await submit()
      expect(errorMessage.value).toBe('Something went wrong. Please try again.')

      await submit()
      expect(errorMessage.value).toBe('')
    })
  })
})