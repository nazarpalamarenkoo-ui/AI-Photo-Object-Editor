import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'

function makeForgotPasswordFlow(recoverPassword: (email: string) => Promise<{ detail: string }>) {
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
      await recoverPassword(email.value.trim())
      status.value = 'success'
    } catch (e: any) {
      status.value = 'error'
      errorMessage.value = e.response?.data?.detail ?? 'Something went wrong. Please try again.'
    }
  }

  return { email, status, errorMessage, submit }
}

describe('ForgotPasswordView', () => {
  let recoverPassword: ReturnType<typeof vi.fn> & ((email: string) => Promise<{ detail: string }>)

  beforeEach(() => {
    recoverPassword = vi.fn() as ReturnType<typeof vi.fn> & ((email: string) => Promise<{ detail: string }>)
  })

  describe('validation', () => {
    it('does not call API when email is empty', async () => {
      const { submit } = makeForgotPasswordFlow(recoverPassword)
      await submit()

      expect(recoverPassword).not.toHaveBeenCalled()
    })

    it('sets error message when email is empty', async () => {
      const { errorMessage, submit } = makeForgotPasswordFlow(recoverPassword)
      await submit()

      expect(errorMessage.value).toBe('Please enter your email')
    })

    it('does not call API when email is whitespace only', async () => {
      const { email, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = '   '
      await submit()

      expect(recoverPassword).not.toHaveBeenCalled()
    })

    it('status stays idle when email is empty', async () => {
      const { status, submit } = makeForgotPasswordFlow(recoverPassword)
      await submit()

      expect(status.value).toBe('idle')
    })
  })

  describe('success', () => {
    it('calls API with trimmed email', async () => {
      recoverPassword.mockResolvedValue({ detail: 'sent' })
      const { email, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = '  user@example.com  '
      await submit()

      expect(recoverPassword).toHaveBeenCalledWith('user@example.com')
      expect(recoverPassword).toHaveBeenCalledOnce()
    })

    it('sets status to success after API call', async () => {
      recoverPassword.mockResolvedValue({ detail: 'sent' })
      const { email, status, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = 'user@example.com'
      await submit()

      expect(status.value).toBe('success')
    })

    it('error message stays empty on success', async () => {
      recoverPassword.mockResolvedValue({ detail: 'sent' })
      const { email, errorMessage, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = 'user@example.com'
      await submit()

      expect(errorMessage.value).toBe('')
    })
  })

  describe('error', () => {
    it('sets status to error when API throws', async () => {
      recoverPassword.mockRejectedValue(new Error('fail'))
      const { email, status, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = 'user@example.com'
      await submit()

      expect(status.value).toBe('error')
    })

    it('shows API detail error message', async () => {
      recoverPassword.mockRejectedValue({ response: { data: { detail: 'User not found' } } })
      const { email, errorMessage, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = 'user@example.com'
      await submit()

      expect(errorMessage.value).toBe('User not found')
    })

    it('shows fallback message when API has no detail', async () => {
      recoverPassword.mockRejectedValue(new Error('Network error'))
      const { email, errorMessage, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = 'user@example.com'
      await submit()

      expect(errorMessage.value).toBe('Something went wrong. Please try again.')
    })

    it('clears previous error message before new submit', async () => {
      recoverPassword.mockRejectedValueOnce(new Error('fail')).mockResolvedValueOnce({ detail: 'sent' })
      const { email, errorMessage, submit } = makeForgotPasswordFlow(recoverPassword)
      email.value = 'user@example.com'

      await submit()
      expect(errorMessage.value).toBe('Something went wrong. Please try again.')

      await submit()
      expect(errorMessage.value).toBe('')
    })
  })
})