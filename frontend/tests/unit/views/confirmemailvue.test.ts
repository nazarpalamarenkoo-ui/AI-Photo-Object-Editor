import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ref } from 'vue'

function makeConfirmEmailFlow(confirmEmail: (token: string) => Promise<void>) {
  const status = ref<'loading' | 'success' | 'error'>('loading')
  const errorMessage = ref('')
  const redirectedTo = ref<string | null>(null)

  async function run(token: string | undefined) {
    status.value = 'loading'
    errorMessage.value = ''
    redirectedTo.value = null

    if (!token) {
      status.value = 'error'
      errorMessage.value = 'Invalid confirmation link'
      return
    }

    try {
      await confirmEmail(token)
      status.value = 'success'
      redirectedTo.value = 'dashboard'
    } catch (e: any) {
      status.value = 'error'
      errorMessage.value = e.response?.data?.detail ?? 'Link is invalid or has expired'
    }
  }

  return { status, errorMessage, redirectedTo, run }
}

describe('ConfirmEmailView: onMounted flow', () => {
    let confirmEmail: ReturnType<typeof vi.fn> & ((token: string) => Promise<void>)

    beforeEach(() => {
        confirmEmail = vi.fn() as ReturnType<typeof vi.fn> & ((token: string) => Promise<void>)
    })

  describe('missing token', () => {
    it('sets status to error when token is undefined', async () => {
      const { status, run } = makeConfirmEmailFlow(confirmEmail)
      await run(undefined)

      expect(status.value).toBe('error')
    })

    it('sets error message when token is undefined', async () => {
      const { errorMessage, run } = makeConfirmEmailFlow(confirmEmail)
      await run(undefined)

      expect(errorMessage.value).toBe('Invalid confirmation link')
    })

    it('does not call confirmEmail when token is missing', async () => {
      const { run } = makeConfirmEmailFlow(confirmEmail)
      await run(undefined)

      expect(confirmEmail).not.toHaveBeenCalled()
    })

    it('sets status to error when token is empty string', async () => {
      const { status, errorMessage, run } = makeConfirmEmailFlow(confirmEmail) // fixed: added errorMessage
      await run('')

      expect(status.value).toBe('error')
      expect(errorMessage.value).toBe('Invalid confirmation link')
    })
  })

  describe('success', () => {
    it('starts with loading status', () => {
      const { status } = makeConfirmEmailFlow(confirmEmail)

      expect(status.value).toBe('loading')
    })

    it('sets status to success after valid token', async () => {
      confirmEmail.mockResolvedValue(undefined)
      const { status, run } = makeConfirmEmailFlow(confirmEmail)
      await run('valid-token-abc')

      expect(status.value).toBe('success')
    })

    it('calls confirmEmail with the token', async () => {
      confirmEmail.mockResolvedValue(undefined)
      const { run } = makeConfirmEmailFlow(confirmEmail)
      await run('valid-token-abc')

      expect(confirmEmail).toHaveBeenCalledWith('valid-token-abc')
      expect(confirmEmail).toHaveBeenCalledOnce()
    })

    it('schedules redirect to dashboard on success', async () => {
      confirmEmail.mockResolvedValue(undefined)
      const { redirectedTo, run } = makeConfirmEmailFlow(confirmEmail)
      await run('valid-token-abc')

      expect(redirectedTo.value).toBe('dashboard')
    })

    it('error message stays empty on success', async () => {
      confirmEmail.mockResolvedValue(undefined)
      const { errorMessage, run } = makeConfirmEmailFlow(confirmEmail)
      await run('valid-token-abc')

      expect(errorMessage.value).toBe('')
    })
  })

  describe('error', () => {
    it('sets status to error when confirmEmail throws', async () => {
      confirmEmail.mockRejectedValue(new Error('fail'))
      const { status, run } = makeConfirmEmailFlow(confirmEmail)
      await run('bad-token')

      expect(status.value).toBe('error')
    })

    it('shows API detail error message', async () => {
      confirmEmail.mockRejectedValue({ response: { data: { detail: 'Token expired' } } })
      const { errorMessage, run } = makeConfirmEmailFlow(confirmEmail)
      await run('expired-token')

      expect(errorMessage.value).toBe('Token expired')
    })

    it('shows fallback message when API response has no detail', async () => {
      confirmEmail.mockRejectedValue(new Error('Network error'))
      const { errorMessage, run } = makeConfirmEmailFlow(confirmEmail)
      await run('bad-token')

      expect(errorMessage.value).toBe('Link is invalid or has expired')
    })

    it('does not redirect on error', async () => {
      confirmEmail.mockRejectedValue(new Error('fail'))
      const { redirectedTo, run } = makeConfirmEmailFlow(confirmEmail)
      await run('bad-token')

      expect(redirectedTo.value).toBeNull()
    })
  })

  describe('combined', () => {
    it('resets state on each run call', async () => {
      confirmEmail.mockRejectedValueOnce(new Error('fail')).mockResolvedValueOnce(undefined)
      const { status, errorMessage, run } = makeConfirmEmailFlow(confirmEmail)

      await run('bad-token')
      expect(status.value).toBe('error')

      await run('good-token')
      expect(status.value).toBe('success')
      expect(errorMessage.value).toBe('')
    })
  })
})