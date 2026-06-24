import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { TokenResponse, SignInArgs, SignUpArgs } from '@/types/Index'

vi.mock('@/api/clients', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn()
  }
}))

import apiClient from '@/api/clients'
import { authApi } from '@/api/auth'

const mockedClient = vi.mocked(apiClient, true)

const signInArgs: SignInArgs = {
  email: 'test@example.com',
  password: 'secret'
}

const signUpArgs: SignUpArgs = {
  username: 'newuser',
  email: 'new@example.com',
  password: 'secret'
}

const fakeTokenResponse: TokenResponse = {
  access_token: 'new-token',
  token_type: 'bearer'
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('authApi: login', () => {
  it('posts credentials to /auth/login and returns the token response', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeTokenResponse })

    const result = await authApi.login(signInArgs)

    expect(mockedClient.post).toHaveBeenCalledWith('/auth/login', signInArgs)
    expect(result).toEqual(fakeTokenResponse)
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.post.mockRejectedValue(new Error('invalid credentials'))

    await expect(authApi.login(signInArgs)).rejects.toThrow('invalid credentials')
  })
})

describe('authApi: signup', () => {
  it('posts the new user payload to /auth/signup and returns the response data', async () => {
    mockedClient.post.mockResolvedValue({ data: { detail: 'check your email' } })

    const result = await authApi.signup(signUpArgs)

    expect(mockedClient.post).toHaveBeenCalledWith('/auth/signup', signUpArgs)
    expect(result).toEqual({ detail: 'check your email' })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.post.mockRejectedValue(new Error('email taken'))

    await expect(authApi.signup(signUpArgs)).rejects.toThrow('email taken')
  })
})

describe('authApi: confirmEmail', () => {
  it('posts the token as a query param and returns the token response', async () => {
    mockedClient.post.mockResolvedValue({ data: fakeTokenResponse })

    const result = await authApi.confirmEmail('confirm-token-123')

    expect(mockedClient.post).toHaveBeenCalledWith('/auth/signup-confirmation?token=confirm-token-123')
    expect(result).toEqual(fakeTokenResponse)
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.post.mockRejectedValue(new Error('invalid or expired token'))

    await expect(authApi.confirmEmail('bad-token')).rejects.toThrow('invalid or expired token')
  })
})

describe('authApi: recoverPassword', () => {
  it('posts to /auth/password-recovery with the email as a query param', async () => {
    mockedClient.post.mockResolvedValue({ data: { detail: 'recovery email sent' } })

    const result = await authApi.recoverPassword('test@example.com')

    expect(mockedClient.post).toHaveBeenCalledWith('/auth/password-recovery', null, {
      params: { email: 'test@example.com' }
    })
    expect(result).toEqual({ detail: 'recovery email sent' })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.post.mockRejectedValue(new Error('user not found'))

    await expect(authApi.recoverPassword('missing@example.com')).rejects.toThrow('user not found')
  })
})

describe('authApi: resetPassword', () => {
  it('patches /auth/reset-password with token and new_password as query params', async () => {
    mockedClient.patch.mockResolvedValue({ data: { detail: 'password updated' } })

    const result = await authApi.resetPassword('reset-token-456', 'new-secret')

    expect(mockedClient.patch).toHaveBeenCalledWith('/auth/reset-password', null, {
      params: { token: 'reset-token-456', new_password: 'new-secret' }
    })
    expect(result).toEqual({ detail: 'password updated' })
  })

  it('propagates the error when the request fails', async () => {
    mockedClient.patch.mockRejectedValue(new Error('invalid or expired token'))

    await expect(authApi.resetPassword('bad-token', 'new-secret')).rejects.toThrow('invalid or expired token')
  })
})