import { describe, it, expect, vi, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import type { TokenResponse, User, SignInArgs, SignUpArgs } from '@/types/Index'

vi.mock('@/api/auth', () => ({
  authApi: {
    login: vi.fn(),
    signup: vi.fn(),
    confirmEmail: vi.fn()
  }
}))

vi.mock('@/api/user', () => ({
  userApi: {
    getMe: vi.fn()
  }
}))

import {authApi} from '@/api/auth'
import {userApi} from '@/api/user'
import {useAuthStore} from '@/stores/auth'

const mockedAuthApi = vi.mocked(authApi, true)
const mockedUserApi = vi.mocked(userApi, true)

const fakeUser: User = {
  id: 1,
  username: 'testuser',
  email: 'test@example.com',
  created_at: '2026-01-01T00:00:00Z'
}

const fakeTokenResponse: TokenResponse = {
  access_token: 'new-token',
  token_type: 'bearer'
}

const signInArgs: SignInArgs = {
  email: 'test@example.com',
  password: 'secret'
}

const signUpArgs: SignUpArgs = {
  username: 'newuser',
  email: 'new@example.com',
  password: 'secret'
}

beforeEach(() => {
  localStorage.clear()
  setActivePinia(createPinia())
  vi.clearAllMocks()
})

describe('useAuthStore: initial state', () => {
  it('reads the token from localStorage on creation', () => {
    localStorage.setItem('access_token', 'stored-token')

    const store = useAuthStore()

    expect(store.token).toBe('stored-token')
    expect(store.isAuthenticated).toBe(true)
  })

  it('has no token when localStorage is empty', () => {
    const store = useAuthStore()

    expect(store.token).toBeNull()
    expect(store.user).toBeNull()
    expect(store.isAuthenticated).toBe(false)
  })
})

describe('useAuthStore: login', () => {
  it('stores the token, persists it and loads the user on success', async () => {
    mockedAuthApi.login.mockResolvedValue(fakeTokenResponse)
    mockedUserApi.getMe.mockResolvedValue(fakeUser)

    const store = useAuthStore()

    await store.login(signInArgs)

    expect(mockedAuthApi.login).toHaveBeenCalledWith(signInArgs)
    expect(mockedUserApi.getMe).toHaveBeenCalledTimes(1)
    expect(store.token).toBe('new-token')
    expect(localStorage.getItem('access_token')).toBe('new-token')
    expect(store.user).toEqual(fakeUser)
    expect(store.isAuthenticated).toBe(true)
  })

  it('logs out if fetching the user fails after a successful login', async () => {
    mockedAuthApi.login.mockResolvedValue(fakeTokenResponse)
    mockedUserApi.getMe.mockRejectedValue(new Error('unauthorized'))

    const store = useAuthStore()

    await store.login(signInArgs)

    expect(store.token).toBeNull()
    expect(store.user).toBeNull()
    expect(localStorage.getItem('access_token')).toBeNull()
  })

  it('propagates the error if authApi.login itself rejects', async () => {
    mockedAuthApi.login.mockRejectedValue(new Error('invalid credentials'))

    const store = useAuthStore()

    await expect(store.login(signInArgs)).rejects.toThrow('invalid credentials')

    expect(store.token).toBeNull()
    expect(mockedUserApi.getMe).not.toHaveBeenCalled()
  })
})

describe('useAuthStore: signup', () => {
  it('calls authApi.signup with the provided args and does not touch auth state', async () => {
    mockedAuthApi.signup.mockResolvedValue(undefined as never)

    const store = useAuthStore()

    await store.signup(signUpArgs)

    expect(mockedAuthApi.signup).toHaveBeenCalledWith(signUpArgs)
    expect(store.token).toBeNull()
    expect(store.user).toBeNull()
  })

  it('propagates errors from authApi.signup', async () => {
    mockedAuthApi.signup.mockRejectedValue(new Error('email taken'))

    const store = useAuthStore()

    await expect(store.signup(signUpArgs)).rejects.toThrow('email taken')
  })
})

describe('useAuthStore: confirmEmail', () => {
  it('stores the token, persists it and loads the user on success', async () => {
    const confirmedToken: TokenResponse = { access_token: 'confirmed-token', token_type: 'bearer' }
    mockedAuthApi.confirmEmail.mockResolvedValue(confirmedToken)
    mockedUserApi.getMe.mockResolvedValue(fakeUser)

    const store = useAuthStore()

    await store.confirmEmail('confirm-token-123')

    expect(mockedAuthApi.confirmEmail).toHaveBeenCalledWith('confirm-token-123')
    expect(store.token).toBe('confirmed-token')
    expect(localStorage.getItem('access_token')).toBe('confirmed-token')
    expect(store.user).toEqual(fakeUser)
  })

  it('logs out if fetching the user fails after confirming the email', async () => {
    const confirmedToken: TokenResponse = { access_token: 'confirmed-token', token_type: 'bearer' }
    mockedAuthApi.confirmEmail.mockResolvedValue(confirmedToken)
    mockedUserApi.getMe.mockRejectedValue(new Error('unauthorized'))

    const store = useAuthStore()

    await store.confirmEmail('confirm-token-123')

    expect(store.token).toBeNull()
    expect(store.user).toBeNull()
    expect(localStorage.getItem('access_token')).toBeNull()
  })
})

describe('useAuthStore: fetchMe', () => {
  it('sets the user on success', async () => {
    mockedUserApi.getMe.mockResolvedValue(fakeUser)

    const store = useAuthStore()
    await store.fetchMe()

    expect(store.user).toEqual(fakeUser)
  })

  it('logs out when the request fails', async () => {
    localStorage.setItem('access_token', 'stale-token')
    mockedUserApi.getMe.mockRejectedValue(new Error('unauthorized'))

    const store = useAuthStore()
    await store.fetchMe()

    expect(store.user).toBeNull()
    expect(store.token).toBeNull()
    expect(localStorage.getItem('access_token')).toBeNull()
  })
})

describe('useAuthStore: logout', () => {
  it('clears token, user and localStorage', () => {
    localStorage.setItem('access_token', 'some-token')
    const store = useAuthStore()
    store.user = fakeUser

    store.logout()

    expect(store.token).toBeNull()
    expect(store.user).toBeNull()
    expect(localStorage.getItem('access_token')).toBeNull()
    expect(store.isAuthenticated).toBe(false)
  })
})

describe('useAuthStore: init', () => {
  it('calls fetchMe when a token already exists', async () => {
    localStorage.setItem('access_token', 'existing-token')
    mockedUserApi.getMe.mockResolvedValue(fakeUser)

    const store = useAuthStore()
    await store.init()

    expect(mockedUserApi.getMe).toHaveBeenCalledTimes(1)
    expect(store.user).toEqual(fakeUser)
  })

  it('does nothing when there is no token', async () => {
    const store = useAuthStore()
    await store.init()

    expect(mockedUserApi.getMe).not.toHaveBeenCalled()
    expect(store.user).toBeNull()
  })
})