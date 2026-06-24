import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { User, UserUpdate, ChangePassword } from '@/types/Index'

vi.mock('@/api/clients', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn()
  }
}))

import apiClient from '@/api/clients'
import { userApi } from '@/api/user'

const mockedClient = vi.mocked(apiClient, true)

const fakeUser: User = {
  id: 1,
  email: 'test@example.com',
  username: 'testuser',
  created_at: '2026-01-01T00:00:00Z'
}

beforeEach(() => {
  vi.clearAllMocks()
})


describe('userApi: getMe', () => {
  it('gets from correct url', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeUser })

    await userApi.getMe()

    expect(mockedClient.get).toHaveBeenCalledWith('/users/me')
  })

  it('returns user', async () => {
    mockedClient.get.mockResolvedValue({ data: fakeUser })

    const result = await userApi.getMe()

    expect(result).toEqual(fakeUser)
  })

  it('propagates error', async () => {
    mockedClient.get.mockRejectedValue(new Error('unauthorized'))

    await expect(userApi.getMe()).rejects.toThrow('unauthorized')
  })
})


describe('userApi: updateMe', () => {
  const updateBody: UserUpdate = { username: 'newname' }

  it('patches to correct url with body', async () => {
    mockedClient.patch.mockResolvedValue({ data: fakeUser })

    await userApi.updateMe(updateBody)

    expect(mockedClient.patch).toHaveBeenCalledWith('/users/me', updateBody)
  })

  it('returns updated user', async () => {
    const updatedUser = { ...fakeUser, username: 'newname' }
    mockedClient.patch.mockResolvedValue({ data: updatedUser })

    const result = await userApi.updateMe(updateBody)

    expect(result.username).toBe('newname')
  })

  it('propagates error', async () => {
    mockedClient.patch.mockRejectedValue(new Error('validation error'))

    await expect(userApi.updateMe(updateBody)).rejects.toThrow('validation error')
  })
})


describe('userApi: changePassword', () => {
  const passwordBody: ChangePassword = {
    old_password: 'oldpass123',
    new_password: 'newpass456'
  }

  it('patches to correct url with body', async () => {
    mockedClient.patch.mockResolvedValue({ data: fakeUser })

    await userApi.changePassword(passwordBody)

    expect(mockedClient.patch).toHaveBeenCalledWith('/users/me/password', passwordBody)
  })

  it('returns user', async () => {
    mockedClient.patch.mockResolvedValue({ data: fakeUser })

    const result = await userApi.changePassword(passwordBody)

    expect(result).toEqual(fakeUser)
  })

  it('propagates error on wrong current password', async () => {
    mockedClient.patch.mockRejectedValue(new Error('incorrect password'))

    await expect(userApi.changePassword(passwordBody)).rejects.toThrow('incorrect password')
  })
})


describe('userApi: deleteMe', () => {
  it('deletes from correct url', async () => {
    mockedClient.delete.mockResolvedValue({ data: fakeUser })

    await userApi.deleteMe()

    expect(mockedClient.delete).toHaveBeenCalledWith('/users/me')
  })

  it('returns deleted user', async () => {
    mockedClient.delete.mockResolvedValue({ data: fakeUser })

    const result = await userApi.deleteMe()

    expect(result).toEqual(fakeUser)
  })

  it('propagates error', async () => {
    mockedClient.delete.mockRejectedValue(new Error('user not found'))

    await expect(userApi.deleteMe()).rejects.toThrow('user not found')
  })
})