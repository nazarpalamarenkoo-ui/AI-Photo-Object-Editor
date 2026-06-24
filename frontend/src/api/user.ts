import apiClient from "./clients";
import type { User, UserUpdate, ChangePassword} from '@/types/Index'

export const userApi = {

    async getMe(): Promise<User> {
        const { data } = await apiClient.get<User>('/users/me')
        return data
    },

    async updateMe(body: UserUpdate): Promise<User> {
        const { data } = await apiClient.patch<User>('/users/me', body)
        return data
    },

    async changePassword(body: ChangePassword): Promise<User> {
        const { data } = await apiClient.patch('/users/me/password', body)
        return data
    },

    async deleteMe(): Promise<User> {
        const { data } = await apiClient.delete('/users/me')
        return data
    }
}