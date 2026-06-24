import apiClient from "./clients";
import type { TokenResponse, SignUpArgs, SignInArgs } from '@/types/Index'

export const authApi = {
    
    async login(body: SignInArgs): Promise<TokenResponse> {
        const { data } = await apiClient.post<TokenResponse>('/auth/login', body)
        return data
    },

    async signup(body: SignUpArgs): Promise<{ detail: string }> {
        const { data } = await apiClient.post('/auth/signup', body)
        return data
    },

    async confirmEmail(token: string): Promise<TokenResponse> {
        const { data } = await apiClient.post<TokenResponse>(`/auth/signup-confirmation?token=${token}`)
        return data
    },

    async recoverPassword(email: string): Promise<{ detail: string}> {
        const { data } = await apiClient.post('/auth/password-recovery', null, {
            params: {email}
        })
        return data  
    },

    async resetPassword(token: string, newPassword: string): Promise<{ detail: string}> {
        const { data } = await apiClient.patch('/auth/reset-password', null, {
            params: {token, new_password: newPassword}
        })
        return data
    }
}