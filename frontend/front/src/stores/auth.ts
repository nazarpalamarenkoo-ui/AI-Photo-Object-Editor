import { defineStore } from "pinia";
import { ref, computed} from 'vue';
import { authApi } from '@/api/auth'
import { userApi } from '@/api/user'
import type { User, SignInArgs, SignUpArgs } from '@/types/Index'

export const useAuthStore = defineStore('auth', () => {

    const token = ref<string | null>(localStorage.getItem('access_token'))
    const user = ref<User | null>(null)

    const isAuthenticated = computed(() => !! token.value)

    async function fetchMe() {
        try {
            user.value = await userApi.getMe()
        } catch {
            logout()
        }
    }

    function logout() {
        token.value = null
        user.value = null
        localStorage.removeItem('access_token')
    }

    async function login(credentials: SignInArgs) {
        
        const response = await authApi.login(credentials)
        token.value = response.access_token
        localStorage.setItem('access_token', response.access_token)
        await fetchMe() 
    }

    async function signup(args: SignUpArgs) {
        await authApi.signup(args)
    }

    async function confirmEmail(confirmToken: string) {

        const response = await authApi.confirmEmail(confirmToken)
        token.value = response.access_token
        localStorage.setItem('access_token', response.access_token)
        await fetchMe() 
    }

    async function init() {
        if (token.value){
            await fetchMe()
        }
    }

    return {
    token,
    user,
    isAuthenticated,
    login,
    signup,
    confirmEmail,
    fetchMe,
    logout,
    init
  }
  
})