import apiClient from "./clients";
import type { Image, PresignedUrlResponse } from '@/types/Index'

export const imagesApi = {

    async upload(file: File): Promise<Image> {
        const formData = new FormData()
        formData.append('file', file)
        const { data } = await apiClient.post<Image>('/images/upload', formData, {
            headers: {'Content-Type': 'multipart/form-data'}
        })
        return data
    },

    async getAll(limit?: number, offset?: number): Promise<Image[]> {
        const { data } = await apiClient.get<Image[]>('/images/', {
            params: {limit, offset}
        })
        return data
    },

    async getById(imageId: number): Promise<Image> {
        const { data } = await apiClient.get<Image>(`/images/${imageId}`)
        return data
    },

    async getPresignedUrl(imageId: number, expiration: 3600): Promise<PresignedUrlResponse> {
        const { data } = await apiClient.get<PresignedUrlResponse>(`/images/${imageId}/url`,{
            params: {expiration}
        })
        return data
    },

    async delete(imageId: number): Promise<void> {
        await apiClient.delete(`/images/${imageId}`)
    },
    
    async download(imageId: number): Promise<void> {
        const response = await apiClient.get(`/images/${imageId}/download`, {
            responseType: 'blob'
        })
        const url = URL.createObjectURL(response.data)
        const a = document.createElement('a')
        a.href = url
        a.download = `image_${imageId}.jpg`
        a.click()
        URL.revokeObjectURL(url)
    }
}