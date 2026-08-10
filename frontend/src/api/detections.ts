import apiClient from "./clients";
import type { Detection, DetectionStats } from '@/types/Index'

export const detectionsApi = {

    async getByImage(imageId: number, versionId?: number, activeOnly = true): Promise<Detection[]> {
        const { data } = await apiClient.get<Detection[]>(`/detections/images/${imageId}`, {
            params: {version_id: versionId, active_only: activeOnly}
        })
        return data
    },

    async getByBboxId(imageId: number, bboxId: number, versionId?: number): Promise<Detection> {
        const { data } = await apiClient.get<Detection>(`/detections/images/${imageId}/bbox/${bboxId}`, {
            params: {version_id: versionId}
        })
        return data
    },

    async getStats(imageId: number): Promise<DetectionStats> {
        const { data } = await apiClient.get<DetectionStats>(`/detections/images/${imageId}/stats`)
        return data
    },

    async deleteByImage(imageId: number): Promise<{ deleted: number}> {
        const { data } = await apiClient.delete(`/detections/images/${imageId}`)
        return data
    }
}