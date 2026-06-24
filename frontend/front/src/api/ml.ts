import apiClient from './clients'
import type { DetectRequest, DetectResponse, LdmConfig, MLResultResponse, Image } from '@/types/Index'

export const PRESETS: Record<string, LdmConfig> = {
  fast:    { ldm_steps: 10, ldm_sampler: 'plms', hd_strategy: 'CROP' },
  quality: { ldm_steps: 25, ldm_sampler: 'plms', hd_strategy: 'CROP' },
}

export const mlApi = {

    async detectObjects(imageId: number, params: DetectRequest = {}): Promise<DetectResponse> {
        const { data } = await apiClient.post<DetectResponse>(`/ml/images/${imageId}/detect`, params)
        return data
    },

    async removeObject(
        imageId: number,
        bboxId: number,
        expandMaskPixels = 5,
        useEdgeBlending = false,
        ldm: LdmConfig = PRESETS.quality
    ): Promise<MLResultResponse> {
        const { data } = await apiClient.post<MLResultResponse>(
            `/ml/images/${imageId}/remove/${bboxId}`,
            {
                expand_mask_pixels: expandMaskPixels,
                use_edge_blending: useEdgeBlending,
                ldm
            }
        )
        return data
    },

    async replaceObject(
        imageId: number,
        bboxId: number,
        replacementFile: File,
        options: {
            expandMaskPixels?: number
            useColorMatching?: boolean
            useEdgeBlending?: boolean
            colorMatchMethod?: 'mean_std' | 'histogram' | 'color_transfer'
            ldmSteps?: number
            ldmSampler?: 'plms' | 'ddim'
            hdStrategy?: 'CROP' | 'RESIZE' | 'ORIGINAL'
        } = {}
    ): Promise<MLResultResponse> {
        const formData = new FormData()
        formData.append('replacement_file', replacementFile)

        const { data } = await apiClient.post<MLResultResponse>(
            `/ml/images/${imageId}/replace/${bboxId}`,
            formData,
            {
                headers: { 'Content-Type': 'multipart/form-data' },
                params: {
                    expand_mask_pixels: options.expandMaskPixels ?? 0,
                    use_color_matching: options.useColorMatching ?? false,
                    use_edge_blending: options.useEdgeBlending ?? false,
                    color_match_method: options.colorMatchMethod ?? 'color_transfer',
                    ldm_steps: options.ldmSteps ?? 25,
                    ldm_sampler: options.ldmSampler ?? 'plms',
                    hd_strategy: options.hdStrategy ?? 'CROP',
                }
            }
        )
        return data
    },

    async removeMultipleObjects(
        imageId: number,
        bboxIds: number[],
        expandMaskPixels = 5,
        useEdgeBlending = false,
        ldm: LdmConfig = PRESETS.quality
    ): Promise<MLResultResponse> {
        const { data } = await apiClient.post<MLResultResponse>(
            `/ml/images/${imageId}/remove-multiple`,
            {
                bbox_ids: bboxIds,
                expand_mask_pixels: expandMaskPixels,
                use_edge_blending: useEdgeBlending,
                ldm
            }
        )
        return data
    },

    async getSupportedClasses(): Promise<string[]> {
        const { data } = await apiClient.get<string[]>('/ml/classes')
        return data
    },

    async saveResult(imageId: number): Promise<Image> {
        const { data } = await apiClient.post<Image>(`/ml/images/${imageId}/save`)
        return data
    },
    async undo(imageId: number): Promise<{ presigned_url: string, label: string, history: string[] }> {
        const { data } = await apiClient.post(`/ml/images/${imageId}/undo`)
        return data
    },

    async redo(imageId: number): Promise<{ presigned_url: string, label: string, history: string[] }> {
        const { data } = await apiClient.post(`/ml/images/${imageId}/redo`)
        return data
    },

    async getHistory(imageId: number): Promise<{ history: string[] }> {
        const { data } = await apiClient.get(`/ml/images/${imageId}/history`)
        return data
    },

    async resetState(imageId: number): Promise<void> {
        await apiClient.post(`/ml/images/${imageId}/reset`)
    }
}