import apiClient from './clients'
import type {
  DetectRequest,
  DetectResponse,
  LdmConfig,
  MLResultResponse,
  Image,
  Bbox,
  ColorMatchMethod,
  SegmentResponse,
  ExtractResponse,
  PasteResponse,
  ReplaceOptions,
  Asset,
  SegmentByPolygonParams,
} from '@/types/Index'

export const PRESETS: Record<string, LdmConfig> = {
  fast:    { ldm_steps: 10, ldm_sampler: 'plms', hd_strategy: 'CROP' },
  quality: { ldm_steps: 25, ldm_sampler: 'plms', hd_strategy: 'CROP' },
}

export const mlApi = {

  async detectObjects(imageId: number, params: DetectRequest = {}): Promise<DetectResponse> {
    const { data } = await apiClient.post<DetectResponse>(`/ml/images/${imageId}/detect`, params)
    return data
  },

  async getSupportedClasses(): Promise<string[]> {
    const { data } = await apiClient.get<string[]>('/ml/classes')
    return data
  },

  async removeObject(
    imageId: number,
    bboxId: number,
    expandMaskPixels = 5,
    useEdgeBlending = false,
    ldm: LdmConfig = PRESETS.quality): Promise<MLResultResponse> {
    const { data } = await apiClient.post<MLResultResponse>(
      `/ml/images/${imageId}/remove/${bboxId}`,
      {
        expand_mask_pixels: expandMaskPixels,
        use_edge_blending: useEdgeBlending,
        ldm,
      }
    )
    return data
  },

  async removeMultipleObjects(
    imageId: number,
    bboxIds: number[],
    expandMaskPixels = 5,
    useEdgeBlending = false,
    ldm: LdmConfig = PRESETS.quality): Promise<MLResultResponse> {
    const { data } = await apiClient.post<MLResultResponse>(
      `/ml/images/${imageId}/remove-multiple`,
      {
        bbox_ids: bboxIds,
        expand_mask_pixels: expandMaskPixels,
        use_edge_blending: useEdgeBlending,
        ldm,
      }
    )
    return data
  },

  async replaceObject(
    imageId: number,
    bboxId: number,
    replacementFile: File,
    options: ReplaceOptions = {}): Promise<MLResultResponse> {
    const formData = new FormData()
    formData.append('replacement_file', replacementFile)

    const ldm = options.ldm ?? PRESETS.quality

    const { data } = await apiClient.post<MLResultResponse>(
      `/ml/images/${imageId}/replace/${bboxId}`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: {
          expand_mask_pixels: options.expandMaskPixels ?? 0,
          use_color_matching: options.useColorMatching ?? false,
          use_edge_blending: options.useEdgeBlending ?? false,
          color_match_method: options.colorMatchMethod ?? 'mean_std',
          ldm_steps: ldm.ldm_steps,
          ldm_sampler: ldm.ldm_sampler,
          hd_strategy: ldm.hd_strategy,
        }
      }
    )
    return data
  },


  async segmentObjects(
    imageId: number,
    minArea = 500,
    maxSegments = 50): Promise<SegmentResponse> {
    const { data } = await apiClient.post<SegmentResponse>(
      `/ml/images/${imageId}/segment`,
      { min_area: minArea, max_segments: maxSegments }
    )
    return data
  },

  async segmentWithPrompt(
    imageId: number,
    params: {
      pointCoords?: [number, number][]
      pointLabels?: number[]
      bbox?: Bbox
      multimask_output?: boolean
    }
  ): Promise<SegmentResponse> {
    const payload = {
      point_coords: params.pointCoords ?? null,
      point_labels: params.pointLabels ?? null,
      bbox: params.bbox
        ? { x1: params.bbox.x1, y1: params.bbox.y1, x2: params.bbox.x2, y2: params.bbox.y2 }
        : null,
        multimask_output: params.multimask_output ?? false,
    }
    const { data } = await apiClient.post<SegmentResponse>(
      `/ml/images/${imageId}/segment/prompt`,
      payload
    )
    return data
  },
  async segmentByPolygon(
    imageId: number,
    params: SegmentByPolygonParams
  ): Promise<SegmentResponse> {
    const { data } = await apiClient.post<SegmentResponse>(
      `/ml/images/${imageId}/segment/polygon`,
      {
        points: params.points,
        smooth: params.smooth ?? true,
        smoothing_factor: params.smoothingFactor ?? 0,
        feather_px: params.featherPx ?? 0,
      }
    )
    return data
  },
  async samRemoveObject(
    imageId: number,
    maskId: number,
    expandMaskPixels = 12,
    useEdgeBlending = false,
    ldm: LdmConfig = PRESETS.quality): Promise<MLResultResponse> {
    const { data } = await apiClient.post<MLResultResponse>(
      `/ml/images/${imageId}/segment/${maskId}/remove`,
      {
        expand_mask_pixels: expandMaskPixels,
        use_edge_blending: useEdgeBlending,
        ldm,
      }
    )
    return data
  },

  async samReplaceObject(
    imageId: number,
    maskId: number,
    replacementFile: File,
    options: ReplaceOptions = {}): Promise<MLResultResponse> {
    const formData = new FormData()
    formData.append('replacement_file', replacementFile)

    const ldm = options.ldm ?? PRESETS.quality

    const { data } = await apiClient.post<MLResultResponse>(
      `/ml/images/${imageId}/segment/${maskId}/replace`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: {
          expand_mask_pixels: options.expandMaskPixels ?? 8,
          use_color_matching: options.useColorMatching ?? false,
          use_edge_blending: options.useEdgeBlending ?? false,
          color_match_method: options.colorMatchMethod ?? 'color_transfer',
          ldm_steps: ldm.ldm_steps,
          ldm_sampler: ldm.ldm_sampler,
          hd_strategy: ldm.hd_strategy,
        }
      }
    )
    return data
  },

  async samReplaceObjectWithAsset(
    imageId: number,
    maskId: number,
    assetId: string,
    options: ReplaceOptions = {}): Promise<MLResultResponse> {
    const ldm = options.ldm ?? PRESETS.quality

    const { data } = await apiClient.post<MLResultResponse>(
      `/ml/images/${imageId}/segment/${maskId}/replace`,
      undefined,
      {
        params: {
          asset_id: assetId,
          expand_mask_pixels: options.expandMaskPixels ?? 8,
          use_color_matching: options.useColorMatching ?? false,
          use_edge_blending: options.useEdgeBlending ?? false,
          color_match_method: options.colorMatchMethod ?? 'color_transfer',
          ldm_steps: ldm.ldm_steps,
          ldm_sampler: ldm.ldm_sampler,
          hd_strategy: ldm.hd_strategy,
        }
      }
    )
    return data
  },

  async extractObject(
    imageId: number,
    maskId: number,
    params: { paddingPixels?: number; label?: string; persistToS3?: boolean } = {}
  ): Promise<ExtractResponse> {
    const { data } = await apiClient.post<ExtractResponse>(
      `/ml/images/${imageId}/segment/${maskId}/extract`,
      {
        padding_pixels: params.paddingPixels ?? 8,
        label: params.label,
        persist_to_s3: params.persistToS3 ?? false,
      }
    )
    return data
  },

  async pasteExtractedObject(
    imageId: number,
    params: {
      assetId?: string
      extractedUrl?: string
      targetBbox: Bbox
      scale?: number
      useColorMatching?: boolean
      useEdgeBlending?: boolean
      colorMatchMethod?: ColorMatchMethod
    }
  ): Promise<PasteResponse> {
    const { data } = await apiClient.post<PasteResponse>(
      `/ml/images/${imageId}/paste`,
      {
        asset_id: params.assetId,
        extracted_url: params.extractedUrl,
        target_bbox: params.targetBbox,
        scale: params.scale ?? 1.0,
        use_color_matching: params.useColorMatching ?? false,
        use_edge_blending: params.useEdgeBlending ?? false,
        color_match_method: params.colorMatchMethod ?? 'color_transfer',
      }
    )
    return data
  },


  async listAssets(limit = 50, offset = 0): Promise<Asset[]> {
    const { data } = await apiClient.get<Asset[]>('/ml/assets', {
      params: { limit, offset },
    })
    return data
  },

  async getAssetThumbnailBlob(assetId: string): Promise<Blob> {
    const { data } = await apiClient.get(`/ml/assets/${assetId}/thumbnail`, {
      responseType: 'blob',
    })
    return data
  },

  async getAssetImageBlob(assetId: string): Promise<Blob> {
    const { data } = await apiClient.get(`/ml/assets/${assetId}/image`, {
      responseType: 'blob',
    })
    return data
  },

  async renameAsset(assetId: string, label: string): Promise<Asset> {
    const { data } = await apiClient.patch<Asset>(`/ml/assets/${assetId}`, { label })
    return data
  },

  async deleteAsset(assetId: string): Promise<void> {
    await apiClient.delete(`/ml/assets/${assetId}`)
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
  },
}