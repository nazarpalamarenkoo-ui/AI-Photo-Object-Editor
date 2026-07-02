export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface SignUpArgs {
  username: string
  email: string
  password: string
}

export interface SignInArgs {
  email: string
  password: string
}

export interface User {
  id: number
  username: string
  email: string
  created_at: string
}

export interface UserUpdate {
  username?: string
  email?: string
}

export interface ChangePassword {
  old_password: string
  new_password: string
}

export interface Image {
  id: number
  filename: string
  storage_path: string
  status: string
  uploaded_at: string
  user_id: number
  [key: string]: any
}

export interface PresignedUrlResponse {
  url: string
  expires_in: number
}
export interface Bbox {
  x1: number
  y1: number
  x2: number
  y2: number
}
export interface Detection {
  id: number
  image_id: number
  bbox_id: number
  x1: number
  y1: number
  x2: number
  y2: number
  detected_class: string
  confidence: number
}
export type EditingMode = 'yolo' | 'sam'

export interface Bbox {
  x1: number
  y1: number
  x2: number
  y2: number
}
export interface RegionItem {
  id: number            
  bbox: Bbox
  label: string          
  confidence?: number
}
export interface DetectionStats {
  total_detections: number
  classes: string[]
  avg_confidence: number
  min_confidence: number
  max_confidence: number
}

export interface LdmConfig {
  ldm_steps: number
  ldm_sampler: 'plms' | 'ddim'
  hd_strategy: 'CROP' | 'RESIZE' | 'ORIGINAL'
}


export interface DetectRequest {
  conf_threshold?: number
  classes?: string[]
}

export interface DetectResponse {
  detections: Detection[]
  image_size: [number, number]
  metrics: Record<string, number | string>
  timestamp: string
  [key: string]: any
}
export type ColorMatchMethod = 'mean_std' | 'histogram' | 'color_transfer'

export interface ReplaceOptions {
  expandMaskPixels?: number
  useColorMatching?: boolean
  useEdgeBlending?: boolean
  colorMatchMethod?: ColorMatchMethod
  ldm?: LdmConfig
}

export interface MLResultResponse {
  result_url: string
  presigned_url: string
  metrics: Record<string, unknown>
  timestamp: string
}

export interface SegmentInfo {
  mask_id: number
  bbox_id: number
  bbox: Bbox
  area: number
  stability_score: number
}

export interface SegmentResponse {
  segments: SegmentInfo[]
  metrics: Record<string, unknown>
  image_size: [number, number]
  timestamp: string
}

export interface ExtractResponse {
  extracted_url: string
  presigned_url: string
  object_size: [number, number]
  area_pixels: number
  cropped_bbox: Bbox
  timestamp: string
}

export interface PasteResponse {
  result_url: string
  presigned_url: string
  paste_bbox: Bbox
  object_size: [number, number]
  timestamp: string
}


export interface ApiError {
  detail: string
}