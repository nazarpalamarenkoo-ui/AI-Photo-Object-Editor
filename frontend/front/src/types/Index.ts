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
}

export interface PresignedUrlResponse {
  url: string
  expires_in: number
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
}

export interface MLResultResponse {
  presigned_url: string
  result_url?: string
  metrics?: Record<string, number | string>
  timestamp?: string
}

export interface ApiError {
  detail: string
}