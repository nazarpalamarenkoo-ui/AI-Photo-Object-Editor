import { ref, onMounted } from 'vue'
import { imagesApi } from '@/api/images'
import { mlApi } from '@/api/ml'
import type { Image, Detection } from '@/types/Index'

export function useImageEditor(imageId: number) {
  const image = ref<Image | null>(null)
  const imageUrl = ref('')
  const originalImageUrl = ref('')
  const isEdited = ref(false)
  const loading = ref(false)
  const imageLoaded = ref(false)
  const naturalSize = ref({ w: 0, h: 0 })
  const detections = ref<Detection[]>([])

  onMounted(async () => {
    loading.value = true
    try {
      image.value = await imagesApi.getById(imageId)

      const { url } = await imagesApi.getPresignedUrl(imageId, 3600)
      originalImageUrl.value = url

      const { presigned_url, is_edited } = await mlApi.getCurrentState(imageId)
      imageUrl.value = presigned_url
      isEdited.value = is_edited
    } catch (e) {
      console.error(e)
    } finally {
      loading.value = false
    }
  })

  function onImageLoad(size: { w: number; h: number }) {
    naturalSize.value = size
    imageLoaded.value = true
  }

  return {
    image,
    imageUrl,
    originalImageUrl,
    isEdited,
    loading,
    imageLoaded,
    naturalSize,
    detections,
    onImageLoad,
  }
}