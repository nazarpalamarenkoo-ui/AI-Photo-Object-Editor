import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { imagesApi } from '@/api/images'
import type { Image } from '@/types/Index'

export function useDashboard() {
  const router = useRouter()
  const images = ref<Image[]>([])
  const imageUrls = ref<Record<number, string>>({})
  const loading = ref(false)
  const uploading = ref(false)

  async function fetchImages() {
    loading.value = true
    try {
      images.value = await imagesApi.getAll()
      for (const image of images.value) {
        try {
          const { url } = await imagesApi.getPresignedUrl(image.id, 3600)
          imageUrls.value[image.id] = url
        } catch {}
      }
    } catch (e) {
      console.error('Failed to fetch images', e)
    } finally {
      loading.value = false
    }
  }

  async function handleFileChange(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file) return
    uploading.value = true
    try {
      const uploaded = await imagesApi.upload(file)
      images.value.unshift(uploaded)
      router.push({ name: 'image-editor', params: { id: uploaded.id } })
    } catch (e) {
      console.error('Upload failed', e)
    } finally {
      uploading.value = false
      input.value = ''
    }
  }

  async function handleDelete(imageId: number) {
    try {
      await imagesApi.delete(imageId)
      images.value = images.value.filter(img => img.id !== imageId)
      delete imageUrls.value[imageId]
    } catch (e) {
      console.error('Delete failed', e)
    }
  }

  function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
    })
  }

  onMounted(fetchImages)

  return {
    images,
    imageUrls,
    loading,
    uploading,
    fetchImages,
    handleFileChange,
    handleDelete,
    formatDate,
  }
}