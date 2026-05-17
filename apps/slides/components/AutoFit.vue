<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

const container = ref(null)
const content = ref(null)
const scaleFactor = ref(1)
let observer = null

const adjust = () => {
  if (!container.value || !content.value) return
  
  // 親の箱の「幅」と「高さ」を測る
  const availableWidth = container.value.clientWidth
  const availableHeight = container.value.clientHeight
  
  // 中身の実際の「幅」と「高さ」を測る
  const contentWidth = content.value.scrollWidth
  const contentHeight = content.value.scrollHeight

  let scaleW = 1
  let scaleH = 1

  // 横にはみ出している場合の縮小率を計算
  if (contentWidth > availableWidth && availableWidth > 0) {
    scaleW = availableWidth / contentWidth
  }
  
  // 縦にはみ出している場合の縮小率を計算
  if (contentHeight > availableHeight && availableHeight > 0) {
    scaleH = availableHeight / contentHeight
  }

  // 縦と横、どちらか「より厳しく縮小しなければならない方」を採用し、0.98を掛けて少し余裕を持たせる
  scaleFactor.value = Math.min(scaleW, scaleH) * 0.98
}

onMounted(() => {
  observer = new ResizeObserver(() => adjust())
  if (container.value) observer.observe(container.value)
  if (content.value) observer.observe(content.value)
  
  setTimeout(adjust, 50)
})

onBeforeUnmount(() => {
  if (observer) observer.disconnect()
})
</script>

<template>
  <div ref="container" class="w-full h-full overflow-hidden flex items-center justify-center">
    
    <div 
      ref="content" 
      class="origin-center transition-transform duration-200 ease-out inline-block max-w-none flex-shrink-0"
      :style="{ transform: `scale(${scaleFactor})` }"
    >
      <slot />
    </div>
    
  </div>
</template>