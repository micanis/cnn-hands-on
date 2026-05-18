<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const predictions = ref([
  { label: 'ネコ', prob: 0.92, color: 'bg-orange-500' },
  { label: 'イヌ', prob: 0.05, color: 'bg-blue-500' },
  { label: 'ウサギ', prob: 0.02, color: 'bg-green-500' },
  { label: 'キツネ', prob: 0.01, color: 'bg-purple-500' }
])

const isAnimating = ref(false)
const showResult = ref(true)

const randomize = () => {
  isAnimating.value = true
  showResult.value = false

  let count = 0
  const interval = setInterval(() => {
    const newProbs = predictions.value.map(() => Math.random())
    const sum = newProbs.reduce((a, b) => a + b, 0)
    predictions.value = predictions.value.map((p, i) => ({
      ...p,
      prob: newProbs[i] / sum
    }))
    count++
    if (count > 10) {
      clearInterval(interval)
      isAnimating.value = false
      showResult.value = true
      predictions.value.sort((a, b) => b.prob - a.prob)
    }
  }, 100)
}

const formatProb = (prob) => {
  return (prob * 100).toFixed(1) + '%'
}
</script>

<template>
  <div class="flex flex-col items-center gap-6">
    <div class="flex items-center gap-8">
      <div class="flex flex-col items-center">
        <div class="text-sm font-bold mb-2 text-gray-500">入力画像</div>
        <div class="w-32 h-32 bg-gradient-to-br from-orange-200 to-orange-400 rounded-lg shadow-lg flex items-center justify-center text-6xl">
          🐱
        </div>
      </div>

      <div class="flex flex-col items-center gap-2">
        <div class="text-4xl text-gray-300 animate-pulse">→</div>
        <div class="text-xs text-gray-400">CNN</div>
      </div>

      <div class="w-64">
        <div class="text-sm font-bold mb-3 text-gray-500 text-center">予測結果</div>
        <div class="space-y-2">
          <div
            v-for="pred in predictions"
            :key="pred.label"
            class="flex items-center gap-2"
          >
            <div class="w-12 text-sm font-medium text-right">{{ pred.label }}</div>
            <div class="flex-1 h-6 bg-gray-200 rounded-full overflow-hidden">
              <div
                class="h-full rounded-full transition-all duration-300"
                :class="[pred.color, isAnimating ? 'opacity-70' : '']"
                :style="{ width: formatProb(pred.prob) }"
              />
            </div>
            <div class="w-14 text-sm font-mono text-right">
              {{ formatProb(pred.prob) }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="showResult && !isAnimating" class="text-center">
      <div class="text-lg">
        判定: <span class="font-bold text-orange-600">{{ predictions[0]?.label }}</span>
        <span class="text-gray-500">({{ formatProb(predictions[0]?.prob) }})</span>
      </div>
    </div>

    <button
      @click="randomize"
      :disabled="isAnimating"
      class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50"
    >
      {{ isAnimating ? '推論中...' : '🔄 再推論' }}
    </button>
  </div>
</template>
