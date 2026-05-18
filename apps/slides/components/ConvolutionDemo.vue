<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  autoPlay: { type: Boolean, default: false }
})

const input = [
  [1, 2, 3, 4, 5],
  [2, 3, 4, 5, 6],
  [3, 4, 5, 6, 7],
  [4, 5, 6, 7, 8],
  [5, 6, 7, 8, 9]
]

const kernel = [
  [1, 0, -1],
  [1, 0, -1],
  [1, 0, -1]
]

const position = ref({ x: 0, y: 0 })
const isPlaying = ref(props.autoPlay)
let intervalId = null

const maxPos = input.length - kernel.length

const currentOutput = computed(() => {
  let sum = 0
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      sum += input[position.value.y + i][position.value.x + j] * kernel[i][j]
    }
  }
  return sum
})

const outputMatrix = computed(() => {
  const result = []
  for (let y = 0; y <= maxPos; y++) {
    const row = []
    for (let x = 0; x <= maxPos; x++) {
      if (y < position.value.y || (y === position.value.y && x <= position.value.x)) {
        let sum = 0
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            sum += input[y + i][x + j] * kernel[i][j]
          }
        }
        row.push(sum)
      } else {
        row.push(null)
      }
    }
    result.push(row)
  }
  return result
})

const isInKernel = (x, y) => {
  return x >= position.value.x && x < position.value.x + 3 &&
         y >= position.value.y && y < position.value.y + 3
}

const step = () => {
  if (position.value.x < maxPos) {
    position.value.x++
  } else if (position.value.y < maxPos) {
    position.value.x = 0
    position.value.y++
  } else {
    position.value = { x: 0, y: 0 }
  }
}

const togglePlay = () => {
  isPlaying.value = !isPlaying.value
  if (isPlaying.value) {
    intervalId = setInterval(step, 800)
  } else {
    clearInterval(intervalId)
  }
}

const reset = () => {
  position.value = { x: 0, y: 0 }
  isPlaying.value = false
  clearInterval(intervalId)
}
</script>

<template>
  <div class="flex flex-col items-center gap-4">
    <div class="flex items-center gap-6">
      <div>
        <div class="text-xs text-center mb-1 font-bold">入力 (5×5)</div>
        <div class="grid grid-cols-5 gap-0.5 p-1 bg-gray-200 rounded">
          <template
            v-for="(row, y) in input"
            :key="'row-' + y"
          >
            <div
              v-for="(val, x) in row"
              :key="'cell-' + x"
              class="w-8 h-8 flex items-center justify-center text-sm font-mono rounded transition-all duration-200"
              :class="isInKernel(x, y) ? 'bg-blue-400 text-white font-bold scale-105' : 'bg-white'"
            >
              {{ val }}
            </div>
          </template>
        </div>
      </div>

      <div class="text-2xl font-bold text-gray-400">×</div>

      <div>
        <div class="text-xs text-center mb-1 font-bold">カーネル (3×3)</div>
        <div class="grid grid-cols-3 gap-0.5 p-1 bg-blue-500 rounded">
          <template
            v-for="(row, y) in kernel"
            :key="'k-row-' + y"
          >
            <div
              v-for="(val, x) in row"
              :key="'k-cell-' + x"
              class="w-8 h-8 flex items-center justify-center text-sm font-mono bg-blue-100 rounded font-bold"
            >
              {{ val }}
            </div>
          </template>
        </div>
      </div>

      <div class="text-2xl font-bold text-gray-400">=</div>

      <div>
        <div class="text-xs text-center mb-1 font-bold">出力 (3×3)</div>
        <div class="grid grid-cols-3 gap-0.5 p-1 bg-gray-200 rounded">
          <template
            v-for="(row, y) in outputMatrix"
            :key="'o-row-' + y"
          >
            <div
              v-for="(val, x) in row"
              :key="'o-cell-' + x"
              class="w-8 h-8 flex items-center justify-center text-sm font-mono rounded transition-all duration-200"
              :class="val !== null ? (x === position.x && y === position.y ? 'bg-green-400 text-white font-bold' : 'bg-green-100') : 'bg-gray-100'"
            >
              {{ val !== null ? val : '' }}
            </div>
          </template>
        </div>
      </div>
    </div>

    <div class="text-sm bg-gray-100 px-4 py-2 rounded font-mono">
      計算: {{ currentOutput }}
    </div>

    <div class="flex gap-2">
      <button
        @click="togglePlay"
        class="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 transition"
      >
        {{ isPlaying ? '⏸ 停止' : '▶ 再生' }}
      </button>
      <button
        @click="step"
        :disabled="isPlaying"
        class="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600 transition disabled:opacity-50"
      >
        → 次へ
      </button>
      <button
        @click="reset"
        class="px-3 py-1 bg-gray-300 text-gray-700 rounded text-sm hover:bg-gray-400 transition"
      >
        ↺ リセット
      </button>
    </div>
  </div>
</template>
