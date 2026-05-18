<script setup>
import { ref, computed } from 'vue'

const activeChannel = ref('all')

const pixelData = [
  [{ r: 255, g: 100, b: 50 }, { r: 200, g: 150, b: 100 }, { r: 150, g: 200, b: 150 }, { r: 100, g: 100, b: 200 }],
  [{ r: 230, g: 80, b: 80 }, { r: 180, g: 180, b: 80 }, { r: 80, g: 180, b: 180 }, { r: 120, g: 80, b: 230 }],
  [{ r: 200, g: 50, b: 100 }, { r: 150, g: 200, b: 50 }, { r: 50, g: 150, b: 200 }, { r: 100, g: 50, b: 180 }],
  [{ r: 180, g: 120, b: 60 }, { r: 120, g: 180, b: 120 }, { r: 60, g: 120, b: 180 }, { r: 150, g: 100, b: 150 }]
]

const getColor = (pixel) => {
  switch (activeChannel.value) {
    case 'r': return `rgb(${pixel.r}, 0, 0)`
    case 'g': return `rgb(0, ${pixel.g}, 0)`
    case 'b': return `rgb(0, 0, ${pixel.b})`
    default: return `rgb(${pixel.r}, ${pixel.g}, ${pixel.b})`
  }
}

const getValue = (pixel) => {
  switch (activeChannel.value) {
    case 'r': return pixel.r
    case 'g': return pixel.g
    case 'b': return pixel.b
    default: return null
  }
}

const channels = [
  { key: 'all', label: 'RGB', color: 'bg-gradient-to-r from-red-500 via-green-500 to-blue-500' },
  { key: 'r', label: 'R', color: 'bg-red-500' },
  { key: 'g', label: 'G', color: 'bg-green-500' },
  { key: 'b', label: 'B', color: 'bg-blue-500' }
]
</script>

<template>
  <div class="flex flex-col items-center gap-4">
    <div class="flex gap-2">
      <button
        v-for="ch in channels"
        :key="ch.key"
        @click="activeChannel = ch.key"
        class="px-4 py-2 rounded text-sm font-bold transition-all"
        :class="[
          ch.color,
          activeChannel === ch.key ? 'ring-2 ring-offset-2 ring-gray-800 text-white' : 'opacity-50 text-white'
        ]"
      >
        {{ ch.label }}
      </button>
    </div>

    <div class="grid grid-cols-4 gap-1 p-2 bg-gray-800 rounded-lg">
      <template
        v-for="(row, y) in pixelData"
        :key="'row-' + y"
      >
        <div
          v-for="(pixel, x) in row"
          :key="'pixel-' + x"
          class="w-16 h-16 rounded flex items-center justify-center text-xs font-mono transition-all duration-300"
          :style="{ backgroundColor: getColor(pixel) }"
        >
          <span v-if="getValue(pixel) !== null" class="text-white font-bold drop-shadow-lg">
            {{ getValue(pixel) }}
          </span>
        </div>
      </template>
    </div>

    <div class="text-sm text-gray-600">
      <span v-if="activeChannel === 'all'">RGB合成画像 - 3チャンネルを重ね合わせた状態</span>
      <span v-else-if="activeChannel === 'r'">Rチャンネル - 赤色成分のみ (0-255)</span>
      <span v-else-if="activeChannel === 'g'">Gチャンネル - 緑色成分のみ (0-255)</span>
      <span v-else>Bチャンネル - 青色成分のみ (0-255)</span>
    </div>
  </div>
</template>
