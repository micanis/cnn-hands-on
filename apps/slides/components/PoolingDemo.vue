<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  type: {
    type: String,
    default: 'max'
  },
  autoPlay: {
    type: Boolean,
    default: true
  }
})

const input = [
  [1, 3, 2, 1],
  [4, 2, 3, 1],
  [2, 4, 1, 2],
  [3, 1, 2, 3]
]

const flatInput = input.flat().map((val, idx) => ({
  val,
  row: Math.floor(idx / 4),
  col: idx % 4
}))

const currentRegion = ref(0)
const regions = [
  { row: 0, col: 0, color: 'blue' },
  { row: 0, col: 1, color: 'green' },
  { row: 1, col: 0, color: 'amber' },
  { row: 1, col: 1, color: 'rose' }
]

const getOutput = (regionIdx) => {
  const region = regions[regionIdx]
  const values = [
    input[region.row * 2][region.col * 2],
    input[region.row * 2][region.col * 2 + 1],
    input[region.row * 2 + 1][region.col * 2],
    input[region.row * 2 + 1][region.col * 2 + 1]
  ]
  if (props.type === 'max') {
    return Math.max(...values)
  }
  return (values.reduce((a, b) => a + b, 0) / 4).toFixed(1)
}

const getRegionIdx = (row, col) => Math.floor(row / 2) * 2 + Math.floor(col / 2)

const isInCurrentRegion = (row, col) => getRegionIdx(row, col) === currentRegion.value

const isMaxCell = (row, col) => {
  if (props.type !== 'max') return false
  const regionIdx = getRegionIdx(row, col)
  return input[row][col] == getOutput(regionIdx) && isInCurrentRegion(row, col)
}

let interval = null

onMounted(() => {
  if (props.autoPlay) {
    interval = setInterval(() => {
      currentRegion.value = (currentRegion.value + 1) % 4
    }, 1200)
  }
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
})
</script>

<template>
  <div class="flex items-center justify-center gap-8">
    <div>
      <div class="text-sm text-center mb-2 font-bold">入力 4×4</div>
      <div class="grid grid-cols-4 gap-0.5 bg-gray-300 p-0.5 rounded">
        <template v-for="cell in flatInput" :key="`${cell.row}-${cell.col}`">
          <div
            class="w-9 h-9 flex items-center justify-center text-sm rounded-sm transition-all duration-300"
            :class="{
              'bg-blue-100 border-blue-400': getRegionIdx(cell.row, cell.col) === 0,
              'bg-green-100 border-green-400': getRegionIdx(cell.row, cell.col) === 1,
              'bg-amber-100 border-amber-400': getRegionIdx(cell.row, cell.col) === 2,
              'bg-rose-100 border-rose-400': getRegionIdx(cell.row, cell.col) === 3,
              'ring-2 ring-gray-700 scale-105 z-10 font-bold': isInCurrentRegion(cell.row, cell.col),
              'bg-blue-300': getRegionIdx(cell.row, cell.col) === 0 && isInCurrentRegion(cell.row, cell.col),
              'bg-green-300': getRegionIdx(cell.row, cell.col) === 1 && isInCurrentRegion(cell.row, cell.col),
              'bg-amber-300': getRegionIdx(cell.row, cell.col) === 2 && isInCurrentRegion(cell.row, cell.col),
              'bg-rose-300': getRegionIdx(cell.row, cell.col) === 3 && isInCurrentRegion(cell.row, cell.col),
              'text-lg font-black underline': isMaxCell(cell.row, cell.col)
            }"
          >
            {{ cell.val }}
          </div>
        </template>
      </div>
    </div>

    <div class="flex flex-col items-center">
      <div class="text-2xl mb-1">→</div>
      <div class="text-xs text-gray-500">{{ type === 'max' ? 'max' : 'avg' }}</div>
    </div>

    <div>
      <div class="text-sm text-center mb-2 font-bold">出力 2×2</div>
      <div class="grid grid-cols-2 gap-0.5 bg-gray-300 p-0.5 rounded">
        <template v-for="(region, idx) in regions" :key="idx">
          <div
            class="w-9 h-9 flex items-center justify-center text-sm font-bold rounded-sm transition-all duration-300"
            :class="{
              'bg-blue-200': idx === 0,
              'bg-green-200': idx === 1,
              'bg-amber-200': idx === 2,
              'bg-rose-200': idx === 3,
              'ring-2 ring-gray-700 scale-110 z-10': currentRegion === idx,
              'bg-blue-400': idx === 0 && currentRegion === idx,
              'bg-green-400': idx === 1 && currentRegion === idx,
              'bg-amber-400': idx === 2 && currentRegion === idx,
              'bg-rose-400': idx === 3 && currentRegion === idx
            }"
          >
            {{ getOutput(idx) }}
          </div>
        </template>
      </div>
    </div>
  </div>
</template>
