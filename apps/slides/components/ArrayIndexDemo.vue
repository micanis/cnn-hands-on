<script setup>
import { ref, computed } from 'vue'

const matrix = [
  [10, 20, 30],
  [40, 50, 60],
  [70, 80, 90]
]

const selectedRow = ref(null)
const selectedCol = ref(null)
const mode = ref('element')

const modes = [
  { key: 'element', label: '要素', desc: 'arr[行, 列]' },
  { key: 'row', label: '行', desc: 'arr[行]' },
  { key: 'col', label: '列', desc: 'arr[:, 列]' },
  { key: 'slice', label: 'スライス', desc: 'arr[1:, 1:]' }
]

const isSelected = (r, c) => {
  if (mode.value === 'element') {
    return r === selectedRow.value && c === selectedCol.value
  } else if (mode.value === 'row') {
    return r === selectedRow.value
  } else if (mode.value === 'col') {
    return c === selectedCol.value
  } else if (mode.value === 'slice') {
    return r >= 1 && c >= 1
  }
  return false
}

const handleClick = (r, c) => {
  if (mode.value === 'element') {
    selectedRow.value = r
    selectedCol.value = c
  } else if (mode.value === 'row') {
    selectedRow.value = r
  } else if (mode.value === 'col') {
    selectedCol.value = c
  }
}

const codeExample = computed(() => {
  if (mode.value === 'element' && selectedRow.value !== null && selectedCol.value !== null) {
    return `arr[${selectedRow.value}, ${selectedCol.value}] = ${matrix[selectedRow.value][selectedCol.value]}`
  } else if (mode.value === 'row' && selectedRow.value !== null) {
    return `arr[${selectedRow.value}] = [${matrix[selectedRow.value].join(', ')}]`
  } else if (mode.value === 'col' && selectedCol.value !== null) {
    return `arr[:, ${selectedCol.value}] = [${matrix.map(r => r[selectedCol.value]).join(', ')}]`
  } else if (mode.value === 'slice') {
    return `arr[1:, 1:] = [[50, 60], [80, 90]]`
  }
  return 'セルをクリックして選択'
})
</script>

<template>
  <div class="flex flex-col items-center gap-4">
    <div class="flex gap-2">
      <button
        v-for="m in modes"
        :key="m.key"
        @click="mode = m.key; selectedRow = null; selectedCol = null"
        class="px-3 py-1.5 rounded text-sm transition-all"
        :class="mode === m.key ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'"
      >
        {{ m.label }}
      </button>
    </div>

    <div class="flex items-start gap-6">
      <div>
        <div class="flex mb-1">
          <div class="w-8" />
          <div v-for="c in 3" :key="'h-' + c" class="w-12 text-center text-xs text-gray-500 font-mono">
            {{ c - 1 }}
          </div>
        </div>
        <div v-for="(row, r) in matrix" :key="'row-' + r" class="flex">
          <div class="w-8 flex items-center justify-center text-xs text-gray-500 font-mono">
            {{ r }}
          </div>
          <div
            v-for="(val, c) in row"
            :key="'cell-' + c"
            @click="handleClick(r, c)"
            class="w-12 h-12 flex items-center justify-center font-mono text-sm border cursor-pointer transition-all duration-200"
            :class="isSelected(r, c) ? 'bg-blue-500 text-white border-blue-600 scale-105' : 'bg-white border-gray-300 hover:bg-gray-100'"
          >
            {{ val }}
          </div>
        </div>
      </div>

      <div class="flex flex-col justify-center h-36">
        <div class="text-sm text-gray-500 mb-1">{{ modes.find(m => m.key === mode)?.desc }}</div>
        <div class="bg-gray-900 text-green-400 px-4 py-2 rounded font-mono text-sm min-w-48">
          {{ codeExample }}
        </div>
      </div>
    </div>
  </div>
</template>
