<script setup>
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([LineChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  type: {
    type: String,
    default: 'relu'
  },
  color: {
    type: String,
    default: '#3b82f6'
  },
  width: {
    type: String,
    default: '100%'
  },
  height: {
    type: String,
    default: '200px'
  }
})

const generateData = (type) => {
  const data = []
  for (let x = -5; x <= 5; x += 0.1) {
    let y
    switch (type) {
      case 'sigmoid':
        y = 1 / (1 + Math.exp(-x))
        break
      case 'tanh':
        y = Math.tanh(x)
        break
      case 'relu':
        y = Math.max(0, x)
        break
      case 'leaky_relu':
        y = x > 0 ? x : 0.01 * x
        break
      default:
        y = Math.max(0, x)
    }
    data.push([x, y])
  }
  return data
}

const option = computed(() => ({
  animation: true,
  grid: {
    left: '12%',
    right: '8%',
    top: '10%',
    bottom: '15%'
  },
  xAxis: {
    type: 'value',
    min: -5,
    max: 5,
    axisLine: { lineStyle: { color: '#666' } },
    splitLine: { show: false },
    axisLabel: { fontSize: 10 }
  },
  yAxis: {
    type: 'value',
    min: props.type === 'relu' || props.type === 'leaky_relu' ? -1 : -1.5,
    max: props.type === 'relu' || props.type === 'leaky_relu' ? 5 : 1.5,
    axisLine: { lineStyle: { color: '#666' } },
    splitLine: { show: false },
    axisLabel: { fontSize: 10 }
  },
  series: [{
    type: 'line',
    data: generateData(props.type),
    smooth: props.type !== 'relu' && props.type !== 'leaky_relu',
    symbol: 'none',
    lineStyle: {
      color: props.color,
      width: 3
    }
  }]
}))
</script>

<template>
  <v-chart :option="option" :style="{ width, height }" autoresize />
</template>
