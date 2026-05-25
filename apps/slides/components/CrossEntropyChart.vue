<script setup>
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkAreaComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([LineChart, GridComponent, TooltipComponent, LegendComponent, MarkAreaComponent, CanvasRenderer])

const props = defineProps({
  width: { type: String, default: '100%' },
  height: { type: String, default: '220px' }
})

const generateLogLoss = () => {
  const data = []
  for (let p = 0.01; p <= 1.0; p += 0.01) {
    data.push([p, -Math.log(p)])
  }
  return data
}

const option = computed(() => ({
  animation: true,
  animationDuration: 1200,
  grid: {
    left: '14%', right: '8%', top: '10%', bottom: '18%'
  },
  tooltip: {
    trigger: 'axis',
    formatter: (params) => {
      const p = params[0].data[0]
      const loss = params[0].data[1]
      return `確率: ${(p * 100).toFixed(0)}%<br/>Loss: ${loss.toFixed(3)}`
    }
  },
  xAxis: {
    type: 'value',
    min: 0, max: 1,
    name: '正解クラスの確率 p',
    nameLocation: 'center',
    nameGap: 28,
    axisLabel: {
      fontSize: 10,
      formatter: (v) => `${(v * 100).toFixed(0)}%`
    },
    nameTextStyle: { fontSize: 11 }
  },
  yAxis: {
    type: 'value',
    min: 0, max: 5,
    name: 'Loss = -log(p)',
    nameLocation: 'center',
    nameGap: 30,
    axisLabel: { fontSize: 10 },
    nameTextStyle: { fontSize: 11 }
  },
  series: [{
    type: 'line',
    data: generateLogLoss(),
    smooth: true,
    symbol: 'none',
    lineStyle: { width: 3, color: '#f97316' },
    areaStyle: {
      color: {
        type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: 'rgba(249,115,22,0.2)' },
          { offset: 1, color: 'rgba(249,115,22,0.02)' }
        ]
      }
    },
    markArea: {
      silent: true,
      data: [
        [
          { xAxis: 0.8, itemStyle: { color: 'rgba(34,197,94,0.08)' } },
          { xAxis: 1.0 }
        ],
        [
          { xAxis: 0.0, itemStyle: { color: 'rgba(239,68,68,0.08)' } },
          { xAxis: 0.2 }
        ]
      ]
    }
  }]
}))
</script>

<template>
  <v-chart :option="option" :style="{ width, height }" autoresize />
</template>
