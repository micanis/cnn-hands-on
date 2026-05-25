<script setup>
import { use } from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([BarChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  type: {
    type: String,
    default: 'good' // 'good' | 'bad' | 'random'
  },
  width: { type: String, default: '100%' },
  height: { type: String, default: '200px' }
})

const configs = {
  good: {
    scores: [0.92, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.003, 0.002],
    correctIdx: 0,
    title: '良い予測（Loss小）'
  },
  bad: {
    scores: [0.02, 0.03, 0.01, 0.01, 0.01, 0.88, 0.01, 0.01, 0.01, 0.01],
    correctIdx: 0,
    title: '悪い予測（Loss大）'
  },
  random: {
    scores: [0.12, 0.09, 0.11, 0.08, 0.13, 0.10, 0.09, 0.11, 0.08, 0.09],
    correctIdx: 0,
    title: 'ランダム予測'
  }
}

const option = computed(() => {
  const config = configs[props.type] || configs.good
  const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

  return {
    animation: true,
    animationDuration: 800,
    grid: {
      left: '10%', right: '5%', top: '15%', bottom: '18%'
    },
    title: {
      text: config.title,
      left: 'center',
      top: 0,
      textStyle: { fontSize: 12, color: '#555' }
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params) => `クラス ${params[0].name}: ${(params[0].data * 100).toFixed(1)}%`
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { fontSize: 10 },
      name: 'クラス',
      nameLocation: 'center',
      nameGap: 22,
      nameTextStyle: { fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      max: 1.0,
      axisLabel: {
        fontSize: 10,
        formatter: (v) => `${(v * 100).toFixed(0)}%`
      }
    },
    series: [{
      type: 'bar',
      data: config.scores.map((v, i) => ({
        value: v,
        itemStyle: {
          color: i === config.correctIdx ? '#22c55e' : '#cbd5e1',
          borderRadius: [3, 3, 0, 0]
        }
      })),
      barWidth: '60%'
    }]
  }
})
</script>

<template>
  <v-chart :option="option" :style="{ width, height }" autoresize />
</template>
