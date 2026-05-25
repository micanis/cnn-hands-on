<script setup>
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([LineChart, GridComponent, TooltipComponent, MarkLineComponent, LegendComponent, CanvasRenderer])

const props = defineProps({
  type: {
    type: String,
    default: 'good' // 'good' | 'overfit'
  },
  width: { type: String, default: '100%' },
  height: { type: String, default: '240px' }
})

const goodData = {
  train: [2.3, 1.2, 0.7, 0.45, 0.32, 0.25, 0.20, 0.17, 0.15, 0.13],
  val:   [2.3, 1.3, 0.8, 0.55, 0.42, 0.36, 0.32, 0.30, 0.28, 0.27]
}

const overfitData = {
  train: [2.3, 1.1, 0.6, 0.35, 0.20, 0.12, 0.07, 0.04, 0.02, 0.01],
  val:   [2.3, 1.2, 0.7, 0.55, 0.50, 0.52, 0.58, 0.65, 0.73, 0.82]
}

const option = computed(() => {
  const data = props.type === 'overfit' ? overfitData : goodData
  const isOverfit = props.type === 'overfit'

  return {
    animation: true,
    animationDuration: 1500,
    grid: {
      left: '14%', right: '8%', top: '18%', bottom: '15%'
    },
    legend: {
      top: 0,
      textStyle: { fontSize: 11 }
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        return params.map(p => `${p.seriesName}: ${p.data.toFixed(3)}`).join('<br>')
      }
    },
    xAxis: {
      type: 'category',
      data: data.train.map((_, i) => i + 1),
      name: 'Epoch',
      nameLocation: 'center',
      nameGap: 25,
      axisLabel: { fontSize: 10 },
      nameTextStyle: { fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: 'Loss',
      nameLocation: 'center',
      nameGap: 35,
      axisLabel: { fontSize: 10 },
      nameTextStyle: { fontSize: 11 }
    },
    series: [
      {
        name: 'Train Loss',
        type: 'line',
        data: data.train,
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { width: 3, color: '#3b82f6' },
        itemStyle: { color: '#3b82f6' }
      },
      {
        name: 'Val Loss',
        type: 'line',
        data: data.val,
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: {
          width: 3,
          color: isOverfit ? '#ef4444' : '#f97316'
        },
        itemStyle: {
          color: isOverfit ? '#ef4444' : '#f97316'
        },
        markLine: isOverfit ? {
          silent: true,
          data: [{ xAxis: 4 }],
          lineStyle: { type: 'dashed', color: '#ef4444' },
          label: {
            formatter: '過学習開始',
            fontSize: 10,
            color: '#ef4444'
          }
        } : undefined
      }
    ]
  }
})
</script>

<template>
  <v-chart :option="option" :style="{ width, height }" autoresize />
</template>
