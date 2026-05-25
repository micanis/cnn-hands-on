<script setup>
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkPointComponent, MarkLineComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([LineChart, GridComponent, TooltipComponent, MarkPointComponent, MarkLineComponent, CanvasRenderer])

const props = defineProps({
  width: { type: String, default: '100%' },
  height: { type: String, default: '260px' },
  showPath: { type: Boolean, default: true }
})

// シンプルな放物線: 最小値が x=0 に明確に存在する
const lossFn = (x) => 0.4 * x * x + 1.5

const generateCurve = () => {
  const data = []
  for (let x = -3.5; x <= 3.5; x += 0.05) {
    data.push([x, lossFn(x)])
  }
  return data
}

// 勾配降下の経路: 大きいステップから小さいステップへ
const gradientPath = [
  [-2.8, lossFn(-2.8)],
  [-2.0, lossFn(-2.0)],
  [-1.3, lossFn(-1.3)],
  [-0.7, lossFn(-0.7)],
  [-0.3, lossFn(-0.3)],
  [-0.1, lossFn(-0.1)],
  [0.0, lossFn(0.0)],
]

const option = computed(() => ({
  animation: true,
  animationDuration: 2000,
  grid: {
    left: '12%', right: '8%', top: '8%', bottom: '15%'
  },
  tooltip: { show: false },
  xAxis: {
    type: 'value',
    min: -3.5, max: 3.5,
    name: 'パラメータ W',
    nameLocation: 'center',
    nameGap: 25,
    axisLabel: { show: false },
    splitLine: { show: false },
    nameTextStyle: { fontSize: 11, color: '#666' }
  },
  yAxis: {
    type: 'value',
    min: 0.5, max: 7,
    name: '損失 L',
    nameLocation: 'center',
    nameGap: 30,
    axisLabel: { show: false },
    splitLine: { show: false },
    nameTextStyle: { fontSize: 11, color: '#666' }
  },
  series: [
    {
      name: '損失関数',
      type: 'line',
      data: generateCurve(),
      smooth: true,
      symbol: 'none',
      lineStyle: { width: 3, color: '#3b82f6' },
      areaStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(59,130,246,0.15)' },
            { offset: 1, color: 'rgba(59,130,246,0.02)' }
          ]
        }
      }
    },
    ...(props.showPath ? [{
      name: '最適化の経路',
      type: 'line',
      data: gradientPath,
      symbol: 'circle',
      symbolSize: 10,
      lineStyle: { width: 2, color: '#ef4444', type: 'dashed' },
      itemStyle: { color: '#ef4444' },
      markPoint: {
        data: [
          {
            coord: gradientPath[0],
            symbol: 'circle',
            symbolSize: 14,
            itemStyle: { color: '#ef4444' },
            label: { show: true, formatter: '開始', position: 'top', fontSize: 11, color: '#ef4444', fontWeight: 'bold' }
          },
          {
            coord: gradientPath[gradientPath.length - 1],
            symbol: 'circle',
            symbolSize: 14,
            itemStyle: { color: '#22c55e' },
            label: { show: true, formatter: '最小値', position: 'bottom', fontSize: 11, color: '#22c55e', fontWeight: 'bold' }
          }
        ]
      }
    }] : [])
  ]
}))
</script>

<template>
  <v-chart :option="option" :style="{ width, height }" autoresize />
</template>
