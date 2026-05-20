<script setup>
import { use } from 'echarts/core'
import { GraphChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([GraphChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  type: {
    type: String,
    default: 'cnn-flow'
  },
  width: {
    type: String,
    default: '100%'
  },
  height: {
    type: String,
    default: '350px'
  }
})

const configs = {
  'cnn-flow': {
    nodes: [
      { name: '入力画像', x: 100, y: 20, color: '#dbeafe', border: '#93c5fd' },
      { name: 'Conv→ReLU→Pool (1)', x: 100, y: 75, color: '#bfdbfe', border: '#60a5fa' },
      { name: 'Conv→ReLU→Pool (2)', x: 100, y: 130, color: '#bfdbfe', border: '#60a5fa' },
      { name: 'Flatten', x: 100, y: 185, color: '#fed7aa', border: '#fdba74' },
      { name: '全結合層', x: 100, y: 240, color: '#fdba74', border: '#f97316' },
      { name: '出力（犬/猫）', x: 100, y: 295, color: '#bbf7d0', border: '#86efac' }
    ]
  },
  'cnn-simple': {
    nodes: [
      { name: '入力 (1,28,28)', x: 100, y: 20, color: '#f3f4f6', border: '#d1d5db' },
      { name: 'Conv→ReLU→Pool', x: 100, y: 95, color: '#dbeafe', border: '#93c5fd' },
      { name: 'Flatten', x: 100, y: 170, color: '#fef3c7', border: '#fcd34d' },
      { name: 'Linear', x: 100, y: 245, color: '#d1fae5', border: '#6ee7b7' },
      { name: '出力 (10,)', x: 100, y: 320, color: '#f3f4f6', border: '#d1d5db' }
    ]
  }
}

const option = computed(() => {
  const config = configs[props.type] || configs['cnn-flow']
  const nodes = config.nodes
  const links = []

  for (let i = 0; i < nodes.length - 1; i++) {
    links.push({ source: i, target: i + 1 })
  }

  return {
    animation: false,
    series: [{
      type: 'graph',
      layout: 'none',
      coordinateSystem: null,
      label: {
        show: true,
        fontSize: 12,
        color: '#333',
        fontWeight: 'bold'
      },
      edgeSymbol: ['none', 'arrow'],
      edgeSymbolSize: [0, 10],
      data: nodes.map((n, i) => ({
        name: n.name,
        x: n.x,
        y: n.y,
        symbol: 'roundRect',
        symbolSize: [140, 28],
        itemStyle: {
          color: n.color,
          borderColor: n.border,
          borderWidth: 2
        }
      })),
      links: links.map(l => ({
        ...l,
        lineStyle: { color: '#999', width: 2 }
      }))
    }]
  }
})
</script>

<template>
  <v-chart :option="option" :style="{ width, height }" autoresize />
</template>
