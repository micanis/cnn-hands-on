<script setup>
import { use } from 'echarts/core'
import { GraphChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([GraphChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  inputNodes: { type: Number, default: 4 },
  outputNodes: { type: Number, default: 2 },
  width: { type: String, default: '100%' },
  height: { type: String, default: '200px' }
})

const option = computed(() => {
  const nodes = []
  const links = []

  const inputSpacing = 80 / (props.inputNodes + 1)
  const outputSpacing = 80 / (props.outputNodes + 1)

  // Input nodes
  for (let i = 0; i < props.inputNodes; i++) {
    nodes.push({
      name: `x${i + 1}`,
      x: 20,
      y: (i + 1) * inputSpacing,
      itemStyle: { color: '#93c5fd', borderColor: '#3b82f6', borderWidth: 2 }
    })
  }

  // Output nodes
  const outputYSpacing = 80 / (props.outputNodes + 1)
  for (let i = 0; i < props.outputNodes; i++) {
    nodes.push({
      name: `y${i + 1}`,
      x: 80,
      y: (i + 1) * outputYSpacing,
      itemStyle: { color: '#fdba74', borderColor: '#f97316', borderWidth: 2 }
    })
  }

  // Links (fully connected)
  for (let i = 0; i < props.inputNodes; i++) {
    for (let j = 0; j < props.outputNodes; j++) {
      links.push({
        source: i,
        target: props.inputNodes + j,
        lineStyle: { color: '#ddd', width: 1 }
      })
    }
  }

  return {
    animation: false,
    series: [{
      type: 'graph',
      layout: 'none',
      coordinateSystem: null,
      label: {
        show: true,
        fontSize: 11,
        color: '#333',
        fontWeight: 'bold'
      },
      data: nodes.map(n => ({
        ...n,
        symbol: 'circle',
        symbolSize: 28
      })),
      links
    }]
  }
})
</script>

<template>
  <v-chart :option="option" :style="{ width, height }" autoresize />
</template>
