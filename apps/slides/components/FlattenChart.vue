<script setup>
import { use } from 'echarts/core'
import { CustomChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, GraphicComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

use([CustomChart, GridComponent, TooltipComponent, GraphicComponent, CanvasRenderer])

const props = defineProps({
  width: { type: String, default: '320px' },
  height: { type: String, default: '280px' }
})

const option = computed(() => {
  const cellSize = 32
  const gap = 4
  const colors3d = ['#93c5fd', '#a5b4fc']
  const colorsFlat = '#fdba74'

  // Center offset
  const centerX = 160
  const centerY = parseInt(props.height) / 2

  // 3D array visualization (2 channels x 2x2)
  const grid3d = []
  const channels = 2
  const gridSize = 2
  const grid3dWidth = gridSize * (cellSize + gap) + (channels - 1) * 12
  const grid3dStartX = centerX - 80 - grid3dWidth / 2

  for (let layer = channels - 1; layer >= 0; layer--) {
    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        const offset = layer * 12
        grid3d.push({
          type: 'rect',
          shape: {
            x: grid3dStartX + col * (cellSize + gap) + offset,
            y: centerY - 20 + row * (cellSize + gap) - offset,
            width: cellSize,
            height: cellSize,
            r: 4
          },
          style: {
            fill: colors3d[layer],
            stroke: '#3b82f6',
            lineWidth: layer === 0 ? 2 : 1,
            opacity: layer === 0 ? 1 : 0.6
          },
          z2: layer === 0 ? 10 : 5
        })
        if (layer === 0) {
          grid3d.push({
            type: 'text',
            style: {
              x: grid3dStartX + col * (cellSize + gap) + cellSize / 2,
              y: centerY - 20 + row * (cellSize + gap) + cellSize / 2,
              text: String(row * gridSize + col + 1),
              fontSize: 14,
              fontWeight: 'bold',
              fill: '#1e40af',
              textAlign: 'center',
              textVerticalAlign: 'middle'
            },
            z2: 15
          })
        }
      }
    }
  }

  // Arrow
  const arrowX = centerX - 10
  const arrow = [
    {
      type: 'polyline',
      shape: {
        points: [[arrowX, centerY + 20], [arrowX + 35, centerY + 20]]
      },
      style: {
        stroke: '#f97316',
        lineWidth: 3
      }
    },
    {
      type: 'polygon',
      shape: {
        points: [[arrowX + 35, centerY + 14], [arrowX + 48, centerY + 20], [arrowX + 35, centerY + 26]]
      },
      style: {
        fill: '#f97316'
      }
    },
    {
      type: 'text',
      style: {
        x: arrowX + 20,
        y: centerY + 50,
        text: 'flatten',
        fontSize: 13,
        fontWeight: 'bold',
        fill: '#f97316',
        textAlign: 'center',
        textVerticalAlign: 'middle'
      }
    }
  ]

  // 1D array visualization - vertical stack
  const flat1d = []
  const totalCells = channels * gridSize * gridSize
  const smallCell = 24
  const flat1dHeight = totalCells * (smallCell + 3)
  const flat1dStartX = centerX + 60
  const flat1dStartY = centerY + 20 - flat1dHeight / 2

  for (let i = 0; i < totalCells; i++) {
    flat1d.push({
      type: 'rect',
      shape: {
        x: flat1dStartX,
        y: flat1dStartY + i * (smallCell + 3),
        width: smallCell,
        height: smallCell,
        r: 3
      },
      style: {
        fill: colorsFlat,
        stroke: '#f97316',
        lineWidth: 2
      }
    })
    flat1d.push({
      type: 'text',
      style: {
        x: flat1dStartX + smallCell / 2,
        y: flat1dStartY + i * (smallCell + 3) + smallCell / 2,
        text: String(i + 1),
        fontSize: 12,
        fontWeight: 'bold',
        fill: '#c2410c',
        textAlign: 'center',
        textVerticalAlign: 'middle'
      }
    })
  }

  // Bracket for 1D array
  flat1d.push({
    type: 'polyline',
    shape: {
      points: [
        [flat1dStartX + smallCell + 8, flat1dStartY],
        [flat1dStartX + smallCell + 18, flat1dStartY],
        [flat1dStartX + smallCell + 18, flat1dStartY + flat1dHeight - 3],
        [flat1dStartX + smallCell + 8, flat1dStartY + flat1dHeight - 3]
      ]
    },
    style: {
      stroke: '#999',
      lineWidth: 2,
      fill: 'none'
    }
  })
  flat1d.push({
    type: 'text',
    style: {
      x: flat1dStartX + smallCell + 30,
      y: flat1dStartY + flat1dHeight / 2,
      text: String(totalCells),
      fontSize: 14,
      fontWeight: 'bold',
      fill: '#666',
      textAlign: 'left',
      textVerticalAlign: 'middle'
    }
  })

  // Labels
  const labels = [
    {
      type: 'text',
      style: {
        x: grid3dStartX + grid3dWidth / 2,
        y: centerY - 55,
        text: '3次元配列',
        fontSize: 13,
        fill: '#666',
        textAlign: 'center',
        textVerticalAlign: 'middle'
      }
    },
    {
      type: 'text',
      style: {
        x: grid3dStartX + grid3dWidth / 2,
        y: centerY + 75,
        text: '(2, 2, 2)',
        fontSize: 12,
        fill: '#999',
        textAlign: 'center',
        textVerticalAlign: 'middle'
      }
    },
    {
      type: 'text',
      style: {
        x: flat1dStartX + smallCell / 2,
        y: flat1dStartY + flat1dHeight + 20,
        text: '1次元 (8,)',
        fontSize: 12,
        fill: '#999',
        textAlign: 'center',
        textVerticalAlign: 'middle'
      }
    }
  ]

  return {
    animation: false,
    graphic: {
      elements: [...grid3d, ...arrow, ...flat1d, ...labels]
    }
  }
})
</script>

<template>
  <div class="flex items-center justify-center w-full h-full">
    <v-chart :option="option" :style="{ width, height }" autoresize />
  </div>
</template>
