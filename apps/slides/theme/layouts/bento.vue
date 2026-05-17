<script setup>
import SlideNumber from '../components/SlideNumber.vue';
import { computed, useAttrs } from 'vue'
import bgImage from "../assets/BackGround.png"

const attrs = useAttrs()

const gridStyle = computed(() => ({
  display: 'grid',
  gridTemplateColumns: `repeat(${attrs.cols || 1}, minmax(0, 1fr))`,
  gridTemplateRows: `repeat(${attrs.rows || 1}, minmax(0, 1fr))`,
  gap: '1.5rem'
}))

const getSizeStyle = (key) => {
  const val = attrs[key]
  if (!val) return {}
  
  let w = 1
  let h = 1
  
  if (typeof val === 'string') {
    const matchW = val.match(/w(\d+)/)
    const matchH = val.match(/h(\d+)/)
    if (matchW) w = matchW[1]
    if (matchH) h = matchH[1]
  }
  
  return {
    gridColumn: `span ${w} / span ${w}`,
    gridRow: `span ${h} / span ${h}`
  }
}

</script>

<template>
  <div class="slidev-layout default h-full w-full  bg-cover bg-no-repeat relative p-0" :style="{backgroundImage: `url(${bgImage})`}">
    
    <div class="absolute title-wrapper">
      <div class="text-3xl font-bold text-neutral-900">
        <slot name="header" />
      </div>
    </div>

    <div class="absolute content-area">
      
      <div class="w-full h-full" :style="gridStyle">
        
        <div v-if="$slots.a" :style="getSizeStyle('sizeA')" class="relative h-full w-full">
          <slot name="a" />
        </div>
        <div v-if="$slots.b" :style="getSizeStyle('sizeB')" class="relative h-full w-full">
          <slot name="b" />
        </div>
        <div v-if="$slots.c" :style="getSizeStyle('sizeC')" class="relative h-full w-full">
          <slot name="c" />
        </div>
        <div v-if="$slots.d" :style="getSizeStyle('sizeD')" class="relative h-full w-full">
          <slot name="d" />
        </div>
        <div v-if="$slots.e" :style="getSizeStyle('sizeE')" class="relative h-full w-full">
          <slot name="e" />
        </div>
        <div v-if="$slots.f" :style="getSizeStyle('sizeF')" class="relative h-full w-full">
          <slot name="f" />
        </div>

      </div>

    </div>

    <slide-number />
  </div>
</template>

<style scoped>
.title-wrapper {
    left: 9%;
    top: 4%;
    width:auto;
    height: 10%;
}

.content-area {
  left: 3.23%;
  top: 18.06%;
  width: 93.75%;
  height: 69.44%;
}


.content-area :deep(p) {
  margin-bottom: 1rem;
}
</style>