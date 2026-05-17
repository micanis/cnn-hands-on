<script setup>
import SlideNumber from '../components/SlideNumber.vue';
import bgImage from "../assets/BackGround.png"
</script>

<template>
  <div class="slidev-layout two-cols h-full w-full  bg-cover bg-no-repeat relative p-0" :style="{backgroundImage: `url(${bgImage})`}">
    
    <div class="absolute title-wrapper">
      <div class="text-3xl font-bold text-neutral-900">
        <slot name="header" />
      </div>
    </div>

    <div class="absolute content-area flex flex-col gap-4">
      
      <div class="flex-grow min-h-0 grid grid-cols-3 grid-rows-1 gap-8">
        
        <div class="relative min-h-0 overflow-hidden text-xl text-neutral-800">
          <slot name="left"/>
        </div>
        
        <div class="relative min-h-0 overflow-hidden text-xl text-neutral-800">
          <slot name="center" />
        </div>

        <div class="relative min-h-0 overflow-hidden text-xl text-neutral-800">
          <slot name="right" />
        </div>

      </div>

      <div v-if="$slots.conc" class="shrink-0 min-h-[2.5rem] w-full text-2xl font-bold flex items-center justify-center">
        <slot name="conc" />
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
/* 先ほど確定した黄金比率コンテナ */
.content-area {
  left: 3.23%;
  top: 18.06%;
  width: 93.75%;
  height: 69.44%;
}

.content-area :deep(p) {
  margin-bottom: 1rem;
}

/* 結論エリアの中のpタグは、中央揃えのために余白を消す */
.content-area :deep(.shrink-0 p) {
  margin: 0 !important;
}
</style>