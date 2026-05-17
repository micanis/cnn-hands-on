<script setup>
import bgImage from "../assets/BackGround.png"
</script>

<template>
  <div class="slidev-layout cover h-full w-full bg-cover bg-no-repeat relative p-0" :style="{backgroundImage: `url(${bgImage})`}">
    
    <div class="absolute top-[40%] left-0 w-full text-center">
      
      <div class="title-wrapper text-4xl font-bold text-neutral-900">
        <slot name="title" />
      </div>
      
      <div class="date-wrapper mt-10 text-xl font-normal text-neutral-800 flex flex-col gap-0">
        <slot name="date" />
        <slot />
      </div>

    </div>

  </div>
</template>

<style scoped>
.date-wrapper :deep(p) {
  margin: 0 !important;
}
</style>