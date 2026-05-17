<script setup>
import SlideNumber from '../components/SlideNumber.vue';
import bgImage from "../assets/BackGround.png"
</script>

<template>
  <div class="slidev-layout toc h-full w-full  bg-cover bg-no-repeat relative p-0" :style="{backgroundImage: `url(${bgImage})`}">
    
    <div class="absolute title-wrapper">
      <div class="text-3xl font-bold text-neutral-900">
        <slot name="header" />
      </div>
    </div>

    <div class="absolute content-area">
      <div class="flex flex-col gap-6 pt-8 pl-12 text-2xl font-bold text-neutral-800">
        <div v-for="(item, index) in $slidev.nav.tocTree.filter(i => $slidev.nav.slides.find(s => s.no === i.no)?.meta?.slide?.frontmatter?.layout === 'section')" :key="item.no" class="flex items-center gap-4">
          <span class="text-neutral-600">{{ index + 1 }}.</span>
          <span>{{ item.title }}</span>
        </div>
      </div>
    </div>

    <SlideNumber />
  </div>
</template>

<style scoped>
.title-wrapper {
  left: 9%;
  top: 4%;
  width: auto;
  height: 10%;
}
.content-area {
  left: 3.23%;
  top: 18.06%;
  width: 93.75%;
  height: 69.44%;
}
</style>