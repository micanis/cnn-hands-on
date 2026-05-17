<script setup>
import SlideNumber from '../components/SlideNumber.vue';
import bgImage from "../assets/BackGround.png"
</script>

<template>
  <div class="slidev-layout default h-full w-full bg-cover bg-no-repeat relative p-0" :style="{backgroundImage: `url(${bgImage})`}">
    
    <div class="absolute title-wrapper">
      <div class="text-3xl font-bold text-neutral-900">
        <slot name="header" />
      </div>
    </div>

    <div class="absolute content-area text-xl text-neutral-800 leading-relaxed">
      <slot />
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
/* Figmaの数値をパーセンテージに変換して完璧に配置 */
.content-area {
  left: 3.23%;
  top: 18.06%;
  width: 93.75%;
  height: 69.44%;
}

/* （任意）Markdownで普通に文字を打った時の段落の余白を整える */
.content-area :deep(p) {
  margin-bottom: 1rem;
}
</style>