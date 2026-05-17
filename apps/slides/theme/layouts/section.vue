<script setup>
import SlideNumber from '../components/SlideNumber.vue';
import bgImage from "../assets/BackGround.png"
</script>

<template>
  <div class="slidev-layout section h-full w-full  bg-cover bg-no-repeat relative p-0" :style="{backgroundImage: `url(${bgImage})`}">
    <div class="hidden">
      <slot />
    </div>
    <div class="absolute inset-0 flex flex-col justify-center items-center text-center px-16">
      <div class="text-3xl font-bold text-neutral-900 tracking-widest">
        <template v-for="(item, index) in $slidev.nav.tocTree.filter(i => $slidev.nav.slides.find(s => s.no === i.no)?.meta?.slide?.frontmatter?.layout === 'section')" :key="item.no">
          <span v-if="item.no === $slidev.nav.currentPage">
            {{ index + 1 }}. {{ item.title }}
          </span>
        </template>
        </div>
      </div>
    <slide-number />
  </div>
</template>