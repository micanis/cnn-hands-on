<script setup>
defineProps({
  // デフォルト幅を700pxにしつつ、外から自由に変更できるようにする
  width: {
    type: String,
    default: '800px'
  }
})
</script>

<template>
  <div class="h-full w-full flex justify-center items-stretch">
    <AutoFit>
      <div :style="{ width: width }">
        <slot />
      </div>
    </AutoFit>
  </div>
</template>