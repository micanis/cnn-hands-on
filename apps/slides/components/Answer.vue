<script setup>
const showAnswers = import.meta.env.VITE_SHOW_ANSWERS === 'true'
</script>

<template>
  <div v-if="showAnswers" class="answer">
    <slot />
  </div>
</template>
