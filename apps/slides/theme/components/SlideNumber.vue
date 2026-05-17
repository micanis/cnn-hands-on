<script setup>
// Propsの定義
defineProps({
    top: {type: String, default: "92.5%"},
    left: {type: String, default: "85%"},
    color: {type: String, default: "text-neutral-900"}
})
</script>

<template>
    <div
        class="absolute text-xl font-bold z-99"
        :class="color"
        :style="{ top, left}"
    >
        {{ $slidev.nav.currentPage }}
    </div>
</template>