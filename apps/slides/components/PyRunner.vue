<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps<{
  init?: string
}>()

const code = ref('')
const output = ref('')
const isRunning = ref(false)
const isLoading = ref(true)
const pyodide = ref<any>(null)

function dedent(str: string): string {
  const lines = str.split('\n')
  const nonEmptyLines = lines.filter(line => line.trim())
  if (nonEmptyLines.length === 0) return str

  const indent = Math.min(...nonEmptyLines.map(line => {
    const match = line.match(/^(\s*)/)
    return match ? match[1].length : 0
  }))

  return lines.map(line => line.slice(indent)).join('\n').trim()
}

onMounted(async () => {
  if (props.init) {
    code.value = dedent(props.init)
  }
  await loadPyodide()
})

async function loadPyodide() {
  if ((window as any).pyodide) {
    pyodide.value = (window as any).pyodide
    isLoading.value = false
    return
  }

  const script = document.createElement('script')
  script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js'
  document.head.appendChild(script)

  await new Promise<void>(resolve => { script.onload = () => resolve() })

  const py = await (window as any).loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
  })

  await py.loadPackage('numpy')
  ;(window as any).pyodide = py
  pyodide.value = py
  isLoading.value = false
}

async function run() {
  if (!pyodide.value || isRunning.value) return

  isRunning.value = true
  output.value = ''

  try {
    pyodide.value.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
`)
    await pyodide.value.runPythonAsync(code.value)

    const stdout = pyodide.value.runPython('sys.stdout.getvalue()')
    const stderr = pyodide.value.runPython('sys.stderr.getvalue()')
    output.value = (stdout + stderr) || '(出力なし)'
  } catch (e: any) {
    output.value = `エラー: ${e.message}`
  } finally {
    isRunning.value = false
  }
}
</script>

<template>
  <div class="runner">
    <textarea
      v-model="code"
      class="editor"
      spellcheck="false"
      placeholder="Pythonコードを入力..."
    />
    <div class="toolbar">
      <button @click="run" :disabled="isLoading || isRunning" class="run-btn">
        {{ isLoading ? '読み込み中...' : isRunning ? '実行中...' : '▶ 実行' }}
      </button>
    </div>
    <pre v-if="output" class="output">{{ output }}</pre>
  </div>
</template>

<style scoped>
.runner {
  border: 1px solid #444;
  border-radius: 6px;
  overflow: hidden;
  background: #1e1e1e;
  font-size: 14px;
}

.editor {
  width: 100%;
  min-height: 100px;
  padding: 12px;
  font-family: 'Fira Code', Consolas, monospace;
  font-size: 14px;
  line-height: 1.5;
  background: #1e1e1e;
  color: #d4d4d4;
  border: none;
  resize: vertical;
  outline: none;
}

.editor:focus {
  background: #252526;
}

.toolbar {
  padding: 8px 12px;
  background: #2d2d2d;
  border-top: 1px solid #404040;
}

.run-btn {
  padding: 6px 16px;
  background: #0e639c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.run-btn:hover:not(:disabled) {
  background: #1177bb;
}

.run-btn:disabled {
  background: #555;
  cursor: not-allowed;
}

.output {
  margin: 0;
  padding: 12px;
  font-family: 'Fira Code', Consolas, monospace;
  font-size: 13px;
  color: #4ec9b0;
  background: #1a1a1a;
  border-top: 1px solid #404040;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
