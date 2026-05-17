import { $ } from "bun";
import { readdir } from "node:fs/promises";
import { resolve, join } from "node:path";

const rootDir = resolve(import.meta.dirname, "..");
const slidesDir = join(rootDir, "slides");
const distDir = join(rootDir, "dist");

const files = await readdir(slidesDir);
const slides = files.filter((f) => f.endsWith(".md"));

console.log(`Found ${slides.length} slides: ${slides.join(", ")}`);

for (const slide of slides) {
  const name = slide.replace(".md", "");
  const input = join(slidesDir, slide);
  const output = join(distDir, name);

  console.log(`\nBuilding ${name}...`);
  await $`bunx slidev build ${input} --base /${name}/ --out ${output}`;
  console.log(`Built: ${output}`);
}

console.log("\nAll slides built successfully!");
