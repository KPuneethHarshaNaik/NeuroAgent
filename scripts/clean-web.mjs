// `vite build` cannot empty frontend/ (it also holds frontend/legacy/, the pre-rewrite tool), so
// stale hashed assets would pile up. Remove exactly what this build emits, and nothing else.
import { rm, readdir } from "node:fs/promises";
import { join, resolve } from "node:path";

const frontend = resolve(import.meta.dirname, "..", "frontend");
const emitted = [join(frontend, "assets"), join(frontend, "index.html"), join(frontend, "app")];

for (const target of emitted) {
  await rm(target, { recursive: true, force: true });
}

if (process.argv.includes("--verbose")) {
  console.log("cleaned", emitted, "->", await readdir(frontend));
}
