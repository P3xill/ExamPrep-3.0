#!/usr/bin/env node
/**
 * Build step for static hosting providers (e.g. Netlify/Vercel).
 * - Copies everything from frontend/ → dist/
 * - Injects EXAMPREP_API_BASE into try.html's data attribute
 *
 * Requires EXAMPREP_API_BASE environment variable.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const srcDir = path.join(projectRoot, "frontend");
const outDir = path.join(projectRoot, "dist");

const apiBase = process.env.EXAMPREP_API_BASE;
if (!apiBase) {
  console.error("EXAMPREP_API_BASE is not set. Add it to your build environment.");
  process.exit(1);
}

await fs.rm(outDir, { recursive: true, force: true });
await fs.mkdir(outDir, { recursive: true });

await copyDir(srcDir, outDir);
console.log(`Frontend copied to ${outDir} with API base ${apiBase}`);

async function copyDir(source, destination) {
  const entries = await fs.readdir(source, { withFileTypes: true });
  for (const entry of entries) {
    const fromPath = path.join(source, entry.name);
    const toPath = path.join(destination, entry.name);

    if (entry.isDirectory()) {
      await fs.mkdir(toPath, { recursive: true });
      await copyDir(fromPath, toPath);
      continue;
    }

    if (entry.isFile() && entry.name === "try.html") {
      const html = await fs.readFile(fromPath, "utf8");
      const updated = injectApiBase(html, apiBase);
      await fs.writeFile(toPath, updated, "utf8");
    } else {
      await fs.copyFile(fromPath, toPath);
    }
  }
}

function injectApiBase(html, base) {
  const pattern = /data-api-base="[^"]*"/;
  if (!pattern.test(html)) {
    throw new Error("try.html is missing data-api-base attribute");
  }
  const cleanBase = base.replace(/\/+$/, "");
  return html.replace(pattern, `data-api-base="${cleanBase}"`);
}

