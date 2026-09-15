// Test-only TS/TSX loader: uses the existing compiler and real React packages.
// No browser, runtime loader, or production dependency is added.
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import ts from "typescript";

const cache = new Map();
export function loadSource(relativePath) {
  return load(fileURLToPath(new URL(relativePath, import.meta.url)));
}
function load(filename) {
  if (cache.has(filename)) return cache.get(filename).exports;
  const module = { exports: {} };
  cache.set(filename, module);
  const source = readFileSync(filename, "utf8").replaceAll("import.meta.env", "({})");
  const compiled = ts.transpileModule(source, {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  const nativeRequire = createRequire(filename);
  const requireSource = (specifier) => {
    if (specifier.startsWith(".")) {
      const base = resolve(dirname(filename), specifier);
      const path = [base, `${base}.ts`, `${base}.tsx`].find(existsSync);
      if (!path) throw new Error(`Test import missing: ${specifier}`);
      return load(path);
    }
    return nativeRequire(specifier);
  };
  new Function("require", "module", "exports", compiled)(requireSource, module, module.exports);
  return module.exports;
}
