import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";


const read = (path) => readFile(new URL(path, import.meta.url), "utf8");


test("frontend metadata identifies the AI Data Scientist workspace", async () => {
  const index = await read("../index.html");
  assert.match(index, /<title>AI Data Scientist<\/title>/);
  assert.match(index, /<div id="root"><\/div>/);
});


test("setup API client retains local upload and target endpoints", async () => {
  const client = await read("../src/api/client.ts");
  assert.match(client, /"\/upload"/);
  assert.match(client, /"\/analysis\/target"/);
  assert.match(client, /VITE_API_BASE_URL/);
  assert.doesNotMatch(client, /openai\.com|googleapis\.com|gemini/i);
});


test("upload control accepts every supported backend dataset format", async () => {
  const app = await read("../src/App.tsx");
  assert.match(app, /accept="\.csv,\.xlsx,\.xls"/);
  assert.match(app, /MAX_UPLOAD_BYTES = 100 \* 1024 \* 1024/);
  assert.match(app, /validateTarget\(upload\.filename, selectedTarget, request\.signal\)/);
});


test("built interface preserves accessible status and error regions", async () => {
  const app = await read("../src/App.tsx");
  assert.match(app, /role="alert"/);
  assert.match(app, /aria-live="polite"/);
  assert.match(app, /aria-label="Analysis workflow"/);
});
