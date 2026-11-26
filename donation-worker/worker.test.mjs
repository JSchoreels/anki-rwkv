import assert from "node:assert/strict";
import test from "node:test";
import worker from "./worker.mjs";

test("only issues links for new collection files in the campaign folder", async (t) => {
  const env = { DROPBOX_APP_KEY: "test", DROPBOX_REFRESH_TOKEN: "test", UPLOAD_LIMIT: { limit: async () => ({success: true}) } };
  const request = (method = "POST", body) => new Request("https://worker.test/v1/upload-link", {method, body});
  const link = "https://content.dropboxapi.com/apitul/1/test";
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({url, options});
    return Response.json(calls.length === 1 ? {access_token: "token"} : {link});
  });
  assert.equal((await worker.fetch(request("GET"), env)).status, 405);
  assert.equal((await worker.fetch(request("POST", "/arbitrary/path"), env)).status, 400);
  assert.equal((await worker.fetch(request(), {})).status, 503);
  assert.equal((await worker.fetch(request(), {...env, UPLOAD_LIMIT: {limit: async () => ({success: false})}})).status, 429);
  assert.equal(calls.length, 0);
  // Qt can send a zero-length stream with Content-Length: 0.
  const response = await worker.fetch(request("POST", new Uint8Array()), env);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.deepEqual(await response.json(), {link});
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, "https://api.dropboxapi.com/2/files/get_temporary_upload_link");
  const args = JSON.parse(calls[1].options.body);
  assert.match(args.commit_info.path, /^\/Anki Data\/anki-[0-9a-f-]{36}\.colpkg$/);
  assert.equal(args.commit_info.mode, "add");
  assert.equal(args.commit_info.strict_conflict, true);
  assert.equal(args.duration, 900);
});
