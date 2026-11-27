import assert from "node:assert/strict";
import test from "node:test";
import worker from "./worker.mjs";

const collectionId = "f8098c7e-e2cd-461d-a57f-a9155a404e4d";
const now = new Date("2026-11-27T17:08:42.123Z");
const snapshot = {
  path: `/Anki Data/anki-${collectionId}-2026-11-27T17-08-42.123Z.colpkg`,
  mode: "add", strict_conflict: true, autorename: true,
};

test("only issues links for new collection files in the campaign folder", async (t) => {
  const env = { DROPBOX_APP_KEY: "test", DROPBOX_REFRESH_TOKEN: "test", UPLOAD_LIMIT: { limit: async () => ({success: true}) } };
  const request = (method = "POST", body) => new Request("https://worker.test/v1/upload-link", {method, body});
  const link = "https://content.dropboxapi.com/apitul/1/test";
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(options.redirect, "manual");
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

test("repeat donations use the synced ID, UTC time and Dropbox conflict renaming", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now });
  const env = { DROPBOX_APP_KEY: "test", DROPBOX_REFRESH_TOKEN: "test", UPLOAD_LIMIT: { limit: async () => ({success: true}) } };
  const commits = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    if (url.endsWith("/oauth2/token")) return Response.json({access_token: "token", expires_in: 14400});
    commits.push(JSON.parse(options.body).commit_info);
    return Response.json({ link: "https://content.dropboxapi.com/apitul/1/test" });
  });
  const request = (id) => new Request("https://worker.test/v1/upload-link", {
    method: "POST", headers: { "X-Collection-ID": id },
  });
  for (const invalid of ["", "../another-folder"]) {
    assert.equal((await worker.fetch(request(invalid), env)).status, 400);
  }
  assert.equal((await worker.fetch(request(collectionId), env)).status, 200);
  assert.equal((await worker.fetch(request(collectionId), env)).status, 200);
  t.mock.timers.tick(1000);
  assert.equal((await worker.fetch(request(collectionId), env)).status, 200);
  assert.deepEqual(commits, [snapshot, snapshot, {...snapshot, path: snapshot.path.replace("42.123Z", "43.123Z")}]);
});

test("signed cursors stream sequential chunks into one campaign file", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now });
  const env = {
    DROPBOX_APP_KEY: "test", DROPBOX_REFRESH_TOKEN: "test",
    UPLOAD_LIMIT: { limit: t.mock.fn(async () => ({ success: true })) },
  };
  const size = 8 * 1024 * 1024 + 3;
  const calls = [];
  const hash = "ab".repeat(32);
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(options.redirect, "manual");
    if (url.endsWith("/oauth2/token")) return Response.json({access_token: "token", expires_in: 14400});
    assert.equal(options.headers.Authorization, "Bearer token");
    const args = JSON.parse(options.headers["Dropbox-API-Arg"]);
    const bytes = await new Response(options.body).arrayBuffer();
    calls.push({url, args, size: bytes.byteLength});
    if (url.endsWith("/start")) return Response.json({session_id: "synthetic"});
    if (url.endsWith("/finish")) return Response.json({size, content_hash: hash});
    assert.ok(url.endsWith("/append_v2"));
    return Response.json(null);
  });
  const send = (route, headers, body) => worker.fetch(new Request("https://worker.test/v1/" + route, {
    method: "POST", headers, body,
  }), env);
  const start = await send("upload-session", {"X-Upload-Size": String(size), "X-Collection-ID": collectionId});
  const first = await start.json();
  assert.equal(first.offset, 0);
  assert.equal(first.chunk_size, size - 3);
  const sendChunk = (cursor, data) => send("upload-chunk", {
    Authorization: "Bearer " + cursor, "Content-Length": String(data.length),
  }, data);
  assert.equal((await sendChunk("invalid", new Uint8Array(3))).status, 403);
  const altered = btoa(JSON.stringify({offset: 0, size, commit: {path: "/other"}})) + "." + first.session.split(".")[1];
  assert.equal((await sendChunk(altered, new Uint8Array(3))).status, 403);
  assert.equal((await sendChunk(first.session, new Uint8Array(3))).status, 400);
  assert.equal(calls.length, 1);
  const next = await (await sendChunk(first.session, new Uint8Array(first.chunk_size))).json();
  assert.equal(next.offset, first.chunk_size);
  const receipt = await sendChunk(next.session, new Uint8Array(3));
  assert.deepEqual(await receipt.json(), {"content-hash": hash});
  assert.deepEqual(calls.map(c => c.size), [0, first.chunk_size, 3]);
  assert.deepEqual(calls.slice(1).map(c => c.args.cursor), [
    {session_id: "synthetic", offset: 0}, {session_id: "synthetic", offset: first.chunk_size},
  ]);
  assert.deepEqual(calls[2].args.commit, snapshot);
  assert.equal(env.UPLOAD_LIMIT.limit.mock.callCount(), 1);
});
