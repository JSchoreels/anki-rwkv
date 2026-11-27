const api = "https://api.dropboxapi.com";
const content = "https://content.dropboxapi.com/2/files/upload_session/";
const chunkBytes = 8 * 1024 * 1024;
const maxSessionBytes = 2 ** 41 - 2 ** 22;
const encoder = new TextEncoder();
let cachedToken;

function reply(status, payload) {
  return Response.json(payload, { status, headers: { "Cache-Control": "no-store" } });
}

function destination(collectionId) {
  const timestamp = new Date().toISOString().replaceAll(":", "-");
  const name = collectionId ? `${collectionId}-${timestamp}` : crypto.randomUUID();
  return { path: `/Anki Data/anki-${name}.colpkg`, mode: "add", strict_conflict: true, autorename: true };
}

async function accessToken(env) {
  if (cachedToken && cachedToken.expires > Date.now()) return cachedToken.value;
  const response = await fetch(api + "/oauth2/token", {
    method: "POST", redirect: "manual",
    body: new URLSearchParams({
      grant_type: "refresh_token", client_id: env.DROPBOX_APP_KEY,
      refresh_token: env.DROPBOX_REFRESH_TOKEN,
    }),
  });
  if (!response.ok) throw new Error("Dropbox authorization failed");
  const { access_token, expires_in } = await response.json();
  if (typeof access_token !== "string" || !access_token) throw new Error("Invalid Dropbox token");
  cachedToken = { value: access_token, expires: Date.now() + (expires_in - 60) * 1000 };
  return access_token;
}

async function dropbox(env, url, headers, body) {
  const token = await accessToken(env);
  const response = await fetch(url, {
    method: "POST", redirect: "manual",
    headers: { ...headers, Authorization: "Bearer " + token }, body,
  });
  if (response.status === 401) cachedToken = undefined;
  if (!response.ok) throw new Error("Dropbox request failed: " + response.status);
  return response;
}

function sessionCall(env, method, args, body) {
  return dropbox(env, content + method, {
    "Content-Type": "application/octet-stream", "Dropbox-API-Arg": JSON.stringify(args),
  }, body);
}

// Only a cursor issued by this Worker can append or commit. The signature binds
// the Dropbox session, destination, declared size, offset and expiry together.
function signingKey(env) {
  return crypto.subtle.importKey("raw", encoder.encode("anki-session-v1:" + env.DROPBOX_REFRESH_TOKEN),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

async function nextChunk(env, state) {
  const payload = btoa(JSON.stringify(state));
  const signature = await crypto.subtle.sign("HMAC", await signingKey(env), encoder.encode(payload));
  const token = payload + "." + btoa(String.fromCharCode(...new Uint8Array(signature)));
  return reply(200, { session: token, offset: state.offset, chunk_size: chunkBytes });
}

async function readCursor(env, request) {
  const header = request.headers.get("Authorization") || "";
  if (!header.startsWith("Bearer ") || header.length > 4096) return null;
  const parts = header.slice(7).split(".");
  if (parts.length !== 2) return null;
  try {
    const signature = Uint8Array.from(atob(parts[1]), c => c.charCodeAt(0));
    if (!await crypto.subtle.verify("HMAC", await signingKey(env), signature, encoder.encode(parts[0]))) return null;
    const state = JSON.parse(atob(parts[0]));
    return state.expires > Date.now() ? state : null;
  } catch {
    return null;
  }
}

async function acceptChunk(env, request) {
  const state = await readCursor(env, request);
  if (!state) return reply(403, { error: "Invalid or expired upload session" });
  const length = Math.min(chunkBytes, state.size - state.offset);
  if (!request.body || request.headers.get("Content-Length") !== String(length)) {
    return reply(400, { error: "Unexpected chunk length" });
  }
  const cursor = { session_id: state.id, offset: state.offset };
  const next = state.offset + length;
  if (next === state.size) {
    const response = await sessionCall(env, "finish", { cursor, commit: state.commit }, request.body);
    const metadata = await response.json();
    if (metadata.size !== state.size || !/^[0-9a-f]{64}$/.test(metadata.content_hash)) {
      throw new Error("Invalid Dropbox receipt");
    }
    return reply(200, { "content-hash": metadata.content_hash });
  }
  await sessionCall(env, "append_v2", { cursor }, request.body);
  return nextChunk(env, { ...state, offset: next });
}

export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (!["/v1/upload-link", "/v1/upload-session", "/v1/upload-chunk"].includes(path)) {
      return reply(404, { error: "Unknown route" });
    }
    if (request.method !== "POST") return new Response(null, { status: 405, headers: { Allow: "POST" } });
    if (!env.DROPBOX_APP_KEY || !env.DROPBOX_REFRESH_TOKEN || !env.UPLOAD_LIMIT) {
      return reply(503, { error: "Campaign uploads are unavailable" });
    }
    try {
      if (path === "/v1/upload-chunk") return await acceptChunk(env, request);
      if (request.body) {
        for await (const chunk of request.body) {
          if (chunk.byteLength) return reply(400, { error: "Send an empty request" });
        }
      }
      const limit = await env.UPLOAD_LIMIT.limit({ key: request.headers.get("CF-Connecting-IP") || "unknown" });
      if (!limit.success) return reply(429, { error: "Try again later" });
      const collectionId = request.headers.get("X-Collection-ID");
      if (collectionId !== null && !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(collectionId)) {
        return reply(400, { error: "Invalid collection ID" });
      }
      if (path === "/v1/upload-session") {
        const size = Number(request.headers.get("X-Upload-Size"));
        if (!Number.isSafeInteger(size) || size <= 0 || size > maxSessionBytes) {
          return reply(400, { error: "Invalid collection size" });
        }
        const response = await sessionCall(env, "start", {}, new Uint8Array());
        const { session_id } = await response.json();
        if (typeof session_id !== "string" || !session_id) throw new Error("Invalid Dropbox session");
        return await nextChunk(env, { id: session_id, offset: 0, size, commit: destination(collectionId), expires: Date.now() + 7 * 86400000 });
      }
      const response = await dropbox(env, api + "/2/files/get_temporary_upload_link",
        { "Content-Type": "application/json" }, JSON.stringify({ commit_info: destination(collectionId), duration: 900 }));
      const { link } = await response.json();
      if (typeof link !== "string" || !link.startsWith("https://content.dropboxapi.com/apitul/1/")) {
        throw new Error("Invalid Dropbox upload link");
      }
      return reply(200, { link });
    } catch (error) {
      console.error("Campaign upload failed", path, error.name);
      return reply(502, { error: "Dropbox upload unavailable" });
    }
  },
};
