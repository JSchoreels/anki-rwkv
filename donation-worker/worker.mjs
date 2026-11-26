const tokenEndpoint = "https://api.dropboxapi.com/oauth2/token";
const linkEndpoint = "https://api.dropboxapi.com/2/files/get_temporary_upload_link";

function reply(status, payload) {
  return Response.json(payload, { status, headers: { "Cache-Control": "no-store" } });
}

export default {
  async fetch(request, env) {
    if (new URL(request.url).pathname !== "/v1/upload-link") {
      return reply(404, { error: "Unknown route" });
    }
    if (request.method !== "POST") {
      return new Response(null, { status: 405, headers: { Allow: "POST" } });
    }
    if (request.body && !(await request.body.getReader().read()).done) {
      return reply(400, { error: "Send an empty request" });
    }
    if (!env.DROPBOX_APP_KEY || !env.DROPBOX_REFRESH_TOKEN || !env.UPLOAD_LIMIT) {
      return reply(503, { error: "Campaign uploads are unavailable" });
    }
    try {
      const { success } = await env.UPLOAD_LIMIT.limit({
        key: request.headers.get("CF-Connecting-IP") || "unknown",
      });
      if (!success) return reply(429, { error: "Try again later" });

      const token = await fetch(tokenEndpoint, {
        method: "POST",
        redirect: "error",
        body: new URLSearchParams({
          grant_type: "refresh_token",
          refresh_token: env.DROPBOX_REFRESH_TOKEN,
          client_id: env.DROPBOX_APP_KEY,
        }),
      });
      if (!token.ok) throw new Error();
      const { access_token } = await token.json();
      if (typeof access_token !== "string" || !access_token) throw new Error();
      const upload = await fetch(linkEndpoint, {
        method: "POST",
        redirect: "error",
        headers: { Authorization: `Bearer ${access_token}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          commit_info: { path: `/Anki Data/anki-${crypto.randomUUID()}.colpkg`, mode: "add", strict_conflict: true },
          duration: 900,
        }),
      });
      if (!upload.ok) throw new Error();
      const { link } = await upload.json();
      if (typeof link !== "string" || !link.startsWith("https://content.dropboxapi.com/apitul/1/")) throw new Error();
      return reply(200, { link });
    } catch {
      return reply(502, { error: "Dropbox upload link unavailable" });
    }
  },
};
