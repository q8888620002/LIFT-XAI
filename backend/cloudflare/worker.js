export default {
  async fetch(request, env) {
    const isValidEmail = (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(value || ""));

    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Content-Type": "application/json",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/api/health") {
      return new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: corsHeaders,
      });
    }

    if (request.method === "POST" && url.pathname === "/api/ratings") {
      let data;
      try {
        data = await request.json();
      } catch {
        return new Response(JSON.stringify({ error: "Request body must be valid JSON" }), {
          status: 400,
          headers: corsHeaders,
        });
      }

      const required = ["cohort", "method", "expertise", "specialty", "rater_email", "ratings"];
      const missing = required.filter((k) => !(k in data));
      if (missing.length > 0) {
        return new Response(
          JSON.stringify({ error: `Missing required keys: ${missing.join(", ")}` }),
          { status: 400, headers: corsHeaders }
        );
      }

      if (!isValidEmail(data.rater_email)) {
        return new Response(JSON.stringify({ error: "Invalid rater_email" }), {
          status: 400,
          headers: corsHeaders,
        });
      }

      const submissionId = crypto.randomUUID();
      const now = new Date().toISOString();
      const key = `ratings:${Date.now()}:${submissionId}`;

      const payload = {
        submission_id: submissionId,
        received_at: now,
        ...data,
      };

      await env.RATINGS.put(key, JSON.stringify(payload));

      return new Response(
        JSON.stringify({
          status: "saved",
          submission_id: submissionId,
          key,
        }),
        { status: 201, headers: corsHeaders }
      );
    }

    return new Response(JSON.stringify({ error: "Not found" }), {
      status: 404,
      headers: corsHeaders,
    });
  },
};
