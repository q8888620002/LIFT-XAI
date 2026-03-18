# Ratings Backend

This docs UI can submit ratings directly to a backend endpoint, so users do not need to download JSON files.

This setup uses anonymous ID only (no auth login flow). Users enter an anonymous ID in the form, and that value is included as `rater_id` in each submission.

## Minimal backend for GitHub Pages

If your frontend is hosted on github.io, use a tiny serverless backend.
This repository includes a Cloudflare Worker example in [backend/cloudflare/worker.js](../backend/cloudflare/worker.js).

### Deploy steps (Cloudflare Worker + KV)

1. Install Wrangler:

```bash
npm install -g wrangler
```

2. Login:

```bash
wrangler login
```

3. Create KV namespaces:

```bash
wrangler kv namespace create RATINGS
wrangler kv namespace create RATINGS --preview
```

4. Copy config template and fill namespace IDs:

```bash
cp backend/cloudflare/wrangler.toml.example backend/cloudflare/wrangler.toml
```

5. Deploy:

```bash
cd backend/cloudflare
wrangler deploy
```

6. Set frontend API URL in [docs/config.js](config.js):

```js
window.RATINGS_API_BASE_URL = "https://<your-worker>.workers.dev";
```

After this, your github.io site can submit ratings directly.

## Start backend

From repo root:

```bash
python tools/ratings_server.py --host 0.0.0.0 --port 8000
```

Optional output directory:

```bash
python tools/ratings_server.py --out-dir data/ratings_submissions
```

## Frontend behavior

- Docs UI sends ratings to `POST /api/ratings`.
- API base URL is configured in [docs/config.js](config.js).
- Local default is `http://localhost:8000`.
- For github.io, set this to your HTTPS worker URL.
- Required payload fields include: `cohort`, `method`, `expertise`, `specialty`, `rater_id`, and `ratings`.

```js
window.RATINGS_API_BASE_URL = "https://your-api-host";
```

## Stored files

Each submission is saved as one JSON file under the output directory, with generated `submission_id` and `received_at` metadata.

Health check:

- `GET /api/health`
