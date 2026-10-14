# Landing page

A single static page for Retrace, framework-free, ready for Cloudflare Pages.

```
site/
  index.html          the page (inline CSS and a few lines of JS; fonts from Google Fonts)
  assets/             screenshots and the hero frame (JPEG)
  _headers            cache and security headers for Cloudflare Pages
```

## Deploy to Cloudflare Pages

Either connect the repository in the Cloudflare dashboard with **build command** empty and
**build output directory** `site`, or push from the command line:

```bash
npx wrangler pages deploy site --project-name retrace
```

The hero frame, crops and screenshots come from OpenCV's public `vtest.avi` sample
(Apache-2.0); regenerate them with `scripts/e2e.sh` and the UI if the design changes.
