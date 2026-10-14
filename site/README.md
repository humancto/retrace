# Landing page

This directory holds one static page for Retrace. It has no framework. It is ready for
Cloudflare Pages.

```
site/
  index.html          the page (inline CSS, some lines of JS, fonts from Google Fonts)
  assets/             the screenshots and the hero frame (JPEG)
  _headers            cache and security headers for Cloudflare Pages
```

## Deploy to Cloudflare Pages

Option 1: connect the repository in the Cloudflare dashboard. Leave the **build command** empty.
Set the **build output directory** to `site`.

Option 2: push from the command line:

```bash
npx wrangler pages deploy site --project-name retrace
```

The hero frame, the crops, and the screenshots come from OpenCV's public sample `vtest.avi`
(Apache-2.0). To make new images, run `scripts/e2e.sh` and the UI.
