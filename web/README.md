# theory.generops.dev

The lessons site: a Vite + React app with lessons written in MDX, math rendered by KaTeX at build time, and interactive demos in TypeScript.

```bash
npm install
npm run dev     # local dev server
npm test        # autograd engine tests (gradient checks, softmax-CE, MLP training)
npm run build   # type-check, then build static files into dist/
```

## Adding a lesson

Create `src/content/<area>/<slug>.mdx`. It's served at `/<area>/<slug>` and listed in the sidebar automatically. `<area>` must be one of the ids in `AREAS` in [src/lessons.ts](src/lessons.ts). Every lesson starts with a `meta` export:

```js
export const meta = {
  title: "Probability spaces",
  summary: "One sentence for the index card.",
  order: 1,            // position within the area
  source: "Durrett, PTE 5th ed., §1.1", // optional
};
```

Use `$...$` for inline math and `$$` on their own lines for display math. Outside math and code, MDX treats `{`, `}` and `<` as JSX, so escape them or keep them inside math. Demos are React components in `src/demos/`, imported at the top of the lesson.

## Deployment (Cloudflare Pages)

The site is static, and `generops.dev`'s DNS is on Cloudflare, so Pages builds it from this repo. One-time setup in the Cloudflare dashboard:

1. **Workers & Pages → Create → Pages → Connect to Git**, then pick `christufa/dev-check`.
2. Build settings:
   - Production branch: `main`
   - Framework preset: None
   - Root directory: `web`
   - Build command: `npm run build`
   - Build output directory: `dist`
3. After the first deploy: **Custom domains → Set up a custom domain → `theory.generops.dev`**. The zone is on Cloudflare, so Pages creates the DNS record and the certificate itself.

After that, every push to `main` deploys to production, and every PR gets a preview URL. Pages serves `index.html` for unknown paths (there's no top-level `404.html`), so client-side routes like `/probability/probability-spaces` work on reload.
