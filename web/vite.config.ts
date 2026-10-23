import { readFile } from "node:fs/promises";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeShiki from "@shikijs/rehype";

/**
 * `lesson.mdx?meta` resolves to just the lesson's `export const meta = {...}` block, so navigation can list every
 * lesson without bundling their bodies.
 */
function mdxMeta(): Plugin {
  return {
    name: "mdx-meta",
    enforce: "pre",
    async load(id) {
      const [file, query] = id.split("?");
      if (!file.endsWith(".mdx") || query !== "meta") return null;
      const source = await readFile(file, "utf8");
      const block = source.match(/^export const meta = \{[\s\S]*?^\};/m);
      if (!block) this.error(`${file} has no "export const meta = { ... };" block`);
      return block[0];
    },
  };
}

export default defineConfig({
  plugins: [
    mdxMeta(),
    {
      enforce: "pre",
      ...mdx({
        remarkPlugins: [remarkMath],
        rehypePlugins: [
          rehypeKatex,
          [rehypeShiki, { themes: { light: "github-light", dark: "github-dark" } }],
        ],
      }),
    },
    react({ include: /\.(mdx|tsx?)$/ }),
  ],
});
