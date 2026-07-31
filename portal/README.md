# Mastering AI Course Portal

This Nextra portal is adapted from the documentation structure in `E:\git\bic\claude-accessibility-plugin\docs`.

## Content Format

Course and workshop source files stay in Markdown. The portal synchronizes them into Nextra's `pages` directory before development and production builds. MDX is used only for pages that embed React components.

Nextra sends both `.md` and `.mdx` through the same MDX compiler in `format: "detect"` mode. Renaming lesson files to `.mdx` does not improve startup or rendering performance.

## Development

```bash
cd portal
npm install
npm run dev
```

The portal runs at `http://localhost:3000` by default.

## Commands

| Command | Purpose |
|---|---|
| `npm run sync-content` | Regenerate course and workshop routes from source Markdown |
| `npm run dev` | Synchronize content and start the development server |
| `npm run typecheck` | Validate TypeScript |
| `npm run build` | Synchronize content and create a production build |
| `npm run preview` | Serve the completed production build on port 3000 |

Generated course and workshop page folders are ignored by Git. Edit the original Markdown lessons outside `portal`.

On Windows machines where Nextra's development page-map compilation is slow, use `npm run build` followed by `npm run preview`. The production server starts quickly because all routes are already compiled.