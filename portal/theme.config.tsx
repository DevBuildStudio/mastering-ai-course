import React from 'react'
import type { DocsThemeConfig } from 'nextra-theme-docs'
import { BrainCircuit } from 'lucide-react'

const config: DocsThemeConfig = {
  logo: (
    <span className="portal-logo">
      <BrainCircuit aria-hidden="true" size={24} />
      <span>Mastering AI</span>
      <span className="portal-logo-subtitle">Course Portal</span>
    </span>
  ),
  color: {
    hue: 345,
    saturation: 70,
  },
  head: (
    <>
      <script
        dangerouslySetInnerHTML={{
          __html: `(() => {
            const param = new URLSearchParams(window.location.search).get("scoutTheme");
            const theme = param || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
            document.documentElement.setAttribute("data-theme", theme);
          })();`,
        }}
      />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta
        name="description"
        content="A practical AI engineering curriculum covering LLM foundations, agent systems, production engineering, and workshops."
      />
    </>
  ),
  sidebar: {
    defaultMenuCollapseLevel: 1,
    autoCollapse: true,
  },
  navigation: true,
  footer: {
    content: (
      <span className="portal-footer">
        Mastering AI Course Portal · Learn by building · {new Date().getFullYear()}
      </span>
    ),
  },
  feedback: {
    content: null,
  },
  editLink: {
    content: null,
  },
  toc: {
    backToTop: true,
  },
  search: {
    placeholder: 'Search lessons and workshops...'
  },
}

export default config