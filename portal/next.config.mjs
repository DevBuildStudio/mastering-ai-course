import nextra from 'nextra'
import { remarkGithubAlerts } from './lib/remark-github-alerts.mjs'

const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.tsx',
  defaultShowCopyCode: true,
  mdxOptions: {
    remarkPlugins: [remarkGithubAlerts],
  },
})

const basePath = process.env.NEXT_PUBLIC_BASE_PATH

export default withNextra({
  reactStrictMode: true,
  images: { unoptimized: true },
  staticPageGenerationTimeout: 300,
  experimental: {
    cpus: 2,
  },
  ...(basePath && {
    basePath,
    assetPrefix: basePath,
    output: 'export',
    trailingSlash: true,
  }),
})