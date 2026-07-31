import nextra from 'nextra'

const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.tsx',
  defaultShowCopyCode: true,
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