/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  basePath: '/agent-imaging-website',
  assetPrefix: '/agent-imaging-website',
  trailingSlash: true,
  transpilePackages: ['react-compare-slider'],
  experimental: {
    esmExternals: 'loose',
  },
  env: {
    NEXT_PUBLIC_BASE_PATH: '/agent-imaging-website',
  },
}

module.exports = nextConfig
