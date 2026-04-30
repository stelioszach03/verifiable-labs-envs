import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Cloudflare Pages target — `next-on-pages` will pick this up at deploy time.
  // Local dev runs the standard Next server.
  poweredByHeader: false,
};

export default nextConfig;
