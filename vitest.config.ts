import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    testTimeout: 90_000,
    env: loadEnv("", process.cwd(), ""),
  },
});
