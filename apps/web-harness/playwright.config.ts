import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for zkGPU web harness browser tests.
 *
 * Launches a Vite dev server and runs tests against it in real
 * Chromium, Firefox, and WebKit browsers with WebGPU support.
 *
 * Run: npx playwright test
 * Run headed: npx playwright test --headed
 * Run single browser: npx playwright test --project=chromium
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false, // Serialize GPU tests to avoid contention
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1, // Single worker — GPU tests shouldn't run concurrently
  reporter: process.env.CI ? "github" : "list",

  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
  },

  // Build wasm (if needed) then start the Vite dev server.
  // This makes `npx playwright test` fully self-bootstrapping —
  // no manual `npm run build:wasm` step required.
  webServer: {
    command: "npm run build:wasm && npx vite --port 5173",
    port: 5173,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000, // wasm release build can take ~15-30s on first compile
  },

  projects: [
    {
      // Default chromium project — uses the platform-native WebGPU
      // backend (Metal on macOS via Dawn). `--enable-unsafe-webgpu`
      // is the only flag needed on current Playwright Chromium; it
      // allows WebGPU on adapters Chrome's allowlist doesn't
      // recognize (Playwright's bundled build counts as unrecognized).
      //
      // Do NOT add `--enable-features=Vulkan` here — it forces
      // Dawn→MoltenVK→Metal on macOS, which measures the MoltenVK
      // translation layer, not Chrome users' actual WebGPU path.
      // For that comparison, use the `chromium-vulkan` project below.
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: ["--enable-unsafe-webgpu"],
        },
      },
    },
    {
      // Chromium forced onto the Vulkan backend (via MoltenVK on
      // macOS, ICD on Linux). Retained for A/B vs the default so
      // the MoltenVK translation cost is measurable rather than
      // silently skewing the default numbers.
      name: "chromium-vulkan",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
          ],
        },
      },
    },
    {
      name: "firefox",
      use: {
        ...devices["Desktop Firefox"],
        launchOptions: {
          firefoxUserPrefs: {
            "dom.webgpu.enabled": true,
          },
        },
      },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
  ],
});
