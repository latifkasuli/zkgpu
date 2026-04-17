/**
 * G.0.2 Browser WebGPU tail A/B — foundation audit closeout.
 *
 * Drives the web harness's wasm `run_suite` via the UI's pre-initialized
 * worker, with forced Stockham family + explicit tail override, mirroring
 * the Android `crossoverStockhamLocalTail` / `crossoverStockhamGlobalTail`
 * methodology.
 *
 * Output: per-browser JSON files compatible with zkgpu-tail-analyze,
 * saved under `test-results/g02/`.
 *
 * The UI's main.ts exposes the worker as `window.__zkgpuWorker` so tests
 * can send arbitrary HarnessRequest payloads without fighting the UI's
 * preset suite selector.
 */
import { test, expect } from "@playwright/test";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const G02_DIR = path.join(__dirname, "../test-results/g02");

const LOGS = [18, 19, 20, 21, 22];

function tailSuiteSpec(tail: "Local" | "Global") {
  const cases = LOGS.flatMap((log_n) => [
    {
      name: `tail_${tail.toLowerCase()}_forward_log${log_n}`,
      log_n,
      direction: "Forward",
      input: "Sequential",
      profile_gpu_timestamps: true,
      iterations: 5,
      warmup_iterations: 1,
    },
    {
      name: `tail_${tail.toLowerCase()}_inverse_log${log_n}`,
      log_n,
      direction: "Inverse",
      input: "Sequential",
      profile_gpu_timestamps: true,
      iterations: 5,
      warmup_iterations: 1,
    },
  ]);
  return {
    kind: "Benchmark",
    cases,
    fail_fast: false,
    family_override: "Stockham",
    stockham_tail_override: tail,
  };
}

async function ensureDir(dir: string) {
  await fs.promises.mkdir(dir, { recursive: true });
}

test.describe("G.0.2 Browser WebGPU tail A/B", () => {
  test.beforeAll(async () => {
    await ensureDir(G02_DIR);
  });

  for (const tail of ["Local", "Global"] as const) {
    test(`stockham ${tail} tail @ log ${LOGS.join(",")}`, async ({ page }, info) => {
      const browserName = info.project.name;

      const consoleLines: string[] = [];
      page.on("console", (msg) => {
        consoleLines.push(`[${msg.type()}] ${msg.text()}`);
      });
      page.on("pageerror", (err) => {
        consoleLines.push(`[pageerror] ${err.message}`);
      });

      await page.goto("/", { waitUntil: "load" });
      await page.waitForLoadState("networkidle");

      // Wait for the UI's worker to finish GPU init.
      const statusLocator = page.locator("#status");
      await statusLocator.waitFor({ state: "visible", timeout: 30_000 });

      // The init flow emits a "success" status class when the device panel
      // is populated, or "error" if WebGPU is unavailable.
      await page
        .waitForFunction(
          () => {
            const el = document.getElementById("status");
            const cls = el?.className ?? "";
            return cls.includes("success") || cls.includes("error");
          },
          { timeout: 60_000 },
        )
        .catch(() => {});

      const statusClass = await statusLocator.getAttribute("class");
      const panelText = (await page.locator("#device-panel").textContent()) ?? "";

      if (statusClass?.includes("error") || !panelText.includes("Device:")) {
        const reportPath = path.join(
          G02_DIR,
          `${browserName}_tail_${tail.toLowerCase()}.json`,
        );
        await fs.promises.writeFile(
          reportPath,
          JSON.stringify(
            {
              browser: browserName,
              tail_strategy: tail,
              webgpu_available: false,
              status_class: statusClass,
              device_panel: panelText,
              console_lines: consoleLines,
              captured_at: new Date().toISOString(),
            },
            null,
            2,
          ),
        );
        console.log(
          `[${browserName} / ${tail}] WebGPU unavailable — diagnostic saved to ${reportPath}`,
        );
        test.skip(true, `${browserName} WebGPU unavailable`);
        return;
      }

      // Drive the UI's worker directly via `window.__zkgpuWorker`.
      const spec = tailSuiteSpec(tail);
      const result = await page.evaluate(async (specInput) => {
        const w = (globalThis as unknown as { __zkgpuWorker?: Worker })
          .__zkgpuWorker;
        if (!w) {
          throw new Error(
            "window.__zkgpuWorker not exposed by main.ts; rebuild required",
          );
        }
        return await new Promise<{ response: unknown }>((resolve, reject) => {
          const onMsg = (ev: MessageEvent) => {
            const m = ev.data;
            if (m?.type === "suite_result") {
              w.removeEventListener("message", onMsg);
              resolve({ response: m.response });
            } else if (m?.type === "suite_error") {
              w.removeEventListener("message", onMsg);
              reject(new Error(m.error || "suite_error"));
            }
          };
          w.addEventListener("message", onMsg);
          // HarnessRequest with a full spec (no `suite` field) so the
          // spec.stockham_tail_override is honored by the wasm runner.
          w.postMessage({
            type: "run_suite",
            request: { spec: specInput },
          });
        });
      }, spec);

      const outPath = path.join(
        G02_DIR,
        `${browserName}_tail_${tail.toLowerCase()}.json`,
      );
      const payload = {
        browser: browserName,
        tail_strategy: tail,
        webgpu_available: true,
        device_panel: panelText,
        captured_at: new Date().toISOString(),
        suite_response: result.response,
      };
      await fs.promises.writeFile(outPath, JSON.stringify(payload, null, 2));
      console.log(`[${browserName} / ${tail}] wrote ${outPath}`);

      const resp = result.response as {
        ok?: boolean;
        report?: { cases?: unknown[] };
        error?: string;
      };
      expect(resp.ok, `Suite failed: ${resp.error ?? "(no msg)"}`).toBe(true);
      expect(resp.report?.cases?.length ?? 0).toBeGreaterThan(0);
    });
  }
});
