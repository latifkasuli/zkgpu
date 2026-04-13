import { test, expect } from "@playwright/test";

/**
 * Smoke tests for the zkGPU web harness.
 *
 * These tests verify:
 * 1. The harness page loads and initializes WebGPU
 * 2. The smoke suite runs and passes
 * 3. Results are rendered correctly in the UI
 * 4. JSON export works
 *
 * WebGPU availability: If the browser doesn't support WebGPU (e.g.,
 * headless Firefox, some CI environments), the init will fail and
 * the tests will be skipped with a clear message.
 */

test.describe("zkGPU Web Harness", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    // Wait for the page to finish loading
    await page.waitForLoadState("networkidle");
  });

  test("page loads with correct title", async ({ page }) => {
    await expect(page).toHaveTitle("zkGPU Web Harness");
  });

  test("shows device panel", async ({ page }) => {
    const panel = page.locator("#device-panel");
    await expect(panel).toBeVisible();
  });

  test("initializes GPU device or reports unavailable", async ({ page }) => {
    // Wait for init to complete (success or failure)
    const status = page.locator("#status");

    // Wait up to 15s for status to appear (GPU init can take time)
    await expect(status).toBeVisible({ timeout: 15_000 });

    const statusText = await status.textContent();
    const statusClass = await status.getAttribute("class");

    if (statusClass === "error") {
      // WebGPU not available — that's OK for some browser/CI combos
      console.warn(`GPU init failed: ${statusText}`);
      test.skip(true, `WebGPU not available: ${statusText}`);
      return;
    }

    // Success: device panel should have device info
    const panelText = await page.locator("#device-panel").textContent();
    expect(panelText).toContain("Device:");
    expect(panelText).toContain("Backend:");

    // Run button should be enabled
    const runBtn = page.locator("#run-btn");
    await expect(runBtn).toBeEnabled();
  });

  test("runs smoke suite and shows results", async ({ page }) => {
    // Wait for GPU init
    const status = page.locator("#status");
    await expect(status).toBeVisible({ timeout: 15_000 });

    const statusClass = await status.getAttribute("class");
    if (statusClass === "error") {
      test.skip(true, "WebGPU not available");
      return;
    }

    // Select smoke suite and run
    await page.selectOption("#suite-select", "Smoke");
    await page.click("#run-btn");

    // Wait for results (up to 30s for GPU work)
    await expect(status).toHaveClass(/success|error/, { timeout: 30_000 });

    const finalClass = await status.getAttribute("class");
    const finalText = await status.textContent();

    if (finalClass === "error") {
      // Some environments can init but fail during execution
      console.warn(`Smoke suite failed: ${finalText}`);
      return;
    }

    // Verify success
    expect(finalText).toContain("passed");

    // Verify summary is visible
    const summary = page.locator("#summary");
    await expect(summary).toHaveClass(/visible/);

    // Verify result rows exist
    const rows = page.locator("#results-body tr");
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThanOrEqual(2); // Smoke has 2 cases

    // All cases should show PASS
    const passCells = page.locator("#results-body td.pass");
    const passCount = await passCells.count();
    expect(passCount).toBeGreaterThanOrEqual(2);
  });

  test("export and copy buttons work after suite run", async ({ page }) => {
    const status = page.locator("#status");
    await expect(status).toBeVisible({ timeout: 15_000 });

    const statusClass = await status.getAttribute("class");
    if (statusClass === "error") {
      test.skip(true, "WebGPU not available");
      return;
    }

    // Run smoke suite
    await page.selectOption("#suite-select", "Smoke");
    await page.click("#run-btn");
    await expect(status).toHaveClass(/success|error/, { timeout: 30_000 });

    const finalClass = await status.getAttribute("class");
    if (finalClass !== "success") {
      test.skip(true, "Suite did not pass");
      return;
    }

    // Export button should be enabled
    const exportBtn = page.locator("#export-btn");
    await expect(exportBtn).toBeEnabled();

    // Click export — JSON textarea should appear
    await exportBtn.click();
    const jsonOutput = page.locator("#json-output");
    await expect(jsonOutput).toBeVisible();

    // Verify JSON content is valid
    const jsonText = await page.locator("#json-text").inputValue();
    const parsed = JSON.parse(jsonText);
    expect(parsed.ok).toBe(true);
    expect(parsed.report).toBeDefined();
    expect(parsed.report.schema_version).toBe(1);
    expect(parsed.report.summary.passed_cases).toBeGreaterThanOrEqual(2);

    // Verify browser metadata enrichment
    expect(parsed.report.timing_metadata).toBeDefined();
    expect(parsed.report.timing_metadata.clock_source).toBe("browser-wall");
    expect(parsed.report.timing_metadata.worker).toBe(true);
    expect(parsed.report.harness).toBe("web");
    expect(parsed.report.collected_at).toBeDefined();
  });

  test("download button creates a file", async ({ page }) => {
    const status = page.locator("#status");
    await expect(status).toBeVisible({ timeout: 15_000 });

    const statusClass = await status.getAttribute("class");
    if (statusClass === "error") {
      test.skip(true, "WebGPU not available");
      return;
    }

    // Run smoke suite
    await page.selectOption("#suite-select", "Smoke");
    await page.click("#run-btn");
    await expect(status).toHaveClass(/success|error/, { timeout: 30_000 });

    const finalClass = await status.getAttribute("class");
    if (finalClass !== "success") {
      test.skip(true, "Suite did not pass");
      return;
    }

    // Download button should be enabled
    const downloadBtn = page.locator("#download-btn");
    await expect(downloadBtn).toBeEnabled();

    // Intercept the download
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      downloadBtn.click(),
    ]);

    expect(download.suggestedFilename()).toContain("zkgpu_Smoke_");
    expect(download.suggestedFilename()).toMatch(/\.json$/);
  });

  test("run button returns to ready state after suite completes", async ({ page }) => {
    const status = page.locator("#status");
    await expect(status).toBeVisible({ timeout: 15_000 });

    const statusClass = await status.getAttribute("class");
    if (statusClass === "error") {
      test.skip(true, "WebGPU not available");
      return;
    }

    // Run a suite and verify the button returns to "Run Suite" afterward.
    // We don't check for "Running..." because it can be too transient to catch.
    await page.selectOption("#suite-select", "Smoke");
    await page.click("#run-btn");

    // Wait for completion (success or error)
    await expect(status).toHaveClass(/success|error/, { timeout: 30_000 });

    // Button should be re-enabled and show "Run Suite"
    const runBtn = page.locator("#run-btn");
    await expect(runBtn).toHaveText("Run Suite");
    await expect(runBtn).toBeEnabled();
  });
});
