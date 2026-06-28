import { expect, test } from "@playwright/test";
import type { Browser, Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const enabled = process.env.RUN_FRONTEND_LOAD_E2E === "1";
const pageCount = envInt("FRONTEND_LOAD_PAGES", 8);
const concurrency = envInt("FRONTEND_LOAD_CONCURRENCY", 4);
const interactionPageCount = envInt("FRONTEND_INTERACTION_PAGES", 4);
const interactionConcurrency = envInt("FRONTEND_INTERACTION_CONCURRENCY", 2);
const busyPageCount = envInt("FRONTEND_BUSY_PAGES", 4);
const busyConcurrency = envInt("FRONTEND_BUSY_CONCURRENCY", 2);
const failedUploadPageCount = envInt("FRONTEND_FAILED_UPLOAD_PAGES", 4);
const failedUploadConcurrency = envInt("FRONTEND_FAILED_UPLOAD_CONCURRENCY", 2);
const token = process.env.FRONTEND_LOAD_TOKEN || "production-rag-fixed-test-login-token";
const startupMaxDomNodes = envInt("FRONTEND_STARTUP_MAX_DOM_NODES", 500);
const startupMaxImageNodes = envInt("FRONTEND_STARTUP_MAX_IMAGE_NODES", 10);
const startupMaxResources = envInt("FRONTEND_STARTUP_MAX_RESOURCES", 100);
const startupMaxTransferKb = envInt("FRONTEND_STARTUP_MAX_TRANSFER_KB", 10_000);
const outputPath = process.env.FRONTEND_LOAD_OUTPUT || "test-results/frontend-load-summary.json";
const interactionOutputPath =
  process.env.FRONTEND_INTERACTION_OUTPUT || "test-results/frontend-interaction-summary.json";
const busyOutputPath = process.env.FRONTEND_BUSY_OUTPUT || "test-results/frontend-busy-summary.json";
const failedUploadOutputPath =
  process.env.FRONTEND_FAILED_UPLOAD_OUTPUT || "test-results/frontend-failed-upload-summary.json";
const testTimeoutMs = envInt("FRONTEND_LOAD_TEST_TIMEOUT_MS", 30_000);

test.skip(!enabled, "Set RUN_FRONTEND_LOAD_E2E=1 to run the browser-level frontend load smoke.");
test.setTimeout(testTimeoutMs);

test.describe("browser-level frontend load smoke", () => {
  test("opens multiple authenticated workspace pages without frontend errors", async ({ browser, baseURL }) => {
    const started = performance.now();
    const samples = await runPages(browser, baseURL || "http://127.0.0.1:5173");
    const wallMs = roundMs(performance.now() - started);
    const failures = samples.filter((sample) => !sample.ok);
    const payload = {
      pages: pageCount,
      concurrency,
      wall_ms: wallMs,
      success: samples.length - failures.length,
      failed: failures.length,
      failure_rate: round(failures.length / Math.max(1, samples.length), 4),
      load_ms: summarize(samples.map((sample) => sample.load_ms)),
      metrics: summarizePageMetrics(samples.map((sample) => sample.metrics)),
      console_errors: samples.reduce((total, sample) => total + sample.console_errors.length, 0),
      page_errors: samples.reduce((total, sample) => total + sample.page_errors.length, 0),
      http_failures: samples.reduce((total, sample) => total + sample.http_failures.length, 0),
      isolated_conversation_requests: samples.reduce(
        (total, sample) => total + sample.isolated_conversation_requests,
        0,
      ),
      failed_samples: failures.slice(0, 10),
      samples: samples.slice(0, 20),
    };
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");

    expect(payload.failed, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.console_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.page_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.http_failures, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.isolated_conversation_requests, JSON.stringify(payload, null, 2)).toBe(pageCount);
  });

  test("uploads a file and renders streamed answers with mocked backend", async ({ browser, baseURL }) => {
    const started = performance.now();
    const samples = await runInteractionPages(browser, baseURL || "http://127.0.0.1:5173");
    const wallMs = roundMs(performance.now() - started);
    const failures = samples.filter((sample) => !sample.ok);
    const payload = {
      pages: interactionPageCount,
      concurrency: interactionConcurrency,
      wall_ms: wallMs,
      success: samples.length - failures.length,
      failed: failures.length,
      failure_rate: round(failures.length / Math.max(1, samples.length), 4),
      total_ms: summarize(samples.map((sample) => sample.total_ms)),
      upload_ready_ms: summarize(samples.map((sample) => sample.upload_ready_ms)),
      query_ms: summarize(samples.map((sample) => sample.query_ms)),
      metrics: summarizePageMetrics(samples.map((sample) => sample.metrics)),
      console_errors: samples.reduce((total, sample) => total + sample.console_errors.length, 0),
      page_errors: samples.reduce((total, sample) => total + sample.page_errors.length, 0),
      http_failures: samples.reduce((total, sample) => total + sample.http_failures.length, 0),
      failed_samples: failures.slice(0, 10),
      samples: samples.slice(0, 20),
    };
    mkdirSync(dirname(interactionOutputPath), { recursive: true });
    writeFileSync(interactionOutputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");

    expect(payload.failed, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.console_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.page_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.http_failures, JSON.stringify(payload, null, 2)).toBe(0);
  });

  test("shows a friendly busy message when streamed answers are rejected", async ({ browser, baseURL }) => {
    const started = performance.now();
    const samples = await runBusyPages(browser, baseURL || "http://127.0.0.1:5173");
    const wallMs = roundMs(performance.now() - started);
    const failures = samples.filter((sample) => !sample.ok);
    const payload = {
      pages: busyPageCount,
      concurrency: busyConcurrency,
      wall_ms: wallMs,
      success: samples.length - failures.length,
      failed: failures.length,
      failure_rate: round(failures.length / Math.max(1, samples.length), 4),
      total_ms: summarize(samples.map((sample) => sample.total_ms)),
      upload_ready_ms: summarize(samples.map((sample) => sample.upload_ready_ms)),
      query_ms: summarize(samples.map((sample) => sample.query_ms)),
      metrics: summarizePageMetrics(samples.map((sample) => sample.metrics)),
      expected_busy_responses: samples.reduce((total, sample) => total + sample.expected_busy_responses, 0),
      expected_busy_console_errors: samples.reduce((total, sample) => total + sample.expected_busy_console_errors, 0),
      console_errors: samples.reduce((total, sample) => total + sample.console_errors.length, 0),
      page_errors: samples.reduce((total, sample) => total + sample.page_errors.length, 0),
      http_failures: samples.reduce((total, sample) => total + sample.http_failures.length, 0),
      failed_samples: failures.slice(0, 10),
      samples: samples.slice(0, 20),
    };
    mkdirSync(dirname(busyOutputPath), { recursive: true });
    writeFileSync(busyOutputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");

    expect(payload.failed, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.expected_busy_responses, JSON.stringify(payload, null, 2)).toBe(busyPageCount);
    expect(payload.expected_busy_console_errors, JSON.stringify(payload, null, 2)).toBe(busyPageCount);
    expect(payload.console_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.page_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.http_failures, JSON.stringify(payload, null, 2)).toBe(0);
  });

  test("keeps failed ingestion sources visible after upload polling", async ({ browser, baseURL }) => {
    const started = performance.now();
    const samples = await runFailedUploadPages(browser, baseURL || "http://127.0.0.1:5173");
    const wallMs = roundMs(performance.now() - started);
    const failures = samples.filter((sample) => !sample.ok);
    const payload = {
      pages: failedUploadPageCount,
      concurrency: failedUploadConcurrency,
      wall_ms: wallMs,
      success: samples.length - failures.length,
      failed: failures.length,
      failure_rate: round(failures.length / Math.max(1, samples.length), 4),
      total_ms: summarize(samples.map((sample) => sample.total_ms)),
      upload_ready_ms: summarize(samples.map((sample) => sample.upload_ready_ms)),
      metrics: summarizePageMetrics(samples.map((sample) => sample.metrics)),
      expected_failed_sources: samples.reduce((total, sample) => total + sample.expected_failed_sources, 0),
      console_errors: samples.reduce((total, sample) => total + sample.console_errors.length, 0),
      page_errors: samples.reduce((total, sample) => total + sample.page_errors.length, 0),
      http_failures: samples.reduce((total, sample) => total + sample.http_failures.length, 0),
      failed_samples: failures.slice(0, 10),
      samples: samples.slice(0, 20),
    };
    mkdirSync(dirname(failedUploadOutputPath), { recursive: true });
    writeFileSync(failedUploadOutputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");

    expect(payload.failed, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.expected_failed_sources, JSON.stringify(payload, null, 2)).toBe(failedUploadPageCount);
    expect(payload.console_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.page_errors, JSON.stringify(payload, null, 2)).toBe(0);
    expect(payload.http_failures, JSON.stringify(payload, null, 2)).toBe(0);
  });

  test("shows stable status text and recovery guidance for ingestion", async ({ page, baseURL }) => {
    const staleSource = {
      ...mockSource("stale-ingestion.txt", 700, "processing"),
      attempt_count: 2,
      created_at: Date.now() - 35 * 60 * 1000,
      updated_at: Date.now() - 31 * 60 * 1000,
    };
    const retryWaitingSource = {
      ...mockSource("retry-waiting.txt", 702, "queued"),
      attempt_count: 1,
      next_attempt_at: Date.now() + 60_000,
    };
    await seedBrowserSession(page, 700);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources: [staleSource, retryWaitingSource] }),
      });
    });

    await page.goto(baseURL || "http://127.0.0.1:5173", { waitUntil: "domcontentloaded" });
    const row = page.locator(".source-row.is-stale-task", { hasText: "stale-ingestion.txt" });
    await expect(row).toBeVisible();
    await expect(row.getByText("处理中")).toBeVisible();
    await expect(row.getByText("处理时间已超过 30 分钟")).toBeVisible();
    await expect(row.getByText("疑似停滞，系统将自动尝试恢复")).toBeVisible();
    await expect(row.getByText("第 2 次尝试")).toBeVisible();
    await expect(row.getByText(/已等待|已处理|完成时间取决于当前队列/)).toHaveCount(0);
    const retryRow = page.locator(".source-row.status-queued", { hasText: "retry-waiting.txt" });
    await expect(retryRow.getByText("等待自动重试 · 已尝试 1 次")).toBeVisible();
    await expect(retryRow.getByText(/秒后|分钟后/)).toHaveCount(0);
  });

  test("requeues a retryable failed ingestion source", async ({ page, baseURL }) => {
    const failedSource = {
      ...mockSource("retryable-ingestion.txt", 701, "failed"),
      retryable: true,
      attempt_count: 3,
      dead_lettered: true,
      error: "Synthetic terminal ingestion failure",
    };
    let currentSource = failedSource;
    let retryRequests = 0;
    await seedBrowserSession(page, 701);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (request.method() === "POST" && path.endsWith(`/${failedSource.doc_id}/retry`)) {
        retryRequests += 1;
        currentSource = {
          ...failedSource,
          status: "queued",
          retryable: false,
          attempt_count: 0,
          next_attempt_at: 0,
          dead_lettered: false,
          error: "",
          updated_at: Date.now(),
        };
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ status: "queued", source: currentSource }),
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources: [currentSource] }),
      });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-701`, {
      waitUntil: "domcontentloaded",
    });
    const row = page.locator(".source-row", { hasText: failedSource.title });
    await expect(row).toHaveClass(/status-failed/);
    await expect(row.getByText("Synthetic terminal ingestion failure")).toBeVisible();
    await expect(row.getByText("已停止自动重试 · 共尝试 3 次，可选择重新处理")).toBeVisible();
    await row.locator(".row-icon-more").click();
    await page.getByRole("button", { name: "重新处理" }).click();
    await expect.poll(() => retryRequests).toBe(1);
    await expect(row).toHaveClass(/status-queued/);
    await expect(row.getByText("排队中")).toBeVisible();
    await expect(row.getByText("Synthetic terminal ingestion failure")).toHaveCount(0);
  });

  test("queues extra uploads beyond the per-page upload limit", async ({ page, baseURL }) => {
    const names = ["queued-upload-1.txt", "queued-upload-2.txt", "queued-upload-3.txt"];
    const uploadRoutes: Array<{ resolve: () => void }> = [];
    const readyUploads = new Set<number>();
    let uploadPostCount = 0;
    let sourceGetInFlight = 0;
    let maxSourceGetInFlight = 0;

    await seedBrowserSession(page, 900);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      const request = route.request();
      const url = request.url();
      if (request.method() === "POST" && url.includes("/sources/upload")) {
        const index = uploadPostCount;
        uploadPostCount += 1;
        await new Promise<void>((resolve) => uploadRoutes.push({ resolve }));
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ sources: [mockSource(names[index], index, "processing")] }),
        });
        return;
      }
      if (request.method() === "GET") {
        sourceGetInFlight += 1;
        maxSourceGetInFlight = Math.max(maxSourceGetInFlight, sourceGetInFlight);
        await page.waitForTimeout(50);
        sourceGetInFlight -= 1;
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            sources: [...readyUploads].map((index) => mockSource(names[index], index, "ready")),
          }),
        });
        return;
      }
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
    });

    await page.goto(baseURL || "http://127.0.0.1:5173", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "来源" })).toBeVisible();

    await chooseUploadFile(page, names[0]);
    await chooseUploadFile(page, names[1]);
    await expect.poll(() => uploadPostCount).toBe(2);
    await chooseUploadFile(page, names[2]);

    await expect(page.locator(".source-row.status-uploading", { hasText: names[0] })).toBeVisible();
    await expect(page.locator(".source-row.status-uploading", { hasText: names[1] })).toBeVisible();
    await expect(page.locator(".source-row.status-queued", { hasText: names[2] })).toBeVisible();
    await expect(page.locator(".source-row.status-queued", { hasText: names[2] }).getByText("排队中")).toBeVisible();
    await expect(page.getByText(/已等待|完成时间取决于当前队列/)).toHaveCount(0);
    expect(uploadPostCount).toBe(2);

    readyUploads.add(0);
    uploadRoutes[0]?.resolve();
    await expect.poll(() => uploadPostCount, { timeout: 8_000 }).toBe(3);
    await expect(page.locator(".source-row.status-uploading", { hasText: names[2] })).toBeVisible();
    expect(maxSourceGetInFlight).toBeLessThanOrEqual(1);
  });

  test("keeps an upload visible when it resolves while the page is refreshing", async ({ page, baseURL }) => {
    const title = "refresh-safe-upload.pdf";
    const pending = {
      ...mockSource(title, 910, "processing"),
      doc_id: "upload-refresh-safe-task",
      source_uri: "mock://refresh-safe-work-copy.pdf",
    };
    const ready = {
      ...mockSource(title, 910, "ready"),
      doc_id: "refresh-safe-logical-source",
      source_uri: "mock://refresh-safe-canonical.pdf",
      child_doc_ids: ["refresh-safe-logical-source/page-1"],
      workspace_alias_ids: [pending.doc_id],
    };
    let uploaded = false;
    let completed = false;

    await seedBrowserSession(page, 910);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      const request = route.request();
      if (request.method() === "POST" && request.url().includes("/sources/upload")) {
        uploaded = true;
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ sources: [pending] }),
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources: !uploaded ? [] : completed ? [ready] : [pending] }),
      });
    });

    await page.goto(baseURL || "http://127.0.0.1:5173", { waitUntil: "domcontentloaded" });
    await chooseUploadFile(page, title);
    await expect(page.locator(".source-row.status-processing", { hasText: title })).toBeVisible();
    await expect.poll(() =>
      page.evaluate(
        (taskId) => Object.values(localStorage).some((value) => value.includes(taskId)),
        pending.doc_id,
      ),
    ).toBe(true);

    completed = true;
    for (let index = 0; index < 7; index += 1) {
      await page.reload({ waitUntil: "domcontentloaded" });
    }
    const resolvedRow = page.locator(".source-row.status-ready", { hasText: title });
    await expect(resolvedRow).toBeVisible();
    await expect(resolvedRow).toContainText("refresh-safe-upload.pdf");
  });

  test("renders large source lists incrementally", async ({ page, baseURL }) => {
    const sources = Array.from({ length: 120 }, (_, index) =>
      mockSource(`bulk-source-${index.toString().padStart(3, "0")}.txt`, index, "ready"),
    );

    await seedBrowserSession(page, 901);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources }),
      });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-901`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.getByRole("heading", { name: "来源" })).toBeVisible();
    await expect(page.locator(".source-row", { hasText: "bulk-source-000.txt" })).toBeVisible();
    await expect(page.locator(".source-row", { hasText: "bulk-source-079.txt" })).toBeVisible();
    await expect(page.locator(".source-row", { hasText: "bulk-source-119.txt" })).toHaveCount(0);

    await page.getByRole("button", { name: /显示更多来源/ }).click();
    await expect(page.locator(".source-row", { hasText: "bulk-source-119.txt" })).toBeVisible();
  });

  test("sends every selected PDF child document to retrieval", async ({ page, baseURL }) => {
    const now = Date.now();
    const sources = ["attention", "autoformer", "third-paper"].map((sourceId, index) => ({
      doc_id: sourceId,
      title: `${sourceId}.pdf`,
      source_type: "pdf",
      source_uri: `mock://${sourceId}.pdf`,
      doc_version: 1,
      chunk_count: 2,
      acl_groups: ["engineering"],
      status: "ready",
      current: true,
      created_at: now + index,
      updated_at: now + index,
      child_doc_ids: [`${sourceId}/page-1`, `${sourceId}/page-2`],
    }));
    let requestedDocIds: string[] = [];

    await seedBrowserSession(page, 902);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources }),
      });
    });
    await page.unroute("**/api/conversations**");
    await page.route("**/api/conversations**", async (route) => {
      const request = route.request();
      if (request.method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ conversations: [] }),
        });
        return;
      }
      const body = JSON.parse(request.postData() || "{}") as {
        title?: string;
        messages?: unknown[];
        source_doc_ids?: string[];
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "multi-source-selection-conversation",
          tenant_id: "queued-upload-tenant",
          title: body.title || "Multi source selection",
          messages: body.messages || [],
          source_doc_ids: body.source_doc_ids || [],
          created_at: now,
          updated_at: Date.now(),
        }),
      });
    });
    await page.route("**/api/query/stream", async (route) => {
      const body = JSON.parse(route.request().postData() || "{}") as { doc_ids?: string[] };
      requestedDocIds = body.doc_ids || [];
      await route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: `${JSON.stringify({
          type: "result",
          request_id: "multi-source-selection-request",
          answer: "All selected documents reached retrieval.",
          citations: [],
          trace: {},
        })}\n`,
      });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-902`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.locator(".source-row input[type='checkbox']:checked")).toHaveCount(3);
    await page.locator("#chat-input-textarea").fill("查找跨文档事实");
    await page.getByRole("button", { name: "发送消息" }).click();
    await expect(page.getByText("All selected documents reached retrieval.")).toBeVisible();
    await expect.poll(() => requestedDocIds).toEqual([
      "attention/page-1",
      "attention/page-2",
      "autoformer/page-1",
      "autoformer/page-2",
      "third-paper/page-1",
      "third-paper/page-2",
    ]);
  });

  test("renders long conversation histories incrementally", async ({ page, baseURL }) => {
    const now = Date.now();
    const messages = Array.from({ length: 120 }, (_, index) => ({
      id: `long-history-message-${index}`,
      role: index % 2 === 0 ? "user" : "assistant",
      content: `long-history-message-${index.toString().padStart(3, "0")}`,
      status: "done",
      request_id: null,
      citations: [],
      image_data_url: null,
      created_at: now + index,
      feedback_rating: null,
      rag_progress: [],
    }));

    await page.route("**/api/**", async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (path.endsWith("/health")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
        return;
      }
      if (path.endsWith("/auth/me")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: "long-history-user",
            username: "long-history-user",
            display_name: "Long History User",
            role: "user",
            tenant_id: "browser-load-tenant-902",
            created_at: now,
            status: "active",
          }),
        });
        return;
      }
      if (path.endsWith("/admin/settings")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ registration_enabled: true, latest_announcement: null }),
        });
        return;
      }
      if (path.endsWith("/announcements")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ announcements: [] }) });
        return;
      }
      if (path.endsWith("/artifacts")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ artifacts: [] }) });
        return;
      }
      if (path.endsWith("/sources")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ sources: [] }) });
        return;
      }
      if (request.method() === "GET" && path.endsWith("/conversations/long-history-conversation")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: "long-history-conversation",
            tenant_id: "browser-load-tenant-902",
            title: "Long history conversation",
            messages,
            source_doc_ids: [],
            created_at: now,
            updated_at: now + messages.length,
          }),
        });
        return;
      }
      if (request.method() === "GET" && path.endsWith("/conversations")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: [
              {
                id: "long-history-conversation",
                tenant_id: "browser-load-tenant-902",
                title: "Long history conversation",
                message_count: messages.length,
                source_doc_ids: [],
                created_at: now,
                updated_at: now + messages.length,
              },
            ],
          }),
        });
        return;
      }
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-902`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.getByRole("heading", { name: "对话" })).toBeVisible();
    await expect(page.getByText("long-history-message-119")).toBeVisible();
    await expect(page.getByText("long-history-message-000")).toHaveCount(0);

    await page.getByRole("button", { name: /显示更早消息/ }).click();
    await expect(page.getByText("long-history-message-000")).toBeVisible();
  });

  test("lazy-loads source reader images", async ({ page, baseURL }) => {
    const source = mockSource("image-heavy-source.pdf", 950, "ready");

    await seedBrowserSession(page, 950);
    await mockStartupApi(page);
    await page.route("**/api/source-assets/**", async (route) => {
      const request = route.request();
      expect(new URL(request.url()).searchParams.has("token")).toBe(false);
      expect(request.headers().authorization).toBe("Bearer browser-load-token-950");
      await route.fulfill({
        status: 200,
        contentType: "image/png",
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
          "base64",
        ),
      });
    });
    await page.route("**/api/sources**", async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (request.method() === "GET" && path.includes("/sources/content/")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            ...source,
            guide: "Source with image blocks.",
            tags: ["image"],
            text: "",
            child_doc_ids: [source.doc_id],
            blocks: [
              {
                type: "image",
                title: "Architecture figure",
                page: "p1",
                url: "/api/source-assets/mock-image-heavy-source-page-1.png?token=browser-load-token-950",
              },
            ],
            suggested_title: source.title,
          }),
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources: [source] }),
      });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-950`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.locator(".source-row", { hasText: source.title })).toBeVisible();
    await page.locator(".source-row", { hasText: source.title }).getByRole("button", { name: source.title }).click();
    const image = page.locator(".source-document-image img");
    await expect(image).toHaveAttribute("loading", "lazy");
    await expect(image).toHaveAttribute("decoding", "async");
  });

  test("keeps expanded long-label mind-map nodes collision free", async ({ page, baseURL }) => {
    const now = Date.now();
    const artifact = {
      id: "mindmap-collision-smoke",
      title: "长文本防碰撞思维导图",
      status: "ready",
      tenant_id: "queued-upload-tenant",
      workspace_id: "default",
      source_doc_ids: [],
      created_at: now,
      updated_at: now,
      artifact_type: "mindmap",
      root: {
        id: "mindmap-root",
        label: "生产级 RAG 系统",
        children: [
          {
            id: "mindmap-retrieval",
            label: "检索链路",
            children: [
              {
                id: "mindmap-retrieval-1",
                label: "这是一个非常长的第三级节点，包含查询重写、混合向量检索、关键词召回以及跨编码器重排序等多个步骤。",
              },
              {
                id: "mindmap-retrieval-2",
                label: "上下文组装需要同时考虑字符预算、每文档片段上限、相关性阈值以及引用证据的多样性。",
              },
              {
                id: "mindmap-retrieval-3",
                label: "long-unbroken-ascii-token-for-browser-layout-collision-regression-check".repeat(3),
              },
            ],
          },
          {
            id: "mindmap-ingestion",
            label: "文档摄取与索引",
            children: [
              {
                id: "mindmap-ingestion-1",
                label: "解析 PDF、抽取图片、生成文本与图片向量，并将版本化片段批量写入 Milvus。",
              },
              {
                id: "mindmap-ingestion-2",
                label: "失败任务采用指数退避重试，长期 processing 任务由恢复流程重新排队，避免永久停滞。",
              },
            ],
          },
          {
            id: "mindmap-observability",
            label: "监控与容量",
            children: [
              {
                id: "mindmap-observability-1",
                label: "Prometheus 采集 HTTP histogram、模型调用、连接池和 ingestion backlog，Grafana 展示 p95 与告警。",
              },
            ],
          },
        ],
      },
      table: null,
    };

    await seedBrowserSession(page, 750);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources: [] }),
      });
    });
    await page.unroute("**/api/artifacts**");
    await page.route("**/api/artifacts**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ artifacts: [artifact] }),
      });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-750`, {
      waitUntil: "domcontentloaded",
    });
    await page.getByText("长文本防碰撞思维导图", { exact: true }).click();
    await expect(page.locator(".mindmap-detail")).toBeVisible();
    const branches = page.locator(".react-flow__node.mindmap-flow-node.branch");
    await expect(branches).toHaveCount(3);
    for (let index = 0; index < 3; index += 1) {
      await branches.nth(index).click();
    }
    const leaves = page.locator(".react-flow__node.mindmap-flow-node.leaf");
    await expect(leaves).toHaveCount(6);
    await page.waitForTimeout(600);

    const geometry = await leaves.evaluateAll((nodes) =>
      nodes.map((node) => {
        const rect = node.getBoundingClientRect();
        return {
          text: node.textContent || "",
          top: rect.top,
          bottom: rect.bottom,
          left: rect.left,
          right: rect.right,
          height: rect.height,
        };
      }),
    );
    expect(geometry.some((item) => item.height > Math.min(...geometry.map((entry) => entry.height)) * 1.8)).toBe(true);
    expect(findOverlappingNodePairs(geometry)).toEqual([]);
  });

  test("keeps final answers stable after high-frequency stages and reload", async ({ page, baseURL }) => {
    const now = Date.now();
    let savedMessages = 0;
    let persistedConversation: {
      id: string;
      tenant_id: string;
      title: string;
      messages: unknown[];
      source_doc_ids: string[];
      created_at: number;
      updated_at: number;
    } | null = null;

    await page.route("**/api/**", async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (path.endsWith("/health")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
        return;
      }
      if (path.endsWith("/auth/me")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: "high-frequency-stage-user",
            username: "high-frequency-stage-user",
            display_name: "High Frequency Stage User",
            role: "user",
            tenant_id: "browser-load-tenant-960",
            created_at: now,
            status: "active",
          }),
        });
        return;
      }
      if (path.endsWith("/admin/settings")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ registration_enabled: true, latest_announcement: null }),
        });
        return;
      }
      if (path.endsWith("/announcements")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ announcements: [] }) });
        return;
      }
      if (path.endsWith("/sources")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ sources: [] }) });
        return;
      }
      if (path.endsWith("/artifacts")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ artifacts: [] }) });
        return;
      }
      if (path.endsWith("/conversations") && request.method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: persistedConversation
              ? [{
                  id: persistedConversation.id,
                  tenant_id: persistedConversation.tenant_id,
                  title: persistedConversation.title,
                  message_count: persistedConversation.messages.length,
                  source_doc_ids: persistedConversation.source_doc_ids,
                  created_at: persistedConversation.created_at,
                  updated_at: persistedConversation.updated_at,
                }]
              : [],
          }),
        });
        return;
      }
      if (path.endsWith("/conversations") && request.method() === "POST") {
        const body = JSON.parse(request.postData() || "{}") as { messages?: unknown[] };
        savedMessages = body.messages?.length || 0;
        persistedConversation = {
          id: "high-frequency-stage-conversation",
          tenant_id: "browser-load-tenant-960",
          title: "High frequency stage conversation",
          messages: body.messages || [],
          source_doc_ids: [],
          created_at: now,
          updated_at: Date.now(),
        };
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(persistedConversation),
        });
        return;
      }
      if (path.endsWith("/conversations/high-frequency-stage-conversation") && request.method() === "GET") {
        await route.fulfill({
          status: persistedConversation ? 200 : 404,
          contentType: "application/json",
          body: JSON.stringify(persistedConversation || { detail: "Conversation not found" }),
        });
        return;
      }
      if (path.endsWith("/query/stream") && request.method() === "POST") {
        const stageEvents = Array.from({ length: 80 }, (_, index) =>
          JSON.stringify({
            type: "stage",
            stage: "answer",
            label: "大模型直接回答",
            detail: `高频阶段 ${index}`,
            status: "active",
            latency_ms: index,
          }),
        );
        const result = JSON.stringify({
          type: "result",
          request_id: "high-frequency-stage-request",
          answer: "Stable final answer after high-frequency stages.",
          citations: [],
          trace: {},
        });
        await route.fulfill({
          status: 200,
          contentType: "application/x-ndjson",
          body: `${[...stageEvents, result].join("\n")}\n`,
        });
        return;
      }
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-960`, {
      waitUntil: "domcontentloaded",
    });
    await page.locator("#chat-input-textarea").fill("触发高频 stage");
    await page.getByRole("button", { name: "发送消息" }).click();
    await expect(page.getByText("Stable final answer after high-frequency stages.")).toBeVisible();
    await page.waitForTimeout(100);
    await expect(page.getByText("Stable final answer after high-frequency stages.")).toBeVisible();
    await expect.poll(() => savedMessages).toBe(2);
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(page.getByText("Stable final answer after high-frequency stages.")).toBeVisible();
    await expect(page.getByText("触发高频 stage")).toBeVisible();
  });

  test("preserves an interrupted answer and completes it after reload", async ({ page, baseURL }) => {
    const now = Date.now();
    let queryAttempts = 0;
    let persistedConversation: {
      id: string;
      tenant_id: string;
      title: string;
      messages: Array<Record<string, unknown>>;
      source_doc_ids: string[];
      created_at: number;
      updated_at: number;
    } | null = null;

    await seedBrowserSession(page, 961);
    await mockStartupApi(page);
    await page.route("**/api/sources**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources: [] }),
      });
    });
    await page.unroute("**/api/conversations**");
    await page.route("**/api/conversations**", async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (request.method() === "GET" && path.endsWith("/conversations")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: persistedConversation
              ? [{
                  id: persistedConversation.id,
                  tenant_id: persistedConversation.tenant_id,
                  title: persistedConversation.title,
                  message_count: persistedConversation.messages.length,
                  source_doc_ids: persistedConversation.source_doc_ids,
                  created_at: persistedConversation.created_at,
                  updated_at: persistedConversation.updated_at,
                }]
              : [],
          }),
        });
        return;
      }
      if (request.method() === "GET" && path.endsWith("/conversations/interrupted-recovery-conversation")) {
        await route.fulfill({
          status: persistedConversation ? 200 : 404,
          contentType: "application/json",
          body: JSON.stringify(persistedConversation || { detail: "Conversation not found" }),
        });
        return;
      }
      if (request.method() === "POST" && path.endsWith("/conversations")) {
        const body = JSON.parse(request.postData() || "{}") as {
          title?: string;
          messages?: Array<Record<string, unknown>>;
          source_doc_ids?: string[];
        };
        persistedConversation = {
          id: "interrupted-recovery-conversation",
          tenant_id: "queued-upload-tenant",
          title: body.title || "Interrupted recovery conversation",
          messages: body.messages || [],
          source_doc_ids: body.source_doc_ids || [],
          created_at: persistedConversation?.created_at || now,
          updated_at: Date.now(),
        };
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(persistedConversation),
        });
        return;
      }
      await route.fulfill({
        status: 404,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Conversation not found" }),
      });
    });
    await page.route("**/api/query/stream", async (route) => {
      queryAttempts += 1;
      if (queryAttempts === 1) {
        await route.abort("failed");
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: `${JSON.stringify({
          type: "result",
          request_id: "interrupted-recovery-request",
          answer: "Recovered answer after browser reload.",
          citations: [],
          trace: {},
        })}\n`,
      });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-961`, {
      waitUntil: "domcontentloaded",
    });
    await page.locator("#chat-input-textarea").fill("测试断流恢复");
    await page.getByRole("button", { name: "发送消息" }).click();
    await expect(page.getByText("连接已中断，刷新页面后将自动恢复回答。")).toBeVisible();
    await expect.poll(() => persistedConversation?.messages.at(-1)?.status).toBe("sending");
    await expect.poll(() => persistedConversation?.messages.at(-1)?.content).toBe(
      "连接已中断，刷新页面后将自动恢复回答。",
    );

    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(page.getByText("Recovered answer after browser reload.")).toBeVisible();
    await expect(page.getByText("测试断流恢复")).toBeVisible();
    await expect.poll(() => queryAttempts).toBe(2);
    await expect.poll(() => persistedConversation?.messages.at(-1)?.status).toBe("done");
    await expect.poll(() => persistedConversation?.messages.at(-1)?.content).toBe(
      "Recovered answer after browser reload.",
    );
  });

  test("compresses large chat image attachments before sending", async ({ page, baseURL }) => {
    const now = Date.now();
    let streamedImageDataUrl = "";
    let savedImageDataUrl = "";

    await page.route("**/api/**", async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (path.endsWith("/health")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
        return;
      }
      if (path.endsWith("/auth/me")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: "image-compression-user",
            username: "image-compression-user",
            display_name: "Image Compression User",
            role: "user",
            tenant_id: "browser-load-tenant-970",
            created_at: now,
            status: "active",
          }),
        });
        return;
      }
      if (path.endsWith("/admin/settings")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ registration_enabled: true, latest_announcement: null }),
        });
        return;
      }
      if (path.endsWith("/announcements")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ announcements: [] }) });
        return;
      }
      if (path.endsWith("/sources")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ sources: [] }) });
        return;
      }
      if (path.endsWith("/artifacts")) {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ artifacts: [] }) });
        return;
      }
      if (path.endsWith("/conversations") && request.method() === "GET") {
        await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ conversations: [] }) });
        return;
      }
      if (path.endsWith("/conversations") && request.method() === "POST") {
        const body = JSON.parse(request.postData() || "{}") as { messages?: Array<{ image_data_url?: string | null }> };
        savedImageDataUrl = body.messages?.find((message) => message.image_data_url)?.image_data_url || "";
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: "image-compression-conversation",
            tenant_id: "browser-load-tenant-970",
            title: "Image compression conversation",
            messages: body.messages || [],
            source_doc_ids: [],
            created_at: now,
            updated_at: Date.now(),
          }),
        });
        return;
      }
      if (path.endsWith("/query/stream") && request.method() === "POST") {
        const body = JSON.parse(request.postData() || "{}") as { image_data_url?: string | null };
        streamedImageDataUrl = body.image_data_url || "";
        await route.fulfill({
          status: 200,
          contentType: "application/x-ndjson",
          body: `${JSON.stringify({
            type: "result",
            request_id: "image-compression-request",
            answer: "Compressed image accepted.",
            citations: [],
            trace: {},
          })}\n`,
        });
        return;
      }
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
    });

    await page.goto(`${(baseURL || "http://127.0.0.1:5173").replace(/\/$/, "")}/#token=browser-load-token-970`, {
      waitUntil: "domcontentloaded",
    });
    const fileChooserPromise = page.waitForEvent("filechooser");
    await page.getByRole("button", { name: "上传图片提问" }).click();
    const fileChooser = await fileChooserPromise;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="3200" height="1800"><rect width="3200" height="1800" fill="#ffffff"/><text x="120" y="220" font-size="120" fill="#111827">large chat image</text></svg>`;
    await fileChooser.setFiles({
      name: "large-chat-image.svg",
      mimeType: "image/svg+xml",
      buffer: Buffer.from(svg, "utf-8"),
    });
    await expect(page.locator(".attachment-preview-button img")).toHaveAttribute("src", /^data:image\/jpeg/);
    await page.locator("#chat-input-textarea").fill("根据这张图回答");
    await page.getByRole("button", { name: "发送消息" }).click();
    await expect(page.getByText("Compressed image accepted.")).toBeVisible();
    expect(streamedImageDataUrl.startsWith("data:image/jpeg")).toBe(true);
    expect(savedImageDataUrl.startsWith("data:image/jpeg")).toBe(true);
    expect(streamedImageDataUrl.length).toBeLessThan(1_000_000);
  });
});

async function runPages(browser: Browser, baseURL: string) {
  const results: PageSample[] = [];
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(concurrency, pageCount) }, async () => {
    while (nextIndex < pageCount) {
      const index = nextIndex;
      nextIndex += 1;
      results.push(await openWorkspacePage(browser, baseURL, index));
    }
  });
  await Promise.all(workers);
  return results.sort((left, right) => left.index - right.index);
}

async function openWorkspacePage(browser: Browser, baseURL: string, index: number): Promise<PageSample> {
  const page = await browser.newPage();
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const httpFailures: string[] = [];
  let isolatedConversationRequests = 0;
  await page.route("**/api/conversations**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "GET" && path.endsWith("/conversations")) {
      isolatedConversationRequests += 1;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ conversations: [] }),
      });
      return;
    }
    await route.continue();
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });
  page.on("response", (response) => {
    if (response.status() >= 500) {
      httpFailures.push(`${response.status()} ${response.url()}`);
    }
  });

  const started = performance.now();
  try {
    await page.goto(`${baseURL.replace(/\/$/, "")}/#token=${encodeURIComponent(token)}`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.getByRole("heading", { name: "来源" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "对话" })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Studio|工作室/ })).toBeVisible();
    await expect(
      page.locator(".message-list .user-message, .message-list .assistant-message"),
    ).toHaveCount(0);
    await waitForNetworkQuiet(page, 500, 10_000);
    const loadMs = roundMs(performance.now() - started);
    const metrics = await collectPageMetrics(page);
    const baselineWithinLimits = startupMetricsWithinLimits(metrics);
    await closePage(page);
    return {
      index,
      ok:
        consoleErrors.length === 0
        && pageErrors.length === 0
        && httpFailures.length === 0
        && isolatedConversationRequests === 1
        && baselineWithinLimits,
      load_ms: loadMs,
      metrics,
      isolated_conversation_requests: isolatedConversationRequests,
      baseline_within_limits: baselineWithinLimits,
      console_errors: consoleErrors,
      page_errors: pageErrors,
      http_failures: httpFailures,
    };
  } catch (error) {
    const loadMs = roundMs(performance.now() - started);
    const metrics = await collectPageMetrics(page).catch(() => null);
    await closePage(page);
    return {
      index,
      ok: false,
      load_ms: loadMs,
      metrics,
      isolated_conversation_requests: isolatedConversationRequests,
      baseline_within_limits: false,
      console_errors: consoleErrors,
      page_errors: pageErrors,
      http_failures: httpFailures,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function runInteractionPages(browser: Browser, baseURL: string) {
  const results: InteractionSample[] = [];
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(interactionConcurrency, interactionPageCount) }, async () => {
    while (nextIndex < interactionPageCount) {
      const index = nextIndex;
      nextIndex += 1;
      results.push(await interactWithWorkspace(browser, baseURL, index, "success"));
    }
  });
  await Promise.all(workers);
  return results.sort((left, right) => left.index - right.index);
}

async function runBusyPages(browser: Browser, baseURL: string) {
  const results: InteractionSample[] = [];
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(busyConcurrency, busyPageCount) }, async () => {
    while (nextIndex < busyPageCount) {
      const index = nextIndex;
      nextIndex += 1;
      results.push(await interactWithWorkspace(browser, baseURL, index, "busy"));
    }
  });
  await Promise.all(workers);
  return results.sort((left, right) => left.index - right.index);
}

async function runFailedUploadPages(browser: Browser, baseURL: string) {
  const results: InteractionSample[] = [];
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(failedUploadConcurrency, failedUploadPageCount) }, async () => {
    while (nextIndex < failedUploadPageCount) {
      const index = nextIndex;
      nextIndex += 1;
      results.push(await interactWithWorkspace(browser, baseURL, index, "ingest-failed"));
    }
  });
  await Promise.all(workers);
  return results.sort((left, right) => left.index - right.index);
}

async function interactWithWorkspace(
  browser: Browser,
  baseURL: string,
  index: number,
  scenario: InteractionScenario,
): Promise<InteractionSample> {
  const page = await browser.newPage();
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const httpFailures: string[] = [];
  let expectedBusyResponses = 0;
  let expectedBusyConsoleErrors = 0;
  let expectedFailedSources = 0;
  page.on("console", (message) => {
    if (message.type() === "error") {
      if (scenario === "busy" && message.text().includes("503") && message.text().includes("Service Unavailable")) {
        expectedBusyConsoleErrors += 1;
        return;
      }
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });
  page.on("response", (response) => {
    if (response.status() >= 500) {
      if (scenario === "busy" && response.status() === 503 && response.url().includes("/api/query/stream")) {
        expectedBusyResponses += 1;
        return;
      }
      httpFailures.push(`${response.status()} ${response.url()}`);
    }
  });

  const started = performance.now();
  let uploadReadyMs = 0;
  let queryMs = 0;
  try {
    await seedBrowserSession(page, index);
    await mockWorkspaceApi(page, index, scenario);

    await page.goto(baseURL.replace(/\/$/, ""), { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "来源" })).toBeVisible();

    const uploadStarted = performance.now();
    await page.getByRole("button", { name: "添加来源" }).click();
    const fileChooserPromise = page.waitForEvent("filechooser");
    await page.getByRole("button", { name: "上传文件" }).click();
    const fileChooser = await fileChooserPromise;
    const fileName = interactionFileName(index);
    await fileChooser.setFiles({
      name: fileName,
      mimeType: "text/plain",
      buffer: Buffer.from(`frontend interaction load test document ${index}`, "utf-8"),
    });
    if (scenario === "ingest-failed") {
      await expect(page.locator(".source-row.status-failed", { hasText: fileName })).toBeVisible({
        timeout: 12_000,
      });
      await expect(page.getByText("Mock ingestion failed")).toBeVisible({ timeout: 12_000 });
      expectedFailedSources = 1;
    } else {
      await expect(page.locator(".source-row.status-ready", { hasText: fileName })).toBeVisible({
        timeout: 12_000,
      });
    }
    uploadReadyMs = roundMs(performance.now() - uploadStarted);

    if (scenario === "ingest-failed") {
      const metrics = await collectPageMetrics(page);
      await closePage(page);
      return {
        index,
        ok: consoleErrors.length === 0 && pageErrors.length === 0 && httpFailures.length === 0 && expectedFailedSources === 1,
        total_ms: roundMs(performance.now() - started),
        upload_ready_ms: uploadReadyMs,
        query_ms: queryMs,
        metrics,
        expected_busy_responses: expectedBusyResponses,
        expected_busy_console_errors: expectedBusyConsoleErrors,
        expected_failed_sources: expectedFailedSources,
        console_errors: consoleErrors,
        page_errors: pageErrors,
        http_failures: httpFailures,
      };
    }

    const queryStarted = performance.now();
    await page.locator("#chat-input-textarea").fill("总结这个文件的核心内容");
    await page.getByRole("button", { name: "发送消息" }).click();
    if (scenario === "busy") {
      await expect(page.getByText("当前服务繁忙，请稍后重试。")).toBeVisible({ timeout: 12_000 });
    } else {
      await expect(page.getByText(`Mock streamed answer ${index}`)).toBeVisible({ timeout: 12_000 });
    }
    queryMs = roundMs(performance.now() - queryStarted);

    const metrics = await collectPageMetrics(page);
    await closePage(page);
    return {
      index,
      ok:
        consoleErrors.length === 0 &&
        pageErrors.length === 0 &&
        httpFailures.length === 0 &&
        (scenario !== "busy" || (expectedBusyResponses > 0 && expectedBusyConsoleErrors > 0)),
      total_ms: roundMs(performance.now() - started),
      upload_ready_ms: uploadReadyMs,
      query_ms: queryMs,
      metrics,
      expected_busy_responses: expectedBusyResponses,
      expected_busy_console_errors: expectedBusyConsoleErrors,
      expected_failed_sources: expectedFailedSources,
      console_errors: consoleErrors,
      page_errors: pageErrors,
      http_failures: httpFailures,
    };
  } catch (error) {
    const metrics = await collectPageMetrics(page).catch(() => null);
    await closePage(page);
    return {
      index,
      ok: false,
      total_ms: roundMs(performance.now() - started),
      upload_ready_ms: uploadReadyMs,
      query_ms: queryMs,
      metrics,
      expected_busy_responses: expectedBusyResponses,
      expected_busy_console_errors: expectedBusyConsoleErrors,
      expected_failed_sources: expectedFailedSources,
      console_errors: consoleErrors,
      page_errors: pageErrors,
      http_failures: httpFailures,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function seedBrowserSession(page: Page, index: number) {
  await page.addInitScript(
    ({ userId, tenantId, sessionToken, displayName }) => {
      const now = Date.now();
      localStorage.setItem(
        "production-rag-settings",
        JSON.stringify({
          apiBaseUrl: "/api",
          token: sessionToken,
          tenantId,
          aclGroups: ["engineering"],
        }),
      );
      localStorage.setItem(
        "production-rag-auth-session",
        JSON.stringify({
          token: sessionToken,
          expires_at: now + 60 * 60 * 1000,
          user: {
            id: userId,
            username: userId,
            display_name: displayName,
            role: "user",
            tenant_id: tenantId,
            created_at: now,
            status: "active",
          },
        }),
      );
    },
    {
      userId: `browser-load-user-${index}`,
      tenantId: `browser-load-tenant-${index}`,
      sessionToken: `browser-load-token-${index}`,
      displayName: `Browser Load User ${index}`,
    },
  );
}

async function mockWorkspaceApi(page: Page, index: number, scenario: InteractionScenario) {
  const now = Date.now();
  const fileName = interactionFileName(index);
  const pendingSource = {
    doc_id: `browser-load-upload-${index}`,
    title: fileName,
    source_type: "txt",
    source_uri: `mock://${fileName}`,
    doc_version: 1,
    chunk_count: 0,
    acl_groups: ["engineering"],
    status: "processing",
    current: false,
    created_at: now,
    updated_at: now,
    child_doc_ids: [],
  };
  const readySource = {
    ...pendingSource,
    doc_id: `browser-load-upload-${index}@sha256-ready`,
    chunk_count: 1,
    status: "ready",
    current: true,
    updated_at: now + 1000,
    child_doc_ids: [`browser-load-upload-${index}@sha256-ready`],
  };
  const failedSource = {
    ...pendingSource,
    status: "failed",
    current: false,
    updated_at: now + 1000,
    error: "Mock ingestion failed",
  };
  let uploaded = false;
  let sourcePollsAfterUpload = 0;

  await page.route("**/api/health", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
  });
  await page.route("**/api/auth/me**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: `browser-load-user-${index}`,
        username: `browser-load-user-${index}`,
        display_name: `Browser Load User ${index}`,
        role: "user",
        tenant_id: `browser-load-tenant-${index}`,
        created_at: now,
        status: "active",
      }),
    });
  });
  await page.route("**/api/admin/settings", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ registration_enabled: true, latest_announcement: null }),
    });
  });
  await page.route("**/api/announcements**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ announcements: [] }) });
  });
  await page.route("**/api/artifacts**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ artifacts: [] }) });
  });
  await page.route("**/api/conversations**", async (route) => {
    const request = route.request();
    if (request.method() === "POST") {
      const body = JSON.parse(request.postData() || "{}") as {
        id?: string | null;
        title?: string;
        messages?: unknown[];
        source_doc_ids?: string[];
        tenant_id?: string;
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: body.id || `browser-load-conversation-${index}`,
          tenant_id: body.tenant_id || `browser-load-tenant-${index}`,
          title: body.title || "Mock conversation",
          messages: body.messages || [],
          source_doc_ids: body.source_doc_ids || [readySource.doc_id],
          created_at: now,
          updated_at: Date.now(),
        }),
      });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ conversations: [] }) });
  });
  await page.route("**/api/sources**", async (route) => {
    const request = route.request();
    const url = request.url();
    if (request.method() === "POST" && url.includes("/sources/upload")) {
      uploaded = true;
      sourcePollsAfterUpload = 0;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ sources: [pendingSource] }),
      });
      return;
    }
    if (request.method() === "GET" && url.includes("/sources/content/")) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ...readySource,
          guide: "A short mock source guide.",
          tags: ["mock"],
          text: "This mock document is ready for browser interaction load testing.",
          blocks: [{ type: "text", text: "This mock document is ready for browser interaction load testing." }],
          suggested_title: fileName,
        }),
      });
      return;
    }
    if (request.method() === "GET") {
      const terminalSource = scenario === "ingest-failed" ? failedSource : readySource;
      const sources = uploaded && sourcePollsAfterUpload > 0 ? [terminalSource] : uploaded ? [pendingSource] : [];
      if (uploaded) {
        sourcePollsAfterUpload += 1;
      }
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ sources }) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
  });
  await page.route("**/api/query/stream", async (route) => {
    if (scenario === "busy") {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Query service is busy. Please retry later." }),
      });
      return;
    }
    const body = [
      JSON.stringify({
        type: "stage",
        stage: "receive",
        label: "接收问题",
        detail: "已收到问题。",
        status: "done",
        latency_ms: 1,
      }),
      JSON.stringify({
        type: "result",
        request_id: `browser-load-request-${index}`,
        answer: `Mock streamed answer ${index}`,
        citations: [
          {
            doc_id: readySource.doc_id,
            title: readySource.title,
            source_uri: readySource.source_uri,
            source_type: readySource.source_type,
            chunk_index: 0,
            score: 1,
            rerank_score: null,
            acl_groups: ["engineering"],
            metadata: {},
            text_preview: "Mock evidence",
          },
        ],
        trace: {},
      }),
    ].join("\n");
    await route.fulfill({ status: 200, contentType: "application/x-ndjson", body: `${body}\n` });
  });
}

async function mockStartupApi(page: Page) {
  const now = Date.now();
  await page.route("**/api/health", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ status: "ok" }) });
  });
  await page.route("**/api/auth/me**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "queued-upload-user",
        username: "queued-upload-user",
        display_name: "Queued Upload User",
        role: "user",
        tenant_id: "queued-upload-tenant",
        created_at: now,
        status: "active",
      }),
    });
  });
  await page.route("**/api/admin/settings", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ registration_enabled: true, latest_announcement: null }),
    });
  });
  await page.route("**/api/announcements**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ announcements: [] }) });
  });
  await page.route("**/api/artifacts**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ artifacts: [] }) });
  });
  await page.route("**/api/conversations**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ conversations: [] }) });
  });
}

async function chooseUploadFile(page: Page, name: string) {
  await page.getByRole("button", { name: "添加来源" }).click();
  const fileChooserPromise = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "上传文件" }).click();
  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles({
    name,
    mimeType: "text/plain",
    buffer: Buffer.from(`queued upload test file ${name}`, "utf-8"),
  });
}

function mockSource(title: string, index: number, status: "processing" | "ready") {
  const now = Date.now();
  const docId = `queued-upload-${index}`;
  return {
    doc_id: status === "ready" ? `${docId}@sha256-ready` : docId,
    title,
    source_type: "txt",
    source_uri: `mock://${title}`,
    doc_version: 1,
    chunk_count: status === "ready" ? 1 : 0,
    acl_groups: ["engineering"],
    status,
    current: status === "ready",
    created_at: now,
    updated_at: now,
    child_doc_ids: status === "ready" ? [`${docId}@sha256-ready`] : [],
  };
}

async function waitForNetworkQuiet(page: Page, quietMs: number, timeoutMs: number) {
  let inflight = 0;
  let lastChange = Date.now();
  const onRequest = () => {
    inflight += 1;
    lastChange = Date.now();
  };
  const onFinished = () => {
    inflight = Math.max(0, inflight - 1);
    lastChange = Date.now();
  };
  page.on("request", onRequest);
  page.on("requestfinished", onFinished);
  page.on("requestfailed", onFinished);
  try {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (inflight === 0 && Date.now() - lastChange >= quietMs) {
        return;
      }
      await page.waitForTimeout(50);
    }
  } finally {
    page.off("request", onRequest);
    page.off("requestfinished", onFinished);
    page.off("requestfailed", onFinished);
  }
}

async function collectPageMetrics(page: Page): Promise<PageMetrics> {
  return await page.evaluate(() => {
    const resources = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
    const memory = (
      performance as Performance & {
        memory?: {
          usedJSHeapSize?: number;
          totalJSHeapSize?: number;
          jsHeapSizeLimit?: number;
        };
      }
    ).memory;
    const transferBytes = resources.reduce((total, item) => total + (item.transferSize || 0), 0);
    const encodedBodyBytes = resources.reduce((total, item) => total + (item.encodedBodySize || 0), 0);
    return {
      dom_nodes: document.querySelectorAll("*").length,
      img_nodes: document.images.length,
      resource_count: resources.length,
      transfer_kb: Math.round(transferBytes / 1024),
      encoded_body_kb: Math.round(encodedBodyBytes / 1024),
      js_heap_used_mb:
        typeof memory?.usedJSHeapSize === "number" ? Math.round(memory.usedJSHeapSize / 1024 / 1024) : null,
      js_heap_total_mb:
        typeof memory?.totalJSHeapSize === "number" ? Math.round(memory.totalJSHeapSize / 1024 / 1024) : null,
    };
  });
}

async function closePage(page: Page) {
  try {
    if (!page.isClosed()) {
      await page.close();
    }
  } catch {
    // A load run can hit the test timeout while pages are closing. The failure
    // should come from the measured sample, not from best-effort cleanup.
  }
}

type PageSample = {
  index: number;
  ok: boolean;
  load_ms: number;
  metrics: PageMetrics | null;
  isolated_conversation_requests: number;
  baseline_within_limits: boolean;
  console_errors: string[];
  page_errors: string[];
  http_failures: string[];
  error?: string;
};

function startupMetricsWithinLimits(metrics: PageMetrics) {
  return (
    metrics.dom_nodes <= startupMaxDomNodes
    && metrics.img_nodes <= startupMaxImageNodes
    && metrics.resource_count <= startupMaxResources
    && metrics.transfer_kb <= startupMaxTransferKb
  );
}

type InteractionSample = {
  index: number;
  ok: boolean;
  total_ms: number;
  upload_ready_ms: number;
  query_ms: number;
  metrics: PageMetrics | null;
  expected_busy_responses: number;
  expected_busy_console_errors: number;
  expected_failed_sources: number;
  console_errors: string[];
  page_errors: string[];
  http_failures: string[];
  error?: string;
};

type InteractionScenario = "success" | "busy" | "ingest-failed";

type PageMetrics = {
  dom_nodes: number;
  img_nodes: number;
  resource_count: number;
  transfer_kb: number;
  encoded_body_kb: number;
  js_heap_used_mb: number | null;
  js_heap_total_mb: number | null;
};

function interactionFileName(index: number) {
  return `browser-load-${index}.txt`;
}

function summarize(values: number[]) {
  if (!values.length) {
    return { avg: 0, p50: 0, p95: 0, min: 0, max: 0 };
  }
  const data = [...values].sort((a, b) => a - b);
  return {
    avg: round(data.reduce((total, value) => total + value, 0) / data.length, 2),
    p50: percentile(data, 50),
    p95: percentile(data, 95),
    min: round(data[0], 2),
    max: round(data[data.length - 1], 2),
  };
}

function summarizeNullable(values: Array<number | null | undefined>) {
  return summarize(values.filter((value): value is number => typeof value === "number" && Number.isFinite(value)));
}

function summarizePageMetrics(metrics: Array<PageMetrics | null>) {
  return {
    dom_nodes: summarize(metrics.map((item) => item?.dom_nodes ?? 0)),
    img_nodes: summarize(metrics.map((item) => item?.img_nodes ?? 0)),
    resource_count: summarize(metrics.map((item) => item?.resource_count ?? 0)),
    transfer_kb: summarize(metrics.map((item) => item?.transfer_kb ?? 0)),
    encoded_body_kb: summarize(metrics.map((item) => item?.encoded_body_kb ?? 0)),
    js_heap_used_mb: summarizeNullable(metrics.map((item) => item?.js_heap_used_mb)),
    js_heap_total_mb: summarizeNullable(metrics.map((item) => item?.js_heap_total_mb)),
  };
}

function percentile(values: number[], pct: number) {
  const index = Math.min(values.length - 1, Math.round((pct / 100) * (values.length - 1)));
  return round(values[index], 2);
}

function findOverlappingNodePairs(
  rectangles: Array<{ text: string; top: number; bottom: number; left: number; right: number }>,
) {
  const overlaps: string[] = [];
  for (let leftIndex = 0; leftIndex < rectangles.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < rectangles.length; rightIndex += 1) {
      const left = rectangles[leftIndex];
      const right = rectangles[rightIndex];
      const horizontalOverlap = left.left < right.right - 0.5 && right.left < left.right - 0.5;
      const verticalOverlap = left.top < right.bottom - 0.5 && right.top < left.bottom - 0.5;
      if (horizontalOverlap && verticalOverlap) {
        overlaps.push(`${left.text} <> ${right.text}`);
      }
    }
  }
  return overlaps;
}

function envInt(name: string, fallback: number) {
  const value = Number.parseInt(process.env[name] || "", 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function roundMs(value: number) {
  return round(value, 2);
}

function round(value: number, digits: number) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
