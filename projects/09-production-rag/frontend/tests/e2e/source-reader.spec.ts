import { expect, test } from "@playwright/test";

test("opens parsed source content from a document-level source row", async ({ page }) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "自然辩证法",
            title: "自然辩证法.pdf",
            source_type: "pdf",
            source_uri: "/object_store/uploads/team_a/regression/自然辩证法.pdf",
            doc_version: 1,
            chunk_count: 6,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: [
              "自然辩证法/page-1",
              "自然辩证法/page-2",
              "自然辩证法/page-3",
              "自然辩证法/page-4",
              "自然辩证法/page-5",
              "自然辩证法/page-6",
            ],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });
  await page.route("**/sources/content/**", async (route) => {
    await route.fulfill({
      json: {
        doc_id: "自然辩证法",
        title: "自然辩证法.pdf",
        source_type: "pdf",
        source_uri: "/object_store/uploads/team_a/regression/自然辩证法.pdf",
        doc_version: 1,
        child_doc_ids: ["自然辩证法/page-1", "自然辩证法/page-2"],
        guide: "这份资料介绍自然辩证法视角下的生态治理实践，并包含引言部分。",
        tags: ["一、引言", "生态治理"],
        text: "第 1 页\n\n一、引言\n\n在 21 世纪全球生态危机日益严峻的背景下，中国生态文明建设受到广泛关注。",
      },
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: /自然辩证法\.pdf/ }).click();

  const reader = page.getByRole("dialog", { name: "自然辩证法.pdf 原始内容" });
  await expect(reader).toBeVisible();
  await expect(reader.getByRole("heading", { name: "自然辩证法.pdf" })).toBeVisible();
  await expect(reader.getByRole("heading", { name: "来源指南" })).toBeVisible();
  await expect(reader.getByText("这份资料介绍自然辩证法视角下的生态治理实践")).toBeVisible();
  await expect(reader.getByText("一、引言").first()).toBeVisible();
  await expect(reader.getByText("在 21 世纪全球生态危机日益严峻的背景下")).toBeVisible();
});

test("expands second-level mind map topics to reveal child outline items", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 800 });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({
      json: {
        artifacts: [
          {
            id: "mindmap-regression",
            title: "实习招聘思维导图",
            status: "ready",
            tenant_id: "team_a",
            source_doc_ids: ["internship-guide/page-1"],
            created_at: Date.now(),
            updated_at: Date.now(),
            root: {
              id: "root",
              label: "创维集团AI研究院实习介绍",
              children: [
                {
                  id: "overview",
                  label: "研究院概况",
                  children: [
                    { id: "positioning", label: "定位：集团技术中枢与AI中台", children: [] },
                    { id: "mission", label: "使命：打造通用AI能力基座", children: [] },
                    { id: "mode", label: "工作模式：自由探索、深度攻坚、平台输出", children: [] },
                  ],
                },
                {
                  id: "research",
                  label: "核心研究方向",
                  children: [{ id: "llm", label: "大语言模型工程化", children: [] }],
                },
              ],
            },
          },
        ],
      },
    });
  });

  await page.goto("/");
  const studioBefore = await page.locator(".studio-panel").boundingBox();
  const chatBefore = await page.locator(".chat-panel").boundingBox();
  await page.getByRole("button", { name: /实习招聘思维导图/ }).click();
  await expect
    .poll(async () => (await page.locator(".studio-panel").boundingBox())?.width ?? 0)
    .toBeGreaterThan((studioBefore?.width ?? 0) + 40);
  const studioAfter = await page.locator(".studio-panel").boundingBox();
  const chatAfter = await page.locator(".chat-panel").boundingBox();

  await expect(page.getByText("创维集团AI研究院实习介绍")).toBeVisible();
  await expect(page.getByText("研究院概况")).toBeVisible();
  await expect(page.getByText("定位：集团技术中枢与AI中台")).toBeHidden();
  await expect(page.locator(".react-flow__controls")).toBeVisible();

  await page.locator(".mindmap-flow-node", { hasText: "研究院概况" }).click();

  await expect(page.getByText("定位：集团技术中枢与AI中台")).toBeVisible();
  await expect(page.getByText("使命：打造通用AI能力基座")).toBeVisible();
  await expect(page.getByText("工作模式：自由探索、深度攻坚、平台输出")).toBeVisible();
  expect(studioBefore?.width).toBeTruthy();
  expect(studioAfter?.width).toBeTruthy();
  expect(chatBefore?.y).toBeTruthy();
  expect(chatAfter?.y).toBeTruthy();
  expect(Math.abs(studioAfter!.y - chatAfter!.y)).toBeLessThan(12);
  expect(studioAfter!.x).toBeGreaterThan(chatAfter!.x + 80);
});

test("resizes source and chat panels by dragging the divider", async ({ page }) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: [] } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  const sourceBefore = await page.locator(".source-panel").boundingBox();
  const divider = page.getByRole("separator", { name: "调整来源和对话宽度" });
  const box = await divider.boundingBox();
  expect(sourceBefore).toBeTruthy();
  expect(box).toBeTruthy();

  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await page.mouse.down();
  await page.mouse.move(box!.x + 96, box!.y + box!.height / 2, { steps: 8 });
  await page.mouse.up();

  const sourceAfter = await page.locator(".source-panel").boundingBox();
  expect(sourceAfter).toBeTruthy();
  expect(sourceAfter!.width).toBeGreaterThan(sourceBefore!.width + 40);
});

test("renders assistant answers with a typewriter reveal", async ({ page }) => {
  const finalMarker = "TYPEWRITER_FINAL_MARKER";
  const answer = [
    "这是一段用于验证打字机效果的回答。",
    "系统会先展示开头内容，再逐步展开后续分析。",
    "中间部分包含较长的解释，用来保证渲染不会在一个帧内全部完成。",
    "自然辩证法的实践案例可以围绕塞罕坝、长江禁渔和流域治理展开。",
    "最终标记：",
    finalMarker,
  ].join("\n\n");

  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "自然辩证法",
            title: "自然辩证法.pdf",
            source_type: "pdf",
            source_uri: "/object_store/uploads/team_a/regression/自然辩证法.pdf",
            doc_version: 1,
            chunk_count: 6,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: ["自然辩证法/page-1"],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: { conversations: [] } });
      return;
    }
    const body = route.request().postDataJSON();
    await route.fulfill({ json: { ...body, id: "conversation-typewriter", updated_at: Date.now() } });
  });
  await page.route("**/query", async (route) => {
    await route.fulfill({
      json: {
        request_id: "typewriter-check",
        answer,
        citations: [
          {
            doc_id: "自然辩证法/page-1",
            title: "自然辩证法 p1",
            source_uri: "/object_store/uploads/team_a/regression/自然辩证法.pdf",
            source_type: "pdf",
            chunk_index: 0,
            score: 0.9,
            rerank_score: 0.8,
            acl_groups: ["engineering"],
            metadata: {},
            text_preview: "实践案例证据片段",
          },
        ],
        trace: {},
      },
    });
  });

  await page.goto("/");
  await page.getByPlaceholder("提问或创作内容").fill("典型的实践案例分析");
  await page.locator(".chat-input button").click();

  await expect(page.getByText("这是一段用于验证打字机效果")).toBeVisible();
  await expect(page.getByText(finalMarker)).toBeHidden();
  await expect(page.getByText(finalMarker)).toBeVisible({ timeout: 5000 });
  await expect(page.getByText("1. 自然辩证法 p1 · chunk 0")).toBeVisible();

  const chatBox = await page.locator(".chat-panel").boundingBox();
  const userBox = await page.locator(".user-message").boundingBox();
  const assistantBox = await page.locator(".assistant-message").boundingBox();
  const inputBox = await page.locator(".chat-input").boundingBox();
  const textareaBox = await page.locator(".chat-input textarea").boundingBox();
  const sendBox = await page.locator(".chat-input button").boundingBox();
  expect(chatBox).toBeTruthy();
  expect(userBox).toBeTruthy();
  expect(assistantBox).toBeTruthy();
  expect(inputBox).toBeTruthy();
  expect(textareaBox).toBeTruthy();
  expect(sendBox).toBeTruthy();
  expect(Math.abs(userBox!.x + userBox!.width - (chatBox!.x + chatBox!.width - 32))).toBeLessThan(90);
  expect(assistantBox!.x - chatBox!.x).toBeLessThan(40);
  expect(inputBox!.height).toBeLessThan(90);
  expect(sendBox!.x + sendBox!.width).toBeLessThanOrEqual(inputBox!.x + inputBox!.width + 1);
  expect(textareaBox!.x).toBeGreaterThanOrEqual(inputBox!.x);
});
