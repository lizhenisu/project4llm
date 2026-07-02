import { expect, test } from "@playwright/test";
import type { Page, Route } from "@playwright/test";

const ONE_PIXEL_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

async function fulfillQueryStream(route: Route, result: Record<string, unknown>) {
  await route.fulfill({
    contentType: "application/x-ndjson",
    body: `${JSON.stringify({ type: "result", ...result })}\n`,
  });
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "production-rag-auth-session",
      JSON.stringify({
        user: {
          id: "test-user",
          username: "tester",
          display_name: "测试用户",
          role: "user",
          tenant_id: "team_a",
          avatar_url: "",
          status: "active",
          created_at: Date.now(),
          last_login_at: Date.now(),
        },
        token: "test-session",
        expires_at: Date.now() + 86_400_000,
      }),
    );
  });
  await page.route("**/announcements?**", async (route) => {
    await route.fulfill({ json: { announcements: [] } });
  });
});

async function mockWorkspaceShell(page: Page) {
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: [] } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });
}

async function mockSourceAssetRoute(page: Page) {
  await page.route("**/api/source-assets/**", async (route) => {
    const request = route.request();
    expect(new URL(request.url()).searchParams.has("token")).toBe(false);
    expect(request.headers().authorization).toBe("Bearer test-session");
    await route.fulfill({
      contentType: "image/png",
      body: Buffer.from(ONE_PIXEL_PNG_BASE64, "base64"),
    });
  });
}

test("sends a normal chat request when no documents are selected", async ({ page }) => {
  await mockWorkspaceShell(page);
  let queryPayload: Record<string, unknown> | null = null;

  await page.route("**/conversations**", async (route) => {
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON();
      await route.fulfill({
        json: {
          ...body,
          id: body.id || "conversation-without-sources",
          created_at: Date.now(),
          updated_at: Date.now(),
        },
      });
      return;
    }
    await route.fulfill({ json: { conversations: [] } });
  });
  await page.route("**/query/stream", async (route) => {
    queryPayload = route.request().postDataJSON();
    await route.fulfill({
      contentType: "application/x-ndjson",
      body: `${JSON.stringify({
        type: "result",
        request_id: "query-without-sources",
        answer: "这是不依赖知识库的普通对话回答。",
        citations: [],
        trace: {},
      })}\n`,
    });
  });

  await page.goto("/");
  await expect(page.getByText("无需选择文档也可以提问")).toBeVisible();
  await expect(page.getByText("0 个来源")).toBeVisible();
  const input = page.getByPlaceholder("提问或创作内容");
  const sendButton = page.getByRole("button", { name: "发送消息" });
  await expect(sendButton).toBeDisabled();

  await input.fill("没有文档时也能聊天吗？");
  await expect(sendButton).toBeEnabled();
  await input.press("Enter");

  await expect(page.getByText("这是不依赖知识库的普通对话回答。")).toBeVisible();
  expect(queryPayload).toMatchObject({
    query: "没有文档时也能聊天吗？",
    doc_ids: [],
    query_mode: "text",
  });
});

test("collapses and restores the top bar and status bar", async ({ page }) => {
  await mockWorkspaceShell(page);
  await page.goto("/");

  const shell = page.locator(".workspace-shell");
  const topbar = page.locator(".topbar");
  const statusbar = page.locator(".statusbar");
  const workspace = page.locator(".workspace-grid");
  const chatHeader = page.locator(".chat-panel > .panel-header");
  const collapseButton = chatHeader.getByRole("button", { name: "折叠顶部栏和状态栏" });
  const expandedWorkspaceHeight = await workspace.evaluate((element) => element.getBoundingClientRect().height);

  await expect(topbar).toBeVisible();
  await expect(statusbar).toBeVisible();
  await expect(topbar.getByRole("button", { name: "折叠顶部栏和状态栏" })).toHaveCount(0);
  await expect(collapseButton.locator("svg")).toHaveClass(/lucide-maximize/);
  await expect(collapseButton).toHaveAttribute("aria-expanded", "true");
  await collapseButton.click();

  const expandButton = chatHeader.getByRole("button", { name: "展开顶部栏和状态栏" });
  await expect(shell).toHaveClass(/chrome-collapsed/);
  await expect(expandButton).toBeVisible();
  await expect(expandButton).toHaveAttribute("aria-expanded", "false");
  await expect.poll(() => topbar.evaluate((element) => element.getBoundingClientRect().height)).toBe(0);
  await expect.poll(() => statusbar.evaluate((element) => element.getBoundingClientRect().height)).toBe(0);
  await expect.poll(() => workspace.evaluate((element) => element.getBoundingClientRect().height)).toBeGreaterThan(
    expandedWorkspaceHeight + 70,
  );

  await expandButton.click();
  await expect(shell).not.toHaveClass(/chrome-collapsed/);
  await expect(page.getByRole("button", { name: "折叠顶部栏和状态栏" })).toBeVisible();
  await expect.poll(() => topbar.evaluate((element) => element.getBoundingClientRect().height)).toBeGreaterThan(50);
  await expect.poll(() => statusbar.evaluate((element) => element.getBoundingClientRect().height)).toBeGreaterThan(20);
});

test("manages historical conversations from the sliding chat drawer", async ({ page }) => {
  await mockWorkspaceShell(page);
  const now = Date.now();
  let conversations = [
    {
      id: "conv-new",
      tenant_id: "team_a",
      workspace_id: "default-workspace",
      title: "最新对话",
      message_count: 2,
      source_doc_ids: [],
      created_at: now - 2_000,
      updated_at: now,
    },
    {
      id: "conv-old",
      tenant_id: "team_a",
      workspace_id: "default-workspace",
      title: "旧对话",
      message_count: 2,
      source_doc_ids: [],
      created_at: now - 10_000,
      updated_at: now - 5_000,
    },
    {
      id: "conv-other-workspace",
      tenant_id: "team_a",
      workspace_id: "workspace-other",
      title: "其他知识库对话",
      message_count: 2,
      source_doc_ids: [],
      created_at: now - 20_000,
      updated_at: now + 10_000,
    },
  ];
  let renamedTitle = "";
  let deletedConversationId = "";

  await page.route("**/conversations**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const conversationId = url.pathname.match(/\/conversations\/([^/]+)$/)?.[1];
    if (!conversationId) {
      expect(url.searchParams.get("workspace_id")).toBe("default-workspace");
      await route.fulfill({
        json: {
          conversations: conversations.filter(
            (conversation) => conversation.workspace_id === url.searchParams.get("workspace_id"),
          ),
        },
      });
      return;
    }
    expect(url.searchParams.get("workspace_id")).toBe("default-workspace");
    if (request.method() === "PATCH") {
      renamedTitle = request.postDataJSON().title;
      conversations = conversations.map((item) =>
        item.id === conversationId ? { ...item, title: renamedTitle, updated_at: now + 1_000 } : item,
      );
      await route.fulfill({
        json: {
          status: "renamed",
          conversation_id: conversationId,
          title: renamedTitle,
          updated_at: now + 1_000,
        },
      });
      return;
    }
    if (request.method() === "DELETE") {
      deletedConversationId = conversationId;
      conversations = conversations.filter((item) => item.id !== conversationId);
      await route.fulfill({ json: { status: "deleted", conversation_id: conversationId } });
      return;
    }
    const item = conversations.find((conversation) => conversation.id === conversationId);
    await route.fulfill({
      json: {
        ...item,
        messages: [
          { id: `${conversationId}-user`, role: "user", content: `${item?.title}问题`, status: "done" },
          { id: `${conversationId}-assistant`, role: "assistant", content: `${item?.title}回答`, status: "done" },
        ],
      },
    });
  });

  await page.goto("/");
  await expect(page.getByText("最新对话回答")).toBeVisible();
  const chatHeader = page.locator(".chat-panel > .panel-header");
  await expect(chatHeader.getByRole("button", { name: "更多" })).toHaveCount(0);
  await chatHeader.getByRole("button", { name: "打开历史对话" }).click();

  const drawer = page.getByRole("complementary", { name: "历史对话" });
  await expect(drawer).toBeVisible();
  await expect(drawer).toHaveCSS("transform", "matrix(1, 0, 0, 1, 0, 0)");
  await expect(drawer.getByText("2 条记录")).toBeVisible();
  await expect(drawer.getByText("其他知识库对话")).toHaveCount(0);
  const newConversationButton = drawer.getByRole("button", { name: "开启新对话" });
  await expect(newConversationButton).toBeVisible();
  expect(
    await newConversationButton.evaluate((element) =>
      element.parentElement?.nextElementSibling?.classList.contains("conversation-history-list"),
    ),
  ).toBe(true);
  await newConversationButton.click();
  await expect(drawer).toBeHidden();
  await expect(page.getByText("最新对话回答")).toHaveCount(0);
  await expect(page.getByText("直接开始对话")).toBeVisible();

  await chatHeader.getByRole("button", { name: "打开历史对话" }).click();
  await drawer.getByRole("button", { name: /旧对话.*2 条消息/ }).click();
  await expect(drawer).toBeHidden();
  await expect(page.getByText("旧对话回答")).toBeVisible();

  await chatHeader.getByRole("button", { name: "打开历史对话" }).click();
  await drawer.getByRole("button", { name: "管理对话：旧对话" }).click();
  const floatingMenu = page.getByRole("menu");
  await expect(floatingMenu).toBeVisible();
  expect(await floatingMenu.evaluate((element) => element.closest("aside"))).toBeNull();
  const menuBox = await floatingMenu.boundingBox();
  const viewport = page.viewportSize();
  expect(menuBox).not.toBeNull();
  expect(viewport).not.toBeNull();
  expect(menuBox!.x).toBeGreaterThanOrEqual(0);
  expect(menuBox!.y).toBeGreaterThanOrEqual(0);
  expect(menuBox!.x + menuBox!.width).toBeLessThanOrEqual(viewport!.width);
  expect(menuBox!.y + menuBox!.height).toBeLessThanOrEqual(viewport!.height);
  await floatingMenu.getByRole("menuitem", { name: "重命名" }).click();
  const renameInput = drawer.getByRole("textbox", { name: "重命名旧对话" });
  await renameInput.fill("项目复盘");
  await expect(drawer.getByRole("button", { name: "保存" })).toHaveCount(0);
  await renameInput.press("Enter");
  await expect(drawer.getByText("项目复盘", { exact: true })).toBeVisible();
  expect(renamedTitle).toBe("项目复盘");

  await drawer.getByRole("button", { name: "管理对话：项目复盘" }).click();
  await page.getByRole("menu").getByRole("menuitem", { name: "重命名" }).click();
  const blurRenameInput = drawer.getByRole("textbox", { name: "重命名项目复盘" });
  await blurRenameInput.fill("失焦复盘");
  await drawer.getByText("历史对话", { exact: true }).click();
  await expect(drawer.getByText("失焦复盘", { exact: true })).toBeVisible();
  expect(renamedTitle).toBe("失焦复盘");

  await drawer.getByRole("button", { name: "管理对话：失焦复盘" }).click();
  await page.getByRole("menu").getByRole("menuitem", { name: "删除" }).click();
  await expect(drawer.getByText("失焦复盘", { exact: true })).toHaveCount(0);
  expect(deletedConversationId).toBe("conv-old");
  await expect(page.getByText("直接开始对话")).toBeVisible();

  await drawer.getByRole("button", { name: "关闭历史对话" }).click();
  await expect(drawer).toBeHidden();
});

test("hides database create/rename controls when not authenticated", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.removeItem("production-rag-auth-session");
  });
  await mockWorkspaceShell(page);

  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "知识库设置" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText("登录后可创建、切换与管理数据库。")).toBeVisible();
  await expect(dialog.getByRole("button", { name: "新建数据库" })).toBeHidden();
  await expect(dialog.getByText("当前数据库名称")).toBeHidden();
  await expect(dialog.getByRole("button", { name: "重命名数据库" })).toBeHidden();
});

test("manages database list in settings without exposing API fields", async ({ page }) => {
  await mockWorkspaceShell(page);

  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "知识库设置" }).or(page.locator(".settings-panel"));
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "知识库设置" })).toBeVisible();
  await expect(dialog.getByText("Production RAG 知识库")).toBeVisible();
  await expect(dialog.getByText("API Base URL")).toBeHidden();
  await expect(dialog.getByText("Token")).toBeHidden();
  await expect(dialog.getByText("Tenant")).toBeHidden();
  await expect(dialog.getByText("ACL Groups")).toBeHidden();

  await page.getByRole("button", { name: "新建数据库" }).click();
  await dialog.locator(".database-list-item.active .icon-button").click();
  await expect(page.getByRole("menuitem", { name: /重命名知识\s*库/ })).toBeVisible();
  await dialog.locator(".settings-section-heading strong", { hasText: "知识库工作区" }).click();
  await expect(page.getByRole("menuitem", { name: /重命名知识\s*库/ })).toBeHidden();
  await dialog.locator(".database-list-item.active .icon-button").click();
  await page.getByRole("menuitem", { name: /重命名知识\s*库/ }).click();
  await dialog.locator(".database-list-item.active .inline-title-input").fill("法规资料库");
  await dialog.locator(".database-list-item.active .inline-title-input").press("Enter");
  await expect(dialog.getByText("法规资料库")).toBeVisible();
  await expect(dialog.getByText("Production RAG 知识库")).toBeVisible();
  await expect(dialog.getByText("当前数据库名称")).toBeHidden();
  await expect(dialog.getByRole("button", { name: "重命名数据库" })).toBeHidden();

  await page.getByRole("button", { name: /Production RAG 知识库/ }).click();
  await expect(dialog.locator(".database-list-item.active")).toContainText("Production RAG 知识库");
});

test("allows deleting the only database and creates a fresh default database", async ({ page }) => {
  let sourceRows = [
    {
      doc_id: "old-source",
      title: "已看12345.txt",
      source_type: "txt",
      source_uri: "/uploads/old-source.txt",
      doc_version: 1,
      chunk_count: 1,
      acl_groups: ["engineering"],
      status: "ready",
      current: true,
      child_doc_ids: [],
    },
  ];
  let artifactRows = [
    {
      id: "old-artifact",
      title: "历史思维导图",
      status: "ready",
      tenant_id: "team_a",
      source_doc_ids: ["old-source"],
      created_at: Date.now(),
      updated_at: Date.now(),
      root: { id: "root", label: "历史思维导图", children: [] },
    },
  ];
  let conversationRows = [
    {
      id: "old-conversation",
      tenant_id: "team_a",
      title: "历史对话",
      message_count: 2,
      source_doc_ids: ["old-source"],
      created_at: Date.now(),
      updated_at: Date.now(),
    },
  ];
  let sourceDeletes = 0;
  let artifactDeletes = 0;
  let conversationDeletes = 0;

  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: sourceRows,
      },
    });
  });
  await page.route("**/sources/old-source?**", async (route) => {
    if (route.request().method() === "DELETE") {
      sourceDeletes += 1;
      sourceRows = [];
      await route.fulfill({ json: { status: "deleted" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({
      json: {
        artifacts: artifactRows,
      },
    });
  });
  await page.route("**/artifacts/old-artifact?**", async (route) => {
    if (route.request().method() === "DELETE") {
      artifactDeletes += 1;
      artifactRows = [];
      await route.fulfill({ json: { status: "deleted", artifact_id: "old-artifact" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({
      json: {
        conversations: conversationRows,
      },
    });
  });
  await page.route("**/conversations/old-conversation?**", async (route) => {
    if (route.request().method() === "DELETE") {
      conversationDeletes += 1;
      conversationRows = [];
      await route.fulfill({ json: { status: "deleted", conversation_id: "old-conversation" } });
      return;
    }
    await route.fulfill({
      json: {
        id: "old-conversation",
        tenant_id: "team_a",
        title: "历史对话",
        source_doc_ids: ["old-source"],
        created_at: Date.now(),
        updated_at: Date.now(),
        messages: [
          { id: "msg-1", role: "user", content: "历史问题", status: "done", citations: [] },
          { id: "msg-2", role: "assistant", content: "历史回答", status: "done", citations: [] },
        ],
      },
    });
  });
  await page.route("**/announcements?**", async (route) => {
    await route.fulfill({ json: { announcements: [] } });
  });

  await page.goto("/");
  await expect(page.locator(".source-row", { hasText: "已看12345.txt" })).toBeVisible();
  await expect(page.getByText("历史思维导图")).toBeVisible();
  await expect(page.getByText("历史回答")).toBeVisible();
  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "知识库设置" });
  await expect(dialog.locator(".database-list-item")).toHaveCount(1);
  await dialog.locator(".database-list-item.active .icon-button").click();
  await page.getByRole("menuitem", { name: "删除知识库" }).click();

  await expect(dialog.locator(".database-list-item")).toHaveCount(1);
  await expect(dialog.locator(".database-list-item.active")).toContainText("未命名知识库");
  await expect(dialog.locator(".database-list-item.active")).toContainText("当前数据库");
  await expect.poll(() => sourceDeletes).toBe(1);
  await expect.poll(() => artifactDeletes).toBe(1);
  await expect.poll(() => conversationDeletes).toBe(1);

  await page.reload();
  await expect(page.locator(".source-row", { hasText: "已看12345.txt" })).toHaveCount(0);
  await expect(page.getByText("历史思维导图")).toHaveCount(0);
  await expect(page.getByText("历史回答")).toHaveCount(0);
});

test("refreshes all workspace panels after deleting the active database", async ({ page }) => {
  const now = Date.now();
  let sourceRows = [
    {
      doc_id: "source-a",
      title: "A 来源.txt",
      source_type: "txt",
      source_uri: "/uploads/source-a.txt",
      doc_version: 1,
      chunk_count: 1,
      acl_groups: ["engineering"],
      status: "ready",
      current: true,
      child_doc_ids: [],
    },
    {
      doc_id: "source-b",
      title: "B 来源.txt",
      source_type: "txt",
      source_uri: "/uploads/source-b.txt",
      doc_version: 1,
      chunk_count: 1,
      acl_groups: ["engineering"],
      status: "ready",
      current: true,
      child_doc_ids: [],
    },
  ];
  let artifactRows = [
    {
      id: "artifact-a",
      title: "A 思维导图",
      status: "ready",
      tenant_id: "team_a",
      source_doc_ids: ["source-a"],
      created_at: now,
      updated_at: now,
      root: { id: "root-a", label: "A 思维导图", children: [] },
    },
    {
      id: "artifact-b",
      title: "B 思维导图",
      status: "ready",
      tenant_id: "team_a",
      source_doc_ids: ["source-b"],
      created_at: now,
      updated_at: now,
      root: { id: "root-b", label: "B 思维导图", children: [] },
    },
  ];
  let conversationRows = [
    {
      id: "conv-a",
      tenant_id: "team_a",
      title: "A 对话",
      message_count: 2,
      source_doc_ids: ["source-a"],
      created_at: now,
      updated_at: now,
    },
    {
      id: "conv-b",
      tenant_id: "team_a",
      title: "B 对话",
      message_count: 2,
      source_doc_ids: ["source-b"],
      created_at: now + 1,
      updated_at: now + 1,
    },
  ];
  await page.addInitScript(() => {
    const createdAt = Date.now();
    localStorage.setItem(
      "production-rag-workspaces:test-user",
      JSON.stringify([
        { id: "workspace-a", name: "知识库 A", user_id: "test-user", created_at: createdAt, updated_at: createdAt },
        { id: "workspace-b", name: "知识库 B", user_id: "test-user", created_at: createdAt, updated_at: createdAt },
      ]),
    );
    localStorage.setItem("production-rag-active-workspace-id:test-user", "workspace-a");
    localStorage.setItem("production-rag-workspace-sources:workspace-a", JSON.stringify(["source-a"]));
    localStorage.setItem("production-rag-workspace-sources:workspace-b", JSON.stringify(["source-b"]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-a", JSON.stringify(["conv-a"]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-b", JSON.stringify(["conv-b"]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-a", JSON.stringify(["artifact-a"]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-b", JSON.stringify(["artifact-b"]));
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/auth/me", async (route) => {
    await route.fulfill({
      json: {
        id: "test-user",
        username: "tester",
        display_name: "测试用户",
        role: "user",
        tenant_id: "team_a",
        avatar_url: "",
        status: "active",
        created_at: now,
        last_login_at: now,
        profile_name_edit_allowed: true,
        avatar_edit_allowed: true,
      },
    });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: sourceRows } });
  });
  await page.route("**/sources/source-a?**", async (route) => {
    if (route.request().method() === "DELETE") {
      sourceRows = sourceRows.filter((source) => source.doc_id !== "source-a");
      await route.fulfill({ json: { status: "deleted" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: artifactRows } });
  });
  await page.route("**/artifacts/artifact-a?**", async (route) => {
    if (route.request().method() === "DELETE") {
      artifactRows = artifactRows.filter((artifact) => artifact.id !== "artifact-a");
      await route.fulfill({ json: { status: "deleted", artifact_id: "artifact-a" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: conversationRows } });
  });
  await page.route("**/conversations/conv-a?**", async (route) => {
    if (route.request().method() === "DELETE") {
      conversationRows = conversationRows.filter((conversation) => conversation.id !== "conv-a");
      await route.fulfill({ json: { status: "deleted", conversation_id: "conv-a" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/conversations/conv-b?**", async (route) => {
    await route.fulfill({
      json: {
        id: "conv-b",
        tenant_id: "team_a",
        title: "B 对话",
        source_doc_ids: ["source-b"],
        created_at: now + 1,
        updated_at: now + 1,
        messages: [
          { id: "msg-b-1", role: "user", content: "B 问题", status: "done", citations: [] },
          { id: "msg-b-2", role: "assistant", content: "B 回答", status: "done", citations: [] },
        ],
      },
    });
  });

  await page.goto("/#token=production-rag-fixed-test-login-token");
  await expect(page.locator(".source-row", { hasText: "A 来源.txt" })).toBeVisible();
  await expect(page.getByText("A 思维导图")).toBeVisible();
  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "知识库设置" });
  await dialog.locator(".database-list-item.active .icon-button").click();
  await page.getByRole("menuitem", { name: "删除知识库" }).click();

  await expect(dialog.locator(".database-list-item.active")).toContainText("知识库 B");
  await expect(page.locator(".source-row", { hasText: "B 来源.txt" })).toBeVisible();
  await expect(page.locator(".source-row", { hasText: "A 来源.txt" })).toHaveCount(0);
  await expect(page.getByText("B 回答")).toBeVisible();
  await expect(page.getByText("A 思维导图")).toHaveCount(0);
  await expect(page.getByText("B 思维导图")).toBeVisible();
  await expect(page.locator(".statusbar span").first()).toContainText("API 已连接");
});

test("removing a shared source only unlinks it from the current database", async ({ page }) => {
  let sourceDeletes = 0;
  await page.addInitScript(() => {
    const now = Date.now();
    localStorage.setItem(
      "production-rag-workspaces:test-user",
      JSON.stringify([
        { id: "workspace-a", name: "资料库 A", user_id: "test-user", created_at: now, updated_at: now },
        { id: "workspace-b", name: "资料库 B", user_id: "test-user", created_at: now, updated_at: now },
      ]),
    );
    localStorage.setItem("production-rag-active-workspace-id:test-user", "workspace-a");
    localStorage.setItem("production-rag-workspace-sources:workspace-a", JSON.stringify(["shared-source"]));
    localStorage.setItem("production-rag-workspace-sources:workspace-b", JSON.stringify(["shared-source"]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-b", JSON.stringify([]));
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "shared-source",
            title: "12345.md",
            source_type: "md",
            source_uri: "/uploads/12345.md",
            doc_version: 1,
            chunk_count: 1,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: [],
          },
        ],
      },
    });
  });
  await page.route("**/sources/shared-source?**", async (route) => {
    if (route.request().method() === "DELETE") {
      sourceDeletes += 1;
      await route.fulfill({ json: { status: "deleted" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await expect(page.locator(".source-row", { hasText: "12345.md" })).toBeVisible();
  await page.locator(".source-row", { hasText: "12345.md" }).locator(".row-icon-more").click();
  await page.getByRole("button", { name: "移除" }).click();
  await page.getByRole("button", { name: "确认移除" }).click();
  await expect(page.locator(".source-row", { hasText: "12345.md" })).toHaveCount(0);
  expect(sourceDeletes).toBe(0);

  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "知识库设置" });
  await page.getByRole("button", { name: /资料库 B/ }).click();
  await expect(dialog.locator(".database-list-item.active")).toContainText("资料库 B");
  await expect(page.locator(".source-row", { hasText: "12345.md" })).toBeVisible();
});

test("renaming a shared source only changes the current database title", async ({ page }) => {
  let sourceRenames = 0;
  await page.addInitScript(() => {
    const now = Date.now();
    localStorage.setItem(
      "production-rag-workspaces:test-user",
      JSON.stringify([
        { id: "workspace-a", name: "资料库 A", user_id: "test-user", created_at: now, updated_at: now },
        { id: "workspace-b", name: "资料库 B", user_id: "test-user", created_at: now, updated_at: now },
      ]),
    );
    localStorage.setItem("production-rag-active-workspace-id:test-user", "workspace-a");
    localStorage.setItem("production-rag-workspace-sources:workspace-a", JSON.stringify(["shared-source"]));
    localStorage.setItem("production-rag-workspace-sources:workspace-b", JSON.stringify(["shared-source"]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-b", JSON.stringify([]));
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "shared-source",
            title: "12345.md",
            source_type: "md",
            source_uri: "/uploads/12345.md",
            doc_version: 1,
            chunk_count: 1,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: [],
          },
        ],
      },
    });
  });
  await page.route("**/sources/shared-source?**", async (route) => {
    if (route.request().method() === "PATCH") {
      sourceRenames += 1;
      await route.fulfill({ json: { status: "renamed", doc_id: "shared-source", title: "资料库 A 标题.md" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await page.locator(".source-row", { hasText: "12345.md" }).locator(".row-icon-more").click();
  await page.getByRole("button", { name: "重命名" }).click();
  await page.locator(".source-row .inline-title-input").fill("资料库 A 标题.md");
  await page.locator(".source-row .inline-title-input").press("Enter");
  await expect(page.locator(".source-row", { hasText: "资料库 A 标题.md" })).toBeVisible();
  expect(sourceRenames).toBe(0);

  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "知识库设置" });
  await page.getByRole("button", { name: /资料库 B/ }).click();
  await expect(dialog.locator(".database-list-item.active")).toContainText("资料库 B");
  await expect(page.locator(".source-row", { hasText: "12345.md" })).toBeVisible();
  await expect(page.locator(".source-row", { hasText: "资料库 A 标题.md" })).toHaveCount(0);
});

test("keeps an in-flight answer scoped to the database where it started", async ({ page }) => {
  let queryResolve: (() => void) | null = null;
  const queryReleased = new Promise<void>((resolve) => {
    queryResolve = resolve;
  });
  let storedConversation: any = null;
  await page.addInitScript(() => {
    const now = Date.now();
    localStorage.setItem(
      "production-rag-workspaces:test-user",
      JSON.stringify([
        { id: "workspace-a", name: "Production RAG 知识库 06/15 11:36", user_id: "test-user", created_at: now, updated_at: now },
        { id: "workspace-b", name: "Production RAG 知识库 06/14 23:54", user_id: "test-user", created_at: now, updated_at: now },
      ]),
    );
    localStorage.setItem("production-rag-active-workspace-id:test-user", "workspace-a");
    localStorage.setItem("production-rag-workspace-sources:workspace-a", JSON.stringify(["natural"]));
    localStorage.setItem("production-rag-workspace-sources:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-b", JSON.stringify([]));
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "natural",
            title: "自然自然辩证法.pdf",
            source_type: "pdf",
            source_uri: "/uploads/natural.pdf",
            doc_version: 1,
            chunk_count: 6,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: ["natural/page-1"],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    if (route.request().method() === "GET") {
      if (route.request().url().includes("/conversations/conv-workspace-race")) {
        await route.fulfill({ json: storedConversation });
        return;
      }
      await route.fulfill({
        json: {
          conversations: storedConversation
            ? [{ id: storedConversation.id, tenant_id: "team_a", title: storedConversation.title, message_count: storedConversation.messages.length, source_doc_ids: storedConversation.source_doc_ids, created_at: storedConversation.created_at, updated_at: storedConversation.updated_at }]
            : [],
        },
      });
      return;
    }
    const body = route.request().postDataJSON();
    storedConversation = {
      ...body,
      id: body.id || "conv-workspace-race",
      created_at: storedConversation?.created_at || Date.now(),
      updated_at: Date.now(),
    };
    await route.fulfill({ json: storedConversation });
  });
  await page.route("**/query**", async (route) => {
    await queryReleased;
    await fulfillQueryStream(route, {
      request_id: "workspace-race",
      answer: "这段回答只属于 06/15 11:36。",
      citations: [],
      trace: {},
    });
  });

  await page.goto("/");
  const queryStarted = page.waitForRequest((request) =>
    new URL(request.url()).pathname.endsWith("/query/stream"),
  );
  await page.getByPlaceholder("提问或创作内容").fill("有哪些关键事实值得关注？");
  await page.getByRole("button", { name: "发送消息" }).click();
  await queryStarted;
  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("button", { name: /Production RAG 知识库 06\/14 23:54/ }).click();
  await page.getByRole("button", { name: "关闭设置" }).click();
  queryResolve?.();

  await expect(page.getByText("这段回答只属于 06/15 11:36。")).toHaveCount(0);
  await expect.poll(async () =>
    page.evaluate(() => JSON.parse(localStorage.getItem("production-rag-workspace-conversations:workspace-a") || "[]")),
  ).toEqual(["conv-workspace-race"]);
  await expect.poll(async () =>
    page.evaluate(() => JSON.parse(localStorage.getItem("production-rag-workspace-conversations:workspace-b") || "[]")),
  ).toEqual([]);

  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("button", { name: /Production RAG 知识库 06\/15 11:36/ }).click();
  await page.getByRole("button", { name: "关闭设置" }).click();
  await expect(page.getByText("这段回答只属于 06/15 11:36。")).toBeVisible();
});

test("keeps an in-flight upload scoped to the database where it started", async ({ page }) => {
  let uploadResolve: (() => void) | null = null;
  const uploadReleased = new Promise<void>((resolve) => {
    uploadResolve = resolve;
  });
  let sourceRows: any[] = [];
  const uploadedSource = {
    doc_id: "uploaded-natural",
    title: "自然自然辩证法.pdf",
    source_type: "pdf",
    source_uri: "/uploads/natural.pdf",
    doc_version: 1,
    chunk_count: 6,
    acl_groups: ["engineering"],
    status: "ready",
    current: true,
    child_doc_ids: ["uploaded-natural/page-1"],
  };
  await page.addInitScript(() => {
    const now = Date.now();
    localStorage.setItem(
      "production-rag-workspaces:test-user",
      JSON.stringify([
        { id: "workspace-a", name: "上传知识库 A", user_id: "test-user", created_at: now, updated_at: now },
        { id: "workspace-b", name: "上传知识库 B", user_id: "test-user", created_at: now, updated_at: now },
      ]),
    );
    localStorage.setItem("production-rag-active-workspace-id:test-user", "workspace-a");
    localStorage.setItem("production-rag-workspace-sources:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-sources:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-b", JSON.stringify([]));
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources/upload", async (route) => {
    await uploadReleased;
    sourceRows = [uploadedSource];
    await route.fulfill({ json: { sources: [uploadedSource] } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: sourceRows } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "添加来源" }).click();
  await page.getByRole("dialog", { name: "添加来源" }).locator('input[type="file"]').setInputFiles({
    name: "自然自然辩证法.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("fake pdf"),
  });
  await expect(page.locator(".source-row", { hasText: "自然自然辩证法.pdf" })).toBeVisible();
  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("button", { name: /上传知识库 B/ }).click();
  await page.getByRole("button", { name: "关闭设置" }).click();
  uploadResolve?.();

  await expect(page.locator(".source-row", { hasText: "自然自然辩证法.pdf" })).toHaveCount(0);
  await expect.poll(async () =>
    page.evaluate(() => JSON.parse(localStorage.getItem("production-rag-workspace-sources:workspace-a") || "[]")),
  ).toContain("uploaded-natural");
  await expect.poll(async () =>
    page.evaluate(() => JSON.parse(localStorage.getItem("production-rag-workspace-sources:workspace-b") || "[]")),
  ).toEqual([]);

  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("button", { name: /上传知识库 A/ }).click();
  await page.getByRole("button", { name: "关闭设置" }).click();
  await expect(page.locator(".source-row", { hasText: "自然自然辩证法.pdf" })).toBeVisible();
});

test("keeps an in-flight studio artifact scoped to the database where it started", async ({ page }) => {
  let artifactResolve: (() => void) | null = null;
  const artifactReleased = new Promise<void>((resolve) => {
    artifactResolve = resolve;
  });
  let artifactRows: any[] = [];
  const readyArtifact = {
    id: "artifact-workspace-race",
    title: "自然自然辩证法.pdf 思维导图",
    status: "ready",
    artifact_type: "mindmap",
    tenant_id: "team_a",
    source_doc_ids: ["natural/page-1"],
    created_at: Date.now(),
    updated_at: Date.now(),
    root: { id: "root", label: "自然辩证法重点", children: [] },
  };
  await page.addInitScript(() => {
    const now = Date.now();
    localStorage.setItem(
      "production-rag-workspaces:test-user",
      JSON.stringify([
        { id: "workspace-a", name: "Studio 知识库 A", user_id: "test-user", created_at: now, updated_at: now },
        { id: "workspace-b", name: "Studio 知识库 B", user_id: "test-user", created_at: now, updated_at: now },
      ]),
    );
    localStorage.setItem("production-rag-active-workspace-id:test-user", "workspace-a");
    localStorage.setItem("production-rag-workspace-sources:workspace-a", JSON.stringify(["natural"]));
    localStorage.setItem("production-rag-workspace-sources:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-conversations:workspace-b", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-a", JSON.stringify([]));
    localStorage.setItem("production-rag-workspace-artifacts:workspace-b", JSON.stringify([]));
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "natural",
            title: "自然自然辩证法.pdf",
            source_type: "pdf",
            source_uri: "/uploads/natural.pdf",
            doc_version: 1,
            chunk_count: 6,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: ["natural/page-1"],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts/mindmap", async (route) => {
    await artifactReleased;
    artifactRows = [readyArtifact];
    await route.fulfill({ json: readyArtifact });
  });
  await page.route("**/artifacts/artifact-workspace-race?**", async (route) => {
    await route.fulfill({ json: readyArtifact });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: artifactRows } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await page.getByRole("button", { name: /思维导图/ }).click();
  await expect(page.getByText("正在生成思维导图...")).toBeVisible();
  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("button", { name: /Studio 知识库 B/ }).click();
  await page.getByRole("button", { name: "关闭设置" }).click();
  artifactResolve?.();

  await expect(page.getByText("自然辩证法重点")).toHaveCount(0);
  await expect(page.getByText("自然自然辩证法.pdf 思维导图")).toHaveCount(0);
  await expect.poll(async () =>
    page.evaluate(() => JSON.parse(localStorage.getItem("production-rag-workspace-artifacts:workspace-a") || "[]")),
  ).toEqual(["artifact-workspace-race"]);
  await expect.poll(async () =>
    page.evaluate(() => JSON.parse(localStorage.getItem("production-rag-workspace-artifacts:workspace-b") || "[]")),
  ).toEqual([]);

  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("button", { name: /Studio 知识库 A/ }).click();
  await page.getByRole("button", { name: "关闭设置" }).click();
  await expect(page.getByText("自然自然辩证法.pdf 思维导图")).toBeVisible();
});

test("clears database rename state after closing settings", async ({ page }) => {
  await mockWorkspaceShell(page);

  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "知识库设置" });
  await dialog.locator(".database-list-item.active .icon-button").click();
  await page.getByRole("menuitem", { name: /重命名知识\s*库/ }).click();
  const input = dialog.locator(".database-list-item.active .inline-title-input");
  await expect(input).toBeVisible();
  await expect(input).toHaveAttribute("id", /workspace-rename-/);
  await expect(input).toHaveAttribute("name", "workspace-name");

  await page.mouse.click(12, 120);
  await expect(dialog).toBeHidden();
  await page.getByRole("button", { name: "设置" }).click();
  await expect(page.getByRole("dialog", { name: "知识库设置" }).locator(".inline-title-input")).toHaveCount(0);
});

test("shows a masked personal login link on the profile page", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await mockWorkspaceShell(page);

  await page.goto("/");
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /个人信息/ }).click();

  const secretInput = page.locator(".secret-field input");
  await expect(secretInput).toHaveValue("********");
  await page.getByRole("button", { name: "显示专属登录链接" }).click();
  await expect(secretInput).toHaveValue(/#token=test-session$/);
  await page.getByRole("button", { name: "复制专属登录链接" }).click();
  await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toContain("#token=test-session");
  await expect(page.getByRole("button", { name: "复制专属登录链接" }).locator("svg")).toHaveCSS("color", "rgb(52, 211, 153)");
  await expect(page.getByText("通过专属登录链接可以实现无密码账户登录，请勿将该链接分享给别人。")).toBeVisible();
  await expect(page.getByRole("link", { name: "打开 GitHub 仓库" })).toHaveAttribute("href", "https://github.com/lizhenisu/project4llm");
});

test("refreshes the personal login link token from the more menu", async ({ page }) => {
  await mockWorkspaceShell(page);
  await page.route("**/auth/token/refresh", async (route) => {
    await route.fulfill({
      json: {
        user: {
          id: "test-user",
          username: "tester",
          display_name: "测试用户",
          role: "user",
          tenant_id: "team_a",
          avatar_url: "",
          status: "active",
          created_at: Date.now(),
          last_login_at: Date.now(),
        },
        token: "new-session-token",
        expires_at: Date.now() + 86_400_000,
      },
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /个人信息/ }).click();
  await page.getByRole("button", { name: "更多专属登录链接选项" }).click();
  await page.getByRole("menuitem", { name: "刷新 token" }).click();

  const secretInput = page.locator(".secret-field input");
  await expect(secretInput).toHaveValue(/#token=new-session-token$/);
  await expect(page.getByText("专属登录链接已刷新")).toBeVisible();
  await expect.poll(async () =>
    page.evaluate(() => JSON.parse(localStorage.getItem("production-rag-auth-session") || "{}").token),
  ).toBe("new-session-token");
});

test("disables personal login token refresh for the fixed test account", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "production-rag-auth-session",
      JSON.stringify({
        user: {
          id: "user-fixed-test",
          username: "test_user",
          display_name: "测试账号",
          role: "user",
          tenant_id: "tenant-fixed-test",
          avatar_url: "",
          status: "active",
          created_at: Date.now(),
          last_login_at: Date.now(),
        },
        token: "production-rag-fixed-test-login-token",
        expires_at: Date.now() + 86_400_000,
      }),
    );
  });
  await mockWorkspaceShell(page);

  await page.goto("/");
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /个人信息/ }).click();
  await page.getByRole("button", { name: "更多专属登录链接选项" }).click();

  await expect(page.getByRole("menuitem", { name: "刷新 token" })).toBeDisabled();
  await expect(page.getByText("测试账号使用固定专属 token，不能刷新。")).toBeVisible();
});

test("renders assistant math formulas with KaTeX", async ({ page }) => {
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
    await route.fulfill({
      json: {
        conversations: [
          {
            id: "math-conversation",
            tenant_id: "team_a",
            title: "公式回答",
            message_count: 1,
            source_doc_ids: [],
            created_at: Date.now(),
            updated_at: Date.now(),
          },
        ],
      },
    });
  });
  await page.route("**/conversations/math-conversation?**", async (route) => {
    await route.fulfill({
      json: {
        id: "math-conversation",
        tenant_id: "team_a",
        title: "公式回答",
        source_doc_ids: [],
        created_at: Date.now(),
        updated_at: Date.now(),
        messages: [
          {
            id: "math-answer",
            role: "assistant",
            content: "行内公式 $E=mc^2$，块公式：\n\n$$\\int_0^1 x^2 dx=\\frac{1}{3}$$",
            status: "done",
            citations: [],
          },
        ],
      },
    });
  });

  await page.goto("/");
  await expect(page.locator(".assistant-message .katex")).toHaveCount(2);
});

test("opens parsed source content from a document-level source row", async ({ page }) => {
  await mockSourceAssetRoute(page);
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
  await page.route("**/conversations**", async (route) => {
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
        blocks: [
          {
            type: "text",
            text: "Page 1\n\nAttention Is All You Need\n\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
          },
          {
            type: "image",
            title: "Image 1",
            page: "Page 1",
            url: "/source-assets/uploads/team_a/regression/paper.assets/page-1-image-1.png?tenant_id=team_a",
          },
        ],
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
  await expect(reader.getByText("Page 1", { exact: true })).toBeVisible();
  await expect(reader.getByText("Attention Is All You Need")).toBeVisible();
  const sourceImage = reader.getByRole("img", { name: "Image 1" });
  await expect(sourceImage).toBeVisible();
  await expect(sourceImage).toHaveAttribute("src", /^blob:/);
  await expect.poll(async () => sourceImage.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await expect(reader.getByText("第 1 页")).toHaveCount(0);
});

test("sends an attached chat image as a multimodal query", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "production-rag-auth-session",
      JSON.stringify({
        user: {
          id: "test-user",
          username: "tester",
          display_name: "测试用户",
          role: "user",
          tenant_id: "team_a",
          avatar_url: "",
          status: "active",
          created_at: Date.now(),
          last_login_at: Date.now(),
        },
        token: "test-session",
        expires_at: Date.now() + 86_400_000,
      }),
    );
  });
  await mockSourceAssetRoute(page);
  let queryPayload: any = null;
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/auth/me", async (route) => {
    await route.fulfill({
      json: {
        id: "test-user",
        username: "tester",
        display_name: "测试用户",
        role: "user",
        tenant_id: "team_a",
        avatar_url: "",
        status: "active",
        created_at: Date.now(),
        last_login_at: Date.now(),
      },
    });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "paper",
            title: "attention is all you need.pdf",
            source_type: "pdf",
            source_uri: "/uploads/paper.pdf",
            doc_version: 1,
            chunk_count: 6,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: ["paper/page-1"],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: { conversations: [] } });
      return;
    }
    const body = route.request().postDataJSON();
    await route.fulfill({ json: { ...body, id: "conv-image-query", updated_at: Date.now() } });
  });
  await page.route("**/query/stream", async (route) => {
    queryPayload = route.request().postDataJSON();
    const result = {
      type: "result",
      request_id: "image-query",
      answer: "已根据图片检索到相关论文图示。",
      citations: [
        {
          doc_id: "paper/page-1",
          title: "attention is all you need p1",
          source_uri: "/uploads/paper.pdf",
          source_type: "pdf",
          chunk_index: 0,
          score: 0.8,
          rerank_score: 0.7,
          acl_groups: ["engineering"],
          metadata: {
            page_no: 1,
            display_blocks: [
              {
                type: "image",
                title: "Figure 1",
                url: "/source-assets/uploads/team_a/regression/paper.assets/page-1-image-1.png?tenant_id=team_a",
              },
            ],
          },
          text_preview: "Figure evidence",
        },
      ],
      trace: {},
    };
    await route.fulfill({
      contentType: "application/x-ndjson",
      body: `${JSON.stringify(result)}\n`,
    });
  });

  await page.goto("/#token=test-session");
  await expect(page.getByPlaceholder("提问或创作内容")).toBeVisible();
  await page.getByRole("button", { name: "上传图片提问" }).click();
  await page.locator('.chat-input input[type="file"]').setInputFiles({
    name: "query.png",
    mimeType: "image/png",
    buffer: Buffer.from(ONE_PIXEL_PNG_BASE64, "base64"),
  });
  await expect(page.getByRole("img", { name: "待发送图片" })).toBeVisible();
  await page.getByRole("img", { name: "待发送图片" }).click();
  const pendingImageDialog = page.getByRole("dialog", { name: "待发送图片" });
  await expect(pendingImageDialog).toBeVisible();
  await expect(pendingImageDialog.getByRole("img", { name: "待发送图片" })).toBeVisible();
  await pendingImageDialog.getByRole("button", { name: "关闭图片预览" }).click();
  await expect(pendingImageDialog).toBeHidden();
  await page.getByPlaceholder("提问或创作内容").fill("这张图和论文中哪部分相关？");
  await page.getByRole("button", { name: "发送消息" }).click();

  await expect(page.getByText("已根据图片检索到相关论文图示。")).toBeVisible();
  const sentImage = page.getByRole("img", { name: "发送的图片" });
  await expect(sentImage).toBeVisible();
  await expect.poll(async () => sentImage.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await page.getByRole("button", { name: "查看发送的图片" }).click();
  const sentImageDialog = page.getByRole("dialog", { name: "发送的图片" });
  await expect(sentImageDialog).toBeVisible();
  await expect(sentImageDialog.getByRole("img", { name: "发送的图片" })).toBeVisible();
  await sentImageDialog.getByRole("button", { name: "关闭图片预览" }).click();

  const citationImage = page.getByRole("img", { name: "Figure 1" });
  await expect(citationImage).toBeVisible();
  await expect(citationImage).toHaveAttribute("src", /^blob:/);
  await expect.poll(async () => citationImage.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await page.getByRole("button", { name: "查看Figure 1" }).click();
  const citationImageDialog = page.getByRole("dialog", { name: "Figure 1" });
  await expect(citationImageDialog).toBeVisible();
  await expect(citationImageDialog.getByRole("img", { name: "Figure 1" })).toBeVisible();
  await citationImageDialog.getByRole("button", { name: "关闭图片预览" }).click();
  expect(queryPayload.query_mode).toBe("multimodal");
  expect(queryPayload.image_data_url).toMatch(/^data:image\/png;base64,/);
  expect(queryPayload.doc_ids).toEqual(["paper/page-1"]);
});

test("keeps source rename input focus shadow unclipped", async ({ page }) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "rename-regression",
            title: "需要重命名的原始文档.pdf",
            source_type: "pdf",
            source_uri: "/object_store/uploads/team_a/rename-regression.pdf",
            doc_version: 1,
            chunk_count: 3,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: [],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await page.locator(".source-row .row-icon-more").click();
  await page.getByRole("button", { name: "重命名", exact: true }).click();

  const row = page.locator(".source-row.is-editing");
  const title = page.locator(".source-title.is-editing");
  const input = page.locator(".inline-title-input");

  await expect(input).toBeFocused();
  await expect(row).toHaveCSS("overflow", "visible");
  await expect(title).toHaveCSS("overflow", "visible");
  await expect(input).not.toHaveCSS("box-shadow", "none");
});

test("does not load protected workspace services before login", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.removeItem("production-rag-auth-session");
  });
  let protectedRequests = 0;
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    protectedRequests += 1;
    await route.fulfill({ status: 401, json: { detail: "请先登录" } });
  });
  await page.route("**/artifacts?**", async (route) => {
    protectedRequests += 1;
    await route.fulfill({ status: 401, json: { detail: "请先登录" } });
  });
  await page.route("**/conversations?**", async (route) => {
    protectedRequests += 1;
    await route.fulfill({ status: 401, json: { detail: "请先登录" } });
  });

  await page.goto("/");
  await expect(page.getByText("请先登录后使用知识库服务")).toBeVisible();
  expect(protectedRequests).toBe(0);
});

test("navigates a guest to the login page when sending from the chat input", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.removeItem("production-rag-auth-session");
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });

  await page.goto("/");
  const input = page.getByPlaceholder("登录后即可发送");
  await input.fill("自然辩证法的引言");
  await expect(page.getByRole("button", { name: "发送消息" })).toBeEnabled();
  await input.press("Enter");
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { name: "登录账号" })).toBeVisible();
  await expect(page.locator(".auth-page")).toHaveCSS("background-color", "rgb(237, 239, 250)");
  await expect(page.locator(".auth-page-panel")).toHaveCSS("background-color", "rgb(255, 255, 255)");
  await expect(page.locator(".auth-page-panel")).toHaveCSS("box-shadow", "none");
  await expect(page.locator(".auth-page-submit")).toHaveCSS("color", "rgb(255, 255, 255)");
  await page.getByLabel("用户名").fill("tester");
  await page.getByLabel("密码").fill("strong-password");
  await expect(page.locator(".auth-page-submit")).toBeEnabled();
  await expect(page.locator(".auth-page-submit")).toHaveCSS("background-color", "rgb(56, 189, 248)");
  await expect(page.locator(".auth-page-submit")).toHaveCSS("color", "rgb(255, 255, 255)");
});

test("switches between dedicated login and register pages", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.removeItem("production-rag-auth-session");
  });

  await page.goto("/login");
  await expect(page.getByRole("heading", { name: "登录账号" })).toBeVisible();
  await page.getByRole("button", { name: "去注册" }).click();
  await expect(page).toHaveURL(/\/register$/);
  await expect(page.getByRole("heading", { name: "注册账号" })).toBeVisible();
  await page.getByRole("button", { name: "去登录" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { name: "登录账号" })).toBeVisible();
});

test("updates workspace status immediately after login", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.removeItem("production-rag-auth-session");
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/auth/login", async (route) => {
    await route.fulfill({
      json: {
        user: {
          id: "user-aak",
          username: "aak",
          display_name: "aak",
          role: "user",
          tenant_id: "team_a",
          avatar_url: "",
          status: "active",
          created_at: Date.now(),
          last_login_at: Date.now(),
        },
        token: "session-aak",
        expires_at: Date.now() + 86_400_000,
      },
    });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: [] } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/login");
  await page.getByLabel("用户名").fill("aak");
  await page.getByLabel("密码").fill("12345678");
  await page.getByRole("button", { name: "登录", exact: true }).click();

  await expect(page).toHaveURL(/\/$/);
  await expect(page.locator(".statusbar")).toContainText("API 已连接");
});

test("logs in directly from a hash token", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.removeItem("production-rag-auth-session");
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/auth/me", async (route) => {
    expect(route.request().headers().authorization).toBe("Bearer abc123abc123abc123abc123");
    await route.fulfill({
      json: {
        id: "token-user",
        username: "tokenuser",
        display_name: "Token 用户",
        role: "user",
        tenant_id: "team_a",
        avatar_url: "",
        status: "active",
        created_at: Date.now(),
        last_login_at: Date.now(),
      },
    });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: [] } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/#token=abc123abc123abc123abc123");
  await expect(page.getByRole("button", { name: "用户头像" })).toHaveText("T");
  await expect(page).not.toHaveURL(/token=/);
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
  await page.getByText("实习招聘思维导图").click();
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

  await page.getByRole("button", { name: "Studio" }).click();
  await expect
    .poll(async () => (await page.locator(".studio-panel").boundingBox())?.width ?? 0)
    .toBeLessThan(studioAfter!.width - 40);
  await expect
    .poll(async () => {
      const studioRestored = await page.locator(".studio-panel").boundingBox();
      return Math.abs((studioRestored?.width ?? 0) - studioBefore!.width);
    })
    .toBeLessThan(32);
  await expect(page.getByText("实习招聘思维导图")).toBeVisible();
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

test("shows marquee feedback for active source and studio tasks", async ({ page }) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "uploading-guide",
            title: "正在解析.pdf",
            source_type: "pdf",
            source_uri: "/uploads/uploading-guide.pdf",
            doc_version: 1,
            chunk_count: 0,
            acl_groups: ["engineering"],
            status: "processing",
            current: false,
            child_doc_ids: [],
            ingestion_stage: "text_embedding",
            progress_percent: 62,
            progress_detail: "36/80 个文本片段",
            eta_seconds: 75,
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({
      json: {
        artifacts: [
          {
            id: "generating-mindmap",
            title: "选中来源思维导图",
            status: "generating",
            tenant_id: "team_a",
            source_doc_ids: ["uploading-guide/page-1"],
            created_at: Date.now(),
            updated_at: Date.now(),
            root: null,
          },
        ],
      },
    });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  const sourceRow = page.locator(".source-row.is-active-task");
  const artifactRow = page.locator(".artifact-row.is-active-task");
  await expect(sourceRow).toBeVisible();
  await expect(artifactRow).toBeVisible();
  await expect(sourceRow.getByText("正在生成文本向量 · 62% · 36/80 个文本片段 · 预计剩余约 1 分钟")).toBeVisible();
  await expect(sourceRow.getByRole("progressbar", { name: "正在解析.pdf 处理进度" })).toHaveAttribute(
    "aria-valuenow",
    "62",
  );
  await expect
    .poll(async () => sourceRow.evaluate((node) => getComputedStyle(node, "::after").animationName))
    .toBe("task-marquee");
  await expect
    .poll(async () => artifactRow.evaluate((node) => getComputedStyle(node, "::after").animationName))
    .toBe("task-marquee");
});

test("removes the upload processing row after the parsed source is ready", async ({ page }) => {
  let sourcesPolls = 0;
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    sourcesPolls += 1;
    await route.fulfill({
      json: {
        sources:
          sourcesPolls < 2
            ? []
            : [
                {
                  doc_id: "upload-task",
                  title: "重复解析.pdf",
                  source_type: "pdf",
                  source_uri: "/uploads/upload-task.pdf",
                  doc_version: 1,
                  chunk_count: 0,
                  acl_groups: ["engineering"],
                  status: "processing",
                  current: false,
                  child_doc_ids: [],
                },
                {
                  doc_id: "parsed-upload",
                  title: "重复解析.pdf",
                  source_type: "pdf",
                  source_uri: "/uploads/parsed-upload.pdf",
                  doc_version: 1,
                  chunk_count: 3,
                  acl_groups: ["engineering"],
                  status: "ready",
                  current: true,
                  child_doc_ids: ["parsed-upload/page-1"],
                },
              ],
      },
    });
  });
  await page.route("**/sources/upload", async (route) => {
    await route.fulfill({
      json: {
        status: "processing",
        document_count: 0,
        chunk_count: 0,
        sources: [
          {
            doc_id: "upload-task",
            title: "重复解析.pdf",
            source_type: "pdf",
            source_uri: "/uploads/upload-task.pdf",
            doc_version: 1,
            chunk_count: 0,
            acl_groups: ["engineering"],
            status: "processing",
            current: false,
            child_doc_ids: [],
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

  await page.goto("/");
  await page.getByRole("button", { name: "添加来源" }).click();
  await page.getByRole("dialog", { name: "添加来源" }).locator('input[type="file"]').setInputFiles({
    name: "重复解析.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("fake pdf"),
  });

  await expect(page.locator(".source-row.status-ready", { hasText: "重复解析.pdf" })).toBeVisible({ timeout: 5_000 });
  await expect(page.locator(".source-row.status-processing", { hasText: "重复解析.pdf" })).toHaveCount(0);
});

test("renames a source from the row menu and persists the new title", async ({ page }) => {
  let sourceTitle = "自然辩证法.pdf";
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "source-natural",
            title: sourceTitle,
            source_type: "pdf",
            source_uri: "/uploads/source-natural.pdf",
            doc_version: 1,
            chunk_count: 6,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: ["source-natural/page-1"],
          },
        ],
      },
    });
  });
  await page.route("**/sources/source-natural?**", async (route) => {
    if (route.request().method() === "PATCH") {
      const body = route.request().postDataJSON();
      sourceTitle = body.title;
      await route.fulfill({ json: { status: "renamed", doc_id: "source-natural", title: sourceTitle } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  const row = page.locator(".source-row", { hasText: "自然辩证法.pdf" });
  await row.locator(".row-icon-more").click();
  await page.getByRole("button", { name: "重命名" }).click();
  await page.locator(".source-row .inline-title-input").fill("自然辩证法-重命名.pdf");
  await page.locator(".source-row .inline-title-input").press("Enter");

  await expect(page.locator(".source-row", { hasText: "自然辩证法-重命名.pdf" })).toBeVisible();
  await expect(page.locator(".source-row", { hasText: "自然辩证法.pdf" })).toHaveCount(0);
});

test("shows only the current source version and deletes the whole source", async ({ page }) => {
  let sources = [
    {
      doc_id: "duplicate-source",
      title: "深大_创维 AI 研究院实习介绍资料(1).pdf",
      source_type: "pdf",
      source_uri: "/uploads/duplicate-v1.pdf",
      doc_version: 1,
      chunk_count: 4,
      acl_groups: ["engineering"],
      status: "ready",
      current: false,
      child_doc_ids: ["duplicate-source/page-1"],
    },
    {
      doc_id: "duplicate-source",
      title: "深大_创维 AI 研究院实习介绍资料(1).pdf",
      source_type: "pdf",
      source_uri: "/uploads/duplicate-v2.pdf",
      doc_version: 2,
      chunk_count: 5,
      acl_groups: ["engineering"],
      status: "ready",
      current: true,
      child_doc_ids: ["duplicate-source/page-1"],
    },
  ];
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources } });
  });
  await page.route("**/sources/duplicate-source?**", async (route) => {
    if (route.request().method() === "DELETE") {
      const url = new URL(route.request().url());
      expect(url.searchParams.get("doc_version")).toBeNull();
      sources = [];
      await route.fulfill({ json: { status: "deleted" } });
      return;
    }
    await route.fallback();
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await expect(page.locator(".source-row", { hasText: "深大_创维 AI 研究院实习介绍资料(1).pdf" })).toHaveCount(1);
  await page.locator(".source-row", { hasText: "深大_创维 AI 研究院实习介绍资料(1).pdf" }).first().locator(".row-icon-more").click();
  await page.getByRole("button", { name: "移除" }).click();
  await page.getByRole("button", { name: "确认移除" }).click();

  await expect(page.locator(".source-row", { hasText: "深大_创维 AI 研究院实习介绍资料(1).pdf" })).toHaveCount(0);
});

test("does not keep unrelated transient sources after upload polling completes", async ({ page }) => {
  let sourcesPolls = 0;
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    sourcesPolls += 1;
    const sources =
      sourcesPolls < 2
        ? []
        : sourcesPolls < 4
          ? [
              {
                doc_id: "stale-internship",
                title: "深大_创维 AI 研究院实习介绍资料(1).pdf",
                source_type: "pdf",
                source_uri: "/uploads/stale.pdf",
                doc_version: 1,
                chunk_count: 2,
                acl_groups: ["engineering"],
                status: "ready",
                current: true,
                child_doc_ids: ["stale-internship/page-1"],
              },
              {
                doc_id: "upload-task-natural",
                title: "自然辩证法.pdf",
                source_type: "pdf",
                source_uri: "/uploads/upload-task-natural.pdf",
                doc_version: 1,
                chunk_count: 0,
                acl_groups: ["engineering"],
                status: "processing",
                current: false,
                child_doc_ids: [],
              },
            ]
          : [
              {
                doc_id: "natural-ready",
                title: "自然辩证法.pdf",
                source_type: "pdf",
                source_uri: "/uploads/natural-ready.pdf",
                doc_version: 1,
                chunk_count: 8,
                acl_groups: ["engineering"],
                status: "ready",
                current: true,
                child_doc_ids: ["natural-ready/page-1"],
              },
            ];
    await route.fulfill({ json: { sources } });
  });
  await page.route("**/sources/upload", async (route) => {
    await route.fulfill({
      json: {
        status: "processing",
        document_count: 0,
        chunk_count: 0,
        sources: [
          {
            doc_id: "upload-task-natural",
            title: "自然辩证法.pdf",
            source_type: "pdf",
            source_uri: "/uploads/upload-task-natural.pdf",
            doc_version: 1,
            chunk_count: 0,
            acl_groups: ["engineering"],
            status: "processing",
            current: false,
            child_doc_ids: [],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "添加来源" }).click();
  await page.getByRole("dialog", { name: "添加来源" }).locator('input[type="file"]').setInputFiles({
    name: "自然辩证法.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("fake pdf"),
  });

  await expect(page.locator(".source-row.status-ready", { hasText: "自然辩证法.pdf" })).toBeVisible({ timeout: 7_000 });
  await expect(page.locator(".source-row", { hasText: "深大_创维 AI 研究院实习介绍资料(1).pdf" })).toHaveCount(0);
  await expect(page.locator(".source-row.status-processing", { hasText: "自然辩证法.pdf" })).toHaveCount(0);
});

test("replaces the visible current version while uploading another copy", async ({ page }) => {
  let sourcesPolls = 0;
  const existingSources = [1, 2, 3].map((version) => ({
    doc_id: "自然辩证法",
    title: "自然辩证法.pdf",
    source_type: "pdf",
    source_uri: `/uploads/natural-v${version}.pdf`,
    doc_version: version,
    chunk_count: 6,
    acl_groups: ["engineering"],
    status: "ready",
    current: version === 3,
    child_doc_ids: [`自然辩证法/page-${version}`],
  }));
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    sourcesPolls += 1;
    const sources =
      sourcesPolls < 3
        ? existingSources
        : sourcesPolls < 5
          ? [
              {
                doc_id: "upload-task-natural-4",
                title: "自然辩证法.pdf",
                source_type: "pdf",
                source_uri: "/uploads/natural-v4-task.pdf",
                doc_version: 4,
                chunk_count: 0,
                acl_groups: ["engineering"],
                status: "processing",
                current: false,
                child_doc_ids: [],
              },
              ...existingSources,
            ]
          : [
              ...existingSources.map((source) => ({ ...source, current: false })),
              {
                doc_id: "自然辩证法",
                title: "自然辩证法.pdf",
                source_type: "pdf",
                source_uri: "/uploads/natural-v4.pdf",
                doc_version: 4,
                chunk_count: 8,
                acl_groups: ["engineering"],
                status: "ready",
                current: true,
                child_doc_ids: ["自然辩证法/page-4"],
              },
            ];
    await route.fulfill({ json: { sources } });
  });
  await page.route("**/sources/upload", async (route) => {
    await route.fulfill({
      json: {
        status: "processing",
        document_count: 0,
        chunk_count: 0,
        sources: [
          {
            doc_id: "upload-task-natural-4",
            title: "自然辩证法.pdf",
            source_type: "pdf",
            source_uri: "/uploads/natural-v4-task.pdf",
            doc_version: 4,
            chunk_count: 0,
            acl_groups: ["engineering"],
            status: "processing",
            current: false,
            child_doc_ids: [],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await expect(page.locator(".source-row", { hasText: "自然辩证法.pdf" })).toHaveCount(1);

  await page.getByRole("button", { name: "添加来源" }).click();
  await page.getByRole("dialog", { name: "添加来源" }).locator('input[type="file"]').setInputFiles({
    name: "自然辩证法.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("fake pdf"),
  });

  await expect(page.locator(".source-row.status-processing", { hasText: "自然辩证法.pdf" })).toHaveCount(1);
  await expect(page.locator(".source-row", { hasText: "自然辩证法.pdf" })).toHaveCount(2);
  await expect(page.locator(".source-row.status-ready", { hasText: "自然辩证法.pdf" })).toHaveCount(1, {
    timeout: 7_000,
  });
  await expect(page.locator(".source-row.status-processing", { hasText: "自然辩证法.pdf" })).toHaveCount(0);
});

test("creates and opens a data table artifact from studio", async ({ page }) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "internship-guide",
            title: "实习介绍资料.pdf",
            source_type: "pdf",
            source_uri: "/uploads/internship-guide.pdf",
            doc_version: 1,
            chunk_count: 1,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            selected: true,
            child_doc_ids: ["internship-guide/page-1"],
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
  await page.route("**/artifacts/table", async (route) => {
    await route.fulfill({
      json: {
        id: "table-regression",
        title: "实习介绍资料.pdf 数据表格",
        status: "ready",
        artifact_type: "table",
        tenant_id: "team_a",
        source_doc_ids: ["internship-guide/page-1"],
        created_at: Date.now(),
        updated_at: Date.now(),
        root: null,
        table: {
          title: "实习岗位数据表格",
          columns: ["岗位", "职责", "要求"],
          rows: [["大模型应用开发实习生", "开发 RAG 与智能体应用", "熟悉 TypeScript"]],
          summary: "该表格用于比较实习岗位的职责和要求。",
        },
      },
    });
  });
  await page.route("**/artifacts/table-regression?**", async (route) => {
    await route.fulfill({
      json: {
        id: "table-regression",
        title: "实习介绍资料.pdf 数据表格",
        status: "ready",
        artifact_type: "table",
        tenant_id: "team_a",
        source_doc_ids: ["internship-guide/page-1"],
        created_at: Date.now(),
        updated_at: Date.now(),
        root: null,
        table: {
          title: "实习岗位数据表格",
          columns: ["岗位", "职责", "要求"],
          rows: [["大模型应用开发实习生", "开发 RAG 与智能体应用", "熟悉 TypeScript"]],
          summary: "该表格用于比较实习岗位的职责和要求。",
        },
      },
    });
  });

  await page.goto("/");
  await expect(page.getByRole("button", { name: /思维导图/ })).toHaveClass(/tone-purple/);
  await expect(page.getByRole("button", { name: /数据表格/ })).toHaveClass(/tone-cyan/);
  await page.getByRole("button", { name: /数据表格/ }).click();
  await expect(page.getByText("实习介绍资料.pdf 数据表格")).toBeVisible();
  await expect(page.getByText("实习岗位数据表格")).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "岗位" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "大模型应用开发实习生" })).toBeVisible();
});

test("rate limits studio artifact generation across mind map and data table tools", async ({ page }) => {
  let mindMapRequests = 0;
  let tableRequests = 0;

  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "studio-source",
            title: "Studio 资料.pdf",
            source_type: "pdf",
            source_uri: "/uploads/studio-source.pdf",
            doc_version: 1,
            chunk_count: 1,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            selected: true,
            child_doc_ids: ["studio-source/page-1"],
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
  await page.route("**/artifacts/mindmap", async (route) => {
    mindMapRequests += 1;
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.fulfill({
      json: {
        id: "mindmap-rate-limit",
        title: "Studio 资料.pdf 思维导图",
        status: "ready",
        artifact_type: "mindmap",
        tenant_id: "team_a",
        source_doc_ids: ["studio-source/page-1"],
        created_at: Date.now(),
        updated_at: Date.now(),
        root: { id: "root", label: "Studio 资料", children: [] },
      },
    });
  });
  await page.route("**/artifacts/mindmap-rate-limit?**", async (route) => {
    await route.fulfill({
      json: {
        id: "mindmap-rate-limit",
        title: "Studio 资料.pdf 思维导图",
        status: "ready",
        artifact_type: "mindmap",
        tenant_id: "team_a",
        source_doc_ids: ["studio-source/page-1"],
        created_at: Date.now(),
        updated_at: Date.now(),
        root: { id: "root", label: "Studio 资料", children: [] },
      },
    });
  });
  await page.route("**/artifacts/table", async (route) => {
    tableRequests += 1;
    await route.fulfill({
      json: {
        id: "table-rate-limit",
        title: "Studio 资料.pdf 数据表格",
        status: "ready",
        artifact_type: "table",
        tenant_id: "team_a",
        source_doc_ids: ["studio-source/page-1"],
        created_at: Date.now(),
        updated_at: Date.now(),
        root: null,
        table: {
          title: "Studio 数据表格",
          columns: ["主题", "摘要"],
          rows: [["速率限制", "每 4 秒最多生成一次"]],
          summary: "用于验证 Studio 生成工具共享冷却。",
        },
      },
    });
  });
  await page.route("**/artifacts/table-rate-limit?**", async (route) => {
    await route.fulfill({
      json: {
        id: "table-rate-limit",
        title: "Studio 资料.pdf 数据表格",
        status: "ready",
        artifact_type: "table",
        tenant_id: "team_a",
        source_doc_ids: ["studio-source/page-1"],
        created_at: Date.now(),
        updated_at: Date.now(),
        root: null,
        table: {
          title: "Studio 数据表格",
          columns: ["主题", "摘要"],
          rows: [["速率限制", "每 4 秒最多生成一次"]],
          summary: "用于验证 Studio 生成工具共享冷却。",
        },
      },
    });
  });

  await page.goto("/");
  const mindMapButton = page.getByRole("button", { name: /思维导图/ });
  const tableButton = page.getByRole("button", { name: /数据表格/ });

  await mindMapButton.click();
  await expect(mindMapButton).toBeDisabled();
  await expect(tableButton).toBeDisabled();
  expect(mindMapRequests).toBe(1);
  expect(tableRequests).toBe(0);

  await expect(page.getByText("Studio 资料.pdf 思维导图")).toBeVisible();
  await page.getByRole("button", { name: "Studio", exact: true }).click();
  await expect(mindMapButton).toBeDisabled();
  await expect(tableButton).toBeDisabled();
  expect(tableRequests).toBe(0);

  await expect(tableButton).toBeEnabled({ timeout: 5_000 });
  await tableButton.click();
  await expect(page.getByText("Studio 数据表格")).toBeVisible();
  expect(mindMapRequests).toBe(1);
  expect(tableRequests).toBe(1);
});

test("registers an admin user from the avatar menu and publishes an announcement", async ({ page }) => {
  const now = Date.now();
  let registrationEnabled = true;
  const usageRequests: URLSearchParams[] = [];
  await page.addInitScript(() => {
    localStorage.removeItem("production-rag-auth-session");
  });
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
  await page.route("**/auth/register", async (route) => {
    await route.fulfill({
      json: {
        user: {
          id: "user-admin",
          username: "admin",
          display_name: "管理员",
          role: "admin",
          tenant_id: "tenant-admin",
          avatar_url: "",
          status: "active",
          created_at: now,
          last_login_at: now,
        },
        token: "session-admin",
        expires_at: now + 86_400_000,
      },
    });
  });
  await page.route("**/auth/me", async (route) => {
    if (route.request().method() === "PATCH") {
      const body = route.request().postDataJSON();
      await route.fulfill({
        json: {
          id: "user-admin",
          username: body.username,
          display_name: body.display_name,
          role: "admin",
          tenant_id: "tenant-admin",
          avatar_url: body.avatar_url,
          status: "active",
          created_at: now,
          last_login_at: now,
        },
      });
      return;
    }
    await route.fulfill({
      json: {
        id: "user-admin",
        username: "admin",
        display_name: "管理员",
        role: "admin",
        tenant_id: "tenant-admin",
        avatar_url: "",
        status: "active",
        created_at: now,
        last_login_at: now,
      },
    });
  });
  await page.route("**/auth/password", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/admin/users?**", async (route) => {
    await route.fulfill({
      json: {
        users: [
          {
            id: "user-admin",
            username: "admin",
            display_name: "管理员",
            role: "admin",
            tenant_id: "tenant-admin",
            avatar_url: "",
            status: "active",
            created_at: now,
            last_login_at: now,
          },
          {
            id: "user-normal",
            username: "reader",
            display_name: "读者",
            role: "user",
            tenant_id: "tenant-reader",
            avatar_url: "",
            status: "active",
            created_at: now,
            last_login_at: null,
          },
        ],
        total: 2,
        limit: 50,
        offset: 0,
        query: "",
      },
    });
  });
  await page.route("**/admin/users/user-normal/status", async (route) => {
    const body = route.request().postDataJSON();
    await route.fulfill({
      json: {
        id: "user-normal",
        username: "reader",
        display_name: "读者",
        role: "user",
        tenant_id: "tenant-reader",
        avatar_url: "",
        status: body.status,
        created_at: now,
        last_login_at: null,
      },
    });
  });
  await page.route("**/admin/settings", async (route) => {
    await route.fulfill({
      json: {
        registration_enabled: registrationEnabled,
        latest_announcement: {
          id: "announcement-old",
          title: "上一条公告",
          content: "这是管理员上一次发布的公告。",
          author_id: "user-admin",
          author_name: "管理员",
          created_at: now - 60_000,
        },
      },
    });
  });
  await page.route("**/admin/settings/registration", async (route) => {
    const body = route.request().postDataJSON();
    registrationEnabled = body.registration_enabled;
    await route.fulfill({
      json: {
        registration_enabled: registrationEnabled,
        latest_announcement: {
          id: "announcement-old",
          title: "上一条公告",
          content: "这是管理员上一次发布的公告。",
          author_id: "user-admin",
          author_name: "管理员",
          created_at: now - 60_000,
        },
      },
    });
  });
  await page.route("**/admin/announcements", async (route) => {
    const body = route.request().postDataJSON();
    await route.fulfill({
      json: {
        id: "announcement-1",
        title: body.title,
        content: body.content,
        author_id: "user-admin",
        author_name: "管理员",
        created_at: now,
      },
    });
  });
  await page.route("**/admin/model-usage**", async (route) => {
    const params = new URL(route.request().url()).searchParams;
    usageRequests.push(new URLSearchParams(params));
    const offset = Number(params.get("offset") || 0);
    const tenantId = params.get("tenant_id") || "";
    const filtered = Boolean(tenantId);
    const rowTenant = tenantId || (offset > 0 ? "tenant-page-2" : "tenant-reader");
    await route.fulfill({
      json: {
        rows: [
          {
            usage_date: filtered ? "2026-06-15" : "2026-07-01",
            tenant_id: rowTenant,
            principal_key: filtered ? "user:filtered" : "user:reader",
            workload: offset > 0 ? "studio_table" : "query",
            provider: "openai-compatible",
            model: offset > 0 ? "table-model" : "answer-model",
            operation: offset > 0 ? "table_generation" : "answer_generation",
            request_count: filtered ? 3 : offset > 0 ? 1 : 12,
            prompt_tokens: filtered ? 120 : offset > 0 ? 50 : 1_200,
            completion_tokens: filtered ? 30 : offset > 0 ? 20 : 300,
            total_tokens: filtered ? 150 : offset > 0 ? 70 : 1_500,
            updated_at: now,
          },
        ],
        totals: filtered
          ? { request_count: 3, prompt_tokens: 120, completion_tokens: 30, total_tokens: 150 }
          : { request_count: 44, prompt_tokens: 4_000, completion_tokens: 1_000, total_tokens: 5_000 },
        total: filtered ? 1 : 26,
        limit: 25,
        offset,
        tenant_id: tenantId,
        start_date: params.get("start_date") || "2026-06-01",
        end_date: params.get("end_date") || "2026-07-01",
      },
    });
  });

  await page.goto("/");
  const guestAvatar = page.getByRole("button", { name: "用户头像" });
  await expect(guestAvatar.locator("svg")).toBeVisible();
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /注册/ }).click();
  await expect(page).toHaveURL(/\/register$/);
  await page.getByLabel("用户名").fill("admin");
  await page.getByLabel("显示名称").fill("管理员");
  await page.getByLabel("密码").fill("strong-password");
  await page.getByRole("button", { name: "注册并登录" }).click();
  await expect(page).toHaveURL(/\/$/);

  await page.getByRole("button", { name: "用户头像" }).click();
  await expect(page.getByText("admin · 管理员")).toBeVisible();
  await page.mouse.click(20, 20);
  await expect(page.getByText("admin · 管理员")).toBeHidden();
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /个人信息/ }).click();
  await expect(page.getByRole("heading", { name: "个人信息" })).toBeVisible();
  await expect(page.getByText("tenant-admin")).toBeVisible();
  await expect(page.getByText("注册日期")).toBeVisible();
  await page.getByLabel("显示名称").fill("管理员二号");
  await page.getByLabel("头像地址").fill("https://example.com/avatar.png");
  await page.getByRole("button", { name: "保存资料" }).click();
  await expect(page.getByText("个人信息已保存")).toBeVisible();
  await expect(page.getByRole("button", { name: "用户头像" }).locator("img")).toHaveAttribute(
    "src",
    "https://example.com/avatar.png",
  );
  await expect(page.getByRole("button", { name: "用户头像" })).not.toHaveCSS("background-color", "rgb(47, 53, 49)");
  await page.getByRole("button", { name: "返回工作台" }).click();
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /管理员控制台/ }).click();
  await expect(page.getByRole("heading", { name: "管理员控制台" })).toBeVisible();
  await expect(page.getByText("当前允许新用户自行注册。")).toBeVisible();
  await page.getByRole("button", { name: "公告管理" }).click();
  await expect(page.getByText("上一条公告")).toBeVisible();
  await expect(page.getByText("这是管理员上一次发布的公告。")).toBeVisible();
  await page.getByRole("button", { name: "系统设置" }).click();
  await expect(page.getByRole("switch", { name: "允许注册" })).toHaveAttribute("aria-checked", "true");
  await page.getByRole("switch", { name: "允许注册" }).click();
  await expect(page.getByText("当前已关闭新用户注册。")).toBeVisible();
  await expect(page.getByRole("switch", { name: "关闭注册" })).toHaveAttribute("aria-checked", "false");
  await page.getByRole("button", { name: "用户管理" }).click();
  await expect(page.getByText("tenant-reader")).toBeVisible();
  await page.getByRole("button", { name: "封禁", exact: true }).click();
  await expect(page.getByText("已封禁")).toBeVisible();

  await page.getByRole("button", { name: "模型用量" }).click();
  await expect(page.getByRole("heading", { name: "模型用量" })).toBeVisible();
  const usageTotals = page.getByLabel("模型用量汇总");
  await expect(usageTotals).toContainText("44");
  await expect(usageTotals).toContainText("5,000");
  await expect(page.getByText("tenant-reader", { exact: true })).toBeVisible();
  await expect(page.getByText("answer-model", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "下一页" }).click();
  await expect(page.getByText("tenant-page-2", { exact: true })).toBeVisible();
  expect(usageRequests.at(-1)?.get("offset")).toBe("25");
  await page.getByLabel("租户 ID").fill("tenant-filtered");
  await page.getByLabel("开始日期（UTC）").fill("2026-06-01");
  await page.getByLabel("结束日期（UTC）").fill("2026-06-30");
  await page.getByRole("button", { name: "应用筛选" }).click();
  await expect(page.getByText("tenant-filtered", { exact: true })).toBeVisible();
  await expect(usageTotals).toContainText("150");
  expect(usageRequests.at(-1)?.get("tenant_id")).toBe("tenant-filtered");
  expect(usageRequests.at(-1)?.get("start_date")).toBe("2026-06-01");
  expect(usageRequests.at(-1)?.get("end_date")).toBe("2026-06-30");
  expect(usageRequests.at(-1)?.get("offset")).toBeNull();

  await page.getByRole("button", { name: "公告管理" }).click();
  await page.getByLabel("公告标题").fill("系统维护");
  await page.getByLabel("公告内容").fill("今晚 23:00 进行例行维护。");
  await page.getByRole("button", { name: "发布公告" }).click();
  await expect(page.locator(".latest-announcement").getByText("系统维护")).toBeVisible();
  const announcementDialog = page.getByRole("dialog").filter({ hasText: "系统维护" });
  await expect(announcementDialog).toBeVisible();
  await expect(announcementDialog.getByText("今晚 23:00 进行例行维护。")).toBeVisible();
  await page.getByRole("button", { name: "关闭公告" }).click();
  await expect(announcementDialog).toBeHidden();
});

test("bulk redrives selected ingestion dead letters from the admin console", async ({ page }) => {
  const now = Date.now();
  let redrivePayload: unknown = null;
  let redriven = false;
  const adminUser = {
    id: "user-ingestion-admin",
    username: "ingestion-admin",
    display_name: "摄取管理员",
    role: "admin",
    tenant_id: "tenant-ingestion-admin",
    avatar_url: "",
    status: "active",
    created_at: now,
    last_login_at: now,
  };
  const deadLetters = [
    {
      tenant_id: "tenant-doc-a",
      task_id: "task-dead-a",
      title: "失败文档 A.pdf",
      source_type: "pdf",
      error: "模型服务连续失败",
      attempt_count: 3,
      dead_lettered_at: now - 20_000,
      updated_at: now - 20_000,
    },
    {
      tenant_id: "tenant-doc-b",
      task_id: "task-dead-b",
      title: "失败文档 B.txt",
      source_type: "txt",
      error: "向量服务不可用",
      attempt_count: 4,
      dead_lettered_at: now - 10_000,
      updated_at: now - 10_000,
    },
  ];

  await page.addInitScript(({ user, timestamp }) => {
    localStorage.setItem(
      "production-rag-auth-session",
      JSON.stringify({
        user,
        token: "session-ingestion-admin",
        expires_at: timestamp + 86_400_000,
      }),
    );
  }, { user: adminUser, timestamp: now });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/auth/me", async (route) => {
    await route.fulfill({ json: adminUser });
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
  await page.route("**/announcements?**", async (route) => {
    await route.fulfill({ json: { announcements: [] } });
  });
  await page.route("**/admin/settings", async (route) => {
    await route.fulfill({
      json: {
        registration_enabled: true,
        latest_announcement: null,
      },
    });
  });
  await page.route("**/admin/users?**", async (route) => {
    await route.fulfill({
      json: {
        users: [adminUser],
        total: 1,
        limit: 1,
        offset: 0,
        query: "",
      },
    });
  });
  await page.route("**/admin/ingestion/audit?**", async (route) => {
    await route.fulfill({
      json: {
        events: redriven
          ? [
              {
                id: "audit-queued",
                actor_user_id: adminUser.id,
                tenant_id: "tenant-doc-a",
                task_id: "task-dead-a",
                operation: "bulk_redrive",
                outcome: "queued",
                detail: "",
                created_at: now,
              },
              {
                id: "audit-not-retryable",
                actor_user_id: adminUser.id,
                tenant_id: "tenant-doc-b",
                task_id: "task-dead-b",
                operation: "bulk_redrive",
                outcome: "not_retryable",
                detail: "",
                created_at: now,
              },
            ]
          : [],
        total: redriven ? 2 : 0,
        limit: 20,
        offset: 0,
      },
    });
  });
  await page.route("**/admin/ingestion/dead-letters?**", async (route) => {
    const offset = Number(new URL(route.request().url()).searchParams.get("offset") || "0");
    const tasks = redriven
      ? [deadLetters[1]]
      : offset >= 20
        ? [
            {
              ...deadLetters[1],
              tenant_id: "tenant-doc-page-2",
              task_id: "task-page-2",
              title: "第二页死信文档.pdf",
            },
          ]
        : deadLetters;
    await route.fulfill({
      json: {
        tasks,
        total: redriven ? 1 : 21,
        limit: 20,
        offset,
      },
    });
  });
  await page.route("**/admin/ingestion/dead-letters/redrive", async (route) => {
    redrivePayload = route.request().postDataJSON();
    redriven = true;
    await route.fulfill({
      json: {
        results: [
          {
            tenant_id: "tenant-doc-a",
            task_id: "task-dead-a",
            outcome: "queued",
          },
          {
            tenant_id: "tenant-doc-b",
            task_id: "task-dead-b",
            outcome: "not_retryable",
          },
        ],
        queued: 1,
        rejected: 1,
      },
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /管理员控制台/ }).click();
  await page.getByRole("button", { name: "摄取运维" }).click();

  await expect(page.getByRole("heading", { name: "摄取运维" })).toBeVisible();
  await expect(page.getByText("失败文档 A.pdf")).toBeVisible();
  await expect(page.getByText("失败文档 B.txt")).toBeVisible();
  await page.getByRole("button", { name: "下一页" }).click();
  await expect(page.getByText("第二页死信文档.pdf")).toBeVisible();
  await page.getByRole("button", { name: "上一页" }).click();
  await expect(page.getByText("失败文档 A.pdf")).toBeVisible();

  const selectPage = page.getByRole("checkbox", { name: "选择当前页全部死信任务" });
  await selectPage.check();
  await expect(page.getByRole("button", { name: "重新处理所选（2）" })).toBeEnabled();
  await page.getByRole("button", { name: "重新处理所选（2）" }).click();

  await expect(page.getByRole("status")).toHaveText("已重新排队 1 个，1 个未处理。");
  expect(redrivePayload).toEqual({
    tasks: [
      { tenant_id: "tenant-doc-a", task_id: "task-dead-a" },
      { tenant_id: "tenant-doc-b", task_id: "task-dead-b" },
    ],
  });
  await expect(page.getByText("失败文档 A.pdf")).toBeHidden();
  await expect(page.getByText("失败文档 B.txt")).toBeVisible();
  await expect(page.getByText("已重新排队", { exact: true })).toBeVisible();
  await expect(page.getByText("状态不可重试", { exact: true })).toBeVisible();
  await expect(selectPage).not.toBeChecked();

  await page.setViewportSize({ width: 390, height: 844 });
  const panelBox = await page.locator(".admin-ingestion-panel").boundingBox();
  expect(panelBox).not.toBeNull();
  expect(panelBox!.x).toBeGreaterThanOrEqual(0);
  expect(panelBox!.x + panelBox!.width).toBeLessThanOrEqual(390);
});

test("clears the saved session after an authenticated request returns 401", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "production-rag-auth-session",
      JSON.stringify({
        user: {
          id: "expired-user",
          username: "expired",
          display_name: "过期用户",
          role: "user",
          tenant_id: "tenant-expired",
          created_at: Date.now(),
        },
        token: "expired-token",
        expires_at: Date.now() + 86_400_000,
      }),
    );
  });
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    if (route.request().headers().authorization === "Bearer expired-token") {
      await route.fulfill({ status: 401, json: { detail: "请先登录" } });
      return;
    }
    await route.fulfill({ json: { sources: [] } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations?**", async (route) => {
    await route.fulfill({ json: { conversations: [] } });
  });

  await page.goto("/");
  await expect
    .poll(async () => page.evaluate(() => localStorage.getItem("production-rag-auth-session")))
    .toBeNull();
  await page.getByRole("button", { name: "用户头像" }).click();
  await expect(page.getByRole("menuitem", { name: /登录/ })).toBeVisible();
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
  await page.route("**/auth/login", async (route) => {
    await route.fulfill({
      json: {
        user: {
          id: "test-user",
          username: "tester",
          display_name: "测试用户",
          role: "user",
          tenant_id: "team_a",
          avatar_url: "",
          status: "active",
          created_at: Date.now(),
          last_login_at: Date.now(),
        },
        token: "test-session",
        expires_at: Date.now() + 86_400_000,
      },
    });
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
  await page.route("**/conversations**", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: { conversations: [] } });
      return;
    }
    const body = route.request().postDataJSON();
    await route.fulfill({ json: { ...body, id: "conversation-typewriter", updated_at: Date.now() } });
  });
  await page.route("**/query**", async (route) => {
    await fulfillQueryStream(route, {
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
          metadata: { page_no: 1, page_start: 1, page_end: 1 },
          text: "实践案例证据片段第一句。这里是完整 chunk 的中段内容。这里是完整 chunk 的末尾内容。",
          text_preview: "实践案例证据片段",
        },
      ],
      trace: {},
    });
  });

  await page.goto("/");
  const chatTextarea = page.locator(".chat-input textarea");
  await expect(chatTextarea).toBeVisible();
  if ((await chatTextarea.getAttribute("placeholder")) === "登录后即可发送") {
    await page.getByRole("button", { name: "用户头像" }).click();
    await page.getByRole("menuitem", { name: /登录/ }).click();
    await page.getByRole("dialog").getByLabel("用户名").fill("tester");
    await page.getByRole("dialog").getByLabel("密码").fill("strong-password");
    await page.getByRole("button", { name: "登录" }).click();
    await expect(page.getByPlaceholder("提问或创作内容")).toBeVisible();
  }
  await page.locator('.chat-input textarea[placeholder="提问或创作内容"]').fill("典型的实践案例分析");
  await page.getByRole("button", { name: "发送消息" }).click();

  await expect(page.getByText("这是一段用于验证打字机效果")).toBeVisible();
  await expect(page.getByText(finalMarker)).toBeHidden();
  await expect(page.getByText(finalMarker)).toBeVisible({ timeout: 5000 });
  await expect(page.getByText("1. 自然辩证法.pdf · 第 1 页 · 重排分数 0.800")).toBeVisible();
  await page.getByText("1. 自然辩证法.pdf · 第 1 页 · 重排分数 0.800").click();
  await expect(page.getByText("这里是完整 chunk 的末尾内容。")).toBeVisible();
  await expect(page.getByText("chunk 0")).toBeHidden();

  const chatBox = await page.locator(".chat-panel").boundingBox();
  const userBox = await page.locator(".user-message").boundingBox();
  const assistantBox = await page.locator(".assistant-message").boundingBox();
  const inputBox = await page.locator(".chat-input").boundingBox();
  const textareaBox = await page.locator(".chat-input textarea").boundingBox();
  const sendBox = await page.getByRole("button", { name: "发送消息" }).boundingBox();
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

test("persists and resumes a pending answer after browser refresh", async ({ page }) => {
  let storedConversation: any = null;
  let queryCalls = 0;

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
            source_uri: "/uploads/natural.pdf",
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
  await page.route("**/conversations**", async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "GET" && url.pathname.endsWith("/conversations")) {
      await route.fulfill({
        json: {
          conversations: storedConversation
            ? [
                {
                  id: storedConversation.id,
                  tenant_id: storedConversation.tenant_id,
                  title: storedConversation.title,
                  message_count: storedConversation.messages.length,
                  source_doc_ids: storedConversation.source_doc_ids,
                  created_at: storedConversation.created_at,
                  updated_at: storedConversation.updated_at,
                },
              ]
            : [],
        },
      });
      return;
    }
    if (route.request().method() === "GET") {
      await route.fulfill({ json: storedConversation });
      return;
    }
    const body = route.request().postDataJSON();
    storedConversation = {
      ...body,
      id: body.id || "conv-resume",
      created_at: Date.now(),
      updated_at: Date.now(),
    };
    await route.fulfill({ json: storedConversation });
  });
  await page.route("**/query**", async (route) => {
    queryCalls += 1;
    if (queryCalls === 1) {
      await new Promise((resolve) => setTimeout(resolve, 2_000));
    }
    await fulfillQueryStream(route, {
      request_id: `resume-${queryCalls}`,
      answer: "刷新后继续完成的回答。",
      citations: [
        {
          doc_id: "自然辩证法/page-1",
          title: "自然辩证法 p1",
          source_uri: "/uploads/natural.pdf",
          source_type: "pdf",
          chunk_index: 0,
          score: 0.9,
          rerank_score: 0.8,
          acl_groups: ["engineering"],
          metadata: { page_no: 1 },
          text_preview: "证据片段",
        },
      ],
      trace: {},
    });
  });

  await page.goto("/");
  await page.getByPlaceholder("提问或创作内容").fill("刷新期间继续处理吗");
  await page.getByRole("button", { name: "发送消息" }).click();
  await expect
    .poll(() => storedConversation?.messages?.some((message: any) => message.status === "sending") ?? false)
    .toBe(true);

  await page.reload();

  await expect(page.getByText("刷新后继续完成的回答。")).toBeVisible({ timeout: 8_000 });
  expect(queryCalls).toBeGreaterThanOrEqual(2);
  expect(storedConversation.messages.at(-1).status).toBe("done");
  expect(storedConversation.messages.at(-1).content).toBe("刷新后继续完成的回答。");
});

test("drops an in-flight answer response after logout", async ({ page }) => {
  let queryResolve: (() => void) | null = null;
  const queryReleased = new Promise<void>((resolve) => {
    queryResolve = resolve;
  });

  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/auth/logout", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({
      json: {
        sources: [
          {
            doc_id: "natural",
            title: "自然辩证法.pdf",
            source_type: "pdf",
            source_uri: "/uploads/natural.pdf",
            doc_version: 1,
            chunk_count: 6,
            acl_groups: ["engineering"],
            status: "ready",
            current: true,
            child_doc_ids: ["natural/page-1"],
          },
        ],
      },
    });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/conversations**", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: { conversations: [] } });
      return;
    }
    const body = route.request().postDataJSON();
    await route.fulfill({ json: { ...body, id: "conv-logout-race", updated_at: Date.now() } });
  });
  await page.route("**/query**", async (route) => {
    await queryReleased;
    await fulfillQueryStream(route, {
      request_id: "logout-race",
      answer: "这段回答不应该在登出后的页面显示。",
      citations: [],
      trace: {},
    });
  });

  await page.goto("/");
  const queryStarted = page.waitForRequest((request) =>
    new URL(request.url()).pathname.endsWith("/query/stream"),
  );
  await page.getByPlaceholder("提问或创作内容").fill("总结当前文章");
  await page.getByRole("button", { name: "发送消息" }).click();
  await queryStarted;
  await page.getByRole("button", { name: "用户头像" }).click();
  await page.getByRole("menuitem", { name: /登出/ }).click();
  queryResolve?.();

  await expect(page.getByPlaceholder("登录后即可发送")).toBeVisible();
  await expect(page.getByText("总结当前文章")).toHaveCount(0);
  await expect(page.getByText("这段回答不应该在登出后的页面显示。")).toHaveCount(0);
});

test("persists assistant feedback rating after browser refresh", async ({ page }) => {
  let storedConversation: any = {
    id: "conv-feedback",
    tenant_id: "team_a",
    title: "反馈测试",
    source_doc_ids: ["自然辩证法"],
    created_at: Date.now(),
    updated_at: Date.now(),
    messages: [
      {
        id: "m-user",
        role: "user",
        content: "自然辩证法的引言",
        status: "done",
        request_id: null,
        citations: [],
        created_at: Date.now() - 2,
        feedback_rating: null,
      },
      {
        id: "m-assistant",
        role: "assistant",
        content: "引言讨论自然观和实践观。",
        status: "done",
        request_id: "feedback-request",
        citations: [
          {
            doc_id: "自然辩证法/page-1",
            title: "自然辩证法 p1",
            source_uri: "/uploads/natural.pdf",
            source_type: "pdf",
            chunk_index: 0,
            score: 0.9,
            rerank_score: 0.8,
            acl_groups: ["engineering"],
            metadata: {
              page_no: 1,
              display_blocks: [
                {
                  type: "image",
                  title: "历史引用图片",
                  url: (
                    "/source-assets/uploads/team_a/regression/history.png" +
                    "?tenant_id=team_a&token=legacy-history-token"
                  ),
                },
              ],
            },
            text_preview: "证据片段",
          },
        ],
        created_at: Date.now() - 1,
        feedback_rating: null,
      },
    ],
  };

  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/sources?**", async (route) => {
    await route.fulfill({ json: { sources: [] } });
  });
  await page.route("**/artifacts?**", async (route) => {
    await route.fulfill({ json: { artifacts: [] } });
  });
  await page.route("**/feedback", async (route) => {
    await route.fulfill({ json: { status: "accepted", request_id: "feedback-request" } });
  });
  await mockSourceAssetRoute(page);
  await page.route("**/conversations**", async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "GET" && url.pathname.endsWith("/conversations")) {
      await route.fulfill({
        json: {
          conversations: [
            {
              id: storedConversation.id,
              tenant_id: storedConversation.tenant_id,
              title: storedConversation.title,
              message_count: storedConversation.messages.length,
              source_doc_ids: storedConversation.source_doc_ids,
              created_at: storedConversation.created_at,
              updated_at: storedConversation.updated_at,
            },
          ],
        },
      });
      return;
    }
    if (route.request().method() === "GET") {
      await route.fulfill({ json: storedConversation });
      return;
    }
    const body = route.request().postDataJSON();
    storedConversation = { ...body, created_at: storedConversation.created_at, updated_at: Date.now() };
    await route.fulfill({ json: storedConversation });
  });

  await page.goto("/");
  const historicalImage = page.getByRole("img", { name: "历史引用图片" });
  await expect(historicalImage).toHaveAttribute("src", /^blob:/);
  await page.getByRole("button").filter({ has: page.locator("svg.lucide-thumbs-up") }).click();
  await expect
    .poll(() => storedConversation.messages[1].feedback_rating)
    .toBe(1);

  await page.reload();

  const likedButton = page.getByRole("button").filter({ has: page.locator("svg.lucide-thumbs-up") });
  await expect(likedButton.locator("svg")).toHaveAttribute("fill", "currentColor");
});
