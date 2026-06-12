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
            source_uri: "/object_store/uploads/team_a/demo/自然辩证法.pdf",
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
        source_uri: "/object_store/uploads/team_a/demo/自然辩证法.pdf",
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
