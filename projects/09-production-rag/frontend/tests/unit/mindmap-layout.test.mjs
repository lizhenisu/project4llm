import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMindMapLayout,
  estimateMindMapNodeHeight,
} from "../../src/components/studio/mindMapLayout.ts";


const root = {
  id: "root",
  label: "生产级 RAG 系统",
  children: [
    {
      id: "retrieval",
      label: "检索链路",
      children: [
        {
          id: "retrieval-1",
          label: "这是一个非常长的第三级节点，包含查询重写、混合向量检索、关键词召回以及跨编码器重排序等多个步骤。",
        },
        {
          id: "retrieval-2",
          label: "第二个长节点包含上下文预算、每文档片段上限和最低相关性阈值，文本换行后高度明显增加。",
        },
        {
          id: "retrieval-3",
          label: "short leaf",
        },
      ],
    },
    {
      id: "ingestion",
      label: "文档摄取与索引",
      children: [
        {
          id: "ingestion-1",
          label: "解析 PDF、抽取图片、生成文本与图片向量，并将版本化片段批量写入 Milvus。",
        },
        {
          id: "ingestion-2",
          label: "失败任务采用指数退避重试，长期 processing 任务由恢复流程重新排队。",
        },
      ],
    },
    {
      id: "monitoring",
      label: "监控",
      children: [{ id: "monitoring-1", label: "Prometheus 与 Grafana" }],
    },
  ],
};


test("expanded long-label leaf nodes occupy non-overlapping vertical ranges", () => {
  const layout = buildMindMapLayout(root, new Set(["retrieval", "ingestion", "monitoring"]));

  assertNoOverlap(layout.filter((node) => node.kind === "leaf"));
  assertNoOverlap(layout.filter((node) => node.kind === "branch"));
  assert.ok(
    layout.find((node) => node.id === "retrieval-1").estimatedHeight
      > layout.find((node) => node.id === "retrieval-3").estimatedHeight,
  );
});


test("expanding a tall branch pushes later branches down by its measured demand", () => {
  const collapsed = buildMindMapLayout(root, new Set());
  const expanded = buildMindMapLayout(root, new Set(["retrieval"]));
  const collapsedIngestionY = collapsed.find((node) => node.id === "ingestion").y;
  const expandedIngestionY = expanded.find((node) => node.id === "ingestion").y;

  assert.ok(expandedIngestionY > collapsedIngestionY + 100);
  assert.equal(expanded.filter((node) => node.kind === "leaf").length, 3);
});


test("height estimation accounts for explicit lines, CJK width, and long ASCII labels", () => {
  const shortHeight = estimateMindMapNodeHeight("短标签");
  assert.ok(estimateMindMapNodeHeight("第一行\n第二行\n第三行") > shortHeight);
  assert.ok(estimateMindMapNodeHeight("这是一段足以换成很多行的中文节点说明文字".repeat(4)) > shortHeight);
  assert.ok(estimateMindMapNodeHeight("long-unbroken-ascii-token-".repeat(10)) > shortHeight);
});


function assertNoOverlap(nodes) {
  const sorted = [...nodes].sort((left, right) => left.y - right.y);
  for (let index = 1; index < sorted.length; index += 1) {
    const previous = sorted[index - 1];
    const current = sorted[index];
    assert.ok(
      current.y >= previous.y + previous.estimatedHeight,
      `${previous.id} overlaps ${current.id}`,
    );
  }
}
