import type { MindMapNode } from "../../lib/types";

export type MindMapLayoutNode = {
  id: string;
  parentId: string | null;
  kind: "root" | "branch" | "leaf";
  label: string;
  x: number;
  y: number;
  estimatedHeight: number;
  canExpand: boolean;
  expanded: boolean;
  toneIndex: number;
};

const ROOT_X = 0;
const BRANCH_X = 330;
const LEAF_X = 690;
const NODE_VERTICAL_PADDING_PX = 24;
const NODE_LINE_HEIGHT_PX = 20;
const NODE_MIN_HEIGHT_PX = 44;
const LABEL_UNITS_PER_LINE = 13;
const LEAF_GAP_PX = 18;
const BRANCH_GROUP_GAP_PX = 30;

export function buildMindMapLayout(
  root: MindMapNode,
  expandedNodeIds: ReadonlySet<string>,
): MindMapLayoutNode[] {
  const output: MindMapLayoutNode[] = [];
  const branches = root.children || [];
  let cursorY = 0;

  for (const [branchIndex, branch] of branches.entries()) {
    const expanded = expandedNodeIds.has(branch.id);
    const children = expanded ? (branch.children || []) : [];
    const branchLabel = `${branch.label}${branch.children?.length ? ` ${expanded ? "⌄" : "›"}` : ""}`;
    const branchHeight = estimateMindMapNodeHeight(branchLabel);
    const childHeights = children.map((child) => estimateMindMapNodeHeight(child.label));
    const childStackHeight =
      childHeights.reduce((total, height) => total + height, 0)
      + Math.max(0, childHeights.length - 1) * LEAF_GAP_PX;
    const groupHeight = Math.max(branchHeight, childStackHeight);
    const branchY = cursorY + (groupHeight - branchHeight) / 2;

    output.push({
      id: branch.id,
      parentId: root.id,
      kind: "branch",
      label: branchLabel,
      x: BRANCH_X,
      y: branchY,
      estimatedHeight: branchHeight,
      canExpand: Boolean(branch.children?.length),
      expanded,
      toneIndex: branchIndex % 6,
    });

    let childY = cursorY;
    children.forEach((child, childIndex) => {
      const childHeight = childHeights[childIndex];
      output.push({
        id: child.id,
        parentId: branch.id,
        kind: "leaf",
        label: child.label,
        x: LEAF_X,
        y: childY,
        estimatedHeight: childHeight,
        canExpand: false,
        expanded: false,
        toneIndex: childIndex % 6,
      });
      childY += childHeight + LEAF_GAP_PX;
    });

    cursorY += groupHeight + BRANCH_GROUP_GAP_PX;
  }

  const totalHeight = Math.max(0, cursorY - (branches.length ? BRANCH_GROUP_GAP_PX : 0));
  const rootHeight = estimateMindMapNodeHeight(root.label);
  output.unshift({
    id: root.id,
    parentId: null,
    kind: "root",
    label: root.label,
    x: ROOT_X,
    y: Math.max(0, (totalHeight - rootHeight) / 2),
    estimatedHeight: rootHeight,
    canExpand: false,
    expanded: false,
    toneIndex: 0,
  });
  return output;
}

export function estimateMindMapNodeHeight(label: string): number {
  const visualLines = String(label || "")
    .split(/\r?\n/)
    .reduce((total, line) => {
      const units = [...line].reduce(
        (lineUnits, character) => lineUnits + (isWideCharacter(character) ? 1 : 0.5),
        0,
      );
      return total + Math.max(1, Math.ceil(units / LABEL_UNITS_PER_LINE));
    }, 0);
  return Math.max(
    NODE_MIN_HEIGHT_PX,
    NODE_VERTICAL_PADDING_PX + Math.max(1, visualLines) * NODE_LINE_HEIGHT_PX,
  );
}

function isWideCharacter(character: string) {
  return /[\u1100-\u115f\u2e80-\ua4cf\uac00-\ud7a3\uf900-\ufaff\ufe10-\ufe6f\uff00-\uffef]/u.test(character);
}
