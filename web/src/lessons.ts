import { lazy, type ComponentType } from "react";

export interface LessonMeta {
  title: string;
  summary: string;
  /** Position within its area; lower comes first. */
  order: number;
  /** Book, paper or notebook the lesson follows, if any. */
  source?: string;
}

export interface Lesson extends LessonMeta {
  area: string;
  slug: string;
  path: string;
  Component: ComponentType;
}

/** Areas in sidebar order. A lesson lives at content/<area>/<slug>.mdx. */
export const AREAS: { id: string; label: string }[] = [
  { id: "probability", label: "Probability" },
  { id: "statistics", label: "Statistics" },
  { id: "deep-learning", label: "Deep learning" },
  { id: "physics", label: "Physics" },
];

// Metadata is bundled up front for navigation; each lesson's body is its own chunk, loaded on first visit.
const metas = import.meta.glob<LessonMeta>("./content/*/*.mdx", { eager: true, import: "meta", query: "?meta" });
const bodies = import.meta.glob<{ default: ComponentType }>("./content/*/*.mdx");

export const LESSONS: Lesson[] = Object.entries(metas)
  .map(([file, meta]) => {
    const [, area, slug] = file.match(/^\.\/content\/([^/]+)\/([^/]+)\.mdx$/)!;
    return { ...meta, area, slug, path: `/${area}/${slug}`, Component: lazy(bodies[file]) };
  })
  .sort((a, b) => a.order - b.order);

export function lessonsIn(area: string): Lesson[] {
  return LESSONS.filter((l) => l.area === area);
}

export function areaLabel(area: string): string {
  return AREAS.find((a) => a.id === area)?.label ?? area;
}
