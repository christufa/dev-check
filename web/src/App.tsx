import { Suspense, useEffect } from "react";
import { Link, NavLink, Route, Routes, useLocation, useParams } from "react-router";
import { AREAS, LESSONS, areaLabel, lessonsIn } from "./lessons";

const REPO_URL = "https://github.com/christufa/dev-check";

export function App() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="shell">
      <header className="masthead">
        <Link to="/" className="brand">
          <span className="brand-mark">∂</span> Dev Check
        </Link>
        <span className="brand-sub">theory notebook</span>
      </header>
      <nav className="sidebar" aria-label="Lessons">
        {AREAS.map((area) => {
          const lessons = lessonsIn(area.id);
          return (
            <section key={area.id}>
              <h2>{area.label}</h2>
              {lessons.length === 0 ? (
                <p className="empty">Nothing here yet</p>
              ) : (
                <ul>
                  {lessons.map((l) => (
                    <li key={l.path}>
                      <NavLink to={l.path}>{l.title}</NavLink>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          );
        })}
      </nav>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/:area/:slug" element={<LessonPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <footer className="footer">
        Notes and code live in <a href={REPO_URL}>christufa/dev-check</a>.
      </footer>
    </div>
  );
}

function Home() {
  return (
    <article className="prose">
      <h1>Dev Check</h1>
      <p className="lede">
        Working notes for building theoretical understanding of probability, advanced statistics, physics and
        deep learning. Proofs and derivations come first, and each idea is checked with code that runs in your
        browser.
      </p>
      {AREAS.filter((a) => lessonsIn(a.id).length > 0).map((area) => (
        <section key={area.id}>
          <h2>{area.label}</h2>
          <div className="cards">
            {lessonsIn(area.id).map((l) => (
              <Link key={l.path} to={l.path} className="card">
                <span className="card-title">{l.title}</span>
                <span className="card-summary">{l.summary}</span>
                {l.source && <span className="card-source">{l.source}</span>}
              </Link>
            ))}
          </div>
        </section>
      ))}
    </article>
  );
}

function LessonPage() {
  const { area, slug } = useParams();
  const lesson = LESSONS.find((l) => l.area === area && l.slug === slug);
  useEffect(() => {
    document.title = lesson ? `${lesson.title} · Dev Check` : "Dev Check · Theory";
  }, [lesson]);
  if (!lesson) return <NotFound />;

  const siblings = lessonsIn(lesson.area);
  const i = siblings.indexOf(lesson);
  const prev = siblings[i - 1];
  const next = siblings[i + 1];
  const { Component } = lesson;

  return (
    <article className="prose">
      <p className="eyebrow">
        {areaLabel(lesson.area)}
        {lesson.source && <> · {lesson.source}</>}
      </p>
      <Suspense fallback={<p className="loading">Loading…</p>}>
        <Component />
      </Suspense>
      {(prev || next) && (
        <nav className="pager" aria-label="More in this area">
          {prev ? <Link to={prev.path}>← {prev.title}</Link> : <span />}
          {next && <Link to={next.path}>{next.title} →</Link>}
        </nav>
      )}
    </article>
  );
}

function NotFound() {
  return (
    <article className="prose">
      <h1>Not found</h1>
      <p>
        There's no page at this address. <Link to="/">Back to the index</Link>.
      </p>
    </article>
  );
}
