# Claude Code Handoff — leeje008.github.io Redesign

이 문서는 `leeje008.github.io` Jekyll 블로그(Chirpy 테마 기반)에 새 디자인을 적용하기 위한 **구체적이고 실행 가능한** 지시서입니다. 디자인 캔버스(`index.html`)에서 본 5개 화면을 그대로 구현하면 됩니다.

사용법:
```bash
cd leeje008.github.io
claude code

# 그리고 다음과 같이 입력:
> CLAUDE_CODE_HANDOFF.md 의 내용대로 블로그 디자인을 단계별로 적용해줘.
> 각 단계가 끝나면 변경 사항을 요약해주고, 내가 확인 후 다음 단계로 넘어가자.
```

---

## 0. 설계 원칙 (Design Principles)

이 리디자인은 **modern editorial / academic-minimal** 미감을 따른다. 화려한 그라데이션·이모지·과한 컬러는 모두 배제하고, 타입 위계와 여백으로 정보를 정리한다.

- **타입**: 한글 본문 가독성을 위해 **Pretendard** 도입, 라틴은 **Inter**, 코드는 **JetBrains Mono**.
- **컬러**: 따뜻한 뉴트럴 톤 (`#FAFAF7` / `#0E0E10`). 액센트는 단일 인디고 `oklch(58% 0.13 252)` (다크 `oklch(72% 0.14 252)`)만 사용. 카테고리 식별용 점(dot)에만 다른 hue 허용.
- **간격**: 본문은 최대 70ch, 행간 1.75. 섹션 간 48–60px의 호흡.
- **모노스페이스 디테일**: 메타 정보(날짜, 메타라벨, 태그, eyebrow)는 모두 mono로. 본문은 sans.
- **이모지·SVG 인포그래픽 없음**. 아이콘은 Chirpy가 이미 쓰는 Font Awesome 그대로 활용.

---

## 1. 디자인 토큰 교체 — `_sass/colors/light-typography.scss`

기존 변수의 **값만** 다음으로 교체. 변수 이름은 그대로 유지 (Chirpy 다른 파일들이 참조).

```scss
@mixin light-scheme {
  /* Framework color */
  --body-bg: #FAFAF7;
  --mask-bg: #c1c3c5;
  --main-wrapper-bg: #FAFAF7;
  --main-border-color: #EFEEE9;

  /* Common color */
  --text-color: #2B2B30;
  --text-muted-color: #5C5C66;
  --heading-color: #111113;
  --blockquote-border-color: oklch(58% 0.13 252 / 0.4);
  --blockquote-text-color: #2B2B30;
  --link-color: oklch(58% 0.13 252);
  --link-underline-color: oklch(58% 0.13 252 / 0.3);
  --button-bg: #FFFFFF;
  --btn-border-color: #E8E6E0;
  --btn-backtotop-color: #5C5C66;
  --btn-backtotop-border-color: #E8E6E0;
  --btn-box-shadow: rgba(0,0,0,0.04);
  --checkbox-color: #c5c5c5;
  --checkbox-checked-color: oklch(58% 0.13 252);

  /* Sidebar */
  --sidebar-bg: #FAFAF7;
  --sidebar-muted-color: #8B8B94;
  --sidebar-active-color: #111113;
  --nav-cursor-color: oklch(58% 0.13 252);
  --sidebar-btn-bg: #FFFFFF;

  /* Topbar */
  --topbar-text-color: #2B2B30;
  --topbar-wrapper-bg: #FAFAF7;
  --search-wrapper-bg: #F2F1EC;
  --search-wrapper-border-color: #E8E6E0;
  --search-tag-bg: #F2F1EC;
  --search-icon-color: #8B8B94;
  --input-focus-border-color: oklch(58% 0.13 252);

  /* Home page */
  --post-list-text-color: #5C5C66;
  --btn-patinator-text-color: #5C5C66;
  --btn-paginator-hover-color: #F2F1EC;
  --btn-paginator-border-color: #E8E6E0;
  --btn-text-color: #2B2B30;
  --pin-bg: oklch(58% 0.13 252 / 0.10);
  --pin-color: oklch(58% 0.13 252);

  /* Posts */
  --btn-share-hover-color: oklch(58% 0.13 252);
  --card-border-color: #E8E6E0;
  --card-box-shadow: rgba(0,0,0,0.03);
  --label-color: #5C5C66;
  --relate-post-date: #8B8B94;
  --footnote-target-bg: oklch(58% 0.13 252 / 0.10);
  --tag-bg: #FFFFFF;
  --tag-border: #E8E6E0;
  --tag-shadow: rgba(0,0,0,0.02);
  --tag-hover: #F2F1EC;
  --tb-odd-bg: #F2F1EC;
  --tb-border-color: #E8E6E0;
  --dash-color: #8B8B94;
  --preview-img-bg: #F2F1EC;
  --kbd-wrap-color: #8B8B94;
  --kbd-text-color: #111113;
  --kbd-bg-color: #FFFFFF;
  --prompt-text-color: #2B2B30;
  --prompt-tip-bg: oklch(65% 0.13 145 / 0.10);
  --prompt-tip-icon-color: oklch(55% 0.13 145);
  --prompt-info-bg: oklch(58% 0.13 252 / 0.10);
  --prompt-info-icon-color: oklch(58% 0.13 252);
  --prompt-warning-bg: oklch(75% 0.13 75 / 0.15);
  --prompt-warning-icon-color: oklch(60% 0.13 75);
  --prompt-danger-bg: oklch(60% 0.15 25 / 0.10);
  --prompt-danger-icon-color: oklch(55% 0.15 25);

  /* Categories */
  --categories-hover-bg: #F2F1EC;
  --categories-icon-hover-color: #111113;

  /* Archive */
  --timeline-color: #E8E6E0;
  --timeline-node-bg: #8B8B94;
  --timeline-year-dot-color: #FFFFFF;
}
```

---

## 2. 다크 모드 토큰 — `_sass/colors/dark-typography.scss`

같은 방식으로 다크 mixin 내부 변수 값 교체:

```scss
@mixin dark-scheme {
  --body-bg: #0E0E10;
  --mask-bg: #232328;
  --main-wrapper-bg: #0E0E10;
  --main-border-color: #1B1B1F;

  --text-color: #D9D8D2;
  --text-muted-color: #9A9A9F;
  --heading-color: #F4F3EE;
  --blockquote-border-color: oklch(72% 0.14 252 / 0.45);
  --blockquote-text-color: #D9D8D2;
  --link-color: oklch(72% 0.14 252);
  --link-underline-color: oklch(72% 0.14 252 / 0.4);
  --button-bg: #16161A;
  --btn-border-color: #232328;

  --sidebar-bg: #0E0E10;
  --sidebar-muted-color: #62626A;
  --sidebar-active-color: #F4F3EE;
  --nav-cursor-color: oklch(72% 0.14 252);
  --sidebar-btn-bg: #16161A;

  --topbar-text-color: #D9D8D2;
  --topbar-wrapper-bg: #0E0E10;
  --search-wrapper-bg: #16161A;
  --search-wrapper-border-color: #232328;

  --post-list-text-color: #9A9A9F;
  --card-border-color: #232328;
  --card-box-shadow: rgba(0,0,0,0.4);
  --pin-bg: oklch(72% 0.14 252 / 0.15);
  --pin-color: oklch(82% 0.14 252);

  --tag-bg: #16161A;
  --tag-border: #232328;
  --tag-hover: #1B1B1F;
  --tb-odd-bg: #16161A;
  --tb-border-color: #232328;
  --kbd-bg-color: #16161A;
  --kbd-text-color: #F4F3EE;

  /* …나머지 변수는 위와 동일한 톤으로 매핑 */
}
```

> **주의**: 위에 명시되지 않은 변수는 기존 다크 테마 값을 유지하되, 채도(saturation)를 줄이고 ink는 `#F4F3EE`, 모든 회색은 위에 정의된 ink-2/ink-3/ink-4 단계로 정렬할 것.

---

## 3. 폰트 시스템 교체

### 3-1. `_sass/addon/variables.scss`

```scss
/* font family */
$font-family-base: "Pretendard", "Inter", -apple-system, BlinkMacSystemFont,
                   "Apple SD Gothic Neo", "Helvetica Neue", sans-serif !default;
$font-family-heading: $font-family-base !default;
$font-family-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas,
                   "Courier New", monospace !default;
```

### 3-2. `_includes/head.html` — `<head>` 끝에 추가

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
```

### 3-3. 본문 행간·자간 (`_sass/addon/commons.scss` 의 `body` 규칙)

```scss
body {
  font-family: $font-family-base;
  font-feature-settings: "ss01", "cv11";
  -webkit-font-smoothing: antialiased;
  letter-spacing: -0.005em;
  line-height: 1.7;
}
```

---

## 4. 홈 레이아웃 — `_layouts/home.html`

기존 페이지 상단에 **profile hero + pinned 카드 행**을 추가하고, 포스트 리스트를 grid-row 형태로 재구성한다.

`_layouts/home.html` 전체를 다음으로 교체:

```liquid
---
layout: page
---

{% include lang.html %}

{% if paginator.page == 1 %}
<section class="home-hero">
  <img src="{{ site.avatar | relative_url }}" alt="profile" class="home-hero__avatar" />
  <div class="home-hero__body">
    <h1 class="home-hero__name">
      {{ site.social.name }}
      {% if site.lang == 'ko' %}<span class="han">고영헌</span>{% endif %}
    </h1>
    <p class="home-hero__role">{{ site.tagline }}</p>
    <p class="home-hero__desc">{{ site.description }}</p>
    <div class="home-hero__stack">
      <span class="stack-pill primary">Python</span>
      <span class="stack-pill primary">PyTorch</span>
      <span class="stack-pill">LangGraph</span>
      <span class="stack-pill">FastAPI</span>
      <span class="stack-pill">PostgreSQL</span>
      <span class="stack-pill">Airflow</span>
      <span class="stack-pill">scikit-learn</span>
    </div>
  </div>
</section>

{% assign pinned_posts = site.posts | where: "pin", "true" %}
{% if pinned_posts.size > 0 %}
<section class="home-pinned">
  <div class="section-head">
    <h2>// Pinned</h2>
    <span class="count">{{ pinned_posts.size | prepend: '0' | slice: -2, 2 }}</span>
  </div>
  <div class="pinned-grid">
    {% for post in pinned_posts limit: 3 %}
    <a href="{{ post.url | relative_url }}" class="pin-card {% if forloop.first %}primary{% endif %}">
      <div class="pin-meta">
        <span class="pin-badge">Pinned</span>
        <span>{{ post.date | date: "%Y.%m.%d" }}</span>
      </div>
      <h3 class="pin-title">{{ post.title }}</h3>
      <p class="pin-snippet">{{ post.content | strip_html | truncate: 100 }}</p>
      <div class="pin-foot">
        {% for category in post.categories limit: 1 %}
          <span class="chip">{{ category }}</span>
        {% endfor %}
      </div>
    </a>
    {% endfor %}
  </div>
</section>
{% endif %}
{% endif %}

<section class="home-list">
  <div class="section-head">
    <h2>// Recent Posts</h2>
    <span class="count">{{ site.posts.size }} posts</span>
  </div>

  {% assign current_year = "" %}
  {% for post in paginator.posts %}
    {% assign post_year = post.date | date: "%Y" %}
    {% if post_year != current_year %}
      {% unless forloop.first %}</div>{% endunless %}
      <div class="year-label">{{ post_year }}</div>
      <div class="year-group">
      {% assign current_year = post_year %}
    {% endif %}

    <article class="post-row">
      <div class="post-row__date">{{ post.date | date: "%b %d" }}</div>
      <div class="post-row__body">
        <h3 class="post-row__title">
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h3>
        <p class="post-row__snippet">{{ post.content | strip_html | truncate: 120 }}</p>
        <div class="post-row__meta">
          {% for category in post.categories limit: 1 %}
            <span class="chip">{{ category }}</span>
          {% endfor %}
          {% for tag in post.tags limit: 3 %}
            <span class="tag">#{{ tag }}</span>
          {% endfor %}
        </div>
      </div>
      <div class="post-row__read">
        {% include read-time.html content=post.content %}
      </div>
    </article>
  {% endfor %}
  {% if paginator.posts.size > 0 %}</div>{% endif %}
</section>

{% if paginator.total_pages > 0 %}
  {% include post-paginator.html %}
{% endif %}
```

---

## 5. 홈 스타일 — `_sass/layout/home.scss`

기존 `#post-list` 블록을 다음 신규 블록으로 교체:

```scss
/* ---- Home hero ---- */
.home-hero {
  display: grid;
  grid-template-columns: 96px 1fr;
  gap: 24px;
  padding: 32px 0 32px;
  border-bottom: 1px solid var(--main-border-color);
  margin-bottom: 40px;

  &__avatar {
    width: 96px; height: 96px;
    border-radius: 22px;
    object-fit: cover;
    border: 1px solid var(--card-border-color);
  }
  &__name {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0 0 6px;
    color: var(--heading-color);

    .han {
      color: var(--text-muted-color);
      font-weight: 500;
      margin-left: 8px;
      font-size: 18px;
    }
  }
  &__role {
    font-size: 14.5px;
    color: var(--text-color);
    margin: 0 0 6px;
  }
  &__desc {
    font-size: 13.5px;
    color: var(--text-muted-color);
    margin: 0 0 14px;
    line-height: 1.55;
    max-width: 60ch;
  }
  &__stack {
    display: flex; flex-wrap: wrap; gap: 6px;
  }
}

.stack-pill {
  font-family: $font-family-mono;
  font-size: 11.5px;
  padding: 3px 8px;
  border-radius: 4px;
  background: var(--tb-odd-bg);
  color: var(--text-color);

  &.primary { background: var(--pin-bg); color: var(--pin-color); }
}

/* ---- Section heads ---- */
.section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 18px;

  h2 {
    font-family: $font-family-mono;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-color);
    margin: 0;
  }
  .count {
    font-family: $font-family-mono;
    font-size: 11px;
    color: var(--text-muted-color);
  }
}

/* ---- Pinned cards ---- */
.home-pinned { margin-bottom: 48px; }
.pinned-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr;
  gap: 14px;
}
.pin-card {
  padding: 18px;
  background: var(--button-bg);
  border: 1px solid var(--card-border-color);
  border-radius: 14px;
  display: flex; flex-direction: column;
  gap: 12px;
  min-height: 180px;
  text-decoration: none !important;
  color: inherit;
  transition: transform 0.15s, border-color 0.15s;

  &:hover {
    transform: translateY(-1px);
    border-color: var(--text-muted-color);
  }

  &.primary {
    background: var(--heading-color);
    color: var(--body-bg);
    border-color: var(--heading-color);
    min-height: 200px;

    .pin-title, .pin-snippet { color: inherit; }
    .pin-meta, .pin-foot .chip { color: rgba(255,255,255,0.6); }
  }
}
.pin-meta {
  font-family: $font-family-mono;
  font-size: 10.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted-color);
  display: flex; gap: 8px; align-items: center;
}
.pin-badge {
  padding: 2px 7px;
  background: var(--pin-bg);
  color: var(--pin-color);
  border-radius: 4px;
  letter-spacing: 0.08em;
}
.pin-title {
  font-size: 17px;
  font-weight: 600;
  line-height: 1.35;
  letter-spacing: -0.02em;
  color: var(--heading-color);
  margin: 0;
}
.pin-card.primary .pin-title { font-size: 22px; }
.pin-snippet {
  font-size: 13px;
  line-height: 1.55;
  color: var(--text-muted-color);
  margin: 0;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  display: -webkit-box;
  overflow: hidden;
}
.pin-foot { margin-top: auto; }

/* ---- Posts list (year-grouped) ---- */
.home-list { margin-bottom: 48px; }
.year-label {
  font-family: $font-family-mono;
  font-size: 11px;
  letter-spacing: 0.1em;
  color: var(--text-muted-color);
  margin: 32px 0 12px;
  display: flex; align-items: center; gap: 12px;

  &::after {
    content: ""; flex: 1; height: 1px;
    background: var(--main-border-color);
  }
}
.post-row {
  display: grid;
  grid-template-columns: 80px 1fr 70px;
  gap: 20px;
  padding: 18px 0;
  border-bottom: 1px solid var(--main-border-color);
  align-items: baseline;

  &__date {
    font-family: $font-family-mono;
    font-size: 11.5px;
    color: var(--text-muted-color);
    white-space: nowrap;
    padding-top: 2px;
  }
  &__title {
    font-size: 15.5px;
    font-weight: 500;
    line-height: 1.45;
    letter-spacing: -0.015em;
    margin: 0 0 6px;
    a {
      color: var(--heading-color);
      text-decoration: none !important;
      &:hover { color: var(--link-color); }
    }
  }
  &__snippet {
    font-size: 13px;
    color: var(--text-muted-color);
    line-height: 1.55;
    margin: 0 0 8px;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    display: -webkit-box;
    overflow: hidden;
  }
  &__meta {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  }
  &__read {
    font-family: $font-family-mono;
    font-size: 11px;
    color: var(--text-muted-color);
    text-align: right;
    white-space: nowrap;
  }
}

/* ---- Chip & tag (재정의) ---- */
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--tb-odd-bg);
  color: var(--text-color);
  font-weight: 500;
  border: 1px solid var(--main-border-color);

  &::before {
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--link-color);
  }
}
.tag {
  font-family: $font-family-mono;
  font-size: 11.5px;
  color: var(--text-muted-color);
  padding: 2px 8px;
  border: 1px solid var(--main-border-color);
  border-radius: 4px;
}

/* 모바일 */
@media (max-width: 830px) {
  .home-hero { grid-template-columns: 64px 1fr; gap: 16px; }
  .home-hero__avatar { width: 64px; height: 64px; border-radius: 16px; }
  .home-hero__name { font-size: 22px; }
  .pinned-grid { grid-template-columns: 1fr; }
  .post-row { grid-template-columns: 1fr; gap: 8px; }
  .post-row__read { text-align: left; }
}
```

---

## 6. 포스트 디테일 — `_sass/layout/post.scss`

기존 파일에 다음을 **추가** (덮어쓰지 말고 append):

```scss
/* ---- Refined post head ---- */
h1[data-toc-skip] {
  font-size: 36px;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.2;
  margin: 8px 0 18px;
  color: var(--heading-color);
  text-wrap: balance;
}

.post-meta {
  font-family: $font-family-mono;
  font-size: 12px;
  color: var(--text-muted-color);

  em { font-style: normal; color: var(--text-color); }
}

/* ---- Body typography ---- */
.post-content {
  font-size: 15.5px;
  line-height: 1.78;
  color: var(--text-color);

  h2 {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 56px 0 16px;
    color: var(--heading-color);
    scroll-margin-top: 80px;
  }
  h3 {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.015em;
    margin: 40px 0 12px;
    color: var(--heading-color);
  }
  p { margin: 0 0 20px; text-wrap: pretty; }
  strong { color: var(--heading-color); font-weight: 600; }

  /* inline code */
  code {
    font-family: $font-family-mono;
    font-size: 0.88em;
    background: var(--tb-odd-bg);
    padding: 1px 6px;
    border-radius: 4px;
    color: var(--heading-color);
  }

  /* code block (rouge) */
  div.highlighter-rouge, figure.highlight {
    background: var(--tb-odd-bg) !important;
    border: 1px solid var(--main-border-color);
    border-radius: 10px;
    padding: 4px 0;
    margin: 0 0 24px;
    overflow: hidden;
  }
  pre {
    background: transparent !important;
    padding: 14px 20px !important;
    font-family: $font-family-mono;
    font-size: 13px;
    line-height: 1.6;
  }

  /* blockquote */
  blockquote {
    margin: 24px 0;
    padding: 4px 0 4px 20px;
    border-left: 2px solid var(--link-color);
    color: var(--text-color);
    font-style: normal;
    p { margin: 0; }
  }

  /* table */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin: 0 0 28px;

    th, td {
      text-align: left;
      padding: 11px 14px;
      border-bottom: 1px solid var(--main-border-color);
    }
    th {
      font-family: $font-family-mono;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-muted-color);
      font-weight: 500;
      background: var(--tb-odd-bg);
    }
  }
}

/* ---- TOC sticky improvement ---- */
#toc-wrapper {
  position: sticky;
  top: 32px;

  .toc-title {
    font-family: $font-family-mono;
    font-size: 10px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted-color) !important;
    margin-bottom: 14px !important;
  }
  nav[data-toggle="toc"] ul {
    border-left: 1px solid var(--main-border-color);

    a.nav-link {
      font-size: 12.5px !important;
      padding: 6px 14px !important;
      color: var(--text-muted-color) !important;
      border-left: 1px solid transparent !important;
      margin-left: -1px;

      &.active {
        color: var(--heading-color) !important;
        font-weight: 500;
        border-left: 1px solid var(--link-color) !important;
      }
    }
  }
}

/* ---- Post nav prev/next as cards ---- */
.post-navigation {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin: 48px 0;

  .btn {
    padding: 18px 20px !important;
    border: 1px solid var(--card-border-color) !important;
    border-radius: 12px !important;
    background: var(--button-bg) !important;
    color: var(--heading-color) !important;
    box-shadow: none !important;

    &:hover { border-color: var(--text-muted-color) !important; }
  }
}

/* related posts */
#related-posts h3 {
  font-family: $font-family-mono;
  font-size: 11px !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted-color);
  margin-bottom: 16px;
}
#related-posts .card {
  border: 1px solid var(--card-border-color) !important;
  border-radius: 12px !important;
  background: var(--button-bg) !important;
  box-shadow: none !important;
}
```

---

## 7. About 페이지 — `_tabs/about.md`

content를 다음 HTML로 교체 (Markdown 위에 HTML 블록):

```markdown
---
title: About
icon: fas fa-info-circle
order: 4
hide_description: true
---

<div class="about-hero">
  <img src="{{ '/assets/img/me.jpg' | relative_url }}" class="about-hero__avatar" alt="profile" />
  <div class="about-hero__body">
    <div class="eyebrow">About</div>
    <h1>고영헌 <span class="latin">Younghun Ko</span></h1>
    <p class="about-hero__tagline">데이터 사이언티스트 & LLM 에이전트 개발자</p>
    <div class="about-hero__actions">
      <a class="btn-primary" href="mailto:leeje008@naver.com">leeje008@naver.com</a>
      <a class="btn-secondary" href="https://github.com/leeje008">github.com/leeje008</a>
    </div>
  </div>
  <div class="about-stats">
    <div class="stat"><div class="stat__num">{{ site.posts.size }}</div><div class="stat__label">Posts</div></div>
    <div class="stat"><div class="stat__num">{{ site.categories.size }}</div><div class="stat__label">Categories</div></div>
    <div class="stat"><div class="stat__num">{{ site.tags.size }}</div><div class="stat__label">Tags</div></div>
  </div>
</div>

<section class="about-section">
  <div class="sec-label">EXPERIENCE</div>
  <div class="timeline">
    <div class="tl-item current">
      <div class="tl-date">2025.01 — Present</div>
      <div class="tl-body">
        <div class="tl-org">에이브랩스 <span class="tl-role">— LLM 에이전트 개발</span></div>
        <ul>
          <li><strong>관리회계 자동화</strong>: SQL + Apache Airflow 기반 손익분석 자동화 (5시간 → 20분)</li>
          <li><strong>MDAI</strong>: LangGraph 기반 Multi-Agent 관리회계 시스템, 9개 계열사 대상</li>
          <li>운영 및 유지보수, Power BI 대시보드</li>
        </ul>
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-date">2024.08 — 2024.09</div>
      <div class="tl-body">
        <div class="tl-org">KB라이프생명 <span class="tl-role">— 인턴, 상품전략</span></div>
        <ul><li>위험률 정비 및 감리 규정 체크, 시장조사</li></ul>
      </div>
    </div>
  </div>
</section>

<section class="about-section">
  <div class="sec-label">EDUCATION</div>
  <div class="timeline">
    <div class="tl-item">
      <div class="tl-date">2021.09 — 2024.02</div>
      <div class="tl-body">
        <div class="tl-org">고려대학교 일반대학원 <span class="tl-role">— 통계학 석사</span></div>
        <ul>
          <li>학위 논문: <em>Penalized Neural Network Sufficient Dimension Reduction</em></li>
          <li>연구 분야: SDR, Neural Network, Sparse Modeling</li>
        </ul>
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-date">2014.03 — 2021.02</div>
      <div class="tl-body">
        <div class="tl-org">중앙대학교 <span class="tl-role">— 응용통계학 학사</span></div>
      </div>
    </div>
  </div>
</section>

<section class="about-section">
  <div class="sec-label">TECH STACK</div>
  <div class="stack-grid">
    <div class="stack-col"><div class="stack-cat">언어</div><div class="stack-row">
      <span class="stack-tile">Python</span><span class="stack-tile">SQL</span>
    </div></div>
    <div class="stack-col"><div class="stack-cat">ML / DL</div><div class="stack-row">
      <span class="stack-tile">PyTorch</span><span class="stack-tile">scikit-learn</span><span class="stack-tile">NumPy</span><span class="stack-tile">Pandas</span>
    </div></div>
    <div class="stack-col"><div class="stack-cat">LLM & Agent</div><div class="stack-row">
      <span class="stack-tile">LangChain</span><span class="stack-tile">LangGraph</span><span class="stack-tile">Claude API</span><span class="stack-tile">Ollama</span><span class="stack-tile">RAG</span>
    </div></div>
    <div class="stack-col"><div class="stack-cat">백엔드</div><div class="stack-row">
      <span class="stack-tile">FastAPI</span><span class="stack-tile">PostgreSQL</span><span class="stack-tile">Alembic</span>
    </div></div>
    <div class="stack-col"><div class="stack-cat">인프라</div><div class="stack-row">
      <span class="stack-tile">Docker</span><span class="stack-tile">Airflow</span><span class="stack-tile">Git</span>
    </div></div>
  </div>
</section>

<section class="about-section">
  <div class="sec-label">RESEARCH INTERESTS</div>
  <div class="interest-grid">
    <div class="interest-card"><div class="interest-num">01</div><h4>LLM Agent & RAG</h4><p>Multi-agent 시스템 설계, retrieval 최적화, production agent 안정성</p></div>
    <div class="interest-card"><div class="interest-num">02</div><h4>Sufficient Dimension Reduction</h4><p>SDR 이론과 neural network 기반 확장, sparse modeling</p></div>
    <div class="interest-card"><div class="interest-num">03</div><h4>Data Pipeline Automation</h4><p>Airflow + SQL ETL 자동화, 관리회계 분석 시스템</p></div>
    <div class="interest-card"><div class="interest-num">04</div><h4>Reinforcement Learning</h4><p>Contextual Bandits, PPO, RLHF / DPO / KTO 정렬 기법</p></div>
  </div>
</section>
```

스타일은 신규 파일 `assets/css/about.scss`를 만들고 디자인 캔버스의 `screens/about.jsx` 안의 `<style>` 블록을 SCSS로 옮긴 후, 컬러는 모두 Chirpy 변수(`var(--heading-color)` 등)로 매핑할 것. `_includes/head.html`에서 about 페이지에서만 로드되도록 분기 또는 전역 로드.

---

## 8. Categories 페이지 — `_layouts/categories.html`

기존 파일을 카드 그리드로 재구성. 디자인 캔버스의 `screens/categories.jsx`의 마크업과 스타일을 참고하여 Liquid로 변환. 핵심:

- 상단 hero에 카테고리 갯수 / 글 갯수
- `.cats-grid` 2-column 카드, 각 카드에 `.cat-dot` (카테고리 별 hue) + 이름 + 설명 + count
- 클릭하면 해당 카테고리의 글 리스트로 (Chirpy 기존 동작)
- 카테고리 hue 매핑: SCSS에 다음 추가
  ```scss
  $cat-hues: (
    "Study Note":            oklch(60% 0.10 30),
    "Statistics":            oklch(60% 0.10 252),
    "Machine Learning":      oklch(60% 0.10 165),
    "Recommender Systems":   oklch(60% 0.10 300),
    "Reinforcement Learning":oklch(60% 0.10 75),
    "Project":               oklch(60% 0.04 252),
  );
  ```

---

## 9. 사이드바 / Topbar 미세 조정

`_sass/addon/commons.scss`의 사이드바 영역에서:

- `#sidebar` 좌측 1px 라인을 `var(--main-border-color)`로
- 사이드바 폰트 사이즈 13px, letter-spacing -0.005em
- 활성 메뉴 표시: 좌측 2px line이 `var(--link-color)`
- topbar 검색 아이콘과 입력 placeholder 톤 균일하게 `var(--search-icon-color)`

---

## 10. 작업 순서 권장

1. **STEP 1** — Section 1, 2, 3 (토큰 + 폰트). 결과: 사이트 전반의 톤이 즉시 바뀜. 확인 후 진행.
2. **STEP 2** — Section 4, 5 (홈). 결과: 메인 페이지 hero + 카드 + 리스트.
3. **STEP 3** — Section 6 (포스트). 결과: 본문 가독성·TOC.
4. **STEP 4** — Section 7 (About). 결과: 자기소개 페이지.
5. **STEP 5** — Section 8, 9 (카테고리·사이드바).

각 STEP 후 `bundle exec jekyll s`로 로컬 미리보기로 확인. 문제 있으면 직전 STEP 롤백.

---

## 11. 검증 체크리스트

- [ ] light/dark 토글이 새 팔레트로 자연스럽게 전환
- [ ] 한글 본문 가독성이 Pretendard로 향상되었는가
- [ ] 홈 hero 아바타가 `assets/img/me.jpg` 로드되는가
- [ ] 포스트 본문 코드블록·표·인용구가 새 스타일 적용
- [ ] TOC가 sticky, active 항목 인디케이터 표시
- [ ] About 페이지의 timeline·stack·interest 카드가 모두 렌더
- [ ] 모바일 (≤830px) 에서 hero/grid가 단일 컬럼으로 reflow
- [ ] 빌드 경고/에러 없음 (`bundle exec jekyll build`)

---

## 12. 참고 — 디자인 캔버스

원본 디자인 캔버스 HTML은 별도 프로젝트에서 다음 파일로 확인 가능:
- `index.html` — 5개 화면 카드 뷰
- `styles/tokens.css` — 모든 색·타입 토큰
- `screens/home.jsx`, `screens/post.jsx`, `screens/about.jsx`, `screens/categories.jsx` — 컴포넌트 마크업

각 화면을 focus 모드로 확대해서 스타일 디테일을 픽셀 단위로 참조하세요.
