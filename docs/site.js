document.documentElement.classList.add('js');
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

// Cubic bezier easing, solved for x by Newton's method (same curve as --ease in the CSS).
const bezier = (x1, y1, x2, y2) => {
  const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
  const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
  const sx = t => ((ax * t + bx) * t + cx) * t;
  const sy = t => ((ay * t + by) * t + cy) * t;
  const dx = t => (3 * ax * t + 2 * bx) * t + cx;
  return x => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    let t = x;
    for (let i = 0; i < 8; i++) {
      const err = sx(t) - x;
      const slope = dx(t);
      if (Math.abs(err) < 1e-5 || Math.abs(slope) < 1e-6) break;
      t = Math.min(1, Math.max(0, t - err / slope));
    }
    return sy(t);
  };
};
const ease = bezier(.22, .61, .36, 1);

// Eased window scroll; instant under reduced motion. Wheel or touch cancels it.
let scrollFrame = 0;
const scrollToY = y => {
  cancelAnimationFrame(scrollFrame);
  const max = document.documentElement.scrollHeight - innerHeight;
  const to = Math.max(0, Math.min(max, y));
  if (reduced) { scrollTo({ top: to, behavior: 'instant' }); return; }
  const from = scrollY;
  const start = performance.now();
  const step = now => {
    const p = Math.min(1, (now - start) / 450);
    scrollTo({ top: from + (to - from) * ease(p), behavior: 'instant' });
    if (p < 1) scrollFrame = requestAnimationFrame(step);
  };
  scrollFrame = requestAnimationFrame(step);
};
const cancelScroll = () => cancelAnimationFrame(scrollFrame);
addEventListener('wheel', cancelScroll, { passive: true });
addEventListener('touchstart', cancelScroll, { passive: true });

// A/B players. Each example has one play button and one or two versions (reference, error).
// Both versions are fetched and decoded on the first tap only; switching version while
// playing jumps to the same point in the other clip (scaled by length) with a short
// crossfade. The playhead and fill move by transform only.
const AC = window.AudioContext || window.webkitAudioContext;
let ctx = null;
const decoded = new Map();
const decode = src => {
  if (!decoded.has(src)) decoded.set(src, fetch(src).then(r => r.arrayBuffer()).then(b => new Promise((ok, no) => ctx.decodeAudioData(b, ok, no))));
  return decoded.get(src);
};
const FADE = 0.015;
let current = null;

const players = [...document.querySelectorAll('.ab')].map(root => {
  const play = root.querySelector('.ab-play');
  const versions = [...root.querySelectorAll('.ab-v')];
  const fill = root.querySelector('.ab-fill');
  const head = root.querySelector('.ab-head');
  const span = Number(root.dataset.span) || 1;
  const title = root.dataset.title || '';
  const dur = versions.map(v => Number(v.dataset.duration) || 1);
  let sel = 0;
  let source = null;
  let gain = null;
  let el = null;
  let startedAt = 0;
  let frame = 0;
  let playing = false;

  const elapsed = () => (el ? el.currentTime : ctx ? ctx.currentTime - startedAt : 0);
  const paint = t => {
    const p = Math.min(1, Math.max(0, t / span));
    fill.style.transform = `scaleX(${p})`;
    head.style.transform = `translateX(${p * 100}%)`;
  };
  const tick = () => {
    if (!playing) return;
    paint(elapsed());
    frame = requestAnimationFrame(tick);
  };
  const select = i => {
    sel = i;
    root.dataset.sel = String(i);
    versions.forEach((v, j) => v.hasAttribute('role') && v.setAttribute('aria-checked', String(j === i)));
  };
  const stopSound = () => {
    if (source) {
      const s = source;
      const g = gain;
      g.gain.setTargetAtTime(0, ctx.currentTime, FADE / 3);
      setTimeout(() => { try { s.stop(); } catch {} }, FADE * 1000 * 3);
      s.onended = null;
    }
    if (el) { el.pause(); el.onended = null; }
    source = gain = el = null;
  };
  const finish = () => {
    stopSound();
    playing = false;
    cancelAnimationFrame(frame);
    root.classList.remove('playing');
    play.setAttribute('aria-label', `Play: ${title}`);
    root.classList.add('resetting');
    paint(0);
    setTimeout(() => root.classList.remove('resetting'), 320);
    if (current === api) current = null;
  };
  const startAt = async offset => {
    if (current && current !== api) current.finish();
    current = api;
    const src = versions[sel].dataset.src;
    playing = true;
    root.classList.add('playing');
    play.setAttribute('aria-label', `Pause: ${title}`);
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(tick);
    if (AC) {
      ctx ??= new AC();
      if (ctx.state === 'suspended') ctx.resume();
      versions.forEach(v => decode(v.dataset.src));
      let buffer;
      try { buffer = await decode(src); } catch { finish(); return; }
      if (!playing || current !== api || versions[sel].dataset.src !== src) return;
      stopSound();
      gain = ctx.createGain();
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(1, ctx.currentTime + FADE);
      gain.connect(ctx.destination);
      source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gain);
      const at = Math.min(offset, Math.max(0, buffer.duration - 0.05));
      startedAt = ctx.currentTime - at;
      const mine = source;
      source.onended = () => { if (source === mine) finish(); };
      source.start(0, at);
    } else {
      stopSound();
      el = new Audio(src);
      el.currentTime = offset;
      el.onended = finish;
      el.play().catch(finish);
    }
  };
  play.addEventListener('click', () => (playing ? finish() : startAt(0)));
  versions.forEach((v, i) => v.addEventListener('click', () => {
    if (playing && i !== sel) {
      const t = elapsed() * (dur[i] / dur[sel]);
      select(i);
      startAt(t);
    } else if (!playing) {
      select(i);
      startAt(0);
    }
  }));
  const api = { root, finish };
  return api;
});

// Positions of the level headings, dimension headings and examples, in page coordinates.
// Measured once and again only when layout changes, so scrolling never reads layout.
const items = [...document.querySelectorAll('.level-head, .dim-head, .example')];
let marks = [];
let span = { top: 0, bottom: 0 };
let pageMax = 0;
const measure = () => {
  marks = items.filter(el => !el.closest('[hidden]')).map(el => {
    const r = el.getBoundingClientRect();
    return { el, top: r.top + scrollY, bottom: r.bottom + scrollY };
  });
  const levels = document.querySelectorAll('.level');
  span = { top: levels[0].getBoundingClientRect().top + scrollY, bottom: levels[levels.length - 1].getBoundingClientRect().bottom + scrollY };
  pageMax = document.documentElement.scrollHeight - innerHeight;
};
const itemAtMiddle = () => {
  const mid = scrollY + innerHeight / 2;
  if (mid < span.top || mid > span.bottom) return null;
  let lo = 0;
  let hi = marks.length - 1;
  while (lo < hi) {
    const m = (lo + hi + 1) >> 1;
    if (marks[m].top <= mid) lo = m; else hi = m - 1;
  }
  const here = marks[lo];
  const next = marks[lo + 1];
  if (!here) return null;
  if (mid <= here.bottom || !next) return here.el;
  return mid - here.bottom <= next.top - mid ? here.el : next.el;
};

// Breadcrumb: a part whose text changes fades out, swaps, and fades back in.
const crumb = document.querySelector('.crumb');
const header = document.querySelector('.top');
const crumbParts = { level: crumb?.querySelector('.crumb-level'), dim: crumb?.querySelector('.crumb-dim') };
const crumbText = { level: '', dim: '' };
const crumbHref = { level: '', dim: '' };
let crumbTimer = 0;
let crumbOn = false;
const setCrumb = (level, dim) => {
  if (!crumb) return;
  const on = Boolean(level);
  if (on !== crumbOn) {
    crumbOn = on;
    crumb.classList.toggle('on', on);
    header?.classList.toggle('has-crumb', on);
  }
  if (!on) return;
  const next = { level: level.dataset.name ?? '', dim: dim?.dataset.name ?? '' };
  Object.assign(crumbHref, { level: `#${level.id}`, dim: dim ? `#${dim.id}` : '' });
  const changed = Object.keys(next).filter(k => crumbText[k] !== next[k]);
  if (!changed.length) return;
  Object.assign(crumbText, next);
  clearTimeout(crumbTimer);
  // Always write the latest text for both parts, so a swap cancelled by a quicker one is not lost.
  const commit = () => {
    Object.keys(crumbText).forEach(k => {
      if (crumbParts[k].textContent !== crumbText[k]) crumbParts[k].textContent = crumbText[k];
      crumbParts[k].setAttribute('href', crumbHref[k] || '#');
    });
    crumb.classList.toggle('has-dim', Boolean(crumbText.dim));
    requestAnimationFrame(() => Object.values(crumbParts).forEach(el => el.classList.remove('swap')));
  };
  if (reduced || !crumbParts.level.textContent) { commit(); return; }
  changed.forEach(k => crumbParts[k].classList.add('swap'));
  crumbTimer = setTimeout(commit, 150);
};
let focusItem = null;
const setFocus = item => {
  if (item === focusItem) return;
  focusItem = item;
  const level = item?.closest('.level') ?? null;
  setCrumb(level, item?.closest('.dim') ?? null);
  // The reading-progress line takes the colour of the level being read.
  if (level) progressBar.dataset.level = level.dataset.level; else delete progressBar.dataset.level;
};

// One scroll handler, one frame: breadcrumb and reading progress.
const progressBar = document.querySelector('.progress span');
let tickFrame = 0;
const onScroll = () => {
  if (tickFrame) return;
  tickFrame = requestAnimationFrame(() => {
    tickFrame = 0;
    setFocus(itemAtMiddle());
    progressBar.style.transform = `scaleX(${pageMax > 0 ? Math.min(1, scrollY / pageMax) : 0})`;
  });
};
addEventListener('scroll', onScroll, { passive: true });
const remeasure = () => { measure(); onScroll(); };
addEventListener('resize', remeasure);
addEventListener('load', remeasure);
new ResizeObserver(remeasure).observe(document.querySelector('main'));
remeasure();

// "More examples" toggles.
document.querySelectorAll('.more').forEach(button => button.addEventListener('click', () => {
  const open = button.getAttribute('aria-expanded') !== 'true';
  button.closest('.dim').querySelectorAll('.example.extra').forEach(card => { card.hidden = !open; });
  button.setAttribute('aria-expanded', String(open));
  const n = Number(button.dataset.more);
  button.textContent = open ? 'Fewer examples' : `${n} more example${n === 1 ? '' : 's'}`;
}));

// Click any heading or example to glide it to the middle; items taller than the
// viewport land just under the header instead.
const centreOf = el => {
  const r = el.getBoundingClientRect();
  return r.height > innerHeight - 160 ? scrollY + r.top - 96 : scrollY + r.top + r.height / 2 - innerHeight / 2;
};
document.addEventListener('click', event => {
  // Controls keep their own behaviour; only a click on the content itself recentres it.
  if (event.target.closest('a, button, summary, input, .why, .ab')) return;
  const item = event.target.closest('.level-head, .dim-head, .example');
  if (item) scrollToY(centreOf(item));
});
// In-page links to a level or dimension centre its heading the same way. The breadcrumb's
// links change as you scroll, so they are matched at click time.
document.addEventListener('click', event => {
  const link = event.target.closest('a[href^="#lvl-"], a[href^="#dim-"]');
  if (!link) return;
  const target = document.getElementById(link.getAttribute('href').slice(1));
  const head = target?.querySelector('.level-head, .dim-head');
  if (!head) return;
  event.preventDefault();
  history.replaceState(null, '', link.getAttribute('href'));
  scrollToY(centreOf(head));
});

// Findings grid: a cell takes you to its dimension.
document.querySelectorAll('.detect td[data-dim]').forEach(cell => cell.addEventListener('click', () => {
  const head = document.getElementById(cell.dataset.dim)?.querySelector('.dim-head');
  if (head) scrollToY(centreOf(head));
}));

// MOS heads tooltip: hover with a mouse, focus with a keyboard, tap on touch. It is
// positioned against the viewport so the grid's scroll box does not clip it.
document.querySelectorAll('.tip').forEach(tip => {
  const button = tip.querySelector('.tip-btn');
  const box = tip.querySelector('.tip-box');
  let shownAt = 0;
  const show = () => {
    const r = button.getBoundingClientRect();
    if (!box.classList.contains('on')) shownAt = performance.now();
    box.classList.add('on');
    button.setAttribute('aria-expanded', 'true');
    const left = Math.min(Math.max(12, r.left), innerWidth - box.offsetWidth - 12);
    const below = r.bottom + 8 + box.offsetHeight < innerHeight - 12;
    box.style.left = `${left}px`;
    box.style.top = `${below ? r.bottom + 8 : r.top - box.offsetHeight - 8}px`;
  };
  const hide = () => {
    box.classList.remove('on');
    button.setAttribute('aria-expanded', 'false');
  };
  tip.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse') show(); });
  tip.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse' && document.activeElement !== button) hide(); });
  // A tap toggles on touch screens; ignore the tap that opened it via a synthetic hover.
  button.addEventListener('click', () => {
    const toggleOff = box.classList.contains('on') && !matchMedia('(hover: hover)').matches && performance.now() - shownAt > 400;
    toggleOff ? hide() : show();
  });
  button.addEventListener('focus', show);
  button.addEventListener('blur', hide);
  button.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
  document.addEventListener('pointerdown', e => { if (!tip.contains(e.target)) hide(); });
  addEventListener('scroll', hide, { passive: true });
});

// Header links reflect the section in view. The hero is watched too (it has no id), so
// scrolling back to the top clears the highlight instead of leaving the last section lit.
const navLinks = [...document.querySelectorAll('.nav a')];
let currentSection = '';
const sectionSpy = new IntersectionObserver(entries => entries.forEach(entry => {
  if (!entry.isIntersecting || entry.target.id === currentSection) return;
  currentSection = entry.target.id;
  navLinks.forEach(a => a.setAttribute('aria-current', String(Boolean(currentSection) && a.getAttribute('href') === `#${currentSection}`)));
}), { rootMargin: '-50% 0px -50% 0px' });
document.querySelectorAll('main > section[id], .hero').forEach(s => sectionSpy.observe(s));

// Copy BibTeX.
const copy = document.querySelector('.copy');
copy?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector('.cite code').textContent);
    copy.textContent = 'Copied';
  } catch {
    copy.textContent = 'Select and copy';
  }
  copy.classList.add('done');
  setTimeout(() => { copy.textContent = 'Copy'; copy.classList.remove('done'); }, 1800);
});

// The levels figure builds once as it comes into view; nothing else animates on scroll.
const stack = document.querySelector('.stack');
if (stack && !reduced) {
  const build = new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) return;
    stack.classList.add('in');
    build.disconnect();
  }, { rootMargin: '0px 0px -15% 0px' });
  build.observe(stack);
} else stack?.classList.add('in');

// Pipeline figure: builds left to right as it comes into view, then data pulses loop along
// the arrows. Under reduced motion it is drawn complete and the pulses never start.
const dg = document.querySelector('.dg');
if (dg && !reduced && 'IntersectionObserver' in window) {
  dg.classList.add('dg-anim');
  const build = new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) return;
    build.disconnect();
    dg.classList.add('in');
    // Correlation values count up as their cells pop in.
    dg.querySelectorAll('.dg-cell text[data-v]').forEach((t, i) => {
      const to = Number(t.dataset.v);
      const places = (t.dataset.v.split('.')[1] || '').length;
      t.textContent = (0).toFixed(places);
      const start = performance.now() + 4600 + i * 80;
      const step = now => {
        const p = Math.min(1, Math.max(0, (now - start) / 900));
        t.textContent = (to * ease(p)).toFixed(places);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    setTimeout(() => {
      dg.classList.add('built');
      document.getElementById('dg-go')?.beginElement?.();
    }, 5600);
  }, { rootMargin: '0px 0px -20% 0px' });
  build.observe(dg);
  // Hold the loops still while the figure is off screen; CSS and SMIL pause together, so they stay in step.
  new IntersectionObserver(([entry]) => {
    dg.classList.toggle('paused', !entry.isIntersecting);
    if (entry.isIntersecting) dg.unpauseAnimations?.(); else dg.pauseAnimations?.();
  }).observe(dg);
}

// Method demo: one tab per level. A new level stops the clip that was playing and builds its
// steps in order.
const demoTabs = [...document.querySelectorAll('.demo-tab')];
const pickLevel = (tab, focus = false) => {
  demoTabs.forEach(t => {
    const on = t === tab;
    const panel = document.getElementById(t.getAttribute('aria-controls'));
    if (!on && !panel.hidden) panel.querySelector('.ab.playing .ab-play')?.click();
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
    panel.hidden = !on;
    panel.classList.toggle('play', on);
  });
  if (focus) tab.focus();
};
demoTabs.forEach((tab, i) => {
  tab.addEventListener('click', () => pickLevel(tab));
  tab.addEventListener('keydown', e => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (e.key === 'Home' || e.key === 'End' || step) {
      e.preventDefault();
      const next = e.key === 'Home' ? 0 : e.key === 'End' ? demoTabs.length - 1 : (i + step + demoTabs.length) % demoTabs.length;
      pickLevel(demoTabs[next], true);
    }
  });
});
// The caption's "Step through" link shows or hides the demo; opening it replays the current level.
const demo = document.getElementById('demo');
const demoToggle = document.querySelector('.cap-link[aria-controls="demo"]');
demoToggle?.addEventListener('click', () => {
  const open = demo.hidden;
  if (!open) demo.querySelector('.ab.playing .ab-play')?.click();
  demo.hidden = !open;
  demoToggle.setAttribute('aria-expanded', String(open));
  demoToggle.textContent = open ? 'Hide steps ▾' : 'Step through ▸';
  if (open) pickLevel(demoTabs.find(t => t.getAttribute('aria-selected') === 'true') || demoTabs[0]);
});

// Pipeline figure: click to enlarge it over a blurred page. The figure itself moves into the dialog,
// so its animations and ids stay intact, and moves back on close. In the enlarged view a click on a
// level keeps only that level lit; a second click, or a click elsewhere in the figure, shows all three.
const zoom = document.querySelector('.zoom');
const dgHome = document.querySelector('.diagram-scroll');
if (zoom && dg && dgHome) {
  const stage = zoom.querySelector('.zoom-stage');
  const focusLevel = row => { if (row) dg.dataset.focus = row; else delete dg.dataset.focus; };
  const openZoom = row => {
    dgHome.style.height = `${dgHome.offsetHeight}px`;
    stage.append(dg);
    focusLevel(row);
    zoom.showModal();
  };
  zoom.addEventListener('close', () => {
    focusLevel(null);
    dgHome.append(dg);
    dgHome.style.height = '';
  });
  dgHome.addEventListener('click', e => openZoom(e.target.closest('[data-row]')?.dataset.row));
  dgHome.addEventListener('keydown', e => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    openZoom();
  });
  zoom.querySelector('.zoom-close').addEventListener('click', () => zoom.close());
  zoom.addEventListener('click', e => {
    if (e.target === zoom) { zoom.close(); return; }
    if (!dg.contains(e.target)) return;
    const row = e.target.closest('[data-row]')?.dataset.row;
    focusLevel(row && dg.dataset.focus !== row ? row : null);
  });
}

// Hero: preview the s-to-z flip once shortly after load, unless the visitor gets there first.
const heroDemo = document.querySelector('.hero-demo');
if (heroDemo && !reduced) {
  let touched = false;
  heroDemo.addEventListener('pointerdown', () => { touched = true; heroDemo.classList.remove('hint'); }, { once: true });
  setTimeout(() => {
    if (touched) return;
    heroDemo.classList.add('hint');
    setTimeout(() => heroDemo.classList.remove('hint'), 2200);
  }, 1400);
}

// Sections and cards rise in once as they are reached. The classes are removed afterwards so
// they never interfere with hover states or the findings highlights.
if (!reduced && 'IntersectionObserver' in window) {
  const revealed = document.querySelectorAll('.section-head, .level-head, .dim-head, .example, .tree-level, .steps li, .class, .conditions li, .finding-list li, .detect, .diagram, .paper-card, .cite');
  const reveal = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const el = entry.target;
    el.classList.add('in');
    reveal.unobserve(el);
    setTimeout(() => el.classList.remove('reveal', 'in'), 1800);
  }), { rootMargin: '0px 0px -8% 0px' });
  revealed.forEach(el => { el.classList.add('reveal'); reveal.observe(el); });
}

// Key numbers count up from zero the first time they are seen.
const stats = document.querySelector('.stats');
if (stats && !reduced && 'IntersectionObserver' in window) {
  const cells = [...stats.querySelectorAll('dt')].map(el => ({ el, to: Number(el.textContent.replace(/,/g, '')) }));
  cells.forEach(c => { c.el.style.minWidth = `${c.el.offsetWidth}px`; c.el.textContent = '0'; });
  const count = new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) return;
    count.disconnect();
    cells.forEach((c, i) => {
      const start = performance.now() + 500 + i * 90;
      const step = now => {
        const p = Math.min(1, Math.max(0, (now - start) / 1300));
        c.el.textContent = Math.round(c.to * ease(p)).toLocaleString('en-US');
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  });
  count.observe(stats);
}
