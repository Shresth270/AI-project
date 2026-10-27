/* =====================================================================
   Crowd simulation for the Overview hero.
   ~1,400 people walk along five streets into a square and leave through
   one exit. "Pressure" (0..1, set from the replay clock's risk level)
   raises arrivals and narrows the exit, so the square fills up.
   Each dot is coloured by how many neighbours it has:
   blue = moving freely, amber = bunching, red = packed.
   The pointer is an obstacle people step around.
   ===================================================================== */
(function () {
  const TAU = Math.PI * 2;

  class CrowdSim {
    constructor(canvas) {
      this.c = canvas;
      this.ctx = canvas.getContext("2d");
      this.pressure = 0.15;
      this.target = 0.15;
      this.mouse = { x: -1e4, y: -1e4 };
      this.agents = [];
      this.running = false;
      this.reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      this._resize = this.resize.bind(this);
      new ResizeObserver(this._resize).observe(canvas);
      canvas.addEventListener("pointermove", (e) => {
        const r = canvas.getBoundingClientRect();
        this.mouse.x = e.clientX - r.left; this.mouse.y = e.clientY - r.top;
      });
      canvas.addEventListener("pointerleave", () => { this.mouse.x = this.mouse.y = -1e4; });
      document.addEventListener("visibilitychange", () => document.hidden ? this.stop() : this.start());
      this.resize();
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = this.c.clientWidth, h = this.c.clientHeight;
      if (!w || !h) return;
      this.w = w; this.h = h; this.dpr = dpr;
      this.c.width = w * dpr; this.c.height = h * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const narrow = w < 700;
      // square (plaza) on the right, text side stays calmer on the left
      // square (plaza) on the right; the exit is on the far right edge, away from arriving people
      this.plaza = narrow ? { x: w * 0.6, y: h * 0.76, r: Math.min(w, h) * 0.2 }
                          : { x: w * 0.72, y: h * 0.52, r: Math.min(w, h) * 0.24 };
      this.exit = { x: w + 10, y: this.plaza.y };
      const P = this.plaza;
      const S = (pts) => pts.map(([x, y]) => ({ x: x * w, y: y * h }));
      this.streets = narrow ? [
        S([[-0.05, 0.7], [0.2, 0.72], [0.4, 0.75]]),
        S([[-0.05, 0.98], [0.22, 0.9], [0.44, 0.82]]),
        S([[0.45, 1.05], [0.52, 0.92], [0.57, 0.86]]),
        S([[0.2, 0.6], [0.36, 0.64], [0.5, 0.7]]),
      ] : [
        S([[-0.05, 0.18], [0.25, 0.24], [0.48, 0.36], [0.6, 0.44]]),
        S([[-0.05, 0.86], [0.22, 0.8], [0.46, 0.68], [0.6, 0.6]]),
        S([[0.52, -0.05], [0.6, 0.18], [0.67, 0.34]]),
        S([[0.56, 1.05], [0.62, 0.84], [0.68, 0.7]]),
        S([[0.86, -0.05], [0.82, 0.2], [0.77, 0.34]]),
      ];
      this.streets.forEach((s) => s.push({ x: P.x, y: P.y }));
      this.cell = 14;
      this.cols = Math.ceil(w / this.cell) + 1; this.rows = Math.ceil(h / this.cell) + 1;
      this.maxAgents = Math.round(Math.min(1500, (w * h) / 420));
      if (!this.agents.length) this.seed();
      if (this.reduce) { for (let i = 0; i < 700; i++) this.step(); this.draw(); }
      else this.start();
    }

    spawn(a) {
      const s = this.streets[(Math.random() * this.streets.length) | 0];
      a = a || {};
      a.street = s; a.wp = 1;
      a.x = s[0].x + (Math.random() - 0.5) * 30; a.y = s[0].y + (Math.random() - 0.5) * 30;
      a.vx = 0; a.vy = 0; a.speed = 0.55 + Math.random() * 0.55; a.lane = (Math.random() - 0.5) * 26;
      a.inPlaza = false; a.dens = 0; a.alive = true;
      return a;
    }

    seed() {
      this.agents = [];
      for (let i = 0; i < this.maxAgents * 0.55; i++) {
        const a = this.spawn();
        // scatter along the streets so the first frame already looks alive
        const t = Math.random();
        const p = this.pointOn(a.street, t);
        a.x = p.x; a.y = p.y; a.wp = p.wp;
        this.agents.push(a);
      }
    }

    pointOn(s, t) {
      const n = s.length - 1, f = t * n, i = Math.min(n - 1, f | 0), k = f - i;
      return { x: s[i].x + (s[i + 1].x - s[i].x) * k, y: s[i].y + (s[i + 1].y - s[i].y) * k, wp: i + 1 };
    }

    setPressure(p) { this.target = Math.max(0, Math.min(1, p)); }

    step() {
      this.pressure += (this.target - this.pressure) * 0.02;
      const P = this.plaza, pr = this.pressure, cs = this.cell, cols = this.cols;
      // arrivals
      const want = this.maxAgents * (0.45 + 0.55 * pr);
      const rate = 1.2 + pr * 3.2;
      for (let i = 0; i < rate && this.agents.length < want; i++) this.agents.push(this.spawn());

      // spatial hash
      const grid = new Map();
      for (const a of this.agents) {
        const key = ((a.y / cs) | 0) * cols + ((a.x / cs) | 0);
        let b = grid.get(key); if (!b) grid.set(key, (b = [])); b.push(a);
      }
      // exit capacity per frame: wide when calm, narrow under pressure
      let capacity = 4.2 - pr * 3.85;
      const gap = P.r * (0.62 - 0.46 * pr);          // half-width of the exit: wide when calm, narrow under pressure
      const ex = this.exit;

      for (const a of this.agents) {
        // target
        let tx, ty;
        if (!a.inPlaza) {
          const wp = a.street[a.wp];
          tx = wp.x; ty = wp.y;
          if (a.wp < a.street.length - 1) {          // stay in a lane along the street
            const prev = a.street[a.wp - 1], dx = wp.x - prev.x, dy = wp.y - prev.y, L = Math.hypot(dx, dy) || 1;
            tx += (-dy / L) * a.lane; ty += (dx / L) * a.lane;
          }
          if (Math.hypot(tx - a.x, ty - a.y) < 18) a.wp++;
          // entering the square: head for the exit instead of its centre point
          if (a.wp >= a.street.length || Math.hypot(a.x - P.x, a.y - P.y) < P.r * 0.85) a.inPlaza = true;
        }
        if (a.inPlaza) { const sp = (a.lane / 13) * P.r * (0.62 - 0.46 * this.pressure) * 0.8;
          if (ex.x > this.w) { tx = ex.x; ty = ex.y + sp; } else { tx = ex.x + sp; ty = ex.y; } }
        let dx = tx - a.x, dy = ty - a.y, d = Math.hypot(dx, dy) || 1;
        let fx = (dx / d) * a.speed * 0.08, fy = (dy / d) * a.speed * 0.08;

        // separation from neighbours + local density
        const gx = (a.x / cs) | 0, gy = (a.y / cs) | 0;
        let n = 0;
        for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
          const b = grid.get((gy + oy) * cols + gx + ox); if (!b) continue;
          for (const o of b) {
            if (o === a) continue;
            const ddx = a.x - o.x, ddy = a.y - o.y, dd = ddx * ddx + ddy * ddy;
            if (dd < 196) { n++; if (dd > 0.01) { const s = (196 - dd) / 196 * 0.05 / Math.sqrt(dd); fx += ddx * s; fy += ddy * s; } }
          }
        }
        a.dens += (n - a.dens) * 0.08;

        // pointer is an obstacle
        const mx = a.x - this.mouse.x, my = a.y - this.mouse.y, md = mx * mx + my * my;
        if (md < 6400 && md > 1) { const s = (6400 - md) / 6400 * 0.9 / Math.sqrt(md); fx += mx * s; fy += my * s; }

        // the square's edge (a soft wall around the plaza on the far side)
        if (a.inPlaza) {
          const px = a.x - P.x, py = a.y - P.y, pd = Math.hypot(px, py);
          const exitSide = (ex.x > this.w) ? px > 0 && Math.abs(py) < gap : py > 0 && Math.abs(px) < gap;
          if (pd > P.r && !exitSide) { fx -= (px / pd) * 0.12; fy -= (py / pd) * 0.12; }
          // the exit lets only so many through
          const nearExit = (ex.x > this.w) ? a.x > P.x + P.r * 0.9 && Math.abs(py) < gap : a.y > P.y + P.r * 0.9 && Math.abs(px) < gap;
          if (nearExit) {
            if (capacity >= 1 || Math.random() < capacity) { capacity -= 1; a.alive = false; }
            else { fx -= dx / d * 0.09; fy -= dy / d * 0.09; }
          }
        }

        a.vx = (a.vx + fx) * 0.9; a.vy = (a.vy + fy) * 0.9;
        const sp = Math.hypot(a.vx, a.vy), max = a.speed * (1.25 - Math.min(0.85, a.dens / 9));
        if (sp > max) { a.vx *= max / sp; a.vy *= max / sp; }
        a.x += a.vx; a.y += a.vy;
        if (a.x < -60 || a.x > this.w + 60 || a.y < -60 || a.y > this.h + 60) a.alive = false;
      }
      this.agents = this.agents.filter((a) => a.alive);
    }

    color(d) {
      // calm blue -> amber -> red as neighbours increase
      const t = Math.min(1, d / 10);
      if (t < 0.5) { const k = t / 0.5; return `rgb(${90 + 152 * k | 0},${167 + 14 * k | 0},${255 - 197 * k | 0})`; }
      const k = (t - 0.5) / 0.5; return `rgb(${242 - 3 * k | 0},${181 - 91 * k | 0},${58 + 18 * k | 0})`;
    }

    draw() {
      const g = this.ctx, w = this.w, h = this.h;
      g.clearRect(0, 0, w, h);
      // streets
      g.lineCap = "round"; g.lineJoin = "round";
      g.beginPath();                                   // one path, so overlapping streets don't stack up brighter
      for (const s of this.streets) {
        g.moveTo(s[0].x, s[0].y);
        for (let i = 1; i < s.length; i++) g.lineTo(s[i].x, s[i].y);
      }
      g.strokeStyle = "rgba(139,154,180,0.07)"; g.lineWidth = 46; g.stroke();
      const P = this.plaza;
      g.beginPath(); g.arc(P.x, P.y, P.r, 0, TAU);
      g.fillStyle = "rgba(139,154,180,0.05)"; g.fill();
      g.strokeStyle = "rgba(139,154,180,0.18)"; g.lineWidth = 1; g.setLineDash([4, 6]); g.stroke(); g.setLineDash([]);

      // heat glow where people pack together
      g.globalCompositeOperation = "lighter";
      for (const a of this.agents) {
        if (a.dens < 6) continue;
        const r = 10 + a.dens * 1.2, al = Math.min(0.035, (a.dens - 6) * 0.009);
        const grd = g.createRadialGradient(a.x, a.y, 0, a.x, a.y, r);
        grd.addColorStop(0, `rgba(239,90,76,${al})`); grd.addColorStop(1, "rgba(239,90,76,0)");
        g.fillStyle = grd; g.fillRect(a.x - r, a.y - r, r * 2, r * 2);
      }
      g.globalCompositeOperation = "source-over";
      for (const a of this.agents) {
        g.fillStyle = this.color(a.dens);
        g.beginPath(); g.arc(a.x, a.y, 1.9, 0, TAU); g.fill();
      }
    }

    loop() {
      if (!this.running) return;
      this.step(); this.draw();
      this.raf = requestAnimationFrame(() => this.loop());
    }
    start() { if (this.reduce || this.running || !this.w) return; this.running = true; this.loop(); }
    stop() { this.running = false; cancelAnimationFrame(this.raf); }
  }

  window.CrowdSim = CrowdSim;
})();
