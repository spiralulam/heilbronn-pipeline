/* free.c - the free engine (no symmetry): warm ruin & recreate from a seed with n points.
 *
 * Site frame: unit square [0,1]^2, triangle (0,0),(1,0),(0,1), or the plane for convex position (kept at
 * centroid 0 and hull area 1).  Every point has a type: interior (x, y free; a quadratic penalty keeps it
 * inside), side k = 1..NS (one parameter along side k) or corner 100 + c (fixed at corner c).
 *
 * The search runs in trees of TREE seconds, each restarting from the incumbent.  One step:
 *   ruin      delete the k most stressed points, k uniform in [10 %, 25 %] of n; the stress of a point is the
 *             sum of exp(-15 (A_t - A_min) / A_min) over its triangles t (times a random factor in [0.9, 1.1]);
 *   recreate  greedy insertion up to n + 0..2 points (best of 4000 random candidates by the smallest new
 *             triangle; candidates may lie inside, on a side with fewer than 2 points, or on a free corner),
 *             a short soft-min polish after every insertion, deletion of the most stressed surplus points,
 *             then soft polishes at beta 20 and 50;
 *   accept    if better than the current state, else with probability 0.15;
 *   polish    a soft value above the incumbent gets the sharp polish (beta 8000) and, for s / t, the LP
 *             relocation + min-norm ascent (reloc.c); it becomes the incumbent if it is better.
 */
#include "heil.h"

#define TREE 30.0          /* seconds per tree */
#define CFL 2.0            /* soft-min scale floor of the sharp polish, in units of 1/n^2 */
#define SUBK 6.0           /* soft polishes use only the triangles below SUBK x the minimum */
#define RR_K 4000          /* candidates per greedy insertion */
#define RR_OV 2            /* the insertion ladder overshoots n by 0..RR_OV points */
#define RR_PB 0.2          /* probability that a candidate lies on a side */
#define RR_PC 0.002        /* probability that a candidate is a corner */
#define RR_STRESS 15.0
#define RUIN_F0 0.10       /* ruin fraction range */
#define RUIN_F1 0.25
#define PACC 0.15          /* probability of accepting a worse state */

static int NS, NC;                              /* sides and corners of the region */
static double SO[4][2], SD[4][2], CO[4][2];     /* side k: SO[k] + t SD[k], t in [0,1]; corners */
static int TYPE[MAXN];                          /* 0 interior, 1..NS side, 100 + c corner c */
static int NV, VOFF[MAXN];                      /* variables and the offset of each point's variables */
static int SUBT[MAXT], NSUB, USE_SUB;           /* near-active triangle subset of the soft polish */

static void setup_sides(void) {
  if (V == 's') {
    NS = 4; NC = 4;
    double o[4][2] = {{0,0},{0,1},{0,0},{1,0}}, d[4][2] = {{1,0},{1,0},{0,1},{0,1}}, c[4][2] = {{0,0},{1,0},{0,1},{1,1}};
    memcpy(SO, o, sizeof o); memcpy(SD, d, sizeof d); memcpy(CO, c, sizeof c);
  } else if (V == 't') {
    NS = 3; NC = 3;
    double o[3][2] = {{0,0},{0,0},{1,0}}, d[3][2] = {{1,0},{0,1},{-1,1}}, c[3][2] = {{0,0},{1,0},{0,1}};
    memcpy(SO, o, sizeof o); memcpy(SD, d, sizeof d); memcpy(CO, c, sizeof c);
  } else { NS = 0; NC = 0; }
}

/* ---------------- geometry ---------------- */
static double minarea(const double *p) {
  double m = 1e300;
  for (int t = 0; t < NT; t++) {
    int i = TI[t], j = TJ[t], k = TK[t];
    double a = fabs((p[2*j]-p[2*i])*(p[2*k+1]-p[2*i+1]) - (p[2*k]-p[2*i])*(p[2*j+1]-p[2*i+1]));
    if (a < m) m = a;
  }
  return 0.5 * m;
}
static double value(const double *p) {
  double m = minarea(p);
  if (V == 't') return 2 * m;
  if (V == 'c') return m / hull_area(p, NULL, NULL, NULL, NULL);
  return m;
}
static double clamp01(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
static void put_inside(double *x, double *y) {
  if (V == 's') { *x = clamp01(*x); *y = clamp01(*y); }
  else if (V == 't') { if (*x < 0) *x = 0; if (*y < 0) *y = 0; double s = *x + *y; if (s > 1) { *x /= s; *y /= s; } }
}
static double side_param(int k, double x, double y) {   /* projection onto side k (1-based) */
  k--; double dx = x - SO[k][0], dy = y - SO[k][1];
  return clamp01((dx * SD[k][0] + dy * SD[k][1]) / (SD[k][0]*SD[k][0] + SD[k][1]*SD[k][1]));
}
static void normalize_convex(double *p) {
  double H = hull_area(p, NULL, NULL, NULL, NULL), cx = 0, cy = 0;
  for (int i = 0; i < N; i++) { cx += p[2*i]; cy += p[2*i+1]; }
  cx /= N; cy /= N; double s = 1 / sqrt(H);
  for (int i = 0; i < N; i++) { p[2*i] = (p[2*i] - cx) * s; p[2*i+1] = (p[2*i+1] - cy) * s; }
}

/* ---------------- parametrisation by point types ---------------- */
static void build_vars(void) {
  NV = 0;
  for (int i = 0; i < N; i++) { VOFF[i] = NV; if (TYPE[i] == 0) NV += 2; else if (TYPE[i] <= NS) NV += 1; }
}
static void p2u(const double *p, double *u) {
  for (int i = 0; i < N; i++) {
    int o = VOFF[i];
    if (TYPE[i] == 0) { u[o] = p[2*i]; u[o+1] = p[2*i+1]; }
    else if (TYPE[i] <= NS) u[o] = side_param(TYPE[i], p[2*i], p[2*i+1]);
  }
}
static void u2p(const double *u, double *p) {
  for (int i = 0; i < N; i++) {
    int o = VOFF[i], ty = TYPE[i];
    if (ty == 0) { p[2*i] = u[o]; p[2*i+1] = u[o+1]; }
    else if (ty <= NS) { int k = ty - 1; p[2*i] = SO[k][0] + u[o] * SD[k][0]; p[2*i+1] = SO[k][1] + u[o] * SD[k][1]; }
    else { int c = ty - 100; p[2*i] = CO[c][0]; p[2*i+1] = CO[c][1]; }
  }
}
static void detect_types(const double *p) {
  for (int i = 0; i < N; i++) {
    TYPE[i] = 0;
    if (V == 'c') continue;
    double x = p[2*i], y = p[2*i+1], e = 1e-7;
    for (int c = 0; c < NC; c++) if (fabs(x - CO[c][0]) < e && fabs(y - CO[c][1]) < e) TYPE[i] = 100 + c;
    if (TYPE[i]) continue;
    if (V == 's') { if (y < e) TYPE[i] = 1; else if (y > 1 - e) TYPE[i] = 2; else if (x < e) TYPE[i] = 3; else if (x > 1 - e) TYPE[i] = 4; }
    else { if (y < e) TYPE[i] = 1; else if (x < e) TYPE[i] = 2; else if (x + y > 1 - e) TYPE[i] = 3; }
  }
  build_vars();
}

/* soft-min objective in the type parameters, plus the penalty MU * (squared distance outside the region) */
static double fgrad(const double *u, double *g) {
  double p[2*MAXN], gp[2*MAXN], gH[2*MAXN];
  u2p(u, p);
  double H = V == 'c' ? hull_area(p, gH, NULL, NULL, NULL) : 1.0;
  double F = softmin(p, H, gH, gp, USE_SUB ? SUBT : NULL, NSUB, 40, NULL, NULL), P = 0;
  for (int i = 0; i < N; i++) {
    int o = VOFF[i], ty = TYPE[i];
    if (ty == 0) {
      double x = p[2*i], y = p[2*i+1], gx = gp[2*i], gy = gp[2*i+1];
      if (V == 's') {
        if (x < 0) { P += x*x; gx += 2*MU*x; } else if (x > 1) { P += (x-1)*(x-1); gx += 2*MU*(x-1); }
        if (y < 0) { P += y*y; gy += 2*MU*y; } else if (y > 1) { P += (y-1)*(y-1); gy += 2*MU*(y-1); }
      } else if (V == 't') {
        if (x < 0) { P += x*x; gx += 2*MU*x; }
        if (y < 0) { P += y*y; gy += 2*MU*y; }
        double e = x + y - 1; if (e > 0) { P += e*e; gx += 2*MU*e; gy += 2*MU*e; }
      }
      g[o] = gx; g[o+1] = gy;
    } else if (ty <= NS) {
      int k = ty - 1; double t = u[o];
      double gt = gp[2*i] * SD[k][0] + gp[2*i+1] * SD[k][1], L2 = SD[k][0]*SD[k][0] + SD[k][1]*SD[k][1];
      if (t < 0) { P += L2*t*t; gt += 2*MU*L2*t; } else if (t > 1) { P += L2*(t-1)*(t-1); gt += 2*MU*L2*(t-1); }
      g[o] = gt;
    }
  }
  return F + MU * P;
}

/* ---------------- polishes ---------------- */
static void polish_prepare(double *p, double *u) {
  if (V == 'c') normalize_convex(p);
  for (int i = 0; i < N; i++) if (TYPE[i] == 0) put_inside(&p[2*i], &p[2*i+1]);
  p2u(p, u); u2p(u, p);
}
static double polish_finish(double *p, const double *u) {
  u2p(u, p);
  if (V == 'c') normalize_convex(p); else for (int i = 0; i < N; i++) put_inside(&p[2*i], &p[2*i+1]);
  return value(p);
}

/* sharp polish: the last two stages of the beta continuation (beta 8000; the scale floor is
   dropped in the final stage) */
static double sharp_polish(double *p) {
  double u[2*MAXN];
  polish_prepare(p, u);
  double d0 = (V == 't' ? 0.7 : 1.0) / sqrt(N);
  for (int b = NBETA - 1; b <= NBETA; b++) {
    u2p(u, p);
    double c = value(p) * (V == 't' ? 0.5 : 1.0);
    double cfl = CFL / (N * N) * (V == 't' ? 0.5 : 1.0);
    if (b < NBETA) { if (c < cfl) c = cfl; BETA = BETAS[b]; }
    else BETA = 8000;
    CREF = c; MU = 50 * BETA * (d0 / c) * (d0 / c);
    lbfgs(u, NV, b < NBETA ? ITS[b] : 150, fgrad);
  }
  return polish_finish(p, u);
}

/* soft polish at sharpness bf relative to the current minimum, over the near-active triangles only */
static double soft_polish(double *p, double bf, int iters) {
  static double a[MAXT];
  double u[2*MAXN];
  polish_prepare(p, u);
  double d0 = (V == 't' ? 0.7 : 1.0) / sqrt(N);
  double c = value(p) * (V == 't' ? 0.5 : 1.0), fl = 0.02 / (N * N) * (V == 't' ? 0.5 : 1.0);
  if (c < fl) c = fl;
  BETA = bf; CREF = c; MU = 50 * BETA * (d0 / c) * (d0 / c);
  double m = 1e300;
  for (int t = 0; t < NT; t++) {
    int i = TI[t], j = TJ[t], k = TK[t];
    a[t] = fabs((p[2*j]-p[2*i])*(p[2*k+1]-p[2*i+1]) - (p[2*k]-p[2*i])*(p[2*j+1]-p[2*i+1])); if (a[t] < m) m = a[t];
  }
  NSUB = 0; for (int t = 0; t < NT; t++) if (a[t] <= SUBK * m) SUBT[NSUB++] = t;
  USE_SUB = 1;
  lbfgs(u, NV, iters, fgrad);
  USE_SUB = 0;
  return polish_finish(p, u);
}

/* LP relocation + min-norm ascent; corner points are held on one adjacent side */
static void reloc_finish(double *p) {
  static int ready = 0;
  if (!ready) { reloc_init(V == 't', rnd64()); ready = 1; }
  double x[MAXN], y[MAXN]; int pin[MAXN];
  static const int sq[5] = {PIN_FREE, PIN_Y0, PIN_Y1, PIN_X0, PIN_X1}, tr[4] = {PIN_FREE, PIN_Y0, PIN_X0, PIN_HYP};
  for (int i = 0; i < N; i++) {
    x[i] = p[2*i]; y[i] = p[2*i+1];
    int ty = TYPE[i];
    if (ty >= 100) pin[i] = V == 's' ? (x[i] < 0.5 ? PIN_X0 : PIN_X1) : (x[i] < 1e-9 ? PIN_X0 : PIN_Y0);
    else pin[i] = V == 's' ? sq[ty] : tr[ty];
  }
  reloc_polish(x, y, pin);
  for (int i = 0; i < N; i++) { p[2*i] = x[i]; p[2*i+1] = y[i]; }
}

/* ---------------- ruin & recreate ---------------- */
/* delete the k most stressed points */
static void ruin(double *p, int k) {
  static double a[MAXT];
  double m = 1e300, st[MAXN];
  for (int t = 0; t < NT; t++) {
    int i = TI[t], j = TJ[t], l = TK[t];
    a[t] = fabs((p[2*j]-p[2*i])*(p[2*l+1]-p[2*i+1]) - (p[2*l]-p[2*i])*(p[2*j+1]-p[2*i+1])); if (a[t] < m) m = a[t];
  }
  if (m < 1e-300) m = 1e-300;
  for (int i = 0; i < N; i++) st[i] = 0;
  for (int t = 0; t < NT; t++) {
    double z = RR_STRESS * (a[t] - m) / m; if (z > 40) continue;
    double w = exp(-z); st[TI[t]] += w; st[TJ[t]] += w; st[TK[t]] += w;
  }
  for (int i = 0; i < N; i++) st[i] *= 0.9 + 0.2 * urand();
  char del[MAXN] = {0};
  for (int r = 0; r < k; r++) {
    int bi = -1;
    for (int i = 0; i < N; i++) if (!del[i] && (bi < 0 || st[i] > st[bi])) bi = i;
    del[bi] = 1;
  }
  int q = 0;
  for (int i = 0; i < N; i++) if (!del[i]) { p[2*q] = p[2*i]; p[2*q+1] = p[2*i+1]; TYPE[q] = TYPE[i]; q++; }
  set_points(q); build_vars();
}

static double LKEY[MAXN * MAXN / 2];
static int cmp_line(const void *a, const void *b) {   /* by length, ties in the original order */
  int x = *(const int *)a, y = *(const int *)b;
  return LKEY[x] < LKEY[y] ? -1 : LKEY[x] > LKEY[y] ? 1 : x - y;
}

/* add the best of RR_K random candidates, scored by the smallest new triangle (convex: / new hull area) */
static void insert_best(double *p) {
  static double LA[MAXN*MAXN/2], LB[MAXN*MAXN/2], LC[MAXN*MAXN/2], A2[MAXN*MAXN/2], B2[MAXN*MAXN/2], C2[MAXN*MAXN/2];
  static int ord[MAXN*MAXN/2];
  double bx = 0, by = 0, bs = -1; int bt = 0;
  double minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
  if (V == 'c')
    for (int i = 0; i < N; i++) {
      if (p[2*i] < minx) minx = p[2*i]; if (p[2*i] > maxx) maxx = p[2*i];
      if (p[2*i+1] < miny) miny = p[2*i+1]; if (p[2*i+1] > maxy) maxy = p[2*i+1];
    }
  int cnt[8] = {0}; for (int i = 0; i < N; i++) if (TYPE[i] >= 1 && TYPE[i] <= NS) cnt[TYPE[i]]++;
  int hv[MAXN], hm = 0; double H0 = 1;
  if (V == 'c') H0 = hull_area(p, NULL, NULL, hv, &hm);
  /* twice the area of (p_i, p_j, c) is |LA x + LB y + LC|; short pairs first, since their thin forbidden
     strips reject most candidates early */
  int NL = 0;
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++) {
      LA[NL] = p[2*i+1] - p[2*j+1]; LB[NL] = p[2*j] - p[2*i]; LC[NL] = p[2*i] * p[2*j+1] - p[2*j] * p[2*i+1];
      LKEY[NL] = LA[NL] * LA[NL] + LB[NL] * LB[NL]; ord[NL] = NL; NL++;
    }
  qsort(ord, NL, sizeof(int), cmp_line);
  for (int q = 0; q < NL; q++) { A2[q] = LA[ord[q]]; B2[q] = LB[ord[q]]; C2[q] = LC[ord[q]]; }
  for (int c = 0; c < RR_K; c++) {
    double x, y; int ty = 0;
    if (V != 'c' && urand() < RR_PC) {   /* a free corner */
      int cc = rnd64() % NC, occ = 0;
      for (int i = 0; i < N; i++) if (TYPE[i] == 100 + cc) occ = 1;
      if (occ) continue;
      x = CO[cc][0]; y = CO[cc][1]; ty = 100 + cc;
    } else if (V != 'c' && urand() < RR_PB) {   /* a side with fewer than 2 points */
      int k = 1 + rnd64() % NS; if (cnt[k] >= 2) continue;
      double t = urand(); x = SO[k-1][0] + t * SD[k-1][0]; y = SO[k-1][1] + t * SD[k-1][1]; ty = k;
    } else if (V == 'c') { x = minx + (maxx - minx) * (urand() * 1.1 - 0.05); y = miny + (maxy - miny) * (urand() * 1.1 - 0.05); }
    else { x = urand(); y = urand(); if (V == 't' && x + y > 1) { x = 1 - x; y = 1 - y; } }
    double sc = 1.0;
    if (V == 'c') {   /* a candidate outside the hull enlarges it */
      int in = 1;
      for (int q = 0; q < hm && in; q++) {
        int a = hv[q], b = hv[(q+1) % hm];
        if ((p[2*b]-p[2*a])*(y-p[2*a+1]) - (p[2*b+1]-p[2*a+1])*(x-p[2*a]) < 0) in = 0;
      }
      if (in) sc = 1.0 / H0;
      else { p[2*N] = x; p[2*N+1] = y; N++; double H = hull_area(p, NULL, NULL, NULL, NULL); N--; sc = 1.0 / H; }
    }
    double mn = 1e300, lim = bs / sc;
    for (int q = 0; q < NL; q += 8) {   /* blocks of 8; stop as soon as the candidate cannot win */
      int e = q + 8 < NL ? q + 8 : NL; double bm = 1e300;
      for (int z = q; z < e; z++) { double a = fabs(A2[z] * x + B2[z] * y + C2[z]); bm = a < bm ? a : bm; }
      if (bm < mn) { mn = bm; if (mn <= lim) break; }
    }
    if (mn * sc > bs) { bs = mn * sc; bx = x; by = y; bt = ty; }
  }
  p[2*N] = bx; p[2*N+1] = by; TYPE[N] = bt; set_points(N + 1); build_vars();
}

/* greedy ladder up to n + 0..RR_OV points, prune back to n, polish; returns the soft value */
static double rebuild(double *p, int n) {
  int tgt = n + (int)(rnd64() % (RR_OV + 1));
  while (N < tgt) { insert_best(p); soft_polish(p, 15, 20); }
  if (N > n) ruin(p, N - n);
  soft_polish(p, 20, 50);
  return soft_polish(p, 50, 50);
}

/* ---------------- driver ---------------- */
typedef struct { double p[2*MAXN]; int type[MAXN]; double v; } State;
static void st_save(State *s, const double *p, double v) { memcpy(s->p, p, sizeof s->p); memcpy(s->type, TYPE, sizeof s->type); s->v = v; }

static void save(const char *fn, const State *s) {
  char hdr[128]; snprintf(hdr, sizeof hdr, "# %c n=%d value=%.15g", V, N, s->v);
  save_points(fn, hdr, s->p, NULL);
}

int run_free(const Opts *o) {
  int n = o->n; double seed[2*MAXN] = {0};
  setup_sides();
  set_points(n);
  if (load_points(o->init, seed, MAXN) != n) { fprintf(stderr, "seed %s does not have %d points\n", o->init, n); return 4; }
  char fn[512]; snprintf(fn, sizeof fn, "%s.txt", o->out);
  State best; best.v = -1;
  double pcur[2*MAXN], ptr[2*MAXN]; int tcur[MAXN];
  memcpy(pcur, seed, sizeof pcur); detect_types(pcur);
  double v0 = sharp_polish(pcur);
  if (v0 > 0) { st_save(&best, pcur, v0); save(fn, &best); }   /* the polished seed is the first incumbent */
  long it = 0, trees = 0; double t0 = now();
  while (now() - t0 < o->secs) {
    double tend = now() + TREE, vcur;
    trees++;
    memcpy(pcur, best.v > 0 ? best.p : seed, sizeof pcur); set_points(n);
    if (best.v > 0) { memcpy(TYPE, best.type, sizeof TYPE); build_vars(); } else detect_types(pcur);
    vcur = soft_polish(pcur, 50, 50);
    memcpy(tcur, TYPE, sizeof tcur);
    while (now() < tend && now() - t0 < o->secs) {
      it++;
      memcpy(ptr, pcur, sizeof ptr); memcpy(TYPE, tcur, sizeof TYPE); set_points(n); build_vars();
      int k = (int)(n * (RUIN_F0 + (RUIN_F1 - RUIN_F0) * urand())); if (k < 1) k = 1;
      ruin(ptr, k);
      double v = rebuild(ptr, n);
      if (v > best.v * (1 + 1e-9)) {   /* soft value above the incumbent: sharp polish + relocation */
        double q[2*MAXN]; memcpy(q, ptr, sizeof q);
        double vq = sharp_polish(q);
        if (V != 'c') { reloc_finish(q); vq = value(q); }
        if (vq < v) { memcpy(q, ptr, sizeof q); vq = v; }
        if (vq > best.v * (1 + 1e-9)) {
          st_save(&best, q, vq); save(fn, &best);
          fprintf(stderr, "[%c%d rr s%llu] tree %ld it %ld t=%.0fs best %.10f\n", V, n, o->seed, trees, it, now() - t0, best.v);
        }
      }
      if (v > vcur || urand() < PACC) { memcpy(pcur, ptr, sizeof pcur); memcpy(tcur, TYPE, sizeof tcur); vcur = v; }
    }
  }
  fprintf(stderr, "[%c%d rr s%llu] done trees %ld iters %ld best %.12f\n", V, n, o->seed, trees, it, best.v);
  printf("%.15g\n", best.v);
  return 0;
}
