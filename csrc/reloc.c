/* reloc.c - LP relocation and min-norm ascent for the square and the triangle, from Nathan Sudermann-Merx's record_hunt
 * (core.c, lp3.c, relocate.c, ascent.c), trimmed to what the engine uses:
 *
 *  1. one-point LP relocation: with all other points fixed, every determinant of a triangle containing
 *     p_i is affine in p_i; fixing the current orientations confines p_i to its face of the line
 *     arrangement, and "max z s.t. sign_t D_t(p_i) >= 2z, p_i in the region" is an LP in 3 variables
 *     (2 for a point held on a side), solved by Seidel's randomised algorithm.  A sweep moves
 *     every point to the optimum of its face; the minimum area never decreases.
 *  2. min-norm ascent: the direction is the minimum-norm element of the convex hull of the eps-active
 *     area gradients (Demyanov-Malozemov, computed by pairwise Frank-Wolfe), which raises all active
 *     triangles at once; along a ray every area is quadratic in the step, and the step goes to the first
 *     local maximum of the lower envelope of these parabolas.
 */
#include <stdint.h>
#include "heil.h"

typedef struct { int n; double x[MAXN], y[MAXN]; int pin[MAXN]; } Cfg;

static int tri_dom;        /* 0 square, 1 triangle */
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rand_u64(void) { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return rng * 0x2545F4914F6CDD1Dull; }

static void project(double *x, double *y) {
  if (*x < 0.0) *x = 0.0;
  if (*y < 0.0) *y = 0.0;
  if (tri_dom) {
    double s = *x + *y;
    if (s > 1.0) { *x -= 0.5 * (s - 1.0); *y -= 0.5 * (s - 1.0); }
    if (*x < 0.0) { *y += *x; *x = 0.0; }
    if (*y < 0.0) { *x += *y; *y = 0.0; }
  } else {
    if (*x > 1.0) *x = 1.0;
    if (*y > 1.0) *y = 1.0;
  }
}

/* twice the signed area of (p_i, p_j, p_k) */
static double det(const Cfg *c, int i, int j, int k) {
  return (c->x[j] - c->x[i]) * (c->y[k] - c->y[i]) - (c->y[j] - c->y[i]) * (c->x[k] - c->x[i]);
}

/* Fan identity: with s_ij = (x_i y_j - x_j y_i)/2 about the centroid, area(i,j,k) = s_ij + s_jk + s_ki. */
static double smat[MAXN][MAXN];
static double min_area(const Cfg *c) {
  const int n = c->n;
  double cx = 0.0, cy = 0.0;
  for (int i = 0; i < n; i++) { cx += c->x[i]; cy += c->y[i]; }
  cx /= n; cy /= n;
  for (int i = 0; i < n; i++) {
    double xi = c->x[i] - cx, yi = c->y[i] - cy;
    smat[i][i] = 0.0;
    for (int j = i + 1; j < n; j++) {
      double v = 0.5 * (xi * (c->y[j] - cy) - (c->x[j] - cx) * yi);
      smat[i][j] = v; smat[j][i] = -v;
    }
  }
  double m = 1.0e300;
  for (int i = 0; i < n - 2; i++)
    for (int j = i + 1; j < n - 1; j++) {
      const double sij = smat[i][j], *sj = smat[j], *si = smat[i];
      for (int k = j + 1; k < n; k++) { double a = fabs(sij + sj[k] - si[k]); if (a < m) m = a; }
    }
  return m;
}

static double min_area_containing(const Cfg *c, int i) {
  double m = 1.0e300;
  for (int j = 0; j < c->n - 1; j++) {
    if (j == i) continue;
    for (int k = j + 1; k < c->n; k++) { if (k == i) continue; double a = fabs(det(c, i, j, k)); if (a < m) m = a; }
  }
  return 0.5 * m;
}

/* ---------------- Seidel's randomised incremental LP, d <= 3 ----------------
   max c.x s.t. A x <= b inside the box [-LP_BIG, LP_BIG]^d; ties prefer the larger coordinate. */
#define LP_BIG 1.0e7
#define LP_EPS 1.0e-12
#define MAXROWS (MAXN * MAXN / 2 + 16)
static int lp_perm[4][MAXROWS];
static double lp_As[4][MAXROWS * 2], lp_bs[4][MAXROWS];

static int lp1(const double *A, const double *b, int m, double c, double *x) {
  double lo = -LP_BIG, hi = LP_BIG;
  for (int i = 0; i < m; i++) {
    double a = A[i];
    if (a > LP_EPS) { double v = b[i] / a; if (v < hi) hi = v; }
    else if (a < -LP_EPS) { double v = b[i] / a; if (v > lo) lo = v; }
    else if (b[i] < -1.0e-9) return 0;
  }
  if (lo > hi) {
    if (lo > hi + 1.0e-9) return 0;
    lo = hi = 0.5 * (lo + hi);
  }
  *x = (c < 0.0) ? lo : hi;
  return 1;
}

static int lp_rec(int d, const double *A, const double *b, int m, const double *c, double *x) {
  if (d == 1) return lp1(A, b, m, c[0], x);
  for (int k = 0; k < d; k++) x[k] = (c[k] < 0.0) ? -LP_BIG : LP_BIG;
  int *perm = lp_perm[d];
  for (int i = 0; i < m; i++) perm[i] = i;
  for (int i = m - 1; i > 0; i--) { int j = (int)(rand_u64() % (uint64_t)(i + 1)), t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  double *As = lp_As[d], *bs = lp_bs[d], xs[3], cs[3];
  int idx[3];
  for (int ii = 0; ii < m; ii++) {
    int i = perm[ii];
    const double *Ai = A + (size_t)i * d;
    double lhs = 0.0;
    for (int k = 0; k < d; k++) lhs += Ai[k] * x[k];
    if (lhs <= b[i] + 1.0e-9 * (1.0 + fabs(b[i]))) continue;
    /* the new optimum lies on Ai.x = b_i: eliminate the variable with the largest coefficient */
    int p = 0; double best = fabs(Ai[0]);
    for (int k = 1; k < d; k++) if (fabs(Ai[k]) > best) { best = fabs(Ai[k]); p = k; }
    if (best < LP_EPS) return 0;
    double aip = Ai[p]; int nd = 0;
    for (int k = 0; k < d; k++) if (k != p) idx[nd++] = k;
    int ms = 0;
    for (int jj = 0; jj < ii; jj++) {
      const double *Aj = A + (size_t)perm[jj] * d;
      double f = Aj[p] / aip;
      for (int t = 0; t < nd; t++) As[(size_t)ms * nd + t] = Aj[idx[t]] - f * Ai[idx[t]];
      bs[ms] = b[perm[jj]] - f * b[i];
      ms++;
    }
    double fc = c[p] / aip;
    for (int t = 0; t < nd; t++) cs[t] = c[idx[t]] - fc * Ai[idx[t]];
    if (!lp_rec(d - 1, As, bs, ms, cs, xs)) return 0;
    double acc = b[i];
    for (int t = 0; t < nd; t++) { x[idx[t]] = xs[t]; acc -= Ai[idx[t]] * xs[t]; }
    x[p] = acc / aip;
  }
  return 1;
}

/* ---------------- one-point relocation ---------------- */
static double Abuf[MAXROWS * 3], bbuf[MAXROWS];

/* move p_i to the LP optimum of its cell; returns min area of the triangles containing i, or -1 */
static double relocate_point(Cfg *cfg, int i) {
  const int n = cfg->n, pin = cfg->pin[i];
  const double xi = cfg->x[i], yi = cfg->y[i];
  int m = 0; double sol[3];
  if (pin == PIN_FREE) {
    for (int j = 0; j < n - 1; j++) {
      if (j == i) continue;
      for (int k = j + 1; k < n; k++) {
        if (k == i) continue;
        double a = cfg->y[j] - cfg->y[k], b = cfg->x[k] - cfg->x[j];
        double c0 = cfg->x[j] * cfg->y[k] - cfg->x[k] * cfg->y[j];
        double D = c0 + a * xi + b * yi, s = (D >= 0.0) ? 1.0 : -1.0;
        Abuf[m*3+0] = -0.5 * s * a; Abuf[m*3+1] = -0.5 * s * b; Abuf[m*3+2] = 1.0; bbuf[m] = 0.5 * s * c0;   /* z - s D/2 <= 0 */
        m++;
      }
    }
    Abuf[m*3+0] = -1; Abuf[m*3+1] = 0; Abuf[m*3+2] = 0; bbuf[m] = 0; m++;
    Abuf[m*3+0] = 0; Abuf[m*3+1] = -1; Abuf[m*3+2] = 0; bbuf[m] = 0; m++;
    if (tri_dom) { Abuf[m*3+0] = 1; Abuf[m*3+1] = 1; Abuf[m*3+2] = 0; bbuf[m] = 1; m++; }
    else {
      Abuf[m*3+0] = 1; Abuf[m*3+1] = 0; Abuf[m*3+2] = 0; bbuf[m] = 1; m++;
      Abuf[m*3+0] = 0; Abuf[m*3+1] = 1; Abuf[m*3+2] = 0; bbuf[m] = 1; m++;
    }
    const double c[3] = {0.0, 0.0, 1.0};
    if (!lp_rec(3, Abuf, bbuf, m, c, sol) || !(sol[2] > 0.0)) return -1.0;
    cfg->x[i] = sol[0] < 0.0 ? 0.0 : (sol[0] > 1.0 ? 1.0 : sol[0]);
    cfg->y[i] = sol[1] < 0.0 ? 0.0 : (sol[1] > 1.0 ? 1.0 : sol[1]);
  } else {   /* one parameter t along the side; on the hypotenuse the point is (t, 1 - t) */
    const int hyp = (pin == PIN_HYP), xfixed = (pin == PIN_X0 || pin == PIN_X1);
    const double vfix = (pin == PIN_X1 || pin == PIN_Y1) ? 1.0 : 0.0;
    const double t0 = hyp ? xi : (xfixed ? yi : xi);
    for (int j = 0; j < n - 1; j++) {
      if (j == i) continue;
      for (int k = j + 1; k < n; k++) {
        if (k == i) continue;
        double a = cfg->y[j] - cfg->y[k], b = cfg->x[k] - cfg->x[j];
        double c0 = cfg->x[j] * cfg->y[k] - cfg->x[k] * cfg->y[j], cst, coef;
        if (hyp) { cst = c0 + b; coef = a - b; }
        else if (xfixed) { cst = c0 + a * vfix; coef = b; }
        else { cst = c0 + b * vfix; coef = a; }
        double D = cst + coef * t0, s = (D >= 0.0) ? 1.0 : -1.0;
        Abuf[m*2+0] = -0.5 * s * coef; Abuf[m*2+1] = 1.0; bbuf[m] = 0.5 * s * cst;
        m++;
      }
    }
    Abuf[m*2+0] = -1; Abuf[m*2+1] = 0; bbuf[m] = 0; m++;
    Abuf[m*2+0] = 1; Abuf[m*2+1] = 0; bbuf[m] = 1; m++;
    const double c[2] = {0.0, 1.0};
    if (!lp_rec(2, Abuf, bbuf, m, c, sol) || !(sol[1] > 0.0)) return -1.0;
    double t = sol[0] < 0.0 ? 0.0 : (sol[0] > 1.0 ? 1.0 : sol[0]);
    if (hyp) { cfg->x[i] = t; cfg->y[i] = 1.0 - t; }
    else if (xfixed) { cfg->x[i] = vfix; cfg->y[i] = t; }
    else { cfg->y[i] = vfix; cfg->x[i] = t; }
  }
  return min_area_containing(cfg, i);
}

/* one pass over all points in random order; returns 1 if some point improved */
static int relocate_sweep(Cfg *cfg) {
  const int n = cfg->n;
  int order[MAXN], improved = 0;
  for (int i = 0; i < n; i++) order[i] = i;
  for (int i = n - 1; i > 0; i--) { int j = (int)(rand_u64() % (uint64_t)(i + 1)), t = order[i]; order[i] = order[j]; order[j] = t; }
  for (int s = 0; s < n; s++) {
    int i = order[s];
    double before = min_area_containing(cfg, i), sx = cfg->x[i], sy = cfg->y[i];
    double after = relocate_point(cfg, i);
    if (after < before - 1e-15 || after < 0.0) { cfg->x[i] = sx; cfg->y[i] = sy; }   /* numerical safeguard */
    else if (after > before + 1e-14) improved = 1;
  }
  return improved;
}

/* ---------------- min-norm ascent ---------------- */
#define MAX_ACT 8192
#define FW_ITERS 8000
#define FW_CHEAP 250
static int a_idx[MAX_ACT][6];
static double a_val[MAX_ACT][6], a_sq[MAX_ACT], fw_lam[MAX_ACT];
static double gvec[2 * MAXN], dvec[2 * MAXN];
static int frozen[2 * MAXN];
static double qa[MAXT + 8], qb[MAXT + 8], qc[MAXT + 8];   /* A_t(tau) = qa tau^2 + qb tau + qc */

/* gradients of the triangles with area <= f (1 + eps); each has 6 nonzeros */
static int collect_active(const Cfg *cfg, double f, double eps) {
  const double thr = f * (1.0 + eps);
  int na = 0;
  for (int t = 0; t < NT && na < MAX_ACT; t++) {
    int i = TI[t], j = TJ[t], k = TK[t];
    double D = det(cfg, i, j, k);
    if (0.5 * fabs(D) > thr) continue;
    double s = (D >= 0.0 ? 0.5 : -0.5), sq = 0.0;
    int *ix = a_idx[na]; double *vl = a_val[na];
    ix[0] = 2*i;   vl[0] = s * (cfg->y[j] - cfg->y[k]);
    ix[1] = 2*i+1; vl[1] = s * (cfg->x[k] - cfg->x[j]);
    ix[2] = 2*j;   vl[2] = s * (cfg->y[k] - cfg->y[i]);
    ix[3] = 2*j+1; vl[3] = s * (cfg->x[i] - cfg->x[k]);
    ix[4] = 2*k;   vl[4] = s * (cfg->y[i] - cfg->y[j]);
    ix[5] = 2*k+1; vl[5] = s * (cfg->x[j] - cfg->x[i]);
    for (int q = 0; q < 6; q++) { if (frozen[ix[q]]) vl[q] = 0.0; sq += vl[q] * vl[q]; }
    a_sq[na++] = sq;
  }
  return na;
}

/* minimum-norm point of conv{a_0..a_{na-1}} by pairwise Frank-Wolfe (linear convergence on a polytope) */
static void minnorm(int na, int nv, double *g, int iters) {
  memset(g, 0, (size_t)nv * sizeof *g);
  memset(fw_lam, 0, (size_t)na * sizeof *fw_lam);
  fw_lam[0] = 1.0;
  for (int q = 0; q < 6; q++) g[a_idx[0][q]] += a_val[0][q];
  for (int it = 0; it < iters; it++) {
    double gg = 0.0;
    for (int v = 0; v < nv; v++) gg += g[v] * g[v];
    if (gg < 1e-26) return;
    int bt = -1, at = -1; double bv = 1e300, av = -1e300;
    for (int t = 0; t < na; t++) {
      double ip = 0.0;
      for (int q = 0; q < 6; q++) ip += g[a_idx[t][q]] * a_val[t][q];
      if (ip < bv) { bv = ip; bt = t; }
      if (fw_lam[t] > 1e-15 && ip > av) { av = ip; at = t; }
    }
    if (bt < 0 || at < 0) return;
    if (bv >= gg - 1e-15 * (1.0 + gg)) return;
    double dd = a_sq[bt] + a_sq[at];   /* d = a_bt - a_at; minimise ||g + gamma d||^2 over [0, lam_at] */
    for (int q = 0; q < 6; q++)
      for (int r = 0; r < 6; r++)
        if (a_idx[bt][q] == a_idx[at][r]) dd -= 2.0 * a_val[bt][q] * a_val[at][r];
    if (dd < 1e-30) return;
    double gam = (av - bv) / dd;
    if (gam > fw_lam[at]) gam = fw_lam[at];
    if (gam <= 0.0) return;
    for (int q = 0; q < 6; q++) g[a_idx[bt][q]] += gam * a_val[bt][q];
    for (int q = 0; q < 6; q++) g[a_idx[at][q]] -= gam * a_val[at][q];
    fw_lam[bt] += gam; fw_lam[at] -= gam;
  }
}

/* smallest root of a t^2 + b t + c = 0 in (lo, hi], else hi */
static double first_root_after(double a, double b, double c, double lo, double hi) {
  if (fabs(a) < 1e-18) {
    if (fabs(b) < 1e-18) return hi;
    double r = -c / b;
    return (r > lo && r <= hi) ? r : hi;
  }
  double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) return hi;
  double sq = sqrt(disc), q = (b >= 0.0) ? -0.5 * (b + sq) : -0.5 * (b - sq);
  double r1 = q / a, r2 = (fabs(q) > 1e-300) ? c / q : hi, best = hi;
  if (r1 > lo && r1 < best) best = r1;
  if (r2 > lo && r2 < best) best = r2;
  return best;
}

/* follow the envelope phi(tau) = min_t A_t(P + tau d) from tau = 0 to its first local maximum on the feasible ray */
static double line_search(const Cfg *cfg, const double *d) {
  const int n = cfg->n, nt = NT;
  double taumax = 1e30;
  for (int p = 0; p < n; p++) {
    double dx = d[2*p], dy = d[2*p+1];
    if (dx < -1e-15) { double t = (0.0 - cfg->x[p]) / dx; if (t < taumax) taumax = t; }
    if (dy < -1e-15) { double t = (0.0 - cfg->y[p]) / dy; if (t < taumax) taumax = t; }
    if (tri_dom) { double ds = dx + dy; if (ds > 1e-15) { double t = (1.0 - cfg->x[p] - cfg->y[p]) / ds; if (t < taumax) taumax = t; } }
    else {
      if (dx > 1e-15) { double t = (1.0 - cfg->x[p]) / dx; if (t < taumax) taumax = t; }
      if (dy > 1e-15) { double t = (1.0 - cfg->y[p]) / dy; if (t < taumax) taumax = t; }
    }
  }
  if (!(taumax > 0.0)) return 0.0;
  if (taumax > 1e29) taumax = 1.0;
  for (int t = 0; t < nt; t++) {
    int i = TI[t], j = TJ[t], k = TK[t];
    double u = cfg->x[j] - cfg->x[i], v = cfg->y[k] - cfg->y[i], w = cfg->y[j] - cfg->y[i], s = cfg->x[k] - cfg->x[i];
    double du = d[2*j] - d[2*i], dv = d[2*k+1] - d[2*i+1], dw = d[2*j+1] - d[2*i+1], ds = d[2*k] - d[2*i];
    double c0 = u * v - w * s, b0 = u * dv + du * v - w * ds - dw * s, a0 = du * dv - dw * ds;
    double sg = (c0 >= 0.0) ? 0.5 : -0.5;
    qa[t] = sg * a0; qb[t] = sg * b0; qc[t] = sg * c0;
  }
  double tau = 0.0, bestt = 0.0, bestv = -1e300;
  for (int ev = 0; ev < 64; ev++) {
    double vmin = 1e300;
    for (int t = 0; t < nt; t++) { double q = (qa[t] * tau + qb[t]) * tau + qc[t]; if (q < vmin) vmin = q; }
    if (vmin > bestv) { bestv = vmin; bestt = tau; }
    int cur = -1; double slope = 1e300, tolv = 1e-13 + 1e-9 * fabs(vmin);   /* right derivative of the envelope */
    for (int t = 0; t < nt; t++) {
      double q = (qa[t] * tau + qb[t]) * tau + qc[t];
      if (q > vmin + tolv) continue;
      double sl = 2.0 * qa[t] * tau + qb[t];
      if (sl < slope) { slope = sl; cur = t; }
    }
    if (cur < 0 || slope <= 1e-16) break;
    if (tau >= taumax - 1e-15) break;
    double tstop = taumax;
    if (qa[cur] < -1e-18) { double apex = -qb[cur] / (2.0 * qa[cur]); if (apex > tau && apex < tstop) tstop = apex; }
    /* only parabolas close to the envelope can overtake it soon (phi is evaluated exactly at every event) */
    const double band = vmin * 4.0 + 1e-12;
    double tnext = tstop;
    for (int t = 0; t < nt; t++) {
      if (t == cur) continue;
      double q = (qa[t] * tau + qb[t]) * tau + qc[t];
      if (q > band && 2.0 * qa[t] * tau + qb[t] >= 0.0) continue;
      double r = first_root_after(qa[t] - qa[cur], qb[t] - qb[cur], qc[t] - qc[cur], tau + 1e-14, tnext);
      if (r < tnext) tnext = r;
    }
    if (!(tnext > tau)) break;
    tau = tnext;
  }
  return bestt;
}

static void ascent(Cfg *cfg, int max_iter) {
  const int n = cfg->n, nv = 2 * n;
  double f = min_area(cfg), eps = 1e-3;
  int fw_budget = FW_CHEAP;
  for (int iter = 0; iter < max_iter; iter++) {
    for (int v = 0; v < nv; v++) frozen[v] = 0;
    for (int p = 0; p < n; p++) {
      if (cfg->pin[p] == PIN_X0 || cfg->pin[p] == PIN_X1) frozen[2*p] = 1;
      if (cfg->pin[p] == PIN_Y0 || cfg->pin[p] == PIN_Y1) frozen[2*p+1] = 1;
    }
    /* coordinates on the boundary whose ascent component points outwards are frozen; re-solve */
    double gn = 0.0; int ok = 0;
    for (int pass = 0; pass < 4 && !ok; pass++) {
      int na = collect_active(cfg, f, eps);
      if (na == 0) break;
      minnorm(na, nv, gvec, fw_budget);
      gn = 0.0;
      for (int v = 0; v < nv; v++) gn += gvec[v] * gvec[v];
      gn = sqrt(gn);
      if (gn < 1e-13) break;
      int added = 0;
      for (int v = 0; v < nv; v++) {
        if (frozen[v]) continue;
        int pt = v / 2;
        double p0 = (v & 1) ? cfg->y[pt] : cfg->x[pt];
        double hi = tri_dom ? 1.0 - ((v & 1) ? cfg->x[pt] : cfg->y[pt]) : 1.0;
        if ((p0 <= 1e-12 && gvec[v] < 0.0) || (p0 >= hi - 1e-12 && gvec[v] > 0.0)) { frozen[v] = 1; added = 1; }
      }
      if (!added) ok = 1;
    }
    if (!ok || gn < 1e-13) { if (eps > 1e-11) { eps *= 0.1; continue; } break; }
    for (int v = 0; v < nv; v++) dvec[v] = gvec[v] / gn;
    double tau = line_search(cfg, dvec);
    if (tau <= 1e-16) {   /* usually an unresolved min-norm point: pay for one long Frank-Wolfe run first */
      if (fw_budget < FW_ITERS) { fw_budget = FW_ITERS; continue; }
      if (eps > 1e-11) { eps *= 0.1; fw_budget = FW_CHEAP; continue; }
      break;
    }
    fw_budget = FW_CHEAP;
    for (int p = 0; p < n; p++) { cfg->x[p] += tau * dvec[2*p]; cfg->y[p] += tau * dvec[2*p+1]; project(&cfg->x[p], &cfg->y[p]); }
    double nf = min_area(cfg);
    if (nf <= f + 1e-16) { f = nf; if (eps > 1e-11) { eps *= 0.1; continue; } break; }
    f = nf;
  }
}

/* ---------------- entry points ---------------- */
void reloc_init(int triangle, unsigned long long seed) { tri_dom = triangle; rng = seed ? seed : 0x9E3779B97F4A7C15ull; }

/* up to 3 rounds of (relocation sweeps until no point improves, at most 20) + 60 ascent steps */
void reloc_polish(double *x, double *y, const int *pin) {
  static Cfg c;
  const int n = c.n = N;
  for (int i = 0; i < n; i++) { c.x[i] = x[i]; c.y[i] = y[i]; c.pin[i] = pin[i]; }
  for (int round = 0; round < 3; round++) {
    for (int k = 0; k < 20; k++) if (!relocate_sweep(&c)) break;
    double before = min_area(&c);
    ascent(&c, 60);
    if (min_area(&c) <= before * (1 + 1e-12)) break;
  }
  for (int i = 0; i < n; i++) { x[i] = c.x[i]; y[i] = c.y[i]; }
}
