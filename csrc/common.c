/* common.c - pieces shared by both engines: triangle list, random numbers, clock, convex hull,
 * the soft-min objective, L-BFGS and configuration files. */
#include <time.h>
#include "heil.h"

char V;
int N, NT;
int TI[MAXT], TJ[MAXT], TK[MAXT];
unsigned long long rs;
double BETA, CREF, MU;
const double BETAS[NBETA] = {8, 30, 120, 500, 2000, 8000};
const int ITS[NBETA] = {60, 60, 60, 80, 100, 200};

void set_points(int n) {
  N = n; NT = 0;
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
      for (int k = j + 1; k < N; k++) { TI[NT] = i; TJ[NT] = j; TK[NT] = k; NT++; }
}

double nrand(void) {
  double u = urand(), v = urand(); if (u < 1e-300) u = 1e-300;
  return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }

/* Andrew's monotone chain */
double hull_area(const double *p, double *gH, const unsigned char *gone, int *hv, int *nh) {
  int idx[MAXN], h[2 * MAXN + 2], k = 0, m = 0;
  for (int i = 0; i < N; i++) if (!gone || !gone[i]) idx[m++] = i;
  for (int i = 1; i < m; i++) {   /* insertion sort by (x, y) */
    int v = idx[i], j = i - 1;
    while (j >= 0 && (p[2*idx[j]] > p[2*v] || (p[2*idx[j]] == p[2*v] && p[2*idx[j]+1] > p[2*v+1]))) { idx[j+1] = idx[j]; j--; }
    idx[j + 1] = v;
  }
#define CR(o, a, b) ((p[2*a]-p[2*o])*(p[2*b+1]-p[2*o+1]) - (p[2*a+1]-p[2*o+1])*(p[2*b]-p[2*o]))
  for (int i = 0; i < m; i++) { while (k >= 2 && CR(h[k-2], h[k-1], idx[i]) <= 0) k--; h[k++] = idx[i]; }
  for (int i = m - 2, t = k + 1; i >= 0; i--) { while (k >= t && CR(h[k-2], h[k-1], idx[i]) <= 0) k--; h[k++] = idx[i]; }
#undef CR
  k--;
  double A = 0;
  for (int i = 0; i < k; i++) { int a = h[i], b = h[(i+1)%k]; A += p[2*a]*p[2*b+1] - p[2*b]*p[2*a+1]; }
  A *= 0.5;
  if (gH) {
    memset(gH, 0, sizeof(double) * 2 * N);
    for (int i = 0; i < k; i++) {
      int a = h[i], nx = h[(i+1)%k], pv = h[(i+k-1)%k];
      gH[2*a] = 0.5 * (p[2*nx+1] - p[2*pv+1]); gH[2*a+1] = 0.5 * (p[2*pv] - p[2*nx]);
    }
  }
  if (hv) { for (int i = 0; i < k; i++) hv[i] = h[i]; *nh = k; }
  return A;
}

/* F = (1/beta) log sum_t w_t exp(-beta (a_t/(CREF H) - amin/(CREF H))) - amin/(CREF H) is minus a smooth
   minimum of the scaled areas; minimising F pushes all near-minimal triangles up at once. */
double softmin(const double *p, double H, const double *gH, double *gp, const int *sub, int nsub,
               double zcut, const float *mul, const unsigned char *skip) {
  static double ar[MAXT];
  double sc = 1.0 / (CREF * H), amin = 1e300;
  int nl = sub ? nsub : NT;
  for (int q = 0; q < nl; q++) {
    int t = sub ? sub[q] : q, i = TI[t], j = TJ[t], k = TK[t];
    double a = 0.5 * ((p[2*j]-p[2*i])*(p[2*k+1]-p[2*i+1]) - (p[2*k]-p[2*i])*(p[2*j+1]-p[2*i+1]));
    ar[t] = a; double aa = fabs(a);
    if (skip && skip[t]) continue;
    if (aa < amin) amin = aa;
  }
  double S = 0, SA = 0, base = amin * sc;
  memset(gp, 0, sizeof(double) * 2 * N);
  for (int q = 0; q < nl; q++) {
    int t = sub ? sub[q] : q;
    if (skip && skip[t]) continue;
    double aa = fabs(ar[t]), z = BETA * (aa * sc - base);
    if (z > zcut) continue;
    double w = exp(-z); if (mul) w *= mul[t];
    S += w; SA += w * aa;
    int i = TI[t], j = TJ[t], k = TK[t];
    double s = (ar[t] >= 0 ? 0.5 : -0.5) * w;
    double xi = p[2*i], yi = p[2*i+1], xj = p[2*j], yj = p[2*j+1], xk = p[2*k], yk = p[2*k+1];
    gp[2*i] += s * (yj - yk); gp[2*i+1] += s * (xk - xj);
    gp[2*j] += s * (yk - yi); gp[2*j+1] += s * (xi - xk);
    gp[2*k] += s * (yi - yj); gp[2*k+1] += s * (xj - xi);
  }
  double F = (-BETA * base + log(S)) / BETA, f = -sc / S;
  if (V == 'c') { double q = SA / H; for (int v = 0; v < 2*N; v++) gp[v] = f * (gp[v] - q * gH[v]); }
  else for (int v = 0; v < 2*N; v++) gp[v] *= f;
  return F;
}

#define LM 8
void lbfgs(double *x, int nv, int maxit, objective fg) {
  static double s[LM][2*MAXN], y[LM][2*MAXN], rho[LM], al[LM];
  double g[2*MAXN], d[2*MAXN], xn[2*MAXN], gn[2*MAXN];
  double f = fg(x, g);
  int k = 0;
  for (int it = 0; it < maxit; it++) {
    for (int v = 0; v < nv; v++) d[v] = -g[v];   /* two-loop recursion */
    int m = k < LM ? k : LM;
    for (int l = 0; l < m; l++) {
      int q = (k - 1 - l) % LM; double a = 0;
      for (int v = 0; v < nv; v++) a += s[q][v] * d[v];
      a *= rho[q]; al[q] = a;
      for (int v = 0; v < nv; v++) d[v] -= a * y[q][v];
    }
    if (m > 0) {
      int q = (k - 1) % LM; double yy = 0, sy = 0;
      for (int v = 0; v < nv; v++) { yy += y[q][v]*y[q][v]; sy += s[q][v]*y[q][v]; }
      double gam = sy / yy; for (int v = 0; v < nv; v++) d[v] *= gam;
    } else {
      double gn2 = 0; for (int v = 0; v < nv; v++) gn2 += g[v]*g[v];
      double sc = 1e-3 / (sqrt(gn2) + 1e-300); for (int v = 0; v < nv; v++) d[v] *= sc;
    }
    for (int l = m - 1; l >= 0; l--) {
      int q = (k - 1 - l) % LM; double b = 0;
      for (int v = 0; v < nv; v++) b += y[q][v] * d[v];
      b *= rho[q];
      for (int v = 0; v < nv; v++) d[v] += s[q][v] * (al[q] - b);
    }
    double gd = 0; for (int v = 0; v < nv; v++) gd += g[v]*d[v];
    if (gd >= 0) {   /* not a descent direction: restart with a short gradient step */
      k = 0; double gn2 = 0; for (int v = 0; v < nv; v++) gn2 += g[v]*g[v];
      double sc = 1e-3 / (sqrt(gn2) + 1e-300);
      for (int v = 0; v < nv; v++) d[v] = -g[v] * sc;
      gd = -gn2 * sc;
    }
    double st = 1, fn = 0; int ok = 0;
    for (int ls = 0; ls < 30; ls++) {
      for (int v = 0; v < nv; v++) xn[v] = x[v] + st * d[v];
      fn = fg(xn, gn);
      if (fn <= f + 1e-4 * st * gd) { ok = 1; break; }
      st *= 0.5;
    }
    if (!ok) break;
    int q = k % LM; double sy = 0;
    for (int v = 0; v < nv; v++) { s[q][v] = xn[v] - x[v]; y[q][v] = gn[v] - g[v]; sy += s[q][v]*y[q][v]; }
    double df = f - fn;
    memcpy(x, xn, sizeof(double) * nv); memcpy(g, gn, sizeof(double) * nv); f = fn;
    if (sy > 1e-30) { rho[q] = 1 / sy; k++; }
    if (df < 1e-14 * (fabs(f) + 1e-12)) break;
  }
}

int load_points(const char *fn, double *p, int maxk) {
  FILE *f = fopen(fn, "r"); if (!f) return 0;
  char line[1024]; int k = 0;
  while (fgets(line, sizeof line, f) && k < maxk) {
    if (line[0] == '#' || line[0] == '\n') continue;
    double x, y; if (sscanf(line, "%lf %lf", &x, &y) == 2) { p[2*k] = x; p[2*k+1] = y; k++; }
  }
  fclose(f); return k;
}

/* written to <fn>.tmp and renamed, so a reader never sees a half-written file */
void save_points(const char *fn, const char *header, const double *p, const unsigned char *gone) {
  char tmp[600]; snprintf(tmp, sizeof tmp, "%s.tmp", fn);
  FILE *f = fopen(tmp, "w"); if (!f) return;
  fprintf(f, "%s\n", header);
  for (int i = 0; i < N; i++) if (!gone || !gone[i]) fprintf(f, "%.17g %.17g\n", p[2*i], p[2*i+1]);
  fclose(f); rename(tmp, fn);
}
