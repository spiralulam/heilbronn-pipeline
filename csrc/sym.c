/* sym.c - the symmetric engine: configurations generated from orbit representatives under a point group G.
 *
 * Native frame: square [-1/2,1/2]^2, equilateral triangle with circumradius 1 centred at 0, or the plane
 * (convex position, centre of symmetry at 0, hull area 1).  value = min area / region area (hull area).
 * Representative kinds: GEN (x, y), AXIS (t along a mirror line), SIDE (t along a region side), CENTER
 * (origin), CORNER (region vertex).  The parameters of representative r start at u[RVO[r]]: packed (2, 1 or
 * 0 slots by kind) in warm runs, two slots each in cold runs (the overshoot shifts them).  The layout changes
 * the rounding inside the vectorised L-BFGS, so it is kept as in the original engine.
 *
 * Modes (run_sym):
 *   cold RR   no start file: every tree draws a random orbit structure (sides carry at most 2 points,
 *             sometimes 1 or a corner orbit, optional centre, 0-5 axis orbits, the rest generic), builds it
 *             by greedy orbit insertion and runs ruin & recreate on it (optionally with defect orbits);
 *   warm RR   start file with G symmetry (any n: whole orbits are added / removed to reach n), every tree
 *             restarts from the incumbent;
 *   walk      start file with n points: hot basin hopping (jitter / relocation / greedy re-insertion /
 *             big ruin / local shake moves, sharp-ish polish from beta stage 1, threshold acceptance).
 * One RR step: mark the 10-30 % most stressed orbits, re-insert them one by one in random order (best of
 * 1000 candidates by the smallest triangle touching the orbit, short soft polish after each), polish at
 * beta 20 and 50; cold runs sometimes insert one extra generic orbit and then drop the most stressed one.
 */
#include "heil.h"

#define MAXG 24
#define MAXR MAXN
#define TREE 30.0          /* seconds per RR tree */
#define CFL 40.0           /* soft-min scale floor of the staged polish, in units of region area / n^2 */
#define SUBK 6.0           /* soft polishes use only the triangles below SUBK x the minimum */
#define SRR_K 1000         /* candidates per orbit re-insertion */
#define SRR_STRESS 15.0
#define RUIN_F0 0.10       /* fraction of the free orbits that is ruined */
#define RUIN_F1 0.30
#define PACC 0.15
#define PSIDE1 0.35        /* square: a side carries a single point */
#define WALK_THR 0.06      /* walk: accept if value > current * (1 - WALK_THR * U) */
#define WALK_RESTART 800   /* walk: back to the incumbent after this many steps without progress */

enum { GEN, AXIS, SIDE, CENTER, CORNER };
typedef struct { double u[2*MAXN]; double v; } St;

/* ---------------- region ---------------- */
static int NSIDE; static double HN[8][2], HH[8], VX[8][2], CAREA = 1;   /* sides n.p <= h, vertices, area */

static void setup_region(void) {
  if (V == 's') { NSIDE = 4; double v[4][2] = {{-.5,-.5},{.5,-.5},{.5,.5},{-.5,.5}}; memcpy(VX, v, sizeof v); CAREA = 1; }
  else if (V == 't') {   /* vertices at 210, 330 and 90 degrees */
    NSIDE = 3;
    for (int k = 0; k < 3; k++) { double a = M_PI / 2 + 2 * M_PI * k / 3 + 2 * M_PI / 3; VX[k][0] = cos(a); VX[k][1] = sin(a); }
    CAREA = 0.5 * fabs((VX[1][0]-VX[0][0])*(VX[2][1]-VX[0][1]) - (VX[2][0]-VX[0][0])*(VX[1][1]-VX[0][1]));
  } else { NSIDE = 0; CAREA = 0; }
  for (int k = 0; k < NSIDE; k++) {
    double *a = VX[k], *b = VX[(k + 1) % NSIDE], dx = b[0] - a[0], dy = b[1] - a[1], L = sqrt(dx*dx + dy*dy);
    HN[k][0] = dy / L; HN[k][1] = -dx / L; HH[k] = HN[k][0] * a[0] + HN[k][1] * a[1];
  }
}
/* native frame <-> site frame (square [0,1]^2, triangle (0,0),(1,0),(0,1), convex unchanged) */
static void to_site(const double *p, double *q, int m) {
  for (int i = 0; i < m; i++) {
    double x = p[2*i], y = p[2*i+1];
    if (V == 's') { q[2*i] = x + 0.5; q[2*i+1] = y + 0.5; }
    else if (V == 't') {
      double ax = VX[1][0]-VX[0][0], ay = VX[1][1]-VX[0][1], bx = VX[2][0]-VX[0][0], by = VX[2][1]-VX[0][1];
      double det = ax*by - ay*bx, dx = x - VX[0][0], dy = y - VX[0][1];
      q[2*i] = (dx*by - dy*bx) / det; q[2*i+1] = (ax*dy - ay*dx) / det;
    } else { q[2*i] = x; q[2*i+1] = y; }
  }
}
static void from_site(const double *q, double *p, int m) {
  for (int i = 0; i < m; i++) {
    double x = q[2*i], y = q[2*i+1];
    if (V == 's') { p[2*i] = x - 0.5; p[2*i+1] = y - 0.5; }
    else if (V == 't') {
      p[2*i] = VX[0][0] + x * (VX[1][0]-VX[0][0]) + y * (VX[2][0]-VX[0][0]);
      p[2*i+1] = VX[0][1] + x * (VX[1][1]-VX[0][1]) + y * (VX[2][1]-VX[0][1]);
    } else { p[2*i] = x; p[2*i+1] = y; }
  }
}
static void put_inside(double *x, double *y) {
  for (int it = 0; it < 4; it++)
    for (int k = 0; k < NSIDE; k++) { double e = HN[k][0] * *x + HN[k][1] * *y - HH[k]; if (e > 0) { *x -= e * HN[k][0]; *y -= e * HN[k][1]; } }
}

/* ---------------- group ---------------- */
static int NG, NMIR; static double GM[MAXG][4], MIRA[MAXG][2]; static char GNAME[16];
static void add_rot(double th) { double c = cos(th), s = sin(th); GM[NG][0] = c; GM[NG][1] = -s; GM[NG][2] = s; GM[NG][3] = c; NG++; }
static void add_mir(double th) {   /* mirror in the line through 0 at angle th */
  double c = cos(2*th), s = sin(2*th); GM[NG][0] = c; GM[NG][1] = s; GM[NG][2] = s; GM[NG][3] = -c; NG++;
  MIRA[NMIR][0] = cos(th); MIRA[NMIR][1] = sin(th); NMIR++;
}
/* square: C1 C2 C4 D1 D1d D2 D2d D4; triangle: C1 C3 D1 D3; convex: Ck, Dk, k <= 12 */
static int setup_group(const char *g) {
  NG = 0; NMIR = 0; snprintf(GNAME, sizeof GNAME, "%s", g);
  int k = atoi(g + 1); if (k < 1) k = 1;
  if (g[0] != 'C' && g[0] != 'D') return 0;
  if (V == 's') {
    if (!strcmp(g, "D1")) { add_rot(0); add_mir(M_PI/2); return 1; }
    if (!strcmp(g, "D1d")) { add_rot(0); add_mir(M_PI/4); return 1; }
    if (!strcmp(g, "D2")) { add_rot(0); add_rot(M_PI); add_mir(0); add_mir(M_PI/2); return 1; }
    if (!strcmp(g, "D2d")) { add_rot(0); add_rot(M_PI); add_mir(M_PI/4); add_mir(3*M_PI/4); return 1; }
    if (g[0] == 'C' && (k == 1 || k == 2 || k == 4)) { for (int j = 0; j < k; j++) add_rot(2*M_PI*j/k); return 1; }
    if (!strcmp(g, "D4")) { for (int j = 0; j < 4; j++) add_rot(M_PI/2*j); for (int j = 0; j < 4; j++) add_mir(M_PI/4*j); return 1; }
    return 0;
  }
  if (V == 't') {   /* mirrors through the vertices */
    if (g[0] == 'C' && (k == 1 || k == 3)) { for (int j = 0; j < k; j++) add_rot(2*M_PI*j/k); return 1; }
    if (!strcmp(g, "D1")) { add_rot(0); add_mir(M_PI/2); return 1; }
    if (!strcmp(g, "D3")) { for (int j = 0; j < 3; j++) add_rot(2*M_PI*j/3); for (int j = 0; j < 3; j++) add_mir(M_PI/2 + M_PI*j/3); return 1; }
    return 0;
  }
  if (k > 12) return 0;
  for (int j = 0; j < k; j++) add_rot(2*M_PI*j/k);
  if (g[0] == 'D') for (int j = 0; j < k; j++) add_mir(M_PI*j/k + M_PI/2);
  return 1;
}
static void apply(int g, double x, double y, double *ox, double *oy) { *ox = GM[g][0]*x + GM[g][1]*y; *oy = GM[g][2]*x + GM[g][3]*y; }

/* ---------------- orbit structure ---------------- */
static int NR, RK[MAXR], RA[MAXR];       /* representatives: kind and mirror / side / vertex index */
static int RSZ[MAXR], RG[MAXR][MAXG];   /* orbit size and the group elements giving its distinct images */
static int PR[MAXN], PG[MAXN];          /* point i = image of representative PR[i] under element PG[i] */
static int NV, RVO[MAXR], SLOT2;        /* parameter count, offset of representative r, two slots each */

static void rep_point(int kind, int aux, const double *u, double *x, double *y) {
  switch (kind) {
    case GEN: *x = u[0]; *y = u[1]; break;
    case AXIS: *x = u[0] * MIRA[aux][0]; *y = u[0] * MIRA[aux][1]; break;
    case SIDE: { const double *a = VX[aux], *b = VX[(aux+1)%NSIDE]; double t = u[0];   /* t in [-1/2, 1/2] */
      *x = 0.5*(a[0]+b[0]) + t * (b[0]-a[0]); *y = 0.5*(a[1]+b[1]) + t * (b[1]-a[1]); break; }
    case CENTER: *x = 0; *y = 0; break;
    default: *x = VX[aux][0]; *y = VX[aux][1];
  }
}
static void rep_base(int r, const double *u, double *x, double *y) { rep_point(RK[r], RA[r], u + RVO[r], x, y); }
static int rep_nvar(int r) { return RK[r] == GEN ? 2 : (RK[r] == AXIS || RK[r] == SIDE) ? 1 : 0; }
static void rep_param(int r, double x, double y, double *u) {   /* parameters of representative r at (x, y) */
  double *o = u + RVO[r];
  if (RK[r] == GEN) { o[0] = x; o[1] = y; }
  else if (RK[r] == AXIS) o[0] = x * MIRA[RA[r]][0] + y * MIRA[RA[r]][1];
  else if (RK[r] == SIDE) {
    const double *a = VX[RA[r]], *b = VX[(RA[r]+1)%NSIDE]; double dx = b[0]-a[0], dy = b[1]-a[1];
    o[0] = ((x - 0.5*(a[0]+b[0]))*dx + (y - 0.5*(a[1]+b[1]))*dy) / (dx*dx + dy*dy);
  }
}
static int orbit_images(double bx, double by, double *sv, int *gl) {   /* distinct images of (bx, by) */
  int n = 0;
  for (int g = 0; g < NG; g++) {
    double x, y; apply(g, bx, by, &x, &y); int dup = 0;
    for (int q = 0; q < n; q++) if (fabs(sv[2*q] - x) < 1e-9 && fabs(sv[2*q+1] - y) < 1e-9) dup = 1;
    if (!dup) { sv[2*n] = x; sv[2*n+1] = y; if (gl) gl[n] = g; n++; }
  }
  return n;
}

/* Triangle orbits: with a nontrivial group only one canonical triangle per orbit enters the soft polish,
   weighted by its multiplicity.  Cached for the last few structures (the overshoot alternates two). */
static unsigned char TCAN[MAXT]; static float TMUL[MAXT]; static int CAN_OK, NOCAN;
static int TIDN = -1, TID[MAXN][MAXN][MAXN];
#define CANC 6
static struct { unsigned long long key; int nt; unsigned char *can; float *mul; int ok; } CANCACHE[CANC]; static int CANNEXT;

static void build_can_raw(void) {
  if (TIDN != N) { int t = 0; for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) for (int k = j+1; k < N; k++) TID[i][j][k] = t++; TIDN = N; }
  CAN_OK = 0;
  double tu[2*MAXN], p[2*MAXN];   /* a generic test configuration */
  for (int r = 0; r < NR; r++) { tu[RVO[r]] = RK[r] == SIDE ? -0.31 + 0.0517 * r : 0.1234567 + 0.0371 * r; tu[RVO[r]+1] = 0.0765432 + 0.0213 * r; }
  for (int i = 0; i < N; i++) { double bx, by; rep_base(PR[i], tu, &bx, &by); apply(PG[i], bx, by, &p[2*i], &p[2*i+1]); }
  static int PERM[MAXG][MAXN];
  for (int g = 0; g < NG; g++)
    for (int i = 0; i < N; i++) {
      double x, y; apply(g, p[2*i], p[2*i+1], &x, &y); int f = -1;
      for (int j = 0; j < N; j++) if (fabs(p[2*j]-x) < 1e-7 && fabs(p[2*j+1]-y) < 1e-7) { f = j; break; }
      if (f < 0) return;   /* not a permutation: use all triangles */
      PERM[g][i] = f;
    }
  for (int t = 0; t < NT; t++) {
    int mn = t, imgs[MAXG], ni = 0;
    for (int g = 0; g < NG; g++) {
      int a = PERM[g][TI[t]], b = PERM[g][TJ[t]], c = PERM[g][TK[t]], x;
      if (a > b) { x = a; a = b; b = x; } if (b > c) { x = b; b = c; c = x; } if (a > b) { x = a; a = b; b = x; }
      int ti = TID[a][b][c], dup = 0; if (ti < mn) mn = ti;
      for (int q = 0; q < ni; q++) if (imgs[q] == ti) dup = 1;
      if (!dup) imgs[ni++] = ti;
    }
    TCAN[t] = (mn == t); TMUL[t] = (float)ni;
  }
  CAN_OK = 1;
}
static void build_can(void) {
  if (NG <= 1 || NOCAN) { CAN_OK = 0; return; }
  unsigned long long key = 1469598103934665603ULL ^ (unsigned long long)N * 1099511628211ULL ^ (unsigned long long)NG;
  for (int r = 0; r < NR; r++) key = (key ^ (unsigned long long)(RK[r] * 64 + RA[r] + 1)) * 1099511628211ULL;
  key ^= (unsigned long long)NR << 48;
  for (int c = 0; c < CANC; c++)
    if (CANCACHE[c].can && CANCACHE[c].key == key && CANCACHE[c].nt == NT) {
      memcpy(TCAN, CANCACHE[c].can, NT); memcpy(TMUL, CANCACHE[c].mul, sizeof(float) * NT); CAN_OK = CANCACHE[c].ok; return;
    }
  build_can_raw();
  int c = CANNEXT; CANNEXT = (CANNEXT + 1) % CANC;
  free(CANCACHE[c].can); free(CANCACHE[c].mul);
  CANCACHE[c].can = malloc(NT); CANCACHE[c].mul = malloc(sizeof(float) * NT);
  memcpy(CANCACHE[c].can, TCAN, NT); memcpy(CANCACHE[c].mul, TMUL, sizeof(float) * NT);
  CANCACHE[c].key = key; CANCACHE[c].nt = NT; CANCACHE[c].ok = CAN_OK;
}
/* orbit sizes and the point list from RK / RA; returns 0 if the orbits do not add up to N points */
static int build_orbits(void) {
  static const double tu[2] = {0.1234567, 0.0765432};
  int np = 0;
  NV = 0;
  for (int r = 0; r < NR; r++) {
    RVO[r] = NV; NV += SLOT2 ? 2 : rep_nvar(r);
    double bx, by, sv[2*MAXG]; rep_point(RK[r], RA[r], tu, &bx, &by);
    RSZ[r] = orbit_images(bx, by, sv, RG[r]);
    for (int q = 0; q < RSZ[r]; q++) { if (np < MAXN) { PR[np] = r; PG[np] = RG[r][q]; } np++; }
  }
  if (np != N) { CAN_OK = 0; return 0; }
  build_can();
  return 1;
}
static void u2p(const double *u, double *p) {
  for (int i = 0; i < N; i++) { double bx, by; rep_base(PR[i], u, &bx, &by); apply(PG[i], bx, by, &p[2*i], &p[2*i+1]); }
}
static void p2u(const double *p, double *u) {   /* the first image of every orbit is the identity image */
  memset(u, 0, sizeof(double) * NV);
  for (int r = 0, i = 0; r < NR; i += RSZ[r++]) rep_param(r, p[2*i], p[2*i+1], u);
}

/* ---------------- masks, geometry ---------------- */
static unsigned char VACP[MAXN]; static int NVAC;   /* defect orbits: vacant images (points that do not exist) */
static unsigned char VACT[MAXT];                   /* triangles touching a vacant point */
static int RMASK[MAXR];                            /* orbits taken out by the ruin step */
static unsigned char SKIP[MAXT]; static int use_skip;   /* triangles ignored: touch a masked orbit or a vacancy */

static void apply_masks(void) {
  int nm = 0; for (int r = 0; r < NR; r++) nm += RMASK[r];
  if (!nm && !NVAC) { use_skip = 0; return; }
  for (int t = 0; t < NT; t++) SKIP[t] = (NVAC && VACT[t]) | RMASK[PR[TI[t]]] | RMASK[PR[TJ[t]]] | RMASK[PR[TK[t]]];
  use_skip = 1;
}
static double tri2(const double *p, int t) {   /* twice the area of triangle t */
  int i = TI[t], j = TJ[t], k = TK[t];
  return fabs((p[2*j]-p[2*i])*(p[2*k+1]-p[2*i+1]) - (p[2*k]-p[2*i])*(p[2*j+1]-p[2*i+1]));
}
static double minarea(const double *p) {
  double m = 1e300;
  for (int t = 0; t < NT; t++) { if (use_skip && SKIP[t]) continue; double a = tri2(p, t); if (a < m) m = a; }
  return 0.5 * m;
}
static double hull(const double *p) { return hull_area(p, NULL, VACP, NULL, NULL); }
static double value(const double *p) { return minarea(p) / (V == 'c' ? hull(p) : CAREA); }

/* stress of every orbit: sum over its (unskipped) triangles of exp(-15 (A_t - A_min) / A_min) */
static void orbit_stress(const double *p, double *srep) {
  static double a[MAXT]; double m = 1e300;
  for (int t = 0; t < NT; t++) { a[t] = tri2(p, t); if (!(use_skip && SKIP[t]) && a[t] < m) m = a[t]; }
  if (m < 1e-300) m = 1e-300;
  for (int r = 0; r < NR; r++) srep[r] = 0;
  for (int t = 0; t < NT; t++) {
    if (use_skip && SKIP[t]) continue;
    double z = SRR_STRESS * (a[t] - m) / m; if (z > 40) continue;
    double w = exp(-z); srep[PR[TI[t]]] += w; srep[PR[TJ[t]]] += w; srep[PR[TK[t]]] += w;
  }
}

/* ---------------- objective and polishes ---------------- */
static int SUBT[MAXT], NSUB, USE_SUB;   /* 1: subset, 2: subset of canonical triangles with multiplicities */

static double fgrad(const double *u, double *g) {
  double p[2*MAXN], gp[2*MAXN], gH[2*MAXN], gb[MAXR][2], P = 0;
  u2p(u, p);
  double H = V == 'c' ? hull_area(p, gH, VACP, NULL, NULL) : CAREA;
  double F = softmin(p, H, gH, gp, USE_SUB ? SUBT : NULL, NSUB, 36, USE_SUB == 2 ? TMUL : NULL, use_skip ? SKIP : NULL);
  memset(g, 0, sizeof(double) * NV);
  memset(gb, 0, sizeof(double) * 2 * NR);
  for (int i = 0; i < N; i++) {   /* chain rule to the representatives: G^T dF/dp */
    int r = PR[i], q = PG[i];
    gb[r][0] += GM[q][0] * gp[2*i] + GM[q][2] * gp[2*i+1];
    gb[r][1] += GM[q][1] * gp[2*i] + GM[q][3] * gp[2*i+1];
  }
  for (int r = 0; r < NR; r++) {   /* plus the penalty MU * (squared distance outside the region) */
    int o = RVO[r];
    if (RK[r] == GEN) {
      double x = u[o], y = u[o+1], gx = gb[r][0], gy = gb[r][1];
      for (int k = 0; k < NSIDE; k++) { double e = HN[k][0]*x + HN[k][1]*y - HH[k]; if (e > 0) { P += e*e; gx += 2*MU*e*HN[k][0]; gy += 2*MU*e*HN[k][1]; } }
      g[o] = gx; g[o+1] = gy;
    } else if (RK[r] == AXIS) {
      double ax = MIRA[RA[r]][0], ay = MIRA[RA[r]][1], t = u[o], gt = ax * gb[r][0] + ay * gb[r][1];
      for (int k = 0; k < NSIDE; k++) {
        double e = (HN[k][0]*ax + HN[k][1]*ay) * t - HH[k];
        if (e > 0) { double d = HN[k][0]*ax + HN[k][1]*ay; P += e*e; gt += 2*MU*e*d; }
      }
      g[o] = gt;
    } else if (RK[r] == SIDE) {
      const double *a = VX[RA[r]], *b = VX[(RA[r]+1)%NSIDE]; double dx = b[0]-a[0], dy = b[1]-a[1], t = u[o];
      double gt = dx * gb[r][0] + dy * gb[r][1], L2 = dx*dx + dy*dy;
      if (t < -0.5) { P += L2*(t+0.5)*(t+0.5); gt += 2*MU*L2*(t+0.5); } else if (t > 0.5) { P += L2*(t-0.5)*(t-0.5); gt += 2*MU*L2*(t-0.5); }
      g[o] = gt;
    }
  }
  return F + MU * P;
}

static void normalize_convex(double *u) {   /* hull area 1; only the trivial group is also re-centred */
  double p[2*MAXN]; u2p(u, p);
  double s = 1 / sqrt(hull(p)), cx = 0, cy = 0;
  if (NG == 1) { for (int i = 0; i < N; i++) { cx += p[2*i]; cy += p[2*i+1]; } cx /= N; cy /= N; }
  for (int r = 0; r < NR; r++) {
    int o = RVO[r];
    if (RK[r] == GEN) { u[o] = (u[o] - cx) * s; u[o+1] = (u[o+1] - cy) * s; } else if (RK[r] == AXIS) u[o] *= s;
  }
}
static void project_u(double *u) {   /* representatives back into the region */
  for (int r = 0; r < NR; r++) {
    int o = RVO[r];
    if (RK[r] == GEN) put_inside(&u[o], &u[o+1]);
    else if (RK[r] == SIDE) { if (u[o] < -0.5) u[o] = -0.5; if (u[o] > 0.5) u[o] = 0.5; }
    else if (RK[r] == AXIS && NSIDE) {
      double x = u[o]*MIRA[RA[r]][0], y = u[o]*MIRA[RA[r]][1], x0 = x, y0 = y; put_inside(&x, &y);
      if (x != x0 || y != y0) u[o] = x*MIRA[RA[r]][0] + y*MIRA[RA[r]][1];
    }
  }
}
static double checked_value(double *u) {   /* 0 for a broken (non-finite, escaped or outside) configuration */
  double p[2*MAXN]; u2p(u, p);
  for (int i = 0; i < 2*N; i++) if (!isfinite(p[i]) || fabs(p[i]) > 10) return 0;
  double v = value(p);
  return isfinite(v) ? v : 0;
}

/* staged polish: beta continuation from stage b0 up to the sharp final stage */
static double local_opt(double *u, int b0) {
  double p[2*MAXN];
  if (V == 'c') normalize_convex(u);
  double area = V == 'c' ? 1.0 : CAREA, d0 = sqrt(area / N);
  for (int b = b0; b <= NBETA; b++) {
    u2p(u, p);
    double c = minarea(p) / (V == 'c' ? hull(p) : 1.0), cfl = CFL / (N * N) * area;
    if (b < NBETA) { if (c < cfl) c = cfl; BETA = BETAS[b]; } else BETA = 8000;
    CREF = c; MU = 50 * BETA * (d0 / c) * (d0 / c);
    lbfgs(u, NV, b < NBETA ? ITS[b] : 150, fgrad);
  }
  project_u(u);
  if (V == 'c') normalize_convex(u);
  u2p(u, p);
  for (int i = 0; i < N && NSIDE; i++)
    for (int k = 0; k < NSIDE; k++) if (HN[k][0]*p[2*i] + HN[k][1]*p[2*i+1] - HH[k] > 1e-9) return 0;
  return checked_value(u);
}

/* soft polish at sharpness bf relative to the current minimum, over the near-active canonical triangles */
static double soft_polish(double *u, double bf, int iters) {
  static double a[MAXT];
  double p[2*MAXN];
  if (V == 'c') normalize_convex(u);
  double area = V == 'c' ? 1.0 : CAREA, d0 = sqrt(area / N);
  u2p(u, p);
  double c = minarea(p) / (V == 'c' ? hull(p) : 1.0), fl = 0.02 / (N * N) * area;
  if (c < fl) c = fl;
  BETA = bf; CREF = c; MU = 50 * BETA * (d0 / c) * (d0 / c);
  double m = 1e300;
  for (int t = 0; t < NT; t++) {
    if (CAN_OK && !TCAN[t]) continue;
    a[t] = tri2(p, t); if (!(use_skip && SKIP[t]) && a[t] < m) m = a[t];
  }
  NSUB = 0;
  for (int t = 0; t < NT; t++) if ((!CAN_OK || TCAN[t]) && !(use_skip && SKIP[t]) && a[t] <= SUBK * m) SUBT[NSUB++] = t;
  if (NSUB > 0) USE_SUB = CAN_OK ? 2 : 1;
  lbfgs(u, NV, iters, fgrad);
  USE_SUB = 0;
  project_u(u);
  if (V == 'c') normalize_convex(u);
  return checked_value(u);
}

/* ---------------- random parameters and structures ---------------- */
static void sample_rep(int r, double *u) {
  double R = V == 'c' ? 0.6 : 1.0, *o = u + RVO[r]; o[0] = 0; o[1] = 0;
  if (RK[r] == GEN) {
    if (V == 'c') { double rr = sqrt(urand()) * R, th = 2*M_PI*urand(); o[0] = rr*cos(th); o[1] = rr*sin(th); }
    else {   /* uniform in the region by rejection from [-1,1]^2 */
      double x, y; int ok;
      do { x = urand()*2-1; y = urand()*2-1; ok = 1; for (int k = 0; k < NSIDE; k++) if (HN[k][0]*x + HN[k][1]*y > HH[k]) ok = 0; } while (!ok);
      o[0] = x; o[1] = y;
    }
  } else if (RK[r] == AXIS) o[0] = (urand()*2-1) * (V == 'c' ? R : 0.5);
  else if (RK[r] == SIDE) o[0] = (urand() - 0.5) * 0.95;
}
static void random_u(double *u) { for (int r = 0; r < NR; r++) sample_rep(r, u); }

static int on_side(double x, double y, int k) { return fabs(HN[k][0]*x + HN[k][1]*y - HH[k]) < 1e-9; }

/* random orbit structure for N points: every side carries at most 2 points (square: sometimes 1),
   sometimes a corner orbit, a centre with probability 1/2, 0-5 axis orbits, the rest generic orbits;
   structures that force a degenerate triangle are rejected */
static int choose_structure(void) {
  for (int attempt = 0; attempt < 200; attempt++) {
    int cap[8], onside[8] = {0};
    NR = 0;
    for (int k = 0; k < 8; k++) cap[k] = (V == 's' && urand() < PSIDE1) ? 1 : 2;
    if (NSIDE && urand() < 0.08) { RK[NR] = CORNER; RA[NR] = rnd64() % NSIDE; NR++; }
    if (NSIDE) {
      if (NR) {   /* the corner orbit's points count for both adjacent sides */
        double sv[2*MAXG]; int ni = orbit_images(VX[RA[0]][0], VX[RA[0]][1], sv, NULL);
        for (int q = 0; q < ni; q++) for (int k = 0; k < NSIDE; k++) onside[k] += on_side(sv[2*q], sv[2*q+1], k);
      }
      int order[8]; for (int k = 0; k < NSIDE; k++) order[k] = k;
      for (int k = NSIDE - 1; k > 0; k--) { int j = rnd64() % (k + 1), t = order[k]; order[k] = order[j]; order[j] = t; }
      for (int pass = 0; pass < 3; pass++)
        for (int q = 0; q < NSIDE; q++) {   /* add a side orbit if no side exceeds its capacity */
          int k = order[q], add[8] = {0}, ok = 1;
          if (onside[k] >= cap[k]) continue;
          double bx, by, t = 0.1234567; rep_point(SIDE, k, &t, &bx, &by);
          for (int g = 0; g < NG; g++) { double x, y; apply(g, bx, by, &x, &y); for (int kk = 0; kk < NSIDE; kk++) add[kk] += on_side(x, y, kk); }
          for (int kk = 0; kk < NSIDE; kk++) if (onside[kk] + add[kk] > cap[kk]) ok = 0;
          if (!ok) continue;
          for (int kk = 0; kk < NSIDE; kk++) onside[kk] += add[kk];
          RK[NR] = SIDE; RA[NR] = k; NR++;
        }
    }
    if (NG > 1 && urand() < 0.5) { RK[NR] = CENTER; RA[NR] = 0; NR++; }
    if (NMIR) { int na = rnd64() % 6; for (int q = 0; q < na; q++) { RK[NR] = AXIS; RA[NR] = rnd64() % NMIR; NR++; } }
    int tot = 0;
    for (int r = 0; r < NR; r++) { double bx, by, sv[2*MAXG], tu[2] = {0.1234567, 0.0765432}; rep_point(RK[r], RA[r], tu, &bx, &by); tot += orbit_images(bx, by, sv, NULL); }
    int rem = N - tot;
    if (rem < 0 || rem % NG) continue;
    for (int q = 0; q < rem / NG; q++) { RK[NR] = GEN; RA[NR] = 0; NR++; }
    if (!build_orbits()) continue;
    double tu[2*MAXN], tp[2*MAXN]; int bad = 0;
    for (int z = 0; z < 2 && !bad; z++) { random_u(tu); u2p(tu, tp); if (minarea(tp) < 1e-12) bad = 1; }
    if (!bad) return 1;
  }
  return 0;
}

/* ---------------- seeds ---------------- */
static int detect_type(double x, double y) {   /* kind << 8 | aux, native coordinates */
  double e = 1e-7;
  for (int k = 0; k < NSIDE; k++) if (fabs(x - VX[k][0]) < e && fabs(y - VX[k][1]) < e) return (CORNER << 8) | k;
  for (int k = 0; k < NSIDE; k++) if (fabs(HN[k][0]*x + HN[k][1]*y - HH[k]) < e) return (SIDE << 8) | k;
  return GEN << 8;
}
/* decompose a configuration (native frame) into orbits of the current group */
static int orbit_decompose(const double *p, double *u) {
  int used[MAXN] = {0}; double rep[MAXR][2];
  NR = 0;
  for (int i = 0; i < N; i++) {
    if (used[i]) continue;
    double x = p[2*i], y = p[2*i+1]; int imgs[MAXG], ni = 0;
    for (int g = 0; g < NG; g++) {
      double gx, gy; apply(g, x, y, &gx, &gy); int f = -1, dup = 0;
      for (int j = 0; j < N; j++) if (fabs(p[2*j]-gx) < 2e-6 && fabs(p[2*j+1]-gy) < 2e-6) { f = j; break; }
      if (f < 0) return 0;
      for (int q = 0; q < ni; q++) if (imgs[q] == f) dup = 1;
      if (!dup) imgs[ni++] = f;
    }
    for (int q = 0; q < ni; q++) used[imgs[q]] = 1;
    int kind = GEN, aux = 0, t = detect_type(x, y);
    if (fabs(x) < 1e-7 && fabs(y) < 1e-7) kind = CENTER;
    else if ((t >> 8) != GEN) { kind = t >> 8; aux = t & 255; }
    else if (ni < NG) { for (int m = 0; m < NMIR; m++) if (fabs(-MIRA[m][1]*x + MIRA[m][0]*y) < 1e-6) { kind = AXIS; aux = m; break; } }
    RK[NR] = kind; RA[NR] = aux; rep[NR][0] = x; rep[NR][1] = y; NR++;
  }
  if (!build_orbits()) return 0;
  memset(u, 0, sizeof(double) * NV);
  for (int r = 0; r < NR; r++) rep_param(r, rep[r][0], rep[r][1], u);
  return 1;
}

static int GK;   /* candidates of greedy_reinsert: 200 (seeding), 800 (walk) */
/* place representative r at the best of K random positions (by the min area over the unskipped triangles;
   convex: / hull area), drawn by sample_rep (one_rep) or from a whole random parameter vector */
static void place_best(double *u, int r, int K, int one_rep) {
  int o = RVO[r], nv = rep_nvar(r);
  if (!nv) return;
  double tmp[2*MAXN], p[2*MAXN], rr[2*MAXN], bestv = -1, bu[2] = {u[o], nv > 1 ? u[o+1] : 0};
  memcpy(tmp, u, sizeof(double) * NV);
  for (int c = 0; c < K; c++) {
    if (one_rep) sample_rep(r, rr); else random_u(rr);
    tmp[o] = rr[o]; if (nv > 1) tmp[o+1] = rr[o+1];
    u2p(tmp, p);
    double v = minarea(p) / (V == 'c' ? hull(p) : 1.0);
    if (v > bestv) { bestv = v; bu[0] = tmp[o]; bu[1] = nv > 1 ? tmp[o+1] : 0; }
  }
  u[o] = bu[0]; if (nv > 1) u[o+1] = bu[1];
}
static void greedy_reinsert(double *u, int r) { place_best(u, r, GK, 0); }

/* Start from a G-symmetric configuration with m != N points: decompose it, then add or remove one or two
   whole orbits whose sizes make up the difference (new orbits placed greedily, 6 tries each), polish from
   beta stage 1 and keep the best option.  Convex seeds are centred, scaled to hull area 1 and rotated
   until the mirrors fit. */
static int seed_other_n(const double *q, int m, St *out) {
  double p[2*MAXN], u0[2*MAXN]; int n = N, found = 0;
  N = m; from_site(q, p, m);
  if (V == 'c') {
    double cx = 0, cy = 0;
    for (int i = 0; i < m; i++) { cx += p[2*i]; cy += p[2*i+1]; }
    cx /= m; cy /= m;
    for (int i = 0; i < m; i++) { p[2*i] -= cx; p[2*i+1] -= cy; }
    double sc = 1 / sqrt(hull(p)); for (int i = 0; i < 2*m; i++) p[i] *= sc;
    int nang = NMIR ? m + m * m : 1;   /* mirror candidates: through a point, or bisecting two at equal radius */
    for (int ai = 0; ai < nang && !found; ai++) {
      double th = 0, pr[2*MAXN];
      if (NMIR) {
        if (ai < m) th = atan2(p[2*ai+1], p[2*ai]);
        else {
          int i = (ai - m) / m, j = (ai - m) % m; if (j <= i) continue;
          if (fabs(hypot(p[2*i], p[2*i+1]) - hypot(p[2*j], p[2*j+1])) > 1e-6) continue;
          th = 0.5 * (atan2(p[2*i+1], p[2*i]) + atan2(p[2*j+1], p[2*j]));
        }
      }
      double rot = M_PI / 2 - th, c = cos(rot), sn = sin(rot);
      for (int i = 0; i < m; i++) { pr[2*i] = c*p[2*i] - sn*p[2*i+1]; pr[2*i+1] = sn*p[2*i] + c*p[2*i+1]; }
      if (orbit_decompose(pr, u0) && NR < m) found = 1;
    }
  } else found = orbit_decompose(p, u0);
  N = n;
  if (!found) { fprintf(stderr, "seed is not %s-symmetric\n", GNAME); return 0; }
  int bNR = NR, bK[MAXR], bA[MAXR], bsz[MAXR], d = N - m; double bp[MAXR][2];
  for (int r = 0; r < NR; r++) { bK[r] = RK[r]; bA[r] = RA[r]; bsz[r] = RSZ[r]; bp[r][0] = u0[RVO[r]]; bp[r][1] = RK[r] == GEN ? u0[RVO[r]+1] : 0; }
  set_points(N);
  int addk[MAXG + 2], adda[MAXG + 2], addsz[MAXG + 2], nadd = 0;   /* orbit kinds that can be added */
  if (d > 0) {
    addk[nadd] = CENTER; adda[nadd] = 0; addsz[nadd++] = 1;
    for (int mi = 0; mi < NMIR; mi++) { addk[nadd] = AXIS; adda[nadd] = mi; addsz[nadd++] = NG / 2; }
    addk[nadd] = GEN; adda[nadd] = 0; addsz[nadd++] = NG;
  }
  int cNR = 0, cK[MAXR], cA[MAXR], tries = 0, lim = d > 0 ? nadd : bNR;
  out->v = -1;
  for (int a1 = -1; a1 < lim; a1++)
    for (int a2 = a1; a2 < lim; a2++) {
      if (a2 < 0) continue;
      NR = 0;
      if (d < 0) {   /* remove orbits a1 (if >= 0) and a2 */
        int tot = 0;
        for (int r = 0; r < bNR; r++) if (r != a1 && r != a2) { RK[NR] = bK[r]; RA[NR] = bA[r]; NR++; } else tot += bsz[r];
        if (tot != -d) continue;
      } else {       /* add orbits of kinds a1 (if >= 0) and a2 */
        for (int r = 0; r < bNR; r++) { RK[NR] = bK[r]; RA[NR] = bA[r]; NR++; }
        if ((a1 >= 0 ? addsz[a1] : 0) + addsz[a2] != d) continue;
        if (a1 >= 0) { RK[NR] = addk[a1]; RA[NR] = adda[a1]; NR++; }
        RK[NR] = addk[a2]; RA[NR] = adda[a2]; NR++;
        int nc = 0; for (int r = 0; r < NR; r++) nc += RK[r] == CENTER;
        if (nc > 1) continue;
      }
      if (!build_orbits()) continue;
      for (int rep = 0; rep < 6; rep++) {
        St c; random_u(c.u);
        int r2 = 0;
        for (int r = 0; r < bNR; r++) {   /* kept orbits keep their parameters */
          if (d < 0 && (r == a1 || r == a2)) continue;
          int o = RVO[r2];
          if (RK[r2] == GEN) { c.u[o] = bp[r][0]; c.u[o+1] = bp[r][1]; } else if (RK[r2] == AXIS || RK[r2] == SIDE) c.u[o] = bp[r][0];
          r2++;
        }
        for (int r = r2; r < NR; r++) greedy_reinsert(c.u, r);
        double pp[2*MAXN]; u2p(c.u, pp); if (minarea(pp) < 1e-12) continue;
        c.v = local_opt(c.u, 1); tries++;
        if (c.v > out->v) { *out = c; cNR = NR; memcpy(cK, RK, sizeof RK); memcpy(cA, RA, sizeof RA); }
        if (d < 0) break;   /* removal is deterministic */
      }
    }
  if (out->v < 0) { fprintf(stderr, "no symmetric option from %d to %d points\n", m, N); return 0; }
  NR = cNR; memcpy(RK, cK, sizeof RK); memcpy(RA, cA, sizeof RA); build_orbits();
  fprintf(stderr, "[%c%d %s] symmetric seed from %d points (%d options): %.10f\n", V, N, GNAME, m, tries, out->v);
  return 1;
}

/* ---------------- output ---------------- */
static char OUTF[512];
static void save(const St *s) {
  double p[2*MAXN], q[2*MAXN]; char hdr[512]; int len;
  u2p(s->u, p); to_site(p, q, N);
  len = snprintf(hdr, sizeof hdr, "# %c n=%d value=%.15g group=%s reps=", V, N - NVAC, s->v, GNAME);
  for (int r = 0; r < NR && len < (int)sizeof hdr - 32; r++) hdr[len++] = "GASXK"[RK[r]];
  hdr[len] = 0;
  if (NVAC) snprintf(hdr + len, sizeof hdr - len, " vacancies=%d", NVAC);
  save_points(OUTF, hdr, q, VACP);
}

/* ---------------- ruin & recreate on orbits ---------------- */
static int TRL[3 * MAXT], TRO[MAXR + 1];   /* (canonical) triangles touching orbit r: TRL[TRO[r] .. TRO[r+1]) */
static void build_trl(void) {
  int pos[MAXR];
  for (int pass = 0; pass < 2; pass++) {   /* count, then fill */
    if (pass) { TRO[0] = 0; for (int r = 0; r < NR; r++) { TRO[r+1] = TRO[r] + pos[r]; pos[r] = TRO[r]; } }
    else for (int r = 0; r < NR; r++) pos[r] = 0;
    for (int t = 0; t < NT; t++) {
      if (CAN_OK && !TCAN[t]) continue;
      int a = PR[TI[t]], b = PR[TJ[t]], c = PR[TK[t]];
      if (pass) { TRL[pos[a]++] = t; if (b != a) TRL[pos[b]++] = t; if (c != a && c != b) TRL[pos[c]++] = t; }
      else { pos[a]++; if (b != a) pos[b]++; if (c != a && c != b) pos[c]++; }
    }
  }
}

/* mask the k most stressed free orbits */
static void ruin_reps(const double *u, int k) {
  double p[2*MAXN], srep[MAXR]; u2p(u, p);
  orbit_stress(p, srep);
  for (int r = 0; r < NR; r++) srep[r] *= 0.9 + 0.2 * urand();
  for (int z = 0; z < k; z++) {
    int bi = -1;
    for (int r = 0; r < NR; r++) if (!RMASK[r] && RK[r] != CENTER && RK[r] != CORNER && (bi < 0 || srep[r] > srep[bi])) bi = r;
    if (bi < 0) break;
    RMASK[bi] = 1;
  }
  apply_masks();
}

typedef struct { double key; int t; } Tkey;
static int cmp_tkey(const void *a, const void *b) {
  const Tkey *x = a, *y = b; return x->key < y->key ? -1 : x->key > y->key ? 1 : x->t - y->t;
}
/* unmask orbit r and put it at the best of SRR_K candidates, scored by the smallest unmasked triangle that
   touches the orbit (convex: / hull area) */
static void reinsert_rep(double *u, int r) {
  static Tkey OT[3 * MAXT];
  RMASK[r] = 0; apply_masks();
  int o = RVO[r], nv = rep_nvar(r);
  if (!nv) return;
  double p[2*MAXN], tmp[2*MAXN], rr[2*MAXN]; u2p(u, p); memcpy(tmp, u, sizeof(double) * NV);
  int st = 0; for (int q = 0; q < r; q++) st += RSZ[q];
  double best = -1, b0 = u[o], b1 = nv > 1 ? u[o+1] : 0;
  /* most dangerous triangles first: several orbit points, then short pairs of fixed points */
  int no = 0;
  for (int q = TRO[r]; q < TRO[r+1]; q++) {
    int t = TRL[q]; if (use_skip && SKIP[t]) continue;
    int i = TI[t], j = TJ[t], l = TK[t];
    int ii = (i >= st && i < st + RSZ[r]), jj = (j >= st && j < st + RSZ[r]), ll = (l >= st && l < st + RSZ[r]);
    double key = 0;
    if (ii + jj + ll == 1) {
      int a = ii ? j : i, b = ll ? j : l; if (jj) { a = i; b = l; }
      key = (p[2*a]-p[2*b])*(p[2*a]-p[2*b]) + (p[2*a+1]-p[2*b+1])*(p[2*a+1]-p[2*b+1]);
    }
    OT[no].key = key; OT[no].t = t; no++;
  }
  qsort(OT, no, sizeof(Tkey), cmp_tkey);
  for (int c = 0; c < SRR_K; c++) {
    sample_rep(r, rr); tmp[o] = rr[o]; if (nv > 1) tmp[o+1] = rr[o+1];
    double bx, by; rep_base(r, tmp, &bx, &by);
    for (int z = 0; z < RSZ[r]; z++) apply(RG[r][z], bx, by, &p[2*(st+z)], &p[2*(st+z)+1]);
    double sc = V == 'c' ? 1.0 / hull(p) : 1.0, lim = best / sc, mn = 1e300;
    for (int q = 0; q < no; q++) { double a = tri2(p, OT[q].t); if (a < mn) { mn = a; if (mn <= lim) break; } }
    if (mn * sc > best) { best = mn * sc; b0 = tmp[o]; b1 = nv > 1 ? tmp[o+1] : 0; }
  }
  u[o] = b0; if (nv > 1) u[o+1] = b1;
}
static void reinsert_masked(double *u) {   /* random order, short soft polish after each orbit */
  int ord[MAXR], m = 0;
  for (int r = 0; r < NR; r++) if (RMASK[r]) ord[m++] = r;
  for (int i = m - 1; i > 0; i--) { int j = rnd64() % (i + 1), t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
  for (int i = 0; i < m; i++) { reinsert_rep(u, ord[i]); soft_polish(u, 15, 20); }
}
static double finalize(double *u) { soft_polish(u, 20, 50); return soft_polish(u, 50, 50); }

/* orbit overshoot (cold runs): one extra generic orbit during recreation, then the most stressed generic
   orbit is dropped */
static int OVERSHOOT;
static void restructure(void) { set_points(N); build_orbits(); build_trl(); }
static void ov_add(double *u) {
  RK[NR] = GEN; RA[NR] = 0; RMASK[NR] = 1; u[2*NR] = 0; u[2*NR+1] = 0; NR++; N += NG;
  restructure(); apply_masks();
}
static void ov_prune(double *u) {
  double p[2*MAXN], srep[MAXR]; u2p(u, p);
  orbit_stress(p, srep);
  int bi = -1;
  for (int r = 0; r < NR; r++) if (RK[r] == GEN) { srep[r] *= 0.9 + 0.2 * urand(); if (bi < 0 || srep[r] > srep[bi]) bi = r; }
  if (bi < 0) return;
  for (int q = bi; q < NR - 1; q++) { RK[q] = RK[q+1]; RA[q] = RA[q+1]; RMASK[q] = RMASK[q+1]; u[2*q] = u[2*q+2]; u[2*q+1] = u[2*q+3]; }   /* SLOT2 */
  NR--; N -= NG; restructure(); apply_masks();
}

/* defect orbits: VAC generic orbits lose one image, FREE generic orbits keep only their identity image */
static int VAC_REQ, FREE_REQ;
static int setup_vacancies(void) {
  int gl[MAXR], ng = 0, st[MAXR], c = 0;
  memset(VACP, 0, sizeof VACP); NVAC = 0;
  for (int r = 0; r < NR; r++) { st[r] = c; c += RSZ[r]; if (RK[r] == GEN && RSZ[r] >= 2) gl[ng++] = r; }
  if (ng < VAC_REQ + FREE_REQ) return 0;
  for (int i = ng - 1; i > 0; i--) { int j = rnd64() % (i + 1), t = gl[i]; gl[i] = gl[j]; gl[j] = t; }
  for (int z = 0; z < FREE_REQ; z++) { int r = gl[z]; for (int q = 1; q < RSZ[r]; q++) VACP[st[r] + q] = 1; }
  for (int z = 0; z < VAC_REQ; z++) { int r = gl[FREE_REQ + z]; VACP[st[r] + 1 + rnd64() % (RSZ[r] - 1)] = 1; }
  for (int i = 0; i < N; i++) NVAC += VACP[i];
  for (int t = 0; t < NT; t++) VACT[t] = VACP[TI[t]] | VACP[TJ[t]] | VACP[TK[t]];
  return 1;
}

static long SRR_IT;
static int same_structure(int nr, const int *rk, const int *ra) {
  return nr == NR && !memcmp(rk, RK, sizeof(int) * NR) && !memcmp(ra, RA, sizeof(int) * NR);
}
static void new_best(St *best, const St *q, const char *what, unsigned long long seed, long tree) {
  *best = *q; save(best);
  fprintf(stderr, "[%c%d %s rr s%llu] tree %ld %s best %.10f\n", V, N, GNAME, seed, tree, what, best->v);
}

/* one tree of ruin & recreate from *cur until tend; at the end the tree's best soft state gets the sharp
   polish */
static void rr_tree(St *cur, St *best, double tend, double tlim, unsigned long long seed, long tree) {
  int nfree = 0; for (int r = 0; r < NR; r++) if (RK[r] != CENTER && RK[r] != CORNER) nfree++;
  St tmp, tb; tb.v = -1;
  int k0[MAXR], a0[MAXR], tbk[MAXR], tba[MAXR], tbn = 0;
  char it[32];
  while (now() < tend && now() < tlim) {
    SRR_IT++; tmp = *cur;
    int nr0 = NR, n0 = N; memcpy(k0, RK, sizeof k0); memcpy(a0, RA, sizeof a0);
    int k = (int)(nfree * (RUIN_F0 + (RUIN_F1 - RUIN_F0) * urand()) + 0.5); if (k < 1) k = 1;
    ruin_reps(tmp.u, k);
    int ov = OVERSHOOT && !NVAC && N + NG <= MAXN && urand() < 0.67;
    if (ov) ov_add(tmp.u);
    reinsert_masked(tmp.u);
    if (ov) ov_prune(tmp.u);
    tmp.v = finalize(tmp.u);
    if (tmp.v > tb.v) { tb = tmp; tbn = NR; memcpy(tbk, RK, sizeof tbk); memcpy(tba, RA, sizeof tba); }
    if (tmp.v > best->v * (1 + 1e-9)) {   /* soft value above the incumbent: sharp polish */
      St q = tmp; double vq = local_opt(q.u, NBETA - 1);
      if (vq < tmp.v) q = tmp; else q.v = vq;
      if (q.v > best->v * (1 + 1e-9)) { snprintf(it, sizeof it, "it %ld", SRR_IT); new_best(best, &q, it, seed, tree); }
    }
    if (tmp.v > cur->v || urand() < PACC) *cur = tmp;
    else if (!same_structure(nr0, k0, a0)) {   /* the rejected trial reordered the orbits: restore */
      NR = nr0; N = n0; memcpy(RK, k0, sizeof k0); memcpy(RA, a0, sizeof a0); restructure();
    }
  }
  if (tb.v > 0 && tb.v > best->v * 0.97) {
    int ck[MAXR], ca[MAXR], cn = NR, sw = !same_structure(tbn, tbk, tba);
    memcpy(ck, RK, sizeof ck); memcpy(ca, RA, sizeof ca);
    if (sw) { memcpy(RK, tbk, sizeof tbk); memcpy(RA, tba, sizeof tba); NR = tbn; build_orbits(); build_trl(); }
    memset(RMASK, 0, sizeof RMASK); apply_masks();
    St q = tb; double vq = local_opt(q.u, NBETA - 1);
    if (vq > best->v * (1 + 1e-9)) { q.v = vq; new_best(best, &q, "end-polish", seed, tree); }
    if (sw) { memcpy(RK, ck, sizeof ck); memcpy(RA, ca, sizeof ca); NR = cn; build_orbits(); build_trl(); apply_masks(); }
  }
}

static int cold_rr(const Opts *o) {
  St cur, best; best.v = -1;
  double t0 = now(); long trees = 0;
  int nreal = N;
  while (now() - t0 < o->secs) {
    trees++; use_skip = 0; NVAC = 0; memset(RMASK, 0, sizeof RMASK); memset(VACP, 0, sizeof VACP);
    if (VAC_REQ || FREE_REQ) {
      int n = nreal + VAC_REQ + FREE_REQ * (NG - 1);
      if (n > MAXN) { fprintf(stderr, "too many points\n"); return 3; }
      set_points(n);
    }
    int ok, tries = 0;
    while (!(ok = choose_structure()) && ++tries <= 50) ;
    if (!ok) { if (best.v < 0) { fprintf(stderr, "no structure for %c%d %s\n", V, nreal, GNAME); return 3; } continue; }
    if ((VAC_REQ || FREE_REQ) && !setup_vacancies()) {
      if (best.v < 0 && trees > 50) { fprintf(stderr, "no defect structure\n"); return 3; }
      continue;
    }
    build_trl();
    random_u(cur.u);   /* build the structure by greedy insertion of all free orbits */
    for (int r = 0; r < NR; r++) RMASK[r] = RK[r] != CENTER && RK[r] != CORNER;
    apply_masks();
    reinsert_masked(cur.u); cur.v = finalize(cur.u);
    rr_tree(&cur, &best, now() + TREE, t0 + o->secs, o->seed, trees);
  }
  fprintf(stderr, "[%c%d rr s%llu] done trees %ld iters %ld best %.12f\n", V, nreal, o->seed, trees, SRR_IT, best.v);
  printf("%.15g\n", best.v);
  return 0;
}

static int warm_rr(const Opts *o, St *best) {
  memset(RMASK, 0, sizeof RMASK); apply_masks(); build_trl();
  double t0 = now(); long trees = 0;
  while (now() - t0 < o->secs) { trees++; St cur = *best; rr_tree(&cur, best, now() + TREE, t0 + o->secs, o->seed, trees); }
  fprintf(stderr, "[%c%d rr s%llu] warm done trees %ld iters %ld best %.12f\n", V, N, o->seed, trees, SRR_IT, best->v);
  printf("%.15g\n", best->v);
  return 0;
}

/* ---------------- hot walk ---------------- */
/* critical-triangle counts per orbit: triangles with area <= f x the minimum */
static void crit_counts(const double *p, double f, int *cnt) {
  double thr = minarea(p) * f;
  memset(cnt, 0, sizeof(int) * NR);
  for (int t = 0; t < NT; t++) if (0.5 * tri2(p, t) <= thr) { cnt[PR[TI[t]]]++; cnt[PR[TJ[t]]]++; cnt[PR[TK[t]]]++; }
}
static int pick_weighted(const int *cnt) {   /* orbit r with probability ~ cnt[r] + 1 */
  int tot = 0; for (int q = 0; q < NR; q++) tot += cnt[q] + 1;
  int w = rnd64() % tot, r = 0;
  while (w >= cnt[r] + 1) { w -= cnt[r] + 1; r++; }
  return r;
}
/* big ruin: remove 1..frac x NR orbits sampled by stress, re-insert each at the best of 400 candidates */
static void big_ruin(double *u, const double *pcur, double frac) {
  double amin = minarea(pcur), srep[MAXR] = {0};
  for (int t = 0; t < NT; t++) {
    double a = 0.5 * tri2(pcur, t), z = (a - amin) / (0.05 * amin); if (z > 30) continue;
    double w = exp(-z); srep[PR[TI[t]]] += w; srep[PR[TJ[t]]] += w; srep[PR[TK[t]]] += w;
  }
  int kmax = (int)(frac * NR); if (kmax < 1) kmax = 1;
  int kk = 1 + rnd64() % kmax, chosen[MAXR], nc = 0;
  memset(RMASK, 0, sizeof RMASK);
  for (int z = 0; z < kk; z++) {
    double tot = 0; for (int r = 0; r < NR; r++) if (!RMASK[r] && RK[r] != CENTER && RK[r] != CORNER) tot += srep[r] + 1e-9;
    if (tot <= 0) break;
    double w = urand() * tot; int r;
    for (r = 0; r < NR; r++) { if (RMASK[r] || RK[r] == CENTER || RK[r] == CORNER) continue; w -= srep[r] + 1e-9; if (w <= 0) break; }
    if (r >= NR) break;
    RMASK[r] = 1; chosen[nc++] = r;
  }
  apply_masks();
  int K = GK / 2 > 50 ? GK / 2 : 50;
  for (int z = 0; z < nc; z++) {   /* candidates: from a whole random vector for C1, one orbit at a time otherwise */
    RMASK[chosen[z]] = 0; apply_masks(); place_best(u, chosen[z], K, NG > 1);
  }
}
/* one of: global jitter (4/20), relocate 1-2 orbits (4/20), greedy re-insertion of 1-3 critical orbits or,
   with probability 0.1, a big ruin (10/20), shake of the orbits near a random point (2/20) */
static void perturb(double *u, double d0, const double *pcur) {
  int mv = (int)(rnd64() % 20), cnt[MAXR];
  if (mv < 4) {
    double sg = d0 * exp(log(0.01) + urand() * (log(0.25) - log(0.01)));
    (void)urand();   /* (a disabled subset-jitter option drew here; kept so seeds reproduce the original walk) */
    for (int v = 0; v < NV; v++) { double e = sg * nrand(); u[v] += e; }   /* rounded product, no FMA */
  } else if (mv < 8) {
    crit_counts(pcur, 1.02, cnt);
    int kk = 1 + rnd64() % 2;
    for (int z = 0; z < kk; z++) {
      int r = urand() < 0.6 ? pick_weighted(cnt) : (int)(rnd64() % NR), o = RVO[r], nv = rep_nvar(r);
      if (!nv) continue;
      if (urand() < 0.5) { double tmp[2*MAXN]; random_u(tmp); u[o] = tmp[o]; if (nv > 1) u[o+1] = tmp[o+1]; }
      else { u[o] += 0.4 * d0 * nrand(); if (nv > 1) u[o+1] += 0.4 * d0 * nrand(); }
    }
  } else if (mv < 18 && urand() < 0.1) {
    big_ruin(u, pcur, urand() < 0.5 ? 0.20 : 0.45);
  } else if (mv < 18) {
    crit_counts(pcur, 1.05, cnt);
    int kk = 1 + rnd64() % 3;
    for (int z = 0; z < kk; z++) { int r = pick_weighted(cnt); greedy_reinsert(u, r); cnt[r] = 0; }
  } else {
    int c = rnd64() % N; double R = d0 * (1 + 2 * urand()), sg = d0 * 0.15, cx = pcur[2*c], cy = pcur[2*c+1];
    for (int r = 0, i = 0; r < NR; i += RSZ[r++]) {
      int near = 0;
      for (int q = 0; q < RSZ[r]; q++) { double ex = pcur[2*(i+q)] - cx, ey = pcur[2*(i+q)+1] - cy; if (ex*ex + ey*ey < R*R) near = 1; }
      if (near) for (int q = 0; q < rep_nvar(r); q++) u[RVO[r]+q] += sg * nrand();
    }
  }
}
static int walk(const Opts *o, St *best) {
  St cur = *best, tmp;
  double d0 = sqrt((V == 'c' ? 1.0 : CAREA) / N), t0 = now(), runbest = cur.v;
  long iter = 0, since = 0;
  while (now() - t0 < o->secs) {
    iter++; since++;
    double pc[2*MAXN]; u2p(cur.u, pc);
    tmp = cur; perturb(tmp.u, d0, pc); tmp.v = local_opt(tmp.u, 1);
    if (tmp.v > cur.v * (1 - WALK_THR * urand())) cur = tmp;
    if (cur.v > runbest * (1 + 1e-5)) { runbest = cur.v; since = 0; }
    if (tmp.v > best->v * (1 + 1e-10)) {
      *best = tmp; save(best);
      fprintf(stderr, "[%c%d %s s%llu] it %ld t=%.0fs best %.10f\n", V, N, GNAME, o->seed, iter, now() - t0, best->v);
    }
    if (since > WALK_RESTART) { since = 0; cur = *best; }
  }
  fprintf(stderr, "[%c%d s%llu] done iters %ld best %.12f\n", V, N, o->seed, iter, best->v);
  printf("%.15g\n", best->v);
  return 0;
}

/* ---------------- driver ---------------- */
int run_sym(const Opts *o) {
  setup_region();
  if (!setup_group(o->group)) { fprintf(stderr, "group %s is not available for %c\n", o->group, V); return 1; }
  VAC_REQ = o->vac; FREE_REQ = o->nfree; NOCAN = VAC_REQ || FREE_REQ;
  GK = o->walk ? 800 : 200;
  OVERSHOOT = !o->init && !NOCAN; SLOT2 = !o->init;
  set_points(o->n);
  snprintf(OUTF, sizeof OUTF, "%s.txt", o->out);
  if (!o->init) return cold_rr(o);

  double q[2*MAXN], p[2*MAXN]; St best;
  int m = load_points(o->init, q, MAXN), b0 = o->walk ? 1 : 2;
  if (m < 3) { fprintf(stderr, "cannot read %s\n", o->init); return 4; }
  if (m != N) {
    if (NG == 1) { fprintf(stderr, "seed has %d points\n", m); return 4; }
    if (!seed_other_n(q, m, &best)) return 4;
  } else if (NG == 1) {   /* trivial group: one orbit per point, side / corner points detected */
    from_site(q, p, N);
    NR = N;
    for (int i = 0; i < N; i++) { int t = V == 'c' ? 0 : detect_type(p[2*i], p[2*i+1]); RK[i] = t >> 8; RA[i] = t & 255; }
    build_orbits(); p2u(p, best.u); best.v = local_opt(best.u, b0);
  } else {
    from_site(q, p, N);
    if (V == 'c') {
      double cx = 0, cy = 0; for (int i = 0; i < N; i++) { cx += p[2*i]; cy += p[2*i+1]; } cx /= N; cy /= N;
      for (int i = 0; i < N; i++) { p[2*i] -= cx; p[2*i+1] -= cy; }
    }
    if (!orbit_decompose(p, best.u)) { fprintf(stderr, "seed is not %s-symmetric\n", GNAME); return 4; }
    best.v = local_opt(best.u, b0);
  }
  save(&best);
  fprintf(stderr, "[%c%d %s s%llu] start %.10f\n", V, N, GNAME, o->seed, best.v);
  return o->walk ? walk(o, &best) : warm_rr(o, &best);
}
