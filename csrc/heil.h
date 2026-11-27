/* heil.h - shared declarations of the Heilbronn search engine `heil`.
 *
 * Problem: place n points in a region so that the smallest triangle they span is as large as possible.
 * Regions: 's' unit square, 't' triangle (0,0),(1,0),(0,1) (value = 2 x min area, i.e. per unit area),
 * 'c' convex position (value = min area / area of the convex hull).
 */
#ifndef HEIL_H
#define HEIL_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 104                                  /* points, including vacant images of defect orbits */
#define MAXT (MAXN * (MAXN - 1) * (MAXN - 2) / 6) /* triangles */

/* ---- current problem ---- */
extern char V;                           /* region: 's', 't' or 'c' */
extern int N, NT;                        /* number of points and of triangles */
extern int TI[MAXT], TJ[MAXT], TK[MAXT]; /* triangle t = (TI[t], TJ[t], TK[t]), i < j < k, lexicographic */
void set_points(int n);                  /* N = n and rebuild the triangle list */

/* ---- random numbers (xorshift64) and clock ---- */
extern unsigned long long rs;
static inline unsigned long long rnd64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline double urand(void) { return (rnd64() >> 11) * (1.0 / 9007199254740992.0); }
double nrand(void);
double now(void);

/* ---- geometry and the smooth objective ---- */
/* Area of the convex hull of the points not flagged in `gone` (NULL: all N points); optional gradient
   gH[2N] and counter-clockwise vertex list hv[0..*nh-1]. */
double hull_area(const double *p, double *gH, const unsigned char *gone, int *hv, int *nh);

/* Soft-min of the triangle areas (log-sum-exp with sharpness BETA, areas measured in units of CREF * H).
   Returns F ~ -min area / (CREF H) and writes dF/dp to gp[2N].  sub/nsub: triangle subset (NULL = all);
   zcut: weights below exp(-zcut) are dropped; mul: optional triangle multiplicities; skip: optional mask
   of ignored triangles.  For convex position H is the hull area and gH its gradient. */
extern double BETA, CREF, MU;
double softmin(const double *p, double H, const double *gH, double *gp, const int *sub, int nsub,
               double zcut, const float *mul, const unsigned char *skip);

/* Limited-memory BFGS (memory 8, Armijo backtracking) on f(u), u in R^nv. */
typedef double (*objective)(const double *u, double *g);
void lbfgs(double *u, int nv, int maxit, objective f);

/* beta continuation: stage b uses BETAS[b] for ITS[b] iterations; stage NBETA is the sharp final polish */
#define NBETA 6
extern const double BETAS[NBETA];
extern const int ITS[NBETA];

/* ---- configuration files: optional '#' header line, then one "x y" pair per line ---- */
int load_points(const char *fn, double *p, int maxk);
void save_points(const char *fn, const char *header, const double *p, const unsigned char *gone);

/* ---- LP relocation and min-norm ascent for square / triangle (reloc.c) ---- */
enum { PIN_FREE = 0, PIN_X0, PIN_X1, PIN_Y0, PIN_Y1, PIN_HYP };   /* side a point is held on */
void reloc_init(int triangle, unsigned long long seed);
void reloc_polish(double *x, double *y, const int *pin);   /* the N current points, site frame */

/* ---- engines ---- */
typedef struct {
  int n;              /* number of points */
  const char *group;  /* symmetry group, "C1" = none */
  double secs;        /* search time after the setup */
  const char *out;    /* best configuration goes to <out>.txt */
  const char *init;   /* start configuration or NULL */
  int walk;           /* hot basin-hopping walk instead of ruin & recreate */
  int vac, nfree;     /* defect classes of the cold symmetric search */
  unsigned long long seed;
} Opts;
int run_free(const Opts *o);   /* free.c: warm ruin & recreate without symmetry */
int run_sym(const Opts *o);    /* sym.c: everything with orbits (cold / warm symmetric RR, walks) */

#endif
