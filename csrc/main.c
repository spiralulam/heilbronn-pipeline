/* main.c - command line of the Heilbronn search engine.
 *
 *   heil V n [--group G] [--seed S] [--time SECS] [--out PREFIX] [--init FILE] [--walk]
 *            [--vacancies K] [--free M]
 *
 * Dispatch: group C1 with a start file -> free warm ruin & recreate (free.c); everything else -> the
 * symmetric engine (sym.c): cold RR without a start file, warm RR with one, --walk for the hot walk.
 */
#include <getopt.h>
#include "heil.h"

static void usage(void) {
  fprintf(stderr,
    "usage: heil V n [options]\n"
    "  V                 s (unit square), t (triangle (0,0),(1,0),(0,1)), c (convex position)\n"
    "  n                 number of points (4..100)\n"
    "  -g, --group G     symmetry group (default C1): s: C1 C2 C4 D1 D1d D2 D2d D4, t: C1 C3 D1 D3, c: Ck Dk (k <= 12)\n"
    "  -s, --seed S      random seed (default 1)\n"
    "  -t, --time SECS   search time after the setup (default 60)\n"
    "  -o, --out PREFIX  best configuration is written to PREFIX.txt (default <V><n>)\n"
    "  -i, --init FILE   start configuration (site coordinates); with C1 it must have n points,\n"
    "                    with another group it must be G-symmetric (any number of points)\n"
    "  -w, --walk        hot basin-hopping walk from --init instead of ruin & recreate\n"
    "  --vacancies K     cold symmetric search with K orbits missing one image\n"
    "  --free M          cold symmetric search with M orbits reduced to one point\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc < 3) usage();
  Opts o = {0};
  o.group = "C1"; o.seed = 1; o.secs = 60;
  V = argv[1][0]; o.n = atoi(argv[2]);
  if ((V != 's' && V != 't' && V != 'c') || argv[1][1] || o.n < 4 || o.n > 100) usage();
  static struct option lo[] = {
    {"group", 1, 0, 'g'}, {"seed", 1, 0, 's'}, {"time", 1, 0, 't'}, {"out", 1, 0, 'o'}, {"init", 1, 0, 'i'},
    {"walk", 0, 0, 'w'}, {"vacancies", 1, 0, 'v'}, {"free", 1, 0, 'f'}, {"help", 0, 0, 'h'}, {0, 0, 0, 0}};
  int c;
  while ((c = getopt_long(argc - 2, argv + 2, "g:s:t:o:i:wh", lo, NULL)) != -1)
    switch (c) {
      case 'g': o.group = optarg; break;
      case 's': o.seed = strtoull(optarg, NULL, 10); break;
      case 't': o.secs = atof(optarg); break;
      case 'o': o.out = optarg; break;
      case 'i': o.init = optarg; break;
      case 'w': o.walk = 1; break;
      case 'v': o.vac = atoi(optarg); break;
      case 'f': o.nfree = atoi(optarg); break;
      default: usage();
    }
  if (optind != argc - 2) usage();
  if (o.walk && !o.init) { fprintf(stderr, "--walk needs --init\n"); return 1; }
  if ((o.vac || o.nfree) && (o.init || !strcmp(o.group, "C1"))) { fprintf(stderr, "defect orbits need a group and no --init\n"); return 1; }
  char out[64];
  if (!o.out) { snprintf(out, sizeof out, "%c%d", V, o.n); o.out = out; }

  rs = 0x9E3779B97F4A7C15ULL ^ (o.seed * 0xD1B54A32D192ED03ULL);
  for (int i = 0; i < 10; i++) rnd64();
  if (!strcmp(o.group, "C1") && o.init && !o.walk) return run_free(&o);
  return run_sym(&o);
}
