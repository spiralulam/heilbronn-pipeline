"""GPU screening kernel (Apple Metal through torch.mps): one thread per chain, float32.

Two kernels act on a batch of B configurations ("chains"), each padded to NM points:
  k_insert  add one point by greedy insertion (best of RRK random candidates for the smallest new triangle), then
            run the soft polish;
  k_polish  soft polish only.
Soft polish: L-BFGS (memory 8, Armijo backtracking) on the soft-min surrogate
    F(P) = -(a_min - c) / c + (1 / beta) log sum_k exp(-beta (|a_k| - a_min) / c),        c = a_min(P_0),
over the triangles that are near-active at the start (|a_k| <= 6 a_min), with a quadratic penalty for leaving the
container.  Points on a side keep one coordinate (their position along the side), corners stay fixed.  For the convex
variant a_k is replaced by a_k / H(P) (H = hull area, with its gradient), points are free and the configuration is
normalised to centroid 0 and hull area 1 before and after.
State per chain: P (B, 2 NM) coordinates, TY (B, NM) point types (0 interior, 1..NS side, 100 + c corner),
NN (B,) current number of points.
"""
import numpy as np
import torch

SRC = r'''
#include <metal_stdlib>
using namespace metal;
#define NM @NM@
#define NS @NS@
#define NC @NC@
#define TRI @TRI@
#define CONV @CONV@
#define IWS @IWS@
#define LM 8
#define NVM (2 * NM)
#define SMAX @SMAX@
#define NLM (NM * (NM - 1) / 2)
#define RRK @RRK@
#define NBK 64
#define WSZ ((2 * LM + 8) * NVM)
constant float2 SO[4] = {@SO@};          // side s: SO[s-1] + t SD[s-1], t in [0, 1]
constant float2 SD[4] = {@SD@};
constant float2 CO[4] = {@CO@};          // corners
constant float PB = 0.2f, PC = 0.002f, SUBK = 6.0f;

inline ulong rnd64(thread ulong &rs) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
inline float urand(thread ulong &rs) { return (float)(rnd64(rs) >> 40) * (1.0f / 16777216.0f); }
inline float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
inline void put_inside(device float* q) {
#if CONV
    return;
#elif TRI
    if (q[0] < 0.0f) q[0] = 0.0f; if (q[1] < 0.0f) q[1] = 0.0f; float s = q[0] + q[1]; if (s > 1.0f) { q[0] /= s; q[1] /= s; }
#else
    q[0] = clamp01(q[0]); q[1] = clamp01(q[1]);
#endif
}
static float mincross(device const float* p, int N) {   // min over all triangles of |cross| = 2 * area
    float m = 1e30f;
    for (int i = 0; i < N; i++) { float xi = p[2*i], yi = p[2*i+1];
        for (int j = i + 1; j < N; j++) { float dx = p[2*j] - xi, dy = p[2*j+1] - yi;
            for (int k = j + 1; k < N; k++) m = fmin(m, fabs(dx * (p[2*k+1] - yi) - (p[2*k] - xi) * dy)); } }
    return m;
}
#if CONV
// hull area by Andrew's monotone chain.  ord: x-order of the points, kept between calls (insertion sort from the previous
// order, nearly linear inside one polish); h: 2N+2 scratch, on return h[0..k-1] = hull vertices ccw; wg: also write gH.
#define CRS(o, a, b) ((p[2*(a)]-p[2*(o)])*(p[2*(b)+1]-p[2*(o)+1]) - (p[2*(a)+1]-p[2*(o)+1])*(p[2*(b)]-p[2*(o)]))
static float hull_area(device const float* p, int N, device int* ord, device int* h, device float* gH, bool wg, thread int &kout) {
    for (int i = 1; i < N; i++) { int v = ord[i]; float xv = p[2*v], yv = p[2*v+1]; int j = i - 1;
        while (j >= 0 && (p[2*ord[j]] > xv || (p[2*ord[j]] == xv && p[2*ord[j]+1] > yv))) { ord[j+1] = ord[j]; j--; }
        ord[j+1] = v; }
    int k = 0;
    for (int i = 0; i < N; i++) { int c = ord[i]; while (k >= 2 && CRS(h[k-2], h[k-1], c) <= 0.0f) k--; h[k++] = c; }
    for (int i = N - 2, t = k + 1; i >= 0; i--) { int c = ord[i]; while (k >= t && CRS(h[k-2], h[k-1], c) <= 0.0f) k--; h[k++] = c; }
    k--;
    float A = 0.0f;
    for (int i = 0; i < k; i++) { int a = h[i], b = h[(i+1)%k]; A += p[2*a]*p[2*b+1] - p[2*b]*p[2*a+1]; }
    A *= 0.5f;
    if (wg) {
        for (int v = 0; v < 2 * N; v++) gH[v] = 0.0f;
        for (int i = 0; i < k; i++) { int a = h[i], nx = h[(i+1)%k], pv = h[(i+k-1)%k];
            gH[2*a] = 0.5f * (p[2*nx+1] - p[2*pv+1]); gH[2*a+1] = 0.5f * (p[2*pv] - p[2*nx]); }
    }
    kout = k; return A;
}
static float normalize_conv(device float* p, int N, device int* hw) {   // centroid 0, hull area 1; hw = ord (NM) + h
    for (int i = 0; i < N; i++) hw[i] = i;
    int k; float H = hull_area(p, N, hw, hw + NM, (device float*)p, false, k);
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < N; i++) { cx += p[2*i]; cy += p[2*i+1]; }
    cx /= (float)N; cy /= (float)N; float s = 1.0f / sqrt(H);
    for (int i = 0; i < N; i++) { p[2*i] = (p[2*i] - cx) * s; p[2*i+1] = (p[2*i+1] - cy) * s; }
    return H;
}
#endif
static int build_vars(device const int* ty, device int* voff, int N) {   // interior: 2 variables, side: 1, corner: 0
    int NV = 0;
    for (int i = 0; i < N; i++) { voff[i] = NV; if (ty[i] == 0) NV += 2; else if (ty[i] <= NS) NV += 1; }
    return NV;
}
static void p2u(device const float* p, device float* u, device const int* ty, device const int* voff, int N) {
    for (int i = 0; i < N; i++) { int o = voff[i], t = ty[i];
        if (t == 0) { u[o] = p[2*i]; u[o+1] = p[2*i+1]; }
        else if (t <= NS) { float2 dd = float2(p[2*i], p[2*i+1]) - SO[t-1]; u[o] = clamp01(dot(dd, SD[t-1]) / dot(SD[t-1], SD[t-1])); } }
}
static void u2p(device const float* u, device float* p, device const int* ty, device const int* voff, int N) {
    for (int i = 0; i < N; i++) { int o = voff[i], t = ty[i];
        if (t == 0) { p[2*i] = u[o]; p[2*i+1] = u[o+1]; }
        else if (t <= NS) { p[2*i] = SO[t-1].x + u[o] * SD[t-1].x; p[2*i+1] = SO[t-1].y + u[o] * SD[t-1].y; }
        else { p[2*i] = CO[t-100].x; p[2*i+1] = CO[t-100].y; } }
}
// soft-min objective over the triangle subset sub and its gradient with respect to the variables u
static float fgrad(device const float* u, device float* g, device float* p, device float* gp, device const int* ty, device const int* voff, int N,
                   device const uint* sub, int nsub, float BETA, float CREF, float MU) {
    u2p(u, p, ty, voff, N);
#if CONV
    int hk; device float* gH = gp + NVM;
    device int* hw = const_cast<device int*>(voff) + NM;   // per-chain ints: voff (NM), hull x-order (NM), hull scratch
    float H = hull_area(p, N, hw, hw + NM, gH, true, hk);
    float sc = 1.0f / (CREF * H), amin = 1e30f, SA = 0.0f;
#else
    float sc = 1.0f / CREF, amin = 1e30f;
#endif
    for (int q = 0; q < nsub; q++) { uint t = sub[q]; uint i = t & 255u, j = (t >> 8) & 255u, k = t >> 16;
        float xi = p[2*i], yi = p[2*i+1];
        amin = fmin(amin, fabs((p[2*j] - xi) * (p[2*k+1] - yi) - (p[2*k] - xi) * (p[2*j+1] - yi))); }
    amin *= 0.5f;
    for (int v = 0; v < 2 * N; v++) gp[v] = 0.0f;
    float S = 0.0f, bs = BETA * sc;
    for (int q = 0; q < nsub; q++) { uint t = sub[q]; uint i = t & 255u, j = (t >> 8) & 255u, k = t >> 16;
        float xi = p[2*i], yi = p[2*i+1], xj = p[2*j], yj = p[2*j+1], xk = p[2*k], yk = p[2*k+1];
        float a = 0.5f * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)), z = bs * (fabs(a) - amin);
        if (z > 40.0f) continue;
        float w = exp(-z); S += w;
#if CONV
        SA += w * fabs(a);
#endif
        float s = a >= 0.0f ? 0.5f * w : -0.5f * w;
        gp[2*i] += s * (yj - yk); gp[2*i+1] += s * (xk - xj);
        gp[2*j] += s * (yk - yi); gp[2*j+1] += s * (xi - xk);
        gp[2*k] += s * (yi - yj); gp[2*k+1] += s * (xj - xi); }
#if CONV
    float F = -(amin * sc - 1.0f) + log(S) / BETA;
    float f = -sc / S, qh = SA / H;
    for (int v = 0; v < 2 * N; v++) gp[v] = f * (gp[v] - qh * gH[v]);   // d(a / H) = da / H - a dH / H^2
#else
    float F = -(amin - CREF) * sc + log(S) / BETA;
    float f = -sc / S;
    for (int v = 0; v < 2 * N; v++) gp[v] *= f;
#endif
    float Pn = 0.0f;
    for (int i = 0; i < N; i++) { int o = voff[i], t = ty[i];
        if (t == 0) {
            float x = p[2*i], y = p[2*i+1], gx = gp[2*i], gy = gp[2*i+1];
#if CONV
            (void)x; (void)y;
#elif TRI
            if (x < 0.0f) { Pn += x*x; gx += 2.0f*MU*x; }
            if (y < 0.0f) { Pn += y*y; gy += 2.0f*MU*y; }
            float e = x + y - 1.0f; if (e > 0.0f) { Pn += e*e; gx += 2.0f*MU*e; gy += 2.0f*MU*e; }
#else
            if (x < 0.0f) { Pn += x*x; gx += 2.0f*MU*x; } else if (x > 1.0f) { Pn += (x-1.0f)*(x-1.0f); gx += 2.0f*MU*(x-1.0f); }
            if (y < 0.0f) { Pn += y*y; gy += 2.0f*MU*y; } else if (y > 1.0f) { Pn += (y-1.0f)*(y-1.0f); gy += 2.0f*MU*(y-1.0f); }
#endif
            g[o] = gx; g[o+1] = gy;
        } else if (t <= NS) {
            float tt = u[o], gt = gp[2*i] * SD[t-1].x + gp[2*i+1] * SD[t-1].y, L2 = dot(SD[t-1], SD[t-1]);
            if (tt < 0.0f) { Pn += L2*tt*tt; gt += 2.0f*MU*L2*tt; } else if (tt > 1.0f) { Pn += L2*(tt-1.0f)*(tt-1.0f); gt += 2.0f*MU*L2*(tt-1.0f); }
            g[o] = gt;
        } }
    return F + MU * Pn;
}
static int lbfgs(device float* x, int maxit, int NV, device float* wk, device const int* ty, device const int* voff, int N,
                 device const uint* sub, int nsub, float BETA, float CREF, float MU) {
    device float* s = wk; device float* y = wk + LM * NVM; device float* g = wk + 2 * LM * NVM; device float* d = g + NVM;
    device float* xn = g + 2 * NVM; device float* gn = g + 3 * NVM; device float* pp = g + 5 * NVM; device float* gp = g + 6 * NVM;
    float rho[LM], al[LM];
    float f = fgrad(x, g, pp, gp, ty, voff, N, sub, nsub, BETA, CREF, MU);
    int k = 0, it;
    for (it = 0; it < maxit; it++) {
        for (int v = 0; v < NV; v++) d[v] = -g[v];
        int m = k < LM ? k : LM;
        for (int l = 0; l < m; l++) {
            int q = (k - 1 - l) % LM; float a = 0.0f;
            for (int v = 0; v < NV; v++) a += s[q*NVM+v] * d[v];
            a *= rho[q]; al[q] = a;
            for (int v = 0; v < NV; v++) d[v] -= a * y[q*NVM+v];
        }
        if (m > 0) {
            int q = (k - 1) % LM; float yy = 0.0f, sy = 0.0f;
            for (int v = 0; v < NV; v++) { yy += y[q*NVM+v] * y[q*NVM+v]; sy += s[q*NVM+v] * y[q*NVM+v]; }
            float gam = sy / yy; for (int v = 0; v < NV; v++) d[v] *= gam;
        } else {
            float gn2 = 0.0f; for (int v = 0; v < NV; v++) gn2 += g[v] * g[v];
            float sc = 1e-3f / (sqrt(gn2) + 1e-30f); for (int v = 0; v < NV; v++) d[v] *= sc;
        }
        for (int l = m - 1; l >= 0; l--) {
            int q = (k - 1 - l) % LM; float bb = 0.0f;
            for (int v = 0; v < NV; v++) bb += y[q*NVM+v] * d[v];
            bb *= rho[q];
            for (int v = 0; v < NV; v++) d[v] += s[q*NVM+v] * (al[q] - bb);
        }
        float gd = 0.0f; for (int v = 0; v < NV; v++) gd += g[v] * d[v];
        if (!(gd < 0.0f)) {
            k = 0; float gn2 = 0.0f; for (int v = 0; v < NV; v++) gn2 += g[v] * g[v];
            float sc = 1e-3f / (sqrt(gn2) + 1e-30f);
            for (int v = 0; v < NV; v++) d[v] = -g[v] * sc;
            gd = -gn2 * sc;
        }
        float st = 1.0f, fn = 0.0f; bool ok = false;
        for (int ls = 0; ls < 30; ls++) {
            for (int v = 0; v < NV; v++) xn[v] = x[v] + st * d[v];
            fn = fgrad(xn, gn, pp, gp, ty, voff, N, sub, nsub, BETA, CREF, MU);
            if (fn <= f + 1e-4f * st * gd) { ok = true; break; }
            st *= 0.5f;
        }
        if (!ok) break;
        int q = k % LM; float sy = 0.0f;
        for (int v = 0; v < NV; v++) { float sv = xn[v] - x[v], yv = gn[v] - g[v]; s[q*NVM+v] = sv; y[q*NVM+v] = yv; sy += sv * yv; }
        float df = f - fn;
        for (int v = 0; v < NV; v++) { x[v] = xn[v]; g[v] = gn[v]; }
        f = fn;
        if (sy > 1e-30f) { rho[q] = 1.0f / sy; k++; }
        if (df < 1e-14f * (fabs(f) + 1e-12f)) break;
    }
    return it;
}
// soft polish with beta relative to the current minimum; returns the true value of the result
static float polish(device float* p, device int* ty, int N, float bf, int iters, device int* voff, device uint* sub, device float* wk) {
#if CONV
    normalize_conv(p, N, voff + NM);   // also leaves the x-order in voff + NM for the hull in fgrad
#endif
    for (int i = 0; i < N; i++) if (ty[i] == 0) put_inside(p + 2 * i);
    int NV = build_vars(ty, voff, N);
    device float* u = wk + 2 * LM * NVM + 4 * NVM;
    p2u(p, u, ty, voff, N); u2p(u, p, ty, voff, N);
    float d0 = (TRI ? 0.7f : 1.0f) / sqrt((float)N);
    float m = mincross(p, N);
    float c = 0.5f * m, fl = 0.02f / (float)(N * N) * (TRI ? 0.5f : 1.0f);
    if (c < fl) c = fl;
    float MU = 50.0f * bf * (d0 / c) * (d0 / c), thr = SUBK * m;
    int ns = 0;
    for (int i = 0; i < N; i++) { float xi = p[2*i], yi = p[2*i+1];
        for (int j = i + 1; j < N; j++) { float dx = p[2*j] - xi, dy = p[2*j+1] - yi;
            for (int k = j + 1; k < N; k++) if (ns < SMAX && fabs(dx * (p[2*k+1] - yi) - (p[2*k] - xi) * dy) <= thr)
                sub[ns++] = (uint)i | ((uint)j << 8) | ((uint)k << 16); } }
    lbfgs(u, iters, NV, wk, ty, voff, N, sub, ns, bf, c, MU);
    u2p(u, p, ty, voff, N);
#if CONV
    normalize_conv(p, N, voff + NM);
    int hk; float H = hull_area(p, N, voff + NM, voff + 2 * NM, p, false, hk);
    return 0.5f * mincross(p, N) / H;
#else
    for (int i = 0; i < N; i++) put_inside(p + 2 * i);
    return mincross(p, N) * (TRI ? 1.0f : 0.5f);
#endif
}
// greedy insertion: best of RRK random candidates by the smallest triangle it forms with an existing pair.  Pairs are
// stored as lines (a, b, c) sorted by length, so a candidate can be dropped as soon as it falls below the best so far.
static int insert_best(device float* p, device int* ty, int N, device float* L, thread ulong &rs, device int* hw) {
    int cnt[5] = {0, 0, 0, 0, 0}; bool occ[4] = {false, false, false, false};
    for (int i = 0; i < N; i++) { int t = ty[i]; if (t >= 1 && t <= NS) cnt[t]++; else if (t >= 100) occ[t - 100] = true; }
#if CONV   // candidates in the bounding box (+5 % each side), score = smallest new area / hull area with the candidate
    float minx = 1e30f, maxx = -1e30f, miny = 1e30f, maxy = -1e30f;
    for (int i = 0; i < N; i++) { minx = fmin(minx, p[2*i]); maxx = fmax(maxx, p[2*i]); miny = fmin(miny, p[2*i+1]); maxy = fmax(maxy, p[2*i+1]); }
    for (int i = 0; i < N; i++) hw[i] = i;
    int hm; float H0 = hull_area(p, N, hw, hw + NM, p, false, hm); device int* hv = hw + NM;
#endif
    int bc[NBK]; for (int q = 0; q < NBK; q++) bc[q] = 0;
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { float la = p[2*i+1] - p[2*j+1], lb = p[2*j] - p[2*i];
        int bq = min(NBK - 1, (int)(sqrt(la * la + lb * lb) * (NBK / 1.4143f))); bc[bq]++; }
    int acc = 0; for (int q = 0; q < NBK; q++) { int c = bc[q]; bc[q] = acc; acc += c; }
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { float la = p[2*i+1] - p[2*j+1], lb = p[2*j] - p[2*i];
        int bq = min(NBK - 1, (int)(sqrt(la * la + lb * lb) * (NBK / 1.4143f))); int pos = bc[bq]++;
        L[3*pos] = la; L[3*pos+1] = lb; L[3*pos+2] = p[2*i] * p[2*j+1] - p[2*j] * p[2*i+1]; }
    int NL = N * (N - 1) / 2;
    float bx = 0.0f, by = 0.0f, bs = -1.0f; int bt = 0;
    for (int c = 0; c < RRK; c++) {
        float x, y; int tc = 0; float sc = 1.0f;
#if CONV
        x = minx + (maxx - minx) * (urand(rs) * 1.1f - 0.05f); y = miny + (maxy - miny) * (urand(rs) * 1.1f - 0.05f);
        float add = 0.0f;   // outside the hull: the hull grows by the triangles over the edges visible from the candidate
        for (int q = 0; q < hm; q++) { int a = hv[q], b2 = hv[(q+1) % hm];
            float cr = (p[2*b2]-p[2*a])*(y-p[2*a+1]) - (p[2*b2+1]-p[2*a+1])*(x-p[2*a]); if (cr < 0.0f) add -= 0.5f * cr; }
        sc = 1.0f / (H0 + add);
#else
        if (urand(rs) < PC) { int cc = (int)(rnd64(rs) % (ulong)NC); if (occ[cc]) continue; x = CO[cc].x; y = CO[cc].y; tc = 100 + cc; }
        else if (urand(rs) < PB) { int k = 1 + (int)(rnd64(rs) % (ulong)NS); if (cnt[k] >= 2) continue;
            float t = urand(rs); x = SO[k-1].x + t * SD[k-1].x; y = SO[k-1].y + t * SD[k-1].y; tc = k; }
        else { x = urand(rs); y = urand(rs);
#if TRI
            if (x + y > 1.0f) { x = 1.0f - x; y = 1.0f - y; }
#endif
        }
#endif
        float mn = 1e30f, lim = bs / sc;
        for (int q = 0; q < NL; q += 8) {
            int e = q + 8 < NL ? q + 8 : NL; float bm = 1e30f;
            for (int z = q; z < e; z++) bm = fmin(bm, fabs(L[3*z] * x + L[3*z+1] * y + L[3*z+2]));
            if (bm < mn) { mn = bm; if (mn <= lim) break; }
        }
        if (mn * sc > bs) { bs = mn * sc; bx = x; by = y; bt = tc; }
    }
    p[2*N] = bx; p[2*N+1] = by; ty[N] = bt;
    return N + 1;
}
inline ulong rs_load(device const uint* RS, uint b) { ulong r = ((ulong)RS[2*b] << 32) | (ulong)RS[2*b+1]; return r ? r : 88172645463325252ul; }
inline void rs_store(device uint* RS, uint b, ulong r) { RS[2*b] = (uint)(r >> 32); RS[2*b+1] = (uint)(r & 0xfffffffful); }

kernel void k_insert(device float* P [[buffer(0)]], device int* TY [[buffer(1)]], device int* NN [[buffer(2)]], device uint* RS [[buffer(3)]],
                     device float* WK [[buffer(4)]], device int* IW [[buffer(5)]], device uint* SUB [[buffer(6)]], device float* LC [[buffer(7)]],
                     device const float* prm [[buffer(8)]], device float* OUT [[buffer(9)]], device const int* TG [[buffer(10)]],
                     uint b [[thread_position_in_grid]]) {   // chains already at their target count TG skip
    if (NN[b] >= TG[b]) return;
    ulong rs = rs_load(RS, b);
    int N = insert_best(P + b * 2 * NM, TY + b * NM, NN[b], LC + b * 3 * NLM, rs, IW + b * IWS + NM);
    NN[b] = N; rs_store(RS, b, rs);
    OUT[b] = polish(P + b * 2 * NM, TY + b * NM, N, prm[0], (int)prm[1], IW + b * IWS, SUB + b * SMAX, WK + b * WSZ);
}
kernel void k_polish(device float* P [[buffer(0)]], device int* TY [[buffer(1)]], device const int* NN [[buffer(2)]],
                     device float* WK [[buffer(3)]], device int* IW [[buffer(4)]], device uint* SUB [[buffer(5)]],
                     device const float* prm [[buffer(6)]], device float* OUT [[buffer(7)]], uint b [[thread_position_in_grid]]) {
    OUT[b] = polish(P + b * 2 * NM, TY + b * NM, NN[b], prm[0], (int)prm[1], IW + b * IWS, SUB + b * SMAX, WK + b * WSZ);
}
'''
GEO = {'s': dict(NS=4, NC=4, SO=[(0, 0), (0, 1), (0, 0), (1, 0)], SD=[(1, 0), (1, 0), (0, 1), (0, 1)], CO=[(0, 0), (1, 0), (0, 1), (1, 1)]),
       't': dict(NS=3, NC=3, SO=[(0, 0), (0, 0), (1, 0), (0, 0)], SD=[(1, 0), (0, 1), (-1, 1), (1, 0)], CO=[(0, 0), (1, 0), (0, 1), (0, 0)]),
       'c': dict(NS=0, NC=0, SO=[(0, 0)] * 4, SD=[(1, 0)] * 4, CO=[(0, 0)] * 4)}


def available():
    return torch.backends.mps.is_available()


class Batch:
    """B chains with per-chain target sizes n (int or sequence, best sorted so that SIMD groups are uniform)."""

    def __init__(self, V, n, B, rrk=4000, smax=8192, seed=None):
        self.V, self.B = V, B; self.dev = d = torch.device('mps'); g = GEO[V]; i32 = torch.int32
        self.nv = torch.as_tensor(np.broadcast_to(np.asarray(n, dtype=np.int32), (B,)).copy(), dtype=i32).to(d)
        self.NM = NM = int(self.nv.max()) + 3
        self.NLM = NM * (NM - 1) // 2; self.SMAX = min(smax, NM * (NM - 1) * (NM - 2) // 6); self.WSZ = 24 * 2 * NM
        self.IWS = 4 * NM + 2 if V == 'c' else NM          # per-chain ints: voff [+ hull x-order + hull scratch]
        fmt = lambda L: ', '.join(f'float2({float(a)}f, {float(b)}f)' for a, b in L)
        src = SRC
        for k, v in dict(NM=NM, NS=g['NS'], NC=g['NC'], TRI=int(V == 't'), CONV=int(V == 'c'), IWS=self.IWS, SMAX=self.SMAX,
                         RRK=rrk, SO=fmt(g['SO']), SD=fmt(g['SD']), CO=fmt(g['CO'])).items():
            src = src.replace(f'@{k}@', str(v))
        self.lib = torch.mps.compile_shader(src)
        self.P = torch.zeros(B, 2 * NM, device=d); self.TY = torch.zeros(B, NM, dtype=i32, device=d); self.NN = self.nv.clone()
        gen = torch.Generator().manual_seed(seed if seed is not None else int(np.random.randint(1, 2**31 - 1)))
        self.RS = torch.randint(1, 2**31 - 1, (2 * B,), dtype=i32, generator=gen).to(d)
        self.WK = torch.zeros(B * self.WSZ, device=d); self.IW = torch.zeros(B * self.IWS, dtype=i32, device=d)
        self.SUB = torch.zeros(B * self.SMAX, dtype=i32, device=d); self.LC = torch.zeros(B * 3 * self.NLM, device=d)
        self.OUT = torch.zeros(B, device=d)

    def load(self, P, TY, NN):
        """P (B, 2 NM) float32, TY (B, NM) int32, NN (B,) current point counts (numpy or torch)"""
        self.P.copy_(torch.as_tensor(P, device=self.dev)); self.TY.copy_(torch.as_tensor(TY, device=self.dev))
        self.NN.copy_(torch.as_tensor(NN, device=self.dev))

    def insert(self, beta, its):
        """one insertion + soft polish in every chain below its target size"""
        self.lib.k_insert(self.P, self.TY, self.NN, self.RS, self.WK, self.IW, self.SUB, self.LC,
                          torch.tensor([beta, its], dtype=torch.float32, device=self.dev), self.OUT, self.nv, threads=self.B)

    def polish(self, beta, its):
        self.lib.k_polish(self.P, self.TY, self.NN, self.WK, self.IW, self.SUB,
                          torch.tensor([beta, its], dtype=torch.float32, device=self.dev), self.OUT, threads=self.B)
        return self.OUT.clone()

    def fill(self):
        """complete every chain to its target size, then polish at beta 20 and 50; returns the values (B,)"""
        for _ in range(int((self.nv - self.NN).max())):
            self.insert(15.0, 20)
        self.polish(20.0, 50)
        return self.polish(50.0, 50)

    def points(self, b):
        n = int(self.NN[b]); return self.P[b, :2 * n].cpu().numpy().astype(float).reshape(n, 2)
