#pragma once

// LeanInfer Phase 3 — CUDA fused kernel C API
//
// Always safe to include; all APIs are no-ops when compiled without CUDA.
//
// SPDX-License-Identifier: MIT

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(GGML_USE_CUDA) || defined(GGML_USE_CUBLAS)

// Fused RMSNorm + matmul: dst[n] = dot(rms_norm(x), W[n,:])
void li_launch_fused_rms_norm_matmul_f32(
        const float * x, const float * gamma, const float * W,
        float * dst, int K, int N, float eps, void * stream);

// Fused RMSNorm + SwiGLU: hidden[n] = silu(dot(x_norm, W_gate[n,:])) * dot(x_norm, W_up[n,:])
void li_launch_fused_rms_norm_swiglu_f32(
        const float * x, const float * gamma,
        const float * W_gate, const float * W_up,
        float * hidden, int K, int N, float eps, void * stream);

// FP16 weight variant of the SwiGLU kernel
void li_launch_fused_rms_norm_swiglu_f16(
        const float * x, const float * gamma,
        const void  * W_gate, const void  * W_up,
        float * hidden, int K, int N, float eps, void * stream);

// Fused RMSNorm + SiLU-gate for DeltaNet gated output
// dst[r][k] = silu(z[r][k]) * (output[r][k] * gamma[k] * rms_scale)
// Replaces 2 kernel launches (RMSNorm + fused_mul_unary) with 1.
void li_launch_fused_rms_norm_silu_gate_f32(
        const float * output, const float * z, const float * gamma,
        float * dst, int K, int n_rows, float eps, void * stream);

// Fused DeltaNet recurrent + output projection
void li_launch_fused_deltanet_recurrent_out_f32(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state_in,
        const float * W_out,
        float * dst, float * proj_out,
        int64_t head_dim, int64_t n_tokens, int64_t n_heads,
        int64_t gqa_ratio, int repeat_type, int64_t n_seqs,
        int64_t hidden_dim,
        size_t vnb1, size_t vnb2, size_t vnb3,
        void * stream);

#else  // no CUDA

// Stubs
static inline void li_launch_fused_rms_norm_silu_gate_f32(
        const float*o,const float*z,const float*g,float*d,int K,int nr,float e,void*s)
        {(void)o;(void)z;(void)g;(void)d;(void)K;(void)nr;(void)e;(void)s;}
static inline void li_launch_fused_rms_norm_matmul_f32(
        const float*x,const float*g,const float*W,float*d,int K,int N,float e,void*s)
        {(void)x;(void)g;(void)W;(void)d;(void)K;(void)N;(void)e;(void)s;}
static inline void li_launch_fused_rms_norm_swiglu_f32(
        const float*x,const float*g,const float*wg,const float*wu,float*h,int K,int N,float e,void*s)
        {(void)x;(void)g;(void)wg;(void)wu;(void)h;(void)K;(void)N;(void)e;(void)s;}
static inline void li_launch_fused_rms_norm_swiglu_f16(
        const float*x,const float*g,const void*wg,const void*wu,float*h,int K,int N,float e,void*s)
        {(void)x;(void)g;(void)wg;(void)wu;(void)h;(void)K;(void)N;(void)e;(void)s;}
static inline void li_launch_fused_deltanet_recurrent_out_f32(
        const float*q,const float*k,const float*v,const float*g,const float*b,const float*s,
        const float*w,float*d,float*p,int64_t hd,int64_t nt,int64_t nh,int64_t gr,int rt,
        int64_t ns,int64_t hid,size_t v1,size_t v2,size_t v3,void*st)
        {(void)q;(void)k;(void)v;(void)g;(void)b;(void)s;(void)w;(void)d;(void)p;
         (void)hd;(void)nt;(void)nh;(void)gr;(void)rt;(void)ns;(void)hid;(void)v1;(void)v2;(void)v3;(void)st;}

#endif

#ifdef __cplusplus
}
#endif
