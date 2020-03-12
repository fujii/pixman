/*
 * Copyright © 2012 Intel Corporation
 *
 * Permission to use, copy, modify, distribute, and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies and that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Red Hat not be used in advertising or
 * publicity pertaining to distribution of the software without specific,
 * written prior permission.  Red Hat makes no representations about the
 * suitability of this software for any purpose.  It is provided "as is"
 * without express or implied warranty.
 *
 * THE COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS, IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
 * AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *
 * Author:  Matt Turner <mattst88@gmail.com>
 *
 * Based on work by Owen Taylor, Søren Sandmann, Rodrigo Kumpera,
 * and André Tupinambá.
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <immintrin.h> /* for AVX2 intrinsics */
#include "pixman-private.h"
#include "pixman-combine32.h"
#include "pixman-inlines.h"

static void
avx2_composite_add_8888_8888 (pixman_implementation_t *imp,
                              pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    uint32_t    *dst_line, *dst;
    uint32_t    *src_line, *src;
    int dst_stride, src_stride;
    int32_t w;

    PIXMAN_IMAGE_GET_LINE (src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);
    PIXMAN_IMAGE_GET_LINE (dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);

    while (height--)
    {
	dst = dst_line;
	dst_line += dst_stride;
	src = src_line;
	src_line += src_stride;
	w = width;

	while (w && (unsigned long)dst & 31)
	{
	    *dst = _mm_cvtsi128_si32(_mm_adds_epu8 (_mm_cvtsi32_si128 (*src),
						    _mm_cvtsi32_si128 (*dst)));
	    dst++;
	    src++;
	    w--;
	}

	while (w >= 8)
	{
	    *(__m256i *)dst = _mm256_adds_epu8 (
					_mm256_lddqu_si256 ((__m256i *)src),
					*(__m256i *)dst);
	    dst += 8;
	    src += 8;
	    w -= 8;
	}

	while (w)
	{
	    *dst = _mm_cvtsi128_si32(_mm_adds_epu8 (_mm_cvtsi32_si128 (*src),
						    _mm_cvtsi32_si128 (*dst)));
	    dst++;
	    src++;
	    w--;
	}
    }
}

#define eqmask(a, b) (_mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) == 0xffff)

#if BILINEAR_INTERPOLATION_BITS < 8

#define BMSK ((1 << BILINEAR_INTERPOLATION_BITS) - 1)

#define BILINEAR_DECLARE_YMM_VARIABLES						\
    const __m256i ymm_wtb = _mm256_set_epi8 (wt, wb, wt, wb, wt, wb, wt, wb,	\
					     wt, wb, wt, wb, wt, wb, wt, wb,	\
					     wt, wb, wt, wb, wt, wb, wt, wb,	\
					     wt, wb, wt, wb, wt, wb, wt, wb);	\
    const __m256i ymm_xorc7 = _mm256_set1_epi32 (BMSK);				\
    const __m256i ymm_addc7 = _mm256_set1_epi32 (1);				\
    const __m256i ymm_ux = _mm256_set1_epi16 (2 * unit_x);			\
    __m256i ymm_x = _mm256_set_m128i (_mm_set1_epi16 (vx + unit_x),		\
                                      _mm_set1_epi16 (vx))

#define BILINEAR_DECLARE_XMM_VARIABLES						\
    const __m128i xmm_wtb = _mm_set_epi8 (wt, wb, wt, wb, wt, wb, wt, wb,	\
					  wt, wb, wt, wb, wt, wb, wt, wb);	\
    const __m128i xmm_xorc7 = _mm_set1_epi32 (BMSK);				\
    const __m128i xmm_addc7 = _mm_set1_epi32 (1);				\
    const __m128i xmm_ux = _mm_set1_epi16 (unit_x);				\
    __m128i xmm_x = _mm_set1_epi16 (vx)

#define BILINEAR_INTERPOLATE_ONE_PIXEL(pix)					\
do {										\
    __m128i xmm_wh, a;								\
    /* fetch 2x2 pixel block into avx2 registers */				\
    __m128i tltr = _mm_loadl_epi64 (						\
			    (__m128i *)&src_top[pixman_fixed_to_int (vx)]);	\
    __m128i blbr = _mm_loadl_epi64 (						\
			    (__m128i *)&src_bottom[pixman_fixed_to_int (vx)]);	\
    vx += unit_x;								\
    /* vertical interpolation */						\
    a = _mm_maddubs_epi16 (_mm_unpacklo_epi8 (blbr, tltr), xmm_wtb);		\
    /* calculate horizontal weights */						\
    xmm_wh = _mm_add_epi16 (xmm_addc7, _mm_xor_si128 (xmm_xorc7,		\
    _mm_srli_epi16 (xmm_x, 16 - BILINEAR_INTERPOLATION_BITS)));			\
    xmm_x = _mm_add_epi16 (xmm_x, xmm_ux);					\
    /* horizontal interpolation */						\
    a = _mm_madd_epi16 (_mm_unpackhi_epi16 (_mm_shuffle_epi32 (			\
	    a, _MM_SHUFFLE (1, 0, 3, 2)), a), xmm_wh);				\
    /* shift and pack the result */						\
    a = _mm_srli_epi32 (a, BILINEAR_INTERPOLATION_BITS * 2);			\
    a = _mm_packs_epi32 (a, a);							\
    a = _mm_packus_epi16 (a, a);						\
    pix = _mm_cvtsi128_si32 (a);						\
} while (0)

#define BILINEAR_INTERPOLATE_TWO_PIXELS(pix1, pix2)				\
do {										\
    __m256i ymm_wh, a;								\
    /* fetch two 2x2 pixel blocks into avx2 registers */			\
    __m256i vindex = _mm256_set_epi64x (&src_bottom[pixman_fixed_to_int (vx + unit_x)],	\
					&src_top   [pixman_fixed_to_int (vx + unit_x)],	\
					&src_bottom[pixman_fixed_to_int (vx)],		\
					&src_top   [pixman_fixed_to_int (vx)]);		\
    __m256i load =  _mm256_i64gather_epi64 (NULL, vindex, 1);			\
    __m256i tltr =  load;							\
    __m256i blbr =  _mm256_permute4x64_epi64 (load, _MM_SHUFFLE (3, 3, 1, 1));	\
    vx += 2 * unit_x;								\
    /* vertical interpolation */						\
    a = _mm256_maddubs_epi16 (_mm256_unpacklo_epi8 (blbr, tltr), ymm_wtb);	\
    /* calculate horizontal weights */						\
    ymm_wh = _mm256_add_epi16 (ymm_addc7, _mm256_xor_si256 (ymm_xorc7,		\
		_mm256_srli_epi16 (ymm_x, 16 - BILINEAR_INTERPOLATION_BITS)));	\
    ymm_x = _mm256_add_epi16 (ymm_x, ymm_ux);					\
    /* horizontal interpolation */						\
    a = _mm256_madd_epi16 (_mm256_unpackhi_epi16 (_mm256_shuffle_epi32 (	\
	    a, _MM_SHUFFLE (1, 0, 3, 2)), a), ymm_wh);				\
    /* shift and pack the result */						\
    a = _mm256_srli_epi32 (a, BILINEAR_INTERPOLATION_BITS * 2);			\
    a = _mm256_packs_epi32 (a, a);						\
    a = _mm256_packus_epi16 (a, a);						\
    _mm_storeu_si64(dst, _mm256_castsi256_si128(a));			\
} while (0)

static force_inline void
scaled_bilinear_scanline_avx2_8888_8888_SRC (uint32_t *       dst,
					     const uint32_t * mask,
					     const uint32_t * src_top,
					     const uint32_t * src_bottom,
					     int32_t          w,
					     int              wt,
					     int              wb,
					     pixman_fixed_t   vx,
					     pixman_fixed_t   unit_x,
					     pixman_fixed_t   max_vx,
					     pixman_bool_t    zero_src)
{
    BILINEAR_DECLARE_YMM_VARIABLES;
    uint32_t pix1, pix2;

    while ((w -= 2) >= 0)
    {
	BILINEAR_INTERPOLATE_TWO_PIXELS (pix1, pix2);
	dst += 2;
    }

    if (w & 1)
    {
	BILINEAR_DECLARE_XMM_VARIABLES;
	BILINEAR_INTERPOLATE_ONE_PIXEL (pix1);
	*dst = pix1;
    }
}

FAST_BILINEAR_MAINLOOP_COMMON (avx2_8888_8888_cover_SRC,
			       scaled_bilinear_scanline_avx2_8888_8888_SRC,
			       uint32_t, uint32_t, uint32_t,
			       COVER, FLAG_NONE)
FAST_BILINEAR_MAINLOOP_COMMON (avx2_8888_8888_pad_SRC,
			       scaled_bilinear_scanline_avx2_8888_8888_SRC,
			       uint32_t, uint32_t, uint32_t,
			       PAD, FLAG_NONE)
FAST_BILINEAR_MAINLOOP_COMMON (avx2_8888_8888_none_SRC,
			       scaled_bilinear_scanline_avx2_8888_8888_SRC,
			       uint32_t, uint32_t, uint32_t,
			       NONE, FLAG_NONE)
FAST_BILINEAR_MAINLOOP_COMMON (avx2_8888_8888_normal_SRC,
			       scaled_bilinear_scanline_avx2_8888_8888_SRC,
			       uint32_t, uint32_t, uint32_t,
			       NORMAL, FLAG_NONE)
#endif

static const pixman_fast_path_t avx2_fast_paths[] =
{
    PIXMAN_STD_FAST_PATH    (ADD,  a8r8g8b8, null,     a8r8g8b8, avx2_composite_add_8888_8888      ),
    PIXMAN_STD_FAST_PATH    (ADD,  a8b8g8r8, null,     a8b8g8r8, avx2_composite_add_8888_8888      ),

#if BILINEAR_INTERPOLATION_BITS < 8
    SIMPLE_BILINEAR_FAST_PATH (SRC, a8r8g8b8, a8r8g8b8, avx2_8888_8888),
    SIMPLE_BILINEAR_FAST_PATH (SRC, a8r8g8b8, x8r8g8b8, avx2_8888_8888),
    SIMPLE_BILINEAR_FAST_PATH (SRC, x8r8g8b8, x8r8g8b8, avx2_8888_8888),
    SIMPLE_BILINEAR_FAST_PATH (SRC, a8b8g8r8, a8b8g8r8, avx2_8888_8888),
    SIMPLE_BILINEAR_FAST_PATH (SRC, a8b8g8r8, x8b8g8r8, avx2_8888_8888),
    SIMPLE_BILINEAR_FAST_PATH (SRC, x8b8g8r8, x8b8g8r8, avx2_8888_8888),
#endif

    { PIXMAN_OP_NONE },
};

#if defined(__GNUC__) && !defined(__x86_64__) && !defined(__amd64__)
__attribute__((__force_align_arg_pointer__))
#endif
pixman_implementation_t *
_pixman_implementation_create_avx2 (pixman_implementation_t *fallback)
{
    pixman_implementation_t *imp = _pixman_implementation_create (fallback, avx2_fast_paths);

    return imp;
}
