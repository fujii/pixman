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

static const pixman_fast_path_t avx2_fast_paths[] =
{
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
