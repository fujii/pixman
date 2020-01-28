#include <stdlib.h>
#include "utils.h"

#define SOURCE_WIDTH 320
#define SOURCE_HEIGHT 240
#define TEST_REPEATS 3

static pixman_image_t *
make_source (void)
{
    size_t n_bytes = (SOURCE_WIDTH + 2) * (SOURCE_HEIGHT + 2) * 4;
    uint32_t *data = malloc (n_bytes);
    pixman_image_t *source;

    prng_randmemset (data, n_bytes, 0);
    
    source = pixman_image_create_bits (
	PIXMAN_a8r8g8b8, SOURCE_WIDTH + 2, SOURCE_HEIGHT + 2,
	data,
	(SOURCE_WIDTH + 2) * 4);

    return source;
}

static void
set_filter (pixman_image_t	*source,
	    double		scale,
	    pixman_filter_t	filter)
{
    pixman_fixed_t *params = NULL;
    int n_params = 0;

    switch (filter) {
    case PIXMAN_FILTER_SEPARABLE_CONVOLUTION:
	params = pixman_filter_create_separable_convolution (
            &n_params,
            pixman_double_to_fixed(scale),
            pixman_double_to_fixed(scale),
            PIXMAN_KERNEL_BOX,
            PIXMAN_KERNEL_BOX,
            PIXMAN_KERNEL_BOX,
            PIXMAN_KERNEL_BOX,
            4, 4);
	break;
    }

    pixman_image_set_filter (source, filter, params, n_params);

    if (params)
	free (params);
}

void
print_help ()
{
    printf ("Options:\n\n"
	    "--filter {n|b|s}   filter\n"
	    "--start <double>   start scale factor\n"
	    "--end <double>     end scale factor\n");
    exit (-1);
}

static void
parse_arguments (int		argc,
		 char		**argv,
		 double		*start_scale,
		 double		*end_scale,
		 pixman_filter_t	*filter)
{
    while (*++argv) {
	if (!strcmp (*argv, "--filter")) {
	    ++argv;
	    if (!strcmp (*argv, "n"))
		*filter = PIXMAN_FILTER_NEAREST;
	    else if (!strcmp (*argv, "b"))
		*filter = PIXMAN_FILTER_BILINEAR;
	    else if (!strcmp (*argv, "s"))
		*filter = PIXMAN_FILTER_SEPARABLE_CONVOLUTION;
	    else {
		printf ("Unknown filter '%s'\n\n", *argv);
		print_help ();
	    }
	} else if (!strcmp (*argv, "--start")) {
	    ++argv;
	    *start_scale = strtod (*argv, NULL);
	} else if (!strcmp (*argv, "--end")) {
	    ++argv;
	    *end_scale = strtod (*argv, NULL);
	} else if (!strcmp (*argv, "-h") || !strcmp (*argv, "--help")) {
	    print_help ();
	} else {
	    printf ("Unknown option '%s'\n\n", *argv);
	    print_help ();
	}
    }
}

int
main (int argc, char **argv)
{
    double scale;
    double start_scale = 0.1, end_scale = 10.005;
    pixman_image_t *src;
    pixman_filter_t filter = PIXMAN_FILTER_BILINEAR;

    parse_arguments (argc, argv, &start_scale, &end_scale, &filter);

    prng_srand (23874);
    
    src = make_source ();
    printf ("# %-6s %-22s   %-14s %-12s\n",
	    "ratio",
	    "resolutions",
	    "time [ms]",
	    "time per pixel [ns]");
    for (scale = start_scale; scale < end_scale; scale += 0.01)
    {
	int i;
	int dest_width = SOURCE_WIDTH * scale + 0.5;
	int dest_height = SOURCE_HEIGHT * scale + 0.5;
	int dest_byte_stride = (dest_width * 4 + 15) & ~15;
	pixman_fixed_t s = (1 / scale) * 65536.0 + 0.5;
	pixman_transform_t transform;
	pixman_image_t *dest;
	double t1, t2, t = -1;
	uint32_t *dest_buf = aligned_malloc (16, dest_byte_stride * dest_height);
	memset (dest_buf, 0, dest_byte_stride * dest_height);

	set_filter (src, 0.4, filter);

	pixman_transform_init_scale (&transform, s, s);
	pixman_image_set_transform (src, &transform);
	
	dest = pixman_image_create_bits (
	    PIXMAN_a8r8g8b8, dest_width, dest_height, dest_buf, dest_byte_stride);

	for (i = 0; i < TEST_REPEATS; i++)
	{
	    t1 = gettime();
	    pixman_image_composite (
		PIXMAN_OP_OVER, src, NULL, dest,
		scale, scale, 0, 0, 0, 0, dest_width, dest_height);
	    t2 = gettime();
	    if (t < 0 || t2 - t1 < t)
		t = t2 - t1;
	}

	printf ("%6.2f : %4dx%-4d => %4dx%-4d : %12.4f : %12.4f\n",
		scale, SOURCE_WIDTH, SOURCE_HEIGHT, dest_width, dest_height,
		t * 1000, (t / (dest_width * dest_height)) * 1000000000);

	pixman_image_unref (dest);
	free (dest_buf);
    }

    return 0;
}
