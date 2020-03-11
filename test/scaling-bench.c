#include <math.h>
#include <stdlib.h>
#include "utils.h"

int dest_width = 320;
int dest_height = 240;
int test_repeats = 3;
double start_scale = 0.1;
double end_scale = 10.005;
double scale_step = 0.01;
pixman_filter_t filter = PIXMAN_FILTER_BILINEAR;
pixman_op_t op = PIXMAN_OP_OVER;

static pixman_image_t *
make_source (void)
{
    int source_width = ceil (dest_width / start_scale);
    int source_height = ceil (dest_height / start_scale);
    size_t n_bytes = source_width * source_height * 4;
    uint32_t *data = malloc (n_bytes);
    pixman_image_t *source;

    prng_randmemset (data, n_bytes, 0);
    
    source = pixman_image_create_bits (
		PIXMAN_a8r8g8b8, source_width, source_height,
		data,
		source_width * 4);

    return source;
}

static pixman_image_t *
make_dest (void)
{
    int dest_byte_stride = (dest_width * 4 + 15) & ~15;
    pixman_image_t *dest;
    uint32_t *dest_buf = aligned_malloc (16, dest_byte_stride * dest_height);

    dest = pixman_image_create_bits (
        PIXMAN_a8r8g8b8, dest_width, dest_height, dest_buf, dest_byte_stride);

    return dest;
}

static void
set_filter (pixman_image_t	*source,
	    double		scale,
	    pixman_filter_t	filter)
{
    pixman_fixed_t *params = NULL;
    int n_params = 0;

    scale = 1 / scale;
    if (scale > 16)
	scale = 16;

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
    default:
	break;
    }

    pixman_image_set_filter (source, filter, params, n_params);
#if 0
    printf ("n_params: %d %dx%d %dx%d\n", n_params, pixman_fixed_to_int(params[0]), pixman_fixed_to_int(params[1]), pixman_fixed_to_int(params[2]), pixman_fixed_to_int(params[3]));

    int width = pixman_fixed_to_int(params[0]);
    int height = pixman_fixed_to_int(params[1]);
    for (int y=0; y < height; y++) {
	for (int x=0; x < width; x++)
	    printf (" %f", pixman_fixed_to_double(params[4+y*width+x]));
	printf ("\n");
    }
#endif		 

    if (params)
	free (params);
}

void
print_help ()
{
    printf ("Options:\n\n"
	    "  --filter {n|b|s}        filter\n"
	    "  --start <double>        start scale factor\n"
	    "  --end <double>          end scale factor\n"
	    "  --step <double>         scale step\n"
	    "  --source-width <int>    source width\n"
	    "  --source-height <int>   source height\n"
	    "  --test-repeats <int>    repeat count\n"
	    "  --op <int>              compositing operator\n");
    exit (-1);
}

static void
parse_arguments (int			argc,
		 char			**argv)
{
    while (*++argv) {
	if (!strcmp (*argv, "--filter")) {
	    ++argv;
	    if (!strcmp (*argv, "n"))
		filter = PIXMAN_FILTER_NEAREST;
	    else if (!strcmp (*argv, "b"))
		filter = PIXMAN_FILTER_BILINEAR;
	    else if (!strcmp (*argv, "s"))
		filter = PIXMAN_FILTER_SEPARABLE_CONVOLUTION;
	    else {
		printf ("Unknown filter '%s'\n\n", *argv);
		print_help ();
	    }
	} else if (!strcmp (*argv, "--start")) {
	    ++argv;
	    start_scale = strtod (*argv, NULL);
	} else if (!strcmp (*argv, "--end")) {
	    ++argv;
	    end_scale = strtod (*argv, NULL);
	} else if (!strcmp (*argv, "--step")) {
	    ++argv;
	    scale_step = strtod (*argv, NULL);
	} else if (!strcmp (*argv, "--dest-width")) {
	    ++argv;
	    dest_width = atoi (*argv);
	} else if (!strcmp (*argv, "--dest-height")) {
	    ++argv;
	    dest_height = atoi (*argv);
	} else if (!strcmp (*argv, "--test-repeats")) {
	    ++argv;
	    test_repeats = atoi (*argv);
	} else if (!strcmp (*argv, "--op")) {
	    ++argv;
	    op = atoi (*argv);
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
    pixman_image_t *dest, *src;

    parse_arguments (argc, argv);

    prng_srand (23874);
    
    src = make_source ();
    dest = make_dest ();

    printf ("# %-6s %-22s   %-14s %-12s\n",
	    "ratio",
	    "resolutions",
	    "time [ms]",
	    "time per pixel [ns]");
    for (scale = start_scale; scale < end_scale; scale += scale_step)
    {
	int i;
	int source_width = dest_width / scale;
	int source_height = dest_height / scale;
	pixman_fixed_t s = (1 / scale) * 65536.0 + 0.5;
	pixman_transform_t transform;
	double t1, t2, t = -1;

	set_filter (src, scale, filter);

	pixman_transform_init_scale (&transform, s, s);
	pixman_image_set_transform (src, &transform);

	pixman_image_composite (
	    PIXMAN_OP_CLEAR, dest, NULL, dest,
	    0, 0, 0, 0, 0, 0, dest_width, dest_height);

	for (i = 0; i < test_repeats; i++)
	{
	    t1 = gettime();
	    pixman_image_composite (
		op, src, NULL, dest,
		0, 0, 0, 0, 0, 0, dest_width, dest_height);
	    t2 = gettime();
	    if (t < 0 || t2 - t1 < t)
		t = t2 - t1;
	}

	printf ("%6.2f : %4dx%-4d => %4dx%-4d : %12.4f : %12.4f\n",
		scale, source_width, source_height, dest_width, dest_height,
		t * 1000, (t / (dest_width * dest_height)) * 1000000000);

    }

    return 0;
}
