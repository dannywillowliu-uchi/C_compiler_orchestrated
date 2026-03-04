/* Adapted from gcc.dg/torture/struct_init.c -- tests struct operations */

struct Config {
	int width;
	int height;
	int depth;
};

int main(void) {
	struct Config c;
	c.width = 640;
	c.height = 480;
	c.depth = 32;

	if (c.width != 640) return 1;
	if (c.height != 480) return 1;
	if (c.depth != 32) return 1;

	int area = c.width * c.height;
	if (area != 307200) return 1;

	return 0;
}
