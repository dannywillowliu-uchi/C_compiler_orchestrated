/* Adapted from gcc.dg/torture/struct_basic.c -- tests struct operations */

struct Point {
	int x;
	int y;
};

int main(void) {
	struct Point p;
	p.x = 10;
	p.y = 20;

	if (p.x != 10) return 1;
	if (p.y != 20) return 1;

	p.x = p.x + p.y;
	if (p.x != 30) return 1;

	return 0;
}
