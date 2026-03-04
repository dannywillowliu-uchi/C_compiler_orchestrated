/* Adapted from gcc.dg/torture/struct_assign.c -- tests struct operations */

struct Vec3 {
	int x;
	int y;
	int z;
};

int main(void) {
	struct Vec3 a;
	a.x = 1;
	a.y = 2;
	a.z = 3;

	struct Vec3 b;
	b = a;  /* struct assignment */

	if (b.x != 1) return 1;
	if (b.y != 2) return 1;
	if (b.z != 3) return 1;

	/* Modify b, a should be unchanged */
	b.x = 99;
	if (a.x != 1) return 1;
	if (b.x != 99) return 1;

	return 0;
}
