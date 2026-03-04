/* Adapted from gcc.dg/torture/struct_large.c -- tests struct operations */
/* Large struct tests copy mechanics */

struct Big {
	int a;
	int b;
	int c;
	int d;
	int e;
	int f;
	int g;
	int h;
};

struct Big make_big(int base) {
	struct Big s;
	s.a = base;
	s.b = base + 1;
	s.c = base + 2;
	s.d = base + 3;
	s.e = base + 4;
	s.f = base + 5;
	s.g = base + 6;
	s.h = base + 7;
	return s;
}

int main(void) {
	struct Big s = make_big(10);
	if (s.a != 10) return 1;
	if (s.d != 13) return 1;
	if (s.h != 17) return 1;

	int sum = s.a + s.b + s.c + s.d + s.e + s.f + s.g + s.h;
	/* 10+11+12+13+14+15+16+17 = 108 */
	if (sum != 108) return 1;

	return 0;
}
