/* Adapted from gcc.dg/torture/struct_pass_value.c -- tests struct operations */

struct Pair {
	int a;
	int b;
};

int sum_pair(struct Pair p) {
	return p.a + p.b;
}

int main(void) {
	struct Pair p;
	p.a = 30;
	p.b = 12;

	int result = sum_pair(p);
	if (result != 42) return 1;

	/* Verify pass-by-value: original unchanged */
	if (p.a != 30) return 1;
	if (p.b != 12) return 1;

	return 0;
}
