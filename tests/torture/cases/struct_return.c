/* Adapted from gcc.dg/torture/struct_return.c -- tests struct operations */

struct Pair {
	int x;
	int y;
};

struct Pair make_pair(int a, int b) {
	struct Pair p;
	p.x = a;
	p.y = b;
	return p;
}

int main(void) {
	struct Pair p = make_pair(42, 99);
	if (p.x != 42) return 1;
	if (p.y != 99) return 1;

	struct Pair q = make_pair(p.y, p.x);
	if (q.x != 99) return 1;
	if (q.y != 42) return 1;

	return 0;
}
