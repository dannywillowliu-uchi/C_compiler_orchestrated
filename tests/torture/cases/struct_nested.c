/* Adapted from gcc.dg/torture/struct_nested.c -- tests struct operations */

struct Inner {
	int value;
};

struct Outer {
	struct Inner inner;
	int extra;
};

int main(void) {
	struct Outer o;
	o.inner.value = 42;
	o.extra = 100;

	if (o.inner.value != 42) return 1;
	if (o.extra != 100) return 1;

	o.inner.value = o.inner.value + o.extra;
	if (o.inner.value != 142) return 1;

	return 0;
}
