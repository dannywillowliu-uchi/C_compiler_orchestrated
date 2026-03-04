/* Adapted from gcc.dg/torture/call_deep_nesting.c -- tests calling convention */

int inc(int x) {
	return x + 1;
}

int main(void) {
	int result = inc(inc(inc(inc(inc(0)))));
	if (result != 5) return 1;

	result = inc(inc(inc(inc(inc(inc(inc(inc(inc(inc(0))))))))));
	if (result != 10) return 1;

	return 0;
}
