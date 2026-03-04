/* Adapted from gcc.dg/torture/misc_conditional_expr.c -- tests miscellaneous */

int abs_val(int x) {
	return x >= 0 ? x : -x;
}

int clamp(int val, int lo, int hi) {
	return val < lo ? lo : (val > hi ? hi : val);
}

int main(void) {
	if (abs_val(5) != 5) return 1;
	if (abs_val(-5) != 5) return 1;
	if (abs_val(0) != 0) return 1;

	if (clamp(5, 0, 10) != 5) return 1;
	if (clamp(-5, 0, 10) != 0) return 1;
	if (clamp(15, 0, 10) != 10) return 1;
	if (clamp(0, 0, 10) != 0) return 1;
	if (clamp(10, 0, 10) != 10) return 1;

	return 0;
}
