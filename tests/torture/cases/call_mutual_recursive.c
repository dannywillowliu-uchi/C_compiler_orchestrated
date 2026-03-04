/* Adapted from gcc.dg/torture/call_mutual_recursive.c -- tests calling convention */

int is_odd(int n);

int is_even(int n) {
	if (n == 0) return 1;
	return is_odd(n - 1);
}

int is_odd(int n) {
	if (n == 0) return 0;
	return is_even(n - 1);
}

int main(void) {
	if (is_even(0) != 1) return 1;
	if (is_even(1) != 0) return 1;
	if (is_even(4) != 1) return 1;
	if (is_even(7) != 0) return 1;
	if (is_odd(0) != 0) return 1;
	if (is_odd(1) != 1) return 1;
	if (is_odd(5) != 1) return 1;
	if (is_odd(6) != 0) return 1;
	return 0;
}
