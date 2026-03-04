/* Adapted from gcc.dg/torture/misc_void_func.c -- tests miscellaneous */

void triple(int *x) {
	*x = *x * 3;
}

void swap(int *a, int *b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

int main(void) {
	int val = 7;
	triple(&val);
	if (val != 21) return 1;

	int a = 10;
	int b = 20;
	swap(&a, &b);
	if (a != 20) return 1;
	if (b != 10) return 1;

	return 0;
}
