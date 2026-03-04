/* Adapted from gcc.dg/torture/ptr_array_equiv.c -- tests pointers */
/* arr[i] == *(arr + i) */

int main(void) {
	int arr[4];
	arr[0] = 100;
	arr[1] = 200;
	arr[2] = 300;
	arr[3] = 400;

	int *p = arr;

	if (arr[0] != *(p + 0)) return 1;
	if (arr[1] != *(p + 1)) return 1;
	if (arr[2] != *(p + 2)) return 1;
	if (arr[3] != *(p + 3)) return 1;

	/* also: i[arr] == arr[i] in C (commutative) */
	/* but let's just verify pointer dereference equivalence */
	for (int i = 0; i < 4; i = i + 1) {
		if (arr[i] != *(arr + i)) return 1;
	}

	return 0;
}
