/* Adapted from gcc.dg/torture/ptr_array_decay.c -- tests pointers */
/* Array decays to pointer when passed to function */

int sum_array(int *arr, int n) {
	int s = 0;
	for (int i = 0; i < n; i = i + 1) {
		s = s + arr[i];
	}
	return s;
}

int first_elem(int *p) {
	return *p;
}

int main(void) {
	int arr[4];
	arr[0] = 1;
	arr[1] = 2;
	arr[2] = 3;
	arr[3] = 4;

	int total = sum_array(arr, 4);
	if (total != 10) return 1;

	int first = first_elem(arr);
	if (first != 1) return 1;

	return 0;
}
