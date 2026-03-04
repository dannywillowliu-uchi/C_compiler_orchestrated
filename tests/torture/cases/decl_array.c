/* Adapted from gcc.dg/torture/decl_array.c -- tests declarations */

int main(void) {
	int arr[5];
	arr[0] = 10;
	arr[1] = 20;
	arr[2] = 30;
	arr[3] = 40;
	arr[4] = 50;

	if (arr[0] != 10) return 1;
	if (arr[2] != 30) return 1;
	if (arr[4] != 50) return 1;

	int sum = 0;
	for (int i = 0; i < 5; i = i + 1) {
		sum = sum + arr[i];
	}
	if (sum != 150) return 1;

	return 0;
}
