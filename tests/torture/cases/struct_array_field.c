/* Adapted from gcc.dg/torture/struct_array_field.c -- tests struct operations */

struct Data {
	int values[4];
	int count;
};

int main(void) {
	struct Data d;
	d.count = 4;
	d.values[0] = 10;
	d.values[1] = 20;
	d.values[2] = 30;
	d.values[3] = 40;

	int sum = 0;
	for (int i = 0; i < d.count; i = i + 1) {
		sum = sum + d.values[i];
	}

	if (sum != 100) return 1;
	if (d.values[2] != 30) return 1;

	return 0;
}
