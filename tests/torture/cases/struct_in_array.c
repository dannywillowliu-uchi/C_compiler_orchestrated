/* Adapted from gcc.dg/torture/struct_in_array.c -- tests struct operations */

struct Item {
	int id;
	int value;
};

int main(void) {
	struct Item items[3];

	items[0].id = 0;
	items[0].value = 100;
	items[1].id = 1;
	items[1].value = 200;
	items[2].id = 2;
	items[2].value = 300;

	int sum = 0;
	for (int i = 0; i < 3; i = i + 1) {
		if (items[i].id != i) return 1;
		sum = sum + items[i].value;
	}

	if (sum != 600) return 1;

	return 0;
}
