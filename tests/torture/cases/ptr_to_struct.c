/* Adapted from gcc.dg/torture/ptr_to_struct.c -- tests pointers */

struct Data {
	int x;
	int y;
};

void swap_fields(struct Data *d) {
	int tmp = d->x;
	d->x = d->y;
	d->y = tmp;
}

int main(void) {
	struct Data d;
	d.x = 10;
	d.y = 20;

	struct Data *p = &d;
	if (p->x != 10) return 1;
	if (p->y != 20) return 1;

	swap_fields(p);
	if (d.x != 20) return 1;
	if (d.y != 10) return 1;

	return 0;
}
