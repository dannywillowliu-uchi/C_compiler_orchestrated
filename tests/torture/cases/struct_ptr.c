/* Adapted from gcc.dg/torture/struct_ptr.c -- tests struct operations */

struct Node {
	int value;
	int flag;
};

void set_value(struct Node *n, int v) {
	n->value = v;
	n->flag = 1;
}

int main(void) {
	struct Node node;
	node.value = 0;
	node.flag = 0;

	struct Node *p = &node;
	set_value(p, 42);

	if (node.value != 42) return 1;
	if (node.flag != 1) return 1;
	if (p->value != 42) return 1;

	return 0;
}
