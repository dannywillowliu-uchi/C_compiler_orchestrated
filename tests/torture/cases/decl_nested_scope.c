/* Adapted from gcc.dg/torture/decl_nested_scope.c -- tests declarations */

int main(void) {
	int x = 10;

	{
		int x = 20;
		if (x != 20) return 1;

		{
			int x = 30;
			if (x != 30) return 1;
		}

		if (x != 20) return 1;
	}

	if (x != 10) return 1;

	return 0;
}
