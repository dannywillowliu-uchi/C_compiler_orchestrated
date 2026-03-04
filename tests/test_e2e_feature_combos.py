"""E2E compile-assemble-link-run tests for recursive data structures, arrays of structs, and multi-feature programs."""

import os
import platform
import subprocess
from pathlib import Path

import pytest

from compiler.__main__ import compile_source
from compiler.linker import (
	ToolchainError,
	assemble,
	compile_and_link,
	detect_toolchain,
	link,
)


def _can_link_x86_64() -> bool:
	"""Check if the system can link x86-64 executables."""
	try:
		tc = detect_toolchain()
	except ToolchainError:
		return False
	import tempfile
	with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
		if platform.system() == "Darwin":
			f.write(".section __TEXT,__text\n.globl _main\n_main:\n\tmovl $42, %eax\n\tretq\n")
		else:
			f.write(".section .text\n.globl main\nmain:\n\tmovl $42, %eax\n\tret\n")
		asm_path = f.name
	obj_path = asm_path.replace(".s", ".o")
	exe_path = asm_path.replace(".s", "")
	try:
		assemble(asm_path, obj_path, toolchain=tc)
		link([obj_path], exe_path, toolchain=tc)
		return True
	except (ToolchainError, FileNotFoundError):
		return False
	finally:
		for p in [asm_path, obj_path, exe_path]:
			try:
				os.remove(p)
			except OSError:
				pass


can_link = pytest.mark.skipif(
	not _can_link_x86_64(),
	reason="x86-64 linker not available on this platform",
)


def _compile_and_run(source: str, tmp_path: Path) -> int:
	"""Compile C source through full pipeline, link, run, and return exit code."""
	asm = compile_source(source)
	exe = tmp_path / "test_exe"
	compile_and_link(asm, str(exe))
	result = subprocess.run([str(exe)], capture_output=True, timeout=10)
	return result.returncode


class TestRecursiveStructPointers:
	"""Recursive functions with struct pointers (linked list traversal)."""

	@can_link
	def test_linked_list_sum_three_nodes(self, tmp_path: Path) -> None:
		"""Build a 3-node linked list on the stack and sum values via recursive traversal."""
		source = """
		struct Node {
			int value;
			struct Node *next;
		};
		int list_sum(struct Node *n) {
			if (n == 0) {
				return 0;
			}
			return n->value + list_sum(n->next);
		}
		int main() {
			struct Node a;
			struct Node b;
			struct Node c;
			a.value = 10;
			b.value = 20;
			c.value = 30;
			c.next = 0;
			b.next = &c;
			a.next = &b;
			return list_sum(&a);
		}
		"""
		# 10 + 20 + 30 = 60
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_linked_list_count_nodes(self, tmp_path: Path) -> None:
		"""Count nodes in a linked list recursively."""
		source = """
		struct Node {
			int value;
			struct Node *next;
		};
		int list_count(struct Node *n) {
			if (n == 0) {
				return 0;
			}
			return 1 + list_count(n->next);
		}
		int main() {
			struct Node a;
			struct Node b;
			struct Node c;
			struct Node d;
			a.value = 1;
			b.value = 2;
			c.value = 3;
			d.value = 4;
			d.next = 0;
			c.next = &d;
			b.next = &c;
			a.next = &b;
			return list_count(&a);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 4

	@can_link
	def test_linked_list_find_max(self, tmp_path: Path) -> None:
		"""Find maximum value in a linked list recursively."""
		source = """
		struct Node {
			int value;
			struct Node *next;
		};
		int list_max(struct Node *n) {
			int rest;
			if (n->next == 0) {
				return n->value;
			}
			rest = list_max(n->next);
			return (n->value > rest) ? n->value : rest;
		}
		int main() {
			struct Node a;
			struct Node b;
			struct Node c;
			a.value = 15;
			b.value = 42;
			c.value = 7;
			c.next = 0;
			b.next = &c;
			a.next = &b;
			return list_max(&a);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42


class TestArraysOfStructs:
	"""Arrays of structs with member access in loops."""

	@can_link
	@pytest.mark.xfail(reason="compiler bug: array subscript on struct type loads value instead of keeping address")
	def test_array_of_points_sum_x(self, tmp_path: Path) -> None:
		"""Sum x coordinates from an array of Point structs."""
		source = """
		struct Point {
			int x;
			int y;
		};
		int main() {
			struct Point pts[3];
			int sum;
			int i;
			pts[0].x = 10;
			pts[0].y = 1;
			pts[1].x = 20;
			pts[1].y = 2;
			pts[2].x = 30;
			pts[2].y = 3;
			sum = 0;
			for (i = 0; i < 3; i++) {
				sum = sum + pts[i].x;
			}
			return sum;
		}
		"""
		# 10 + 20 + 30 = 60
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	@pytest.mark.xfail(reason="compiler bug: array subscript on struct type loads value instead of keeping address")
	def test_array_of_structs_sum_both_fields(self, tmp_path: Path) -> None:
		"""Sum both fields from an array of structs."""
		source = """
		struct Pair {
			int a;
			int b;
		};
		int main() {
			struct Pair items[4];
			int total;
			int i;
			items[0].a = 1;
			items[0].b = 2;
			items[1].a = 3;
			items[1].b = 4;
			items[2].a = 5;
			items[2].b = 6;
			items[3].a = 7;
			items[3].b = 8;
			total = 0;
			for (i = 0; i < 4; i++) {
				total = total + items[i].a + items[i].b;
			}
			return total;
		}
		"""
		# 1+2+3+4+5+6+7+8 = 36
		assert _compile_and_run(source, tmp_path) == 36

	@can_link
	@pytest.mark.xfail(reason="compiler bug: array subscript on struct type loads value instead of keeping address")
	def test_array_of_structs_find_max_field(self, tmp_path: Path) -> None:
		"""Find the maximum value field from an array of structs."""
		source = """
		struct Entry {
			int key;
			int val;
		};
		int main() {
			struct Entry arr[3];
			int max;
			int i;
			arr[0].key = 1;
			arr[0].val = 50;
			arr[1].key = 2;
			arr[1].val = 80;
			arr[2].key = 3;
			arr[2].val = 30;
			max = arr[0].val;
			for (i = 1; i < 3; i++) {
				if (arr[i].val > max) {
					max = arr[i].val;
				}
			}
			return max;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 80


class TestSwitchStructPointerArray:
	"""Programs combining switch + struct + pointer + array features."""

	@can_link
	def test_switch_dispatches_struct_operations(self, tmp_path: Path) -> None:
		"""Use switch to select operation on struct fields."""
		source = """
		struct Vec2 {
			int x;
			int y;
		};
		int operate(struct Vec2 *v, int op) {
			int result;
			switch (op) {
				case 0:
					result = v->x + v->y;
					break;
				case 1:
					result = v->x - v->y;
					break;
				case 2:
					result = v->x * v->y;
					break;
				default:
					result = 0;
					break;
			}
			return result;
		}
		int main() {
			struct Vec2 v;
			v.x = 12;
			v.y = 5;
			return operate(&v, 2);
		}
		"""
		# 12 * 5 = 60
		assert _compile_and_run(source, tmp_path) == 60

	@can_link
	def test_switch_over_array_with_struct_result(self, tmp_path: Path) -> None:
		"""Use switch to pick from array, store result in struct."""
		source = """
		struct Result {
			int code;
			int value;
		};
		int main() {
			int scores[4];
			struct Result r;
			int selector;
			scores[0] = 10;
			scores[1] = 25;
			scores[2] = 50;
			scores[3] = 75;
			selector = 2;
			switch (selector) {
				case 0:
					r.value = scores[0];
					r.code = 1;
					break;
				case 1:
					r.value = scores[1];
					r.code = 2;
					break;
				case 2:
					r.value = scores[2];
					r.code = 3;
					break;
				default:
					r.value = scores[3];
					r.code = 4;
					break;
			}
			return r.value + r.code;
		}
		"""
		# scores[2]=50, code=3 => 53
		assert _compile_and_run(source, tmp_path) == 53

	@can_link
	def test_switch_loop_struct_array_accumulate(self, tmp_path: Path) -> None:
		"""Loop over array, use switch on values, accumulate into struct."""
		source = """
		struct Acc {
			int even_sum;
			int odd_sum;
		};
		int main() {
			int vals[6];
			struct Acc acc;
			int i;
			int tag;
			vals[0] = 4;
			vals[1] = 7;
			vals[2] = 2;
			vals[3] = 9;
			vals[4] = 6;
			vals[5] = 3;
			acc.even_sum = 0;
			acc.odd_sum = 0;
			for (i = 0; i < 6; i++) {
				tag = vals[i] % 2;
				switch (tag) {
					case 0:
						acc.even_sum = acc.even_sum + vals[i];
						break;
					case 1:
						acc.odd_sum = acc.odd_sum + vals[i];
						break;
				}
			}
			return acc.even_sum + acc.odd_sum;
		}
		"""
		# 4+7+2+9+6+3 = 31
		assert _compile_and_run(source, tmp_path) == 31


class TestNestedFunctionCallsWithStructPointers:
	"""Nested function calls with struct arguments passed by pointer."""

	@can_link
	def test_nested_struct_transforms(self, tmp_path: Path) -> None:
		"""Pass struct pointer through chain of functions that modify fields."""
		source = """
		struct Data {
			int x;
			int y;
		};
		void double_x(struct Data *d) {
			d->x = d->x * 2;
		}
		void add_to_y(struct Data *d, int amount) {
			d->y = d->y + amount;
		}
		int sum_fields(struct Data *d) {
			return d->x + d->y;
		}
		int main() {
			struct Data d;
			d.x = 10;
			d.y = 5;
			double_x(&d);
			add_to_y(&d, 15);
			return sum_fields(&d);
		}
		"""
		# x: 10*2=20, y: 5+15=20, sum=40
		assert _compile_and_run(source, tmp_path) == 40

	@can_link
	def test_struct_pointer_swap_fields(self, tmp_path: Path) -> None:
		"""Swap fields of two structs via pointer parameters."""
		source = """
		struct Pair {
			int first;
			int second;
		};
		void swap_first(struct Pair *a, struct Pair *b) {
			int tmp;
			tmp = a->first;
			a->first = b->first;
			b->first = tmp;
		}
		int main() {
			struct Pair p1;
			struct Pair p2;
			p1.first = 100;
			p1.second = 1;
			p2.first = 200;
			p2.second = 2;
			swap_first(&p1, &p2);
			return p1.first - p2.first;
		}
		"""
		# After swap: p1.first=200, p2.first=100 => 200-100=100
		assert _compile_and_run(source, tmp_path) == 100

	@can_link
	def test_nested_calls_accumulate_struct(self, tmp_path: Path) -> None:
		"""Multiple nested function calls that read/write the same struct."""
		source = """
		struct Counter {
			int count;
			int total;
		};
		void increment(struct Counter *c) {
			c->count = c->count + 1;
		}
		void add_value(struct Counter *c, int v) {
			c->total = c->total + v;
			increment(c);
		}
		int main() {
			struct Counter c;
			c.count = 0;
			c.total = 0;
			add_value(&c, 10);
			add_value(&c, 20);
			add_value(&c, 30);
			return c.count + c.total;
		}
		"""
		# count=3, total=60 => 63
		assert _compile_and_run(source, tmp_path) == 63

	@can_link
	def test_compute_via_struct_pointer_chain(self, tmp_path: Path) -> None:
		"""Chain of functions each reading a struct pointer and returning a computed value."""
		source = """
		struct Rect {
			int width;
			int height;
		};
		int area(struct Rect *r) {
			return r->width * r->height;
		}
		int perimeter(struct Rect *r) {
			return 2 * (r->width + r->height);
		}
		int main() {
			struct Rect r;
			r.width = 5;
			r.height = 8;
			return area(&r) + perimeter(&r);
		}
		"""
		# area=40, perimeter=26 => 66
		assert _compile_and_run(source, tmp_path) == 66


class TestEnumAsArrayIndices:
	"""Programs using enum values as array indices."""

	@can_link
	def test_enum_indices_basic(self, tmp_path: Path) -> None:
		"""Use enum constants to index into an array."""
		source = """
		enum Color { RED, GREEN, BLUE };
		int main() {
			int palette[3];
			palette[RED] = 10;
			palette[GREEN] = 20;
			palette[BLUE] = 30;
			return palette[GREEN];
		}
		"""
		assert _compile_and_run(source, tmp_path) == 20

	@can_link
	def test_enum_indices_sum(self, tmp_path: Path) -> None:
		"""Sum array elements indexed by enum constants."""
		source = """
		enum Dir { NORTH, SOUTH, EAST, WEST, NUM_DIRS };
		int main() {
			int speeds[4];
			int total;
			speeds[NORTH] = 10;
			speeds[SOUTH] = 20;
			speeds[EAST] = 30;
			speeds[WEST] = 40;
			total = speeds[NORTH] + speeds[SOUTH] + speeds[EAST] + speeds[WEST];
			return total;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100

	@can_link
	def test_enum_with_explicit_values_as_indices(self, tmp_path: Path) -> None:
		"""Enum with explicit values used as array indices."""
		source = """
		enum Slot { FIRST = 0, SECOND = 1, THIRD = 2 };
		int main() {
			int data[3];
			data[FIRST] = 11;
			data[SECOND] = 22;
			data[THIRD] = 33;
			return data[FIRST] + data[THIRD];
		}
		"""
		# 11 + 33 = 44
		assert _compile_and_run(source, tmp_path) == 44

	@can_link
	def test_enum_index_in_loop(self, tmp_path: Path) -> None:
		"""Use enum constant as loop bound and index."""
		source = """
		enum Size { SMALL = 0, MEDIUM = 1, LARGE = 2, NUM_SIZES = 3 };
		int main() {
			int weights[3];
			int total;
			int i;
			weights[SMALL] = 5;
			weights[MEDIUM] = 15;
			weights[LARGE] = 25;
			total = 0;
			for (i = 0; i < NUM_SIZES; i++) {
				total = total + weights[i];
			}
			return total;
		}
		"""
		# 5 + 15 + 25 = 45
		assert _compile_and_run(source, tmp_path) == 45

	@can_link
	def test_enum_switch_and_array(self, tmp_path: Path) -> None:
		"""Combine enum with switch and array access."""
		source = """
		enum Level { LOW = 0, MED = 1, HIGH = 2 };
		int main() {
			int thresholds[3];
			int level;
			int result;
			thresholds[LOW] = 10;
			thresholds[MED] = 50;
			thresholds[HIGH] = 90;
			level = MED;
			switch (level) {
				case LOW:
					result = thresholds[LOW];
					break;
				case MED:
					result = thresholds[MED];
					break;
				case HIGH:
					result = thresholds[HIGH];
					break;
				default:
					result = 0;
					break;
			}
			return result;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50
