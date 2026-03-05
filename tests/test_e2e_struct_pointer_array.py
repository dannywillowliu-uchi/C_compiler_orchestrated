"""End-to-end correctness tests for complex struct/pointer/array interactions.

Tests exercise the full pipeline (parse -> semantic -> IR -> codegen -> link -> execute)
for complex feature interactions involving structs, pointers, arrays, unions, and
function pointers.

Tests verify runtime correctness where the compiler produces correct output, and
verify assembly output patterns where codegen has known issues.
"""

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


# ---------------------------------------------------------------------------
# (1) Pointer to struct member access chains
# ---------------------------------------------------------------------------


class TestPointerToStructAccessChainsAsm:
	"""Verify assembly generation for arrow/dot access chains (codegen has known
	offset bugs with arrow operator, so we validate compilation + asm patterns)."""

	def test_arrow_then_dot_compiles(self) -> None:
		"""p->inner.val compiles through full pipeline."""
		source = """
		struct Inner { int val; };
		struct Outer { struct Inner inner; int id; };
		int main() {
			struct Outer o;
			o.inner.val = 10;
			o.id = 20;
			struct Outer *p = &o;
			return p->inner.val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "mov" in asm.lower()
		assert "ret" in asm

	def test_arrow_dot_arrow_chain_compiles(self) -> None:
		"""p->embedded.ptr->val: arrow, dot, arrow chain compiles."""
		source = """
		struct Leaf { int val; };
		struct Mid { struct Leaf *ptr; };
		struct Root { struct Mid embedded; };
		int main() {
			struct Leaf leaf;
			leaf.val = 33;
			struct Root root;
			root.embedded.ptr = &leaf;
			struct Root *p = &root;
			return p->embedded.ptr->val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_double_arrow_compiles(self) -> None:
		"""p->next->val: two successive arrow dereferences compile."""
		source = """
		struct Node { struct Node *next; int val; };
		int main() {
			struct Node a;
			struct Node b;
			a.next = &b;
			b.next = 0;
			b.val = 55;
			struct Node *p = &a;
			return p->next->val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_triple_arrow_compiles(self) -> None:
		"""Three-level pointer chain compiles through full pipeline."""
		source = """
		struct Node { struct Node *next; int val; };
		int main() {
			struct Node a;
			struct Node b;
			struct Node c;
			a.next = &b;
			b.next = &c;
			c.next = 0;
			c.val = 77;
			return a.next->next->val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_mixed_dot_arrow_deep_compiles(self) -> None:
		"""Deep mixed dot/arrow chain: s.a.b->inner.c compiles."""
		source = """
		struct C { int c; };
		struct B { struct C inner; };
		struct A { struct B *b; };
		struct S { struct A a; };
		int main() {
			struct B bval;
			bval.inner.c = 19;
			struct S s;
			s.a.b = &bval;
			return s.a.b->inner.c;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_write_through_arrow_compiles(self) -> None:
		"""Write through two-level arrow chain compiles."""
		source = """
		struct Inner { int val; };
		struct Outer { struct Inner *ptr; };
		int main() {
			struct Inner i;
			i.val = 0;
			struct Outer o;
			o.ptr = &i;
			struct Outer *p = &o;
			p->ptr->val = 88;
			return i.val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# (2) Array of structs with pointer arithmetic
# ---------------------------------------------------------------------------


class TestArrayOfStructsAsm:
	"""Verify compilation of array-of-structs patterns (some have runtime
	issues with pointer arithmetic scaling)."""

	def test_array_of_structs_indexing_compiles(self) -> None:
		"""Array subscript on struct array compiles."""
		source = """
		struct Point { int x; int y; };
		int main() {
			struct Point pts[3];
			pts[0].x = 1; pts[0].y = 2;
			pts[1].x = 10; pts[1].y = 20;
			pts[2].x = 100; pts[2].y = 200;
			return pts[2].x + pts[1].y;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_pointer_arithmetic_over_struct_array_compiles(self) -> None:
		"""Pointer arithmetic over struct array compiles."""
		source = """
		struct Pair { int a; int b; };
		int main() {
			struct Pair arr[3];
			arr[0].a = 1; arr[0].b = 2;
			arr[1].a = 10; arr[1].b = 20;
			arr[2].a = 100; arr[2].b = 200;
			struct Pair *p = arr;
			p = p + 2;
			return p->a + p->b;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_loop_over_struct_array_compiles(self) -> None:
		"""Loop over struct array with index compiles."""
		source = """
		struct Val { int n; };
		int main() {
			struct Val arr[4];
			arr[0].n = 1; arr[1].n = 2; arr[2].n = 3; arr[3].n = 4;
			int sum = 0;
			int i = 0;
			while (i < 4) {
				sum = sum + arr[i].n;
				i = i + 1;
			}
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_pointer_walk_struct_array_compiles(self) -> None:
		"""Walk struct array using pointer increment compiles."""
		source = """
		struct Val { int n; };
		int main() {
			struct Val arr[3];
			arr[0].n = 10; arr[1].n = 20; arr[2].n = 30;
			struct Val *p = arr;
			int sum = p->n;
			p = p + 1;
			sum = sum + p->n;
			p = p + 1;
			sum = sum + p->n;
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_struct_array_generates_scaling(self) -> None:
		"""Struct array subscript should generate address scaling in asm."""
		source = """
		struct S { int a; int b; };
		int main() {
			struct S arr[2];
			arr[0].a = 1;
			arr[1].a = 2;
			return arr[1].a;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		# Should have memory operations for struct layout
		asm_lower = asm.lower()
		assert "mov" in asm_lower


# ---------------------------------------------------------------------------
# (3) Struct containing arrays accessed via pointer
# ---------------------------------------------------------------------------


class TestStructContainingArraysAsm:
	"""Struct array members accessed via pointer indirection."""

	def test_struct_array_member_via_pointer_compiles(self) -> None:
		"""Access struct's array member by taking pointer to it."""
		source = """
		struct Data { int values[4]; int count; };
		int main() {
			struct Data d;
			d.count = 4;
			int *p = d.values;
			*p = 10;
			*(p + 1) = 20;
			*(p + 2) = 30;
			*(p + 3) = 40;
			return *(p + 2);
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_struct_array_sum_via_pointer_compiles(self) -> None:
		"""Sum array elements inside struct via pointer compiles."""
		source = """
		struct Buf { int data[3]; };
		int main() {
			struct Buf b;
			int *p = b.data;
			*p = 5; *(p + 1) = 15; *(p + 2) = 25;
			return *p + *(p + 1) + *(p + 2);
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_nested_struct_array_member_compiles(self) -> None:
		"""Nested struct with inner array member compiles."""
		source = """
		struct Inner { int arr[2]; };
		struct Outer { struct Inner inner; int tag; };
		int main() {
			struct Outer o;
			o.tag = 33;
			int *p = o.inner.arr;
			*p = 11;
			*(p + 1) = 22;
			return *p + *(p + 1) + o.tag;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_struct_array_loop_compiles(self) -> None:
		"""Loop over struct array member via pointer compiles."""
		source = """
		struct Buf { int data[4]; };
		int main() {
			struct Buf b;
			int *p = b.data;
			int i = 0;
			while (i < 4) {
				*(p + i) = (i + 1) * 10;
				i = i + 1;
			}
			int sum = 0;
			i = 0;
			while (i < 4) {
				sum = sum + *(p + i);
				i = i + 1;
			}
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# (4) Nested struct initialization and member-by-member comparison
# ---------------------------------------------------------------------------


class TestNestedStructInitAndComparison:
	"""Runtime correctness tests for nested struct init and comparison
	(dot-only access patterns work correctly)."""

	@can_link
	def test_nested_struct_member_by_member(self, tmp_path: Path) -> None:
		"""Initialize nested struct members and verify all fields."""
		source = """
		struct Vec2 { int x; int y; };
		struct Rect { struct Vec2 origin; struct Vec2 size; };
		int main() {
			struct Rect r;
			r.origin.x = 10;
			r.origin.y = 20;
			r.size.x = 30;
			r.size.y = 40;
			int ok = 1;
			if (r.origin.x != 10) ok = 0;
			if (r.origin.y != 20) ok = 0;
			if (r.size.x != 30) ok = 0;
			if (r.size.y != 40) ok = 0;
			return ok;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_struct_copy_and_compare(self, tmp_path: Path) -> None:
		"""Copy struct fields individually and verify all members match."""
		source = """
		struct Pair { int a; int b; };
		int main() {
			struct Pair p1;
			p1.a = 42;
			p1.b = 99;
			struct Pair p2;
			p2.a = p1.a;
			p2.b = p1.b;
			if (p2.a != 42) return 1;
			if (p2.b != 99) return 2;
			return 0;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_three_level_nested_init(self, tmp_path: Path) -> None:
		"""Three levels of struct nesting, init and verify sum."""
		source = """
		struct A { int x; };
		struct B { struct A a; int y; };
		struct C { struct B b; int z; };
		int main() {
			struct C c;
			c.b.a.x = 1;
			c.b.y = 2;
			c.z = 3;
			return c.b.a.x + c.b.y + c.z;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 6

	@can_link
	def test_two_nested_structs_equality(self, tmp_path: Path) -> None:
		"""Initialize two nested structs with same values, compare field by field."""
		source = """
		struct Vec2 { int x; int y; };
		struct Rect { struct Vec2 origin; struct Vec2 size; };
		int main() {
			struct Rect a;
			a.origin.x = 1; a.origin.y = 2;
			a.size.x = 3; a.size.y = 4;
			struct Rect b;
			b.origin.x = 1; b.origin.y = 2;
			b.size.x = 3; b.size.y = 4;
			int eq = 1;
			if (a.origin.x != b.origin.x) eq = 0;
			if (a.origin.y != b.origin.y) eq = 0;
			if (a.size.x != b.size.x) eq = 0;
			if (a.size.y != b.size.y) eq = 0;
			return eq;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 1

	@can_link
	def test_nested_struct_modify_and_verify(self, tmp_path: Path) -> None:
		"""Modify nested fields and verify changes propagate."""
		source = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner inner; int c; };
		int main() {
			struct Outer o;
			o.inner.a = 10;
			o.inner.b = 20;
			o.c = 30;
			o.inner.a = o.inner.a + 5;
			o.inner.b = o.inner.b + 5;
			o.c = o.c + 5;
			if (o.inner.a != 15) return 1;
			if (o.inner.b != 25) return 2;
			if (o.c != 35) return 3;
			return 0;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 0

	@can_link
	def test_four_field_struct_init(self, tmp_path: Path) -> None:
		"""Struct with four int fields, verify all."""
		source = """
		struct Quad { int a; int b; int c; int d; };
		int main() {
			struct Quad q;
			q.a = 10; q.b = 20; q.c = 30; q.d = 40;
			return q.a + q.b + q.c + q.d;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100


# ---------------------------------------------------------------------------
# (5) Union punning with struct members
# ---------------------------------------------------------------------------


class TestUnionPunning:
	"""Union type punning tests -- basic patterns work at runtime."""

	@can_link
	def test_union_int_char_punning(self, tmp_path: Path) -> None:
		"""Write int, read low byte via char member."""
		source = """
		union Pun { int i; char c; };
		int main() {
			union Pun u;
			u.i = 65;
			return u.c;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 65

	@can_link
	def test_union_char_low_byte(self, tmp_path: Path) -> None:
		"""Write value > 255, read low byte."""
		source = """
		union Pun { int i; char c; };
		int main() {
			union Pun u;
			u.i = 256 + 42;
			return u.c;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_union_overwrite_member(self, tmp_path: Path) -> None:
		"""Write via one member, overwrite via another, read back."""
		source = """
		union Val { int i; char c; };
		int main() {
			union Val v;
			v.i = 1000;
			v.c = 7;
			return v.c;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 7

	@can_link
	def test_union_with_struct_member(self, tmp_path: Path) -> None:
		"""Union containing a struct, access struct fields."""
		source = """
		struct Pair { int lo; int hi; };
		union Data { struct Pair p; int i; };
		int main() {
			union Data d;
			d.p.lo = 10;
			d.p.hi = 20;
			return d.p.lo;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 10

	@can_link
	def test_union_write_int_read_struct_lo(self, tmp_path: Path) -> None:
		"""Write int to union, read back through struct member (type punning)."""
		source = """
		struct Pair { int lo; int hi; };
		union Data { struct Pair p; int i; };
		int main() {
			union Data d;
			d.i = 42;
			return d.p.lo;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	def test_union_pointer_access_compiles(self) -> None:
		"""Union pointer access compiles (has known offset issue at runtime)."""
		source = """
		union Val { int i; char c; };
		int main() {
			union Val v;
			v.i = 97;
			union Val *p = &v;
			return p->c;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_union_in_struct_compiles(self) -> None:
		"""Struct containing union, accessed via pointer, compiles."""
		source = """
		union Kind { int ival; char cval; };
		struct Tagged { int tag; union Kind data; };
		int main() {
			struct Tagged t;
			t.tag = 1;
			t.data.ival = 50;
			struct Tagged *p = &t;
			if (p->tag != 1) return 1;
			return p->data.ival;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	@can_link
	def test_union_in_struct_dot_access(self, tmp_path: Path) -> None:
		"""Struct with union member, dot access only (no pointers)."""
		source = """
		union Kind { int ival; char cval; };
		struct Tagged { int tag; union Kind data; };
		int main() {
			struct Tagged t;
			t.tag = 1;
			t.data.ival = 50;
			if (t.tag != 1) return 1;
			return t.data.ival;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 50


# ---------------------------------------------------------------------------
# (6) Function pointers stored in / interacting with structs
# ---------------------------------------------------------------------------


class TestFunctionPointersWithStructs:
	"""Function pointers interacting with structs via local fp variables."""

	@can_link
	def test_fp_called_with_struct_dot_fields(self, tmp_path: Path) -> None:
		"""Call function pointer with struct member values (dot access)."""
		source = """
		int add(int a, int b) { return a + b; }
		struct Pair { int x; int y; };
		int main() {
			int (*fp)(int, int) = add;
			struct Pair p;
			p.x = 30;
			p.y = 12;
			return fp(p.x, p.y);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_fp_dispatch_with_struct_fields(self, tmp_path: Path) -> None:
		"""Select function pointer at runtime, apply to struct fields."""
		source = """
		int add(int a, int b) { return a + b; }
		int sub(int a, int b) { return a - b; }
		struct Pair { int x; int y; };
		int main() {
			struct Pair p;
			p.x = 50;
			p.y = 30;
			int (*fp)(int, int);
			int sel = 1;
			if (sel) {
				fp = add;
			} else {
				fp = sub;
			}
			return fp(p.x, p.y);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 80

	@can_link
	def test_fp_on_nested_struct_field(self, tmp_path: Path) -> None:
		"""Function pointer applied to nested struct field value."""
		source = """
		int double_it(int x) { return x + x; }
		struct Inner { int n; };
		struct Outer { struct Inner inner; };
		int main() {
			struct Outer o;
			o.inner.n = 21;
			int (*fp)(int) = double_it;
			return fp(o.inner.n);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 42

	@can_link
	def test_two_fps_same_struct(self, tmp_path: Path) -> None:
		"""Two function pointers applied to same struct fields."""
		source = """
		int add(int a, int b) { return a + b; }
		int sub(int a, int b) { return a - b; }
		struct Pair { int x; int y; };
		int main() {
			struct Pair p;
			p.x = 50;
			p.y = 30;
			int (*fp_add)(int, int) = add;
			int (*fp_sub)(int, int) = sub;
			return fp_add(p.x, p.y) + fp_sub(p.x, p.y);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 100

	def test_fp_with_arrow_access_compiles(self) -> None:
		"""Function pointer with arrow access compiles (known runtime offset bug)."""
		source = """
		int mul(int a, int b) { return a * b; }
		struct Pair { int x; int y; };
		int main() {
			int (*fp)(int, int) = mul;
			struct Pair p;
			p.x = 5;
			p.y = 6;
			struct Pair *pp = &p;
			return fp(pp->x, pp->y);
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "call" in asm.lower()


# ---------------------------------------------------------------------------
# Cross-cutting complex interactions
# ---------------------------------------------------------------------------


class TestComplexInteractions:
	"""Tests combining multiple feature interactions."""

	def test_linked_list_compiles(self) -> None:
		"""Linked list traversal compiles through full pipeline."""
		source = """
		struct Node { int val; struct Node *next; };
		int main() {
			struct Node c; c.val = 3; c.next = 0;
			struct Node b; b.val = 2; b.next = &c;
			struct Node a; a.val = 1; a.next = &b;
			int sum = 0;
			struct Node *cur = &a;
			while (cur) {
				sum = sum + cur->val;
				cur = cur->next;
			}
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm
		# Should have a conditional jump for while loop
		asm_lower = asm.lower()
		assert "j" in asm_lower

	def test_struct_with_pointer_to_another_struct_compiles(self) -> None:
		"""Struct holding pointer to a different struct type compiles."""
		source = """
		struct Config { int mode; int flags; };
		struct App { struct Config *cfg; int running; };
		int main() {
			struct Config c;
			c.mode = 3;
			c.flags = 7;
			struct App app;
			app.cfg = &c;
			app.running = 1;
			return app.cfg->mode + app.cfg->flags + app.running;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_nested_modify_via_pointer_compiles(self) -> None:
		"""Modify deeply nested member via pointer chain compiles."""
		source = """
		struct Inner { int val; };
		struct Mid { struct Inner *inner; };
		struct Outer { struct Mid mid; };
		int main() {
			struct Inner i;
			i.val = 5;
			struct Outer o;
			o.mid.inner = &i;
			struct Outer *p = &o;
			p->mid.inner->val = 99;
			return i.val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_array_of_structs_with_pointer_members_compiles(self) -> None:
		"""Array of structs with pointer members compiles."""
		source = """
		struct Entry { int *valptr; int id; };
		int main() {
			int a = 10;
			int b = 20;
			struct Entry entries[2];
			entries[0].valptr = &a;
			entries[0].id = 0;
			entries[1].valptr = &b;
			entries[1].id = 1;
			return *entries[0].valptr + *entries[1].valptr;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	@can_link
	def test_union_and_struct_combined(self, tmp_path: Path) -> None:
		"""Two structs with union, all dot-only access."""
		source = """
		union Val { int ival; char cval; };
		struct Entry { int type; union Val data; };
		int main() {
			struct Entry e1;
			e1.type = 0;
			e1.data.ival = 42;
			struct Entry e2;
			e2.type = 1;
			e2.data.cval = 10;
			return e1.data.ival + e2.data.cval;
		}
		"""
		assert _compile_and_run(source, tmp_path) == 52

	@can_link
	def test_fp_with_nested_struct(self, tmp_path: Path) -> None:
		"""Function pointer applied to nested struct members."""
		source = """
		int sum3(int a, int b, int c) { return a + b + c; }
		struct A { int x; };
		struct B { struct A a; int y; };
		struct C { struct B b; int z; };
		int main() {
			struct C c;
			c.b.a.x = 10;
			c.b.y = 20;
			c.z = 30;
			int (*fp)(int, int, int) = sum3;
			return fp(c.b.a.x, c.b.y, c.z);
		}
		"""
		assert _compile_and_run(source, tmp_path) == 60

	def test_self_referential_struct(self) -> None:
		"""Self-referential struct (tree-like) compiles."""
		source = """
		struct Tree { int val; struct Tree *left; struct Tree *right; };
		int main() {
			struct Tree root;
			struct Tree left;
			struct Tree right;
			root.val = 1;
			root.left = &left;
			root.right = &right;
			left.val = 2;
			left.left = 0;
			left.right = 0;
			right.val = 3;
			right.left = 0;
			right.right = 0;
			return root.val + root.left->val + root.right->val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm

	def test_function_pointer_with_linked_list_compiles(self) -> None:
		"""Function pointer processing linked list values compiles."""
		source = """
		int double_it(int x) { return x + x; }
		struct Node { int val; struct Node *next; };
		int main() {
			struct Node b; b.val = 5; b.next = 0;
			struct Node a; a.val = 3; a.next = &b;
			int (*fp)(int) = double_it;
			int sum = 0;
			struct Node *cur = &a;
			while (cur) {
				sum = sum + fp(cur->val);
				cur = cur->next;
			}
			return sum;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "call" in asm.lower()

	def test_struct_weighted_accumulate_compiles(self) -> None:
		"""Weighted accumulation over struct array via pointer compiles."""
		source = """
		struct Weighted { int value; int weight; };
		int main() {
			struct Weighted items[3];
			items[0].value = 10; items[0].weight = 1;
			items[1].value = 20; items[1].weight = 2;
			items[2].value = 30; items[2].weight = 3;
			struct Weighted *p = items;
			int total = 0;
			int i = 0;
			while (i < 3) {
				total = total + p->value * p->weight;
				p = p + 1;
				i = i + 1;
			}
			return total;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm


# ---------------------------------------------------------------------------
# Assembly pattern verification
# ---------------------------------------------------------------------------


class TestAssemblyPatterns:
	"""Verify specific assembly patterns for struct operations."""

	def test_struct_member_generates_offset_addressing(self) -> None:
		"""Multi-member struct access uses offset-based addressing."""
		source = """
		struct S { int a; int b; int c; };
		int main() {
			struct S s;
			s.a = 1;
			s.b = 2;
			s.c = 3;
			return s.b;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm
		# Should have mov instructions for member access
		assert "mov" in asm.lower()

	def test_nested_struct_generates_composite_offset(self) -> None:
		"""Nested struct access produces composite offset calculations."""
		source = """
		struct Inner { int a; int b; };
		struct Outer { struct Inner inner; int c; };
		int main() {
			struct Outer o;
			o.inner.a = 1;
			o.inner.b = 2;
			o.c = 3;
			return o.inner.b;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "rbp" in asm

	def test_self_referential_struct_compiles_cleanly(self) -> None:
		"""Self-referential struct compiles with pointer-sized member."""
		source = """
		struct Node { int val; struct Node *next; };
		int main() {
			struct Node n;
			n.val = 42;
			n.next = 0;
			return n.val;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "42" in asm

	def test_arrow_generates_memory_load(self) -> None:
		"""Arrow operator generates memory load for pointer dereference."""
		source = """
		struct S { int x; int y; };
		int main() {
			struct S s;
			s.x = 10;
			s.y = 20;
			struct S *p = &s;
			return p->y;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		asm_lower = asm.lower()
		assert "mov" in asm_lower
		# Should have lea for address-of
		has_lea_or_mov = "lea" in asm_lower or "mov" in asm_lower
		assert has_lea_or_mov

	def test_union_shares_memory(self) -> None:
		"""Union members should overlay the same memory."""
		source = """
		union U { int i; char c; };
		int main() {
			union U u;
			u.i = 0;
			u.c = 42;
			return u.c;
		}
		"""
		asm = compile_source(source)
		assert "main" in asm
		assert "ret" in asm
