"""Tests for struct member access IR type correctness.

Covers:
- char/short member loads use correct ir_type (not default INT)
- char/short member stores use the member's declared type
- nested struct member access (outer.inner.x) returns address for struct members
- mixed-width struct layouts
- compound assignment on narrow members
"""

from compiler.ast_nodes import (
	Assignment,
	CompoundAssignment,
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	MemberAccess,
	Program,
	ReturnStmt,
	StructDecl,
	StructMember,
	TypeSpec,
	VarDecl,
)
from compiler.ir import (
	IRLoad,
	IRStore,
	IRType,
)
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _char_type() -> TypeSpec:
	return TypeSpec(base_type="char")


def _short_type() -> TypeSpec:
	return TypeSpec(base_type="short", width_modifier="short")


def _make_program(*funcs: FunctionDecl) -> Program:
	return Program(declarations=list(funcs))


class TestMemberLoadWidth:
	"""IRLoad for member access should use the member's declared ir_type."""

	def test_char_member_load_uses_char_ir_type(self) -> None:
		"""struct s { int a; char b; }; s.b should emit IRLoad with ir_type=CHAR."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="s",
							members=[
								StructMember(type_spec=_int_type(), name="a"),
								StructMember(type_spec=_char_type(), name="b"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
						ReturnStmt(
							expression=MemberAccess(
								object=Identifier(name="val"),
								member="b",
								is_arrow=False,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 1
		assert loads[0].ir_type == IRType.CHAR

	def test_short_member_load_uses_short_ir_type(self) -> None:
		"""struct s { char a; short b; }; s.b should emit IRLoad with ir_type=SHORT."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="s",
							members=[
								StructMember(type_spec=_char_type(), name="a"),
								StructMember(type_spec=_short_type(), name="b"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
						ReturnStmt(
							expression=MemberAccess(
								object=Identifier(name="val"),
								member="b",
								is_arrow=False,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 1
		assert loads[0].ir_type == IRType.SHORT

	def test_int_member_load_uses_int_ir_type(self) -> None:
		"""struct s { char a; int b; }; s.b should emit IRLoad with ir_type=INT."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="s",
							members=[
								StructMember(type_spec=_char_type(), name="a"),
								StructMember(type_spec=_int_type(), name="b"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
						ReturnStmt(
							expression=MemberAccess(
								object=Identifier(name="val"),
								member="b",
								is_arrow=False,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 1
		assert loads[0].ir_type == IRType.INT


class TestMemberStoreWidth:
	"""IRStore for member assignment should use the member's declared ir_type."""

	def test_char_member_store_uses_char_ir_type(self) -> None:
		"""s.b = 42 where b is char should emit IRStore with ir_type=CHAR."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="s",
							members=[
								StructMember(type_spec=_int_type(), name="a"),
								StructMember(type_spec=_char_type(), name="b"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
						ExprStmt(
							expression=Assignment(
								target=MemberAccess(
									object=Identifier(name="val"),
									member="b",
									is_arrow=False,
								),
								value=IntLiteral(value=42),
							)
						),
						ReturnStmt(expression=IntLiteral(value=0)),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].ir_type == IRType.CHAR

	def test_short_member_store_uses_short_ir_type(self) -> None:
		"""s.b = 42 where b is short should emit IRStore with ir_type=SHORT."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="s",
							members=[
								StructMember(type_spec=_int_type(), name="a"),
								StructMember(type_spec=_short_type(), name="b"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
						ExprStmt(
							expression=Assignment(
								target=MemberAccess(
									object=Identifier(name="val"),
									member="b",
									is_arrow=False,
								),
								value=IntLiteral(value=42),
							)
						),
						ReturnStmt(expression=IntLiteral(value=0)),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].ir_type == IRType.SHORT


class TestCompoundAssignmentMemberWidth:
	"""Compound assignment (+=, etc.) on members should use the member's ir_type."""

	def test_char_member_compound_assign(self) -> None:
		"""s.b += 1 where b is char should emit IRLoad and IRStore with ir_type=CHAR."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="s",
							members=[
								StructMember(type_spec=_int_type(), name="a"),
								StructMember(type_spec=_char_type(), name="b"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
						ExprStmt(
							expression=CompoundAssignment(
								target=MemberAccess(
									object=Identifier(name="val"),
									member="b",
									is_arrow=False,
								),
								op="+=",
								value=IntLiteral(value=1),
							)
						),
						ReturnStmt(expression=IntLiteral(value=0)),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(loads) == 1
		assert loads[0].ir_type == IRType.CHAR
		assert len(stores) == 1
		assert stores[0].ir_type == IRType.CHAR


class TestNestedStructMemberAccess:
	"""Nested struct member access (outer.inner.x) should not load the inner struct."""

	def test_nested_struct_no_load_for_inner(self) -> None:
		"""struct outer { struct inner i; }; outer.i.x -- inner access returns address, not scalar load."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="inner",
							members=[
								StructMember(type_spec=_int_type(), name="x"),
								StructMember(type_spec=_int_type(), name="y"),
							],
						),
						StructDecl(
							name="outer",
							members=[
								StructMember(
									type_spec=TypeSpec(base_type="struct inner"),
									name="i",
								),
								StructMember(type_spec=_int_type(), name="z"),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="outer"), name="o"),
						ReturnStmt(
							expression=MemberAccess(
								object=MemberAccess(
									object=Identifier(name="o"),
									member="i",
									is_arrow=False,
								),
								member="x",
								is_arrow=False,
							)
						),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		loads = [i for i in body if isinstance(i, IRLoad)]
		# Should have exactly 1 load: for x (int). NOT 2 (one for inner struct, one for x).
		assert len(loads) == 1
		assert loads[0].ir_type == IRType.INT

	def test_nested_struct_assign_inner_field(self) -> None:
		"""outer.i.x = 5 should store with ir_type=INT and no load for the inner struct."""
		prog = _make_program(
			FunctionDecl(
				return_type=_int_type(),
				name="f",
				body=CompoundStmt(
					statements=[
						StructDecl(
							name="inner",
							members=[
								StructMember(type_spec=_int_type(), name="x"),
								StructMember(type_spec=_char_type(), name="y"),
							],
						),
						StructDecl(
							name="outer",
							members=[
								StructMember(
									type_spec=TypeSpec(base_type="struct inner"),
									name="i",
								),
							],
						),
						VarDecl(type_spec=TypeSpec(base_type="outer"), name="o"),
						ExprStmt(
							expression=Assignment(
								target=MemberAccess(
									object=MemberAccess(
										object=Identifier(name="o"),
										member="i",
										is_arrow=False,
									),
									member="x",
									is_arrow=False,
								),
								value=IntLiteral(value=5),
							)
						),
						ReturnStmt(expression=IntLiteral(value=0)),
					]
				),
			)
		)
		ir = IRGenerator().generate(prog)
		body = ir.functions[0].body
		stores = [i for i in body if isinstance(i, IRStore)]
		assert len(stores) == 1
		assert stores[0].ir_type == IRType.INT
		# No loads for the inner struct address
		loads = [i for i in body if isinstance(i, IRLoad)]
		assert len(loads) == 0


class TestMixedWidthStructLayout:
	"""Struct with mixed char/short/int members should all get correct ir_types."""

	def test_mixed_width_loads(self) -> None:
		"""struct s { char a; short b; int c; }; accessing each member uses correct ir_type."""
		members = [
			StructMember(type_spec=_char_type(), name="a"),
			StructMember(type_spec=_short_type(), name="b"),
			StructMember(type_spec=_int_type(), name="c"),
		]

		for member_name, expected_type in [("a", IRType.CHAR), ("b", IRType.SHORT), ("c", IRType.INT)]:
			prog = _make_program(
				FunctionDecl(
					return_type=_int_type(),
					name="f",
					body=CompoundStmt(
						statements=[
							StructDecl(name="s", members=list(members)),
							VarDecl(type_spec=TypeSpec(base_type="s"), name="val"),
							ReturnStmt(
								expression=MemberAccess(
									object=Identifier(name="val"),
									member=member_name,
									is_arrow=False,
								)
							),
						]
					),
				)
			)
			ir = IRGenerator().generate(prog)
			body = ir.functions[0].body
			loads = [i for i in body if isinstance(i, IRLoad)]
			assert len(loads) == 1, f"Expected 1 load for member {member_name}"
			assert loads[0].ir_type == expected_type, (
				f"Member {member_name}: expected ir_type={expected_type}, got {loads[0].ir_type}"
			)
