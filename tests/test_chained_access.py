"""Tests for chained arrow/dot member access and prefix inc/dec on MemberAccess."""

from compiler.ast_nodes import (
	Assignment,
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	MemberAccess,
	ParamDecl,
	Program,
	ReturnStmt,
	StructDecl,
	StructMember,
	TypeSpec,
	UnaryOp,
	VarDecl,
)
from compiler.ir import (
	IRBinOp,
	IRConst,
	IRLoad,
	IRStore,
)
from compiler.ir_gen import IRGenerator


def _int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def _void_type() -> TypeSpec:
	return TypeSpec(base_type="void")


def _struct_type(name: str) -> TypeSpec:
	return TypeSpec(base_type=f"struct {name}")


def _struct_ptr_type(name: str) -> TypeSpec:
	return TypeSpec(base_type=f"struct {name}", pointer_count=1)


def _make_program(*decls) -> Program:
	return Program(declarations=list(decls))


def _get_instrs(prog: Program) -> list:
	gen = IRGenerator()
	ir_prog = gen.generate(prog)
	assert len(ir_prog.functions) > 0
	return ir_prog.functions[0].body


# ------------------------------------------------------------------
# Struct definitions used across tests
# ------------------------------------------------------------------


def _struct_inner() -> StructDecl:
	"""struct Inner { int val; int extra; };"""
	return StructDecl(
		name="Inner",
		members=[
			StructMember(name="val", type_spec=_int_type()),
			StructMember(name="extra", type_spec=_int_type()),
		],
	)


def _struct_outer() -> StructDecl:
	"""struct Outer { struct Inner *inner; int id; };"""
	return StructDecl(
		name="Outer",
		members=[
			StructMember(name="inner", type_spec=_struct_ptr_type("Inner")),
			StructMember(name="id", type_spec=_int_type()),
		],
	)


def _struct_top() -> StructDecl:
	"""struct Top { struct Outer *outer; };"""
	return StructDecl(
		name="Top",
		members=[
			StructMember(name="outer", type_spec=_struct_ptr_type("Outer")),
		],
	)


def _struct_embed_inner() -> StructDecl:
	"""struct EmbedInner { int x; int y; };"""
	return StructDecl(
		name="EmbedInner",
		members=[
			StructMember(name="x", type_spec=_int_type()),
			StructMember(name="y", type_spec=_int_type()),
		],
	)


def _struct_embed_outer() -> StructDecl:
	"""struct EmbedOuter { struct EmbedInner sub; int z; };"""
	return StructDecl(
		name="EmbedOuter",
		members=[
			StructMember(name="sub", type_spec=_struct_type("EmbedInner")),
			StructMember(name="z", type_spec=_int_type()),
		],
	)


def _struct_point() -> StructDecl:
	"""struct Point { int x; int y; };"""
	return StructDecl(
		name="Point",
		members=[
			StructMember(name="x", type_spec=_int_type()),
			StructMember(name="y", type_spec=_int_type()),
		],
	)


# ------------------------------------------------------------------
# Chained arrow access: a->b->c
# ------------------------------------------------------------------


class TestChainedArrowAccess:
	def test_two_level_arrow(self) -> None:
		"""p->inner->val should resolve through two pointer dereferences."""
		prog = _make_program(
			_struct_inner(),
			_struct_outer(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_struct_ptr_type("Outer"))],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=MemberAccess(
						object=MemberAccess(
							object=Identifier(name="p"),
							member="inner",
							is_arrow=True,
						),
						member="val",
						is_arrow=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		# Should have at least 2 loads: one for p->inner, one for ->val
		assert len(loads) >= 2, f"Expected >= 2 loads for chained arrow, got {len(loads)}"

	def test_two_level_arrow_second_member(self) -> None:
		"""p->inner->extra should access the second member of Inner."""
		prog = _make_program(
			_struct_inner(),
			_struct_outer(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_struct_ptr_type("Outer"))],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=MemberAccess(
						object=MemberAccess(
							object=Identifier(name="p"),
							member="inner",
							is_arrow=True,
						),
						member="extra",
						is_arrow=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		assert len(loads) >= 2
		# The second-level access needs an offset computation for 'extra'
		add_ops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(add_ops) >= 1, "Should compute offset for second member"

	def test_three_level_arrow(self) -> None:
		"""t->outer->inner->val should chain three levels deep."""
		prog = _make_program(
			_struct_inner(),
			_struct_outer(),
			_struct_top(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="t", type_spec=_struct_ptr_type("Top"))],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=MemberAccess(
						object=MemberAccess(
							object=MemberAccess(
								object=Identifier(name="t"),
								member="outer",
								is_arrow=True,
							),
							member="inner",
							is_arrow=True,
						),
						member="val",
						is_arrow=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		# At least 3 loads: t->outer, ->inner, ->val
		assert len(loads) >= 3, f"Expected >= 3 loads for 3-level chain, got {len(loads)}"

	def test_chained_arrow_assignment(self) -> None:
		"""p->inner->val = 42 should store through chained access."""
		prog = _make_program(
			_struct_inner(),
			_struct_outer(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_struct_ptr_type("Outer"))],
				body=CompoundStmt(statements=[
					ExprStmt(expression=Assignment(
						target=MemberAccess(
							object=MemberAccess(
								object=Identifier(name="p"),
								member="inner",
								is_arrow=True,
							),
							member="val",
							is_arrow=True,
						),
						value=IntLiteral(value=42),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(stores) >= 1, "Should store value through chained arrow"


# ------------------------------------------------------------------
# Mixed dot/arrow chains: s.sub.x, p->sub.x, etc.
# ------------------------------------------------------------------


class TestMixedDotArrowChains:
	def test_dot_then_dot(self) -> None:
		"""s.sub.x -- embedded struct accessed via dot."""
		prog = _make_program(
			_struct_embed_inner(),
			_struct_embed_outer(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("EmbedOuter")),
					ReturnStmt(expression=MemberAccess(
						object=MemberAccess(
							object=Identifier(name="s"),
							member="sub",
							is_arrow=False,
						),
						member="x",
						is_arrow=False,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		assert len(loads) >= 1, "Should load the nested member value"

	def test_arrow_then_dot(self) -> None:
		"""p->sub.x -- pointer to struct with embedded struct, then dot access."""
		prog = _make_program(
			_struct_embed_inner(),
			_struct_embed_outer(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_struct_ptr_type("EmbedOuter"))],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=MemberAccess(
						object=MemberAccess(
							object=Identifier(name="p"),
							member="sub",
							is_arrow=True,
						),
						member="x",
						is_arrow=False,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		assert len(loads) >= 1

	def test_dot_then_dot_second_member(self) -> None:
		"""s.sub.y -- access second member of embedded struct."""
		prog = _make_program(
			_struct_embed_inner(),
			_struct_embed_outer(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("EmbedOuter")),
					ReturnStmt(expression=MemberAccess(
						object=MemberAccess(
							object=Identifier(name="s"),
							member="sub",
							is_arrow=False,
						),
						member="y",
						is_arrow=False,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		assert len(loads) >= 1
		# Needs offset computation for 'y' (second member)
		add_ops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(add_ops) >= 1

	def test_dot_dot_assignment(self) -> None:
		"""s.sub.x = 10 -- assign through nested dot access."""
		prog = _make_program(
			_struct_embed_inner(),
			_struct_embed_outer(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("EmbedOuter")),
					ExprStmt(expression=Assignment(
						target=MemberAccess(
							object=MemberAccess(
								object=Identifier(name="s"),
								member="sub",
								is_arrow=False,
							),
							member="x",
							is_arrow=False,
						),
						value=IntLiteral(value=10),
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(stores) >= 1


# ------------------------------------------------------------------
# Prefix ++/-- on MemberAccess targets
# ------------------------------------------------------------------


class TestPrefixIncDecMember:
	def test_prefix_increment_member(self) -> None:
		"""++s.x should load member, add 1, store back, return new value."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ReturnStmt(expression=UnaryOp(
						op="++",
						operand=MemberAccess(object=Identifier(name="s"), member="x"),
						prefix=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		add_ops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		assert len(loads) >= 1, "Should load current member value"
		assert len(stores) >= 1, "Should store incremented value back"
		assert len(add_ops) >= 1, "Should have an add operation"

	def test_prefix_decrement_member(self) -> None:
		"""--s.y should load member, subtract 1, store back, return new value."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ReturnStmt(expression=UnaryOp(
						op="--",
						operand=MemberAccess(object=Identifier(name="s"), member="y"),
						prefix=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		sub_ops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "-"]
		assert len(loads) >= 1, "Should load current member value"
		assert len(stores) >= 1, "Should store decremented value back"
		assert len(sub_ops) >= 1, "Should have a sub operation"

	def test_prefix_increment_stores_back(self) -> None:
		"""++s.x as expr stmt: ensure the store happens even when result is unused."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=UnaryOp(
						op="++",
						operand=MemberAccess(object=Identifier(name="s"), member="x"),
						prefix=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(stores) >= 1, "Prefix ++ must store the result back to member"

	def test_prefix_inc_arrow_member(self) -> None:
		"""++p->x should work on arrow-accessed member."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[ParamDecl(name="p", type_spec=_struct_ptr_type("Point"))],
				body=CompoundStmt(statements=[
					ReturnStmt(expression=UnaryOp(
						op="++",
						operand=MemberAccess(
							object=Identifier(name="p"),
							member="x",
							is_arrow=True,
						),
						prefix=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		loads = [i for i in instrs if isinstance(i, IRLoad)]
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(loads) >= 1
		assert len(stores) >= 1

	def test_prefix_dec_second_member(self) -> None:
		"""--s.y where y is the second member (non-zero offset)."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_int_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ReturnStmt(expression=UnaryOp(
						op="--",
						operand=MemberAccess(object=Identifier(name="s"), member="y"),
						prefix=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		# Accessing 'y' requires an offset computation
		add_ops = [i for i in instrs if isinstance(i, IRBinOp) and i.op == "+"]
		# The offset for y should produce an add with const 4
		assert any(
			isinstance(i.right, IRConst) and i.right.value == 4
			for i in add_ops
		), "Should compute offset 4 for second int member 'y'"

	def test_multiple_prefix_ops(self) -> None:
		"""++s.x; --s.y; both in same function."""
		prog = _make_program(
			_struct_point(),
			FunctionDecl(
				return_type=_void_type(),
				name="test",
				params=[],
				body=CompoundStmt(statements=[
					VarDecl(name="s", type_spec=_struct_type("Point")),
					ExprStmt(expression=UnaryOp(
						op="++",
						operand=MemberAccess(object=Identifier(name="s"), member="x"),
						prefix=True,
					)),
					ExprStmt(expression=UnaryOp(
						op="--",
						operand=MemberAccess(object=Identifier(name="s"), member="y"),
						prefix=True,
					)),
				]),
			),
		)
		instrs = _get_instrs(prog)
		stores = [i for i in instrs if isinstance(i, IRStore)]
		assert len(stores) >= 2, "Both prefix ops should store back"
