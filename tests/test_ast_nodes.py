"""Tests for AST node definitions."""

from compiler.ast_nodes import (
	ArraySubscript,
	ASTNode,
	ASTVisitor,
	Assignment,
	BinaryOp,
	CaseClause,
	CharLiteral,
	CompoundStmt,
	ExprStmt,
	ForStmt,
	FunctionCall,
	FunctionDecl,
	Identifier,
	IfStmt,
	IntLiteral,
	ParamDecl,
	PostfixExpr,
	Program,
	ReturnStmt,
	SizeofExpr,
	SourceLocation,
	StringLiteral,
	SwitchStmt,
	TernaryExpr,
	TypeSpec,
	UnaryOp,
	VarDecl,
	WhileStmt,
)


# --- Construction tests ---


class TestSourceLocation:
	def test_construction(self) -> None:
		loc = SourceLocation(line=10, col=5)
		assert loc.line == 10
		assert loc.col == 5


class TestTypeSpec:
	def test_basic_type(self) -> None:
		t = TypeSpec(base_type="int")
		assert t.base_type == "int"
		assert t.pointer_count == 0

	def test_pointer_type(self) -> None:
		t = TypeSpec(base_type="char", pointer_count=2)
		assert t.base_type == "char"
		assert t.pointer_count == 2


class TestLiterals:
	def test_int_literal(self) -> None:
		node = IntLiteral(value=42, loc=SourceLocation(1, 1))
		assert node.value == 42
		assert node.loc.line == 1

	def test_string_literal(self) -> None:
		node = StringLiteral(value="hello")
		assert node.value == "hello"

	def test_char_literal(self) -> None:
		node = CharLiteral(value="a")
		assert node.value == "a"


class TestIdentifier:
	def test_construction(self) -> None:
		node = Identifier(name="x", loc=SourceLocation(3, 7))
		assert node.name == "x"
		assert node.loc.col == 7


class TestBinaryOp:
	def test_construction(self) -> None:
		left = IntLiteral(value=1)
		right = IntLiteral(value=2)
		node = BinaryOp(left=left, op="+", right=right)
		assert node.op == "+"
		assert node.left == left
		assert node.right == right


class TestUnaryOp:
	def test_prefix(self) -> None:
		operand = Identifier(name="x")
		node = UnaryOp(op="-", operand=operand, prefix=True)
		assert node.op == "-"
		assert node.prefix is True

	def test_postfix(self) -> None:
		operand = Identifier(name="i")
		node = UnaryOp(op="++", operand=operand, prefix=False)
		assert node.prefix is False


class TestAssignment:
	def test_construction(self) -> None:
		target = Identifier(name="x")
		value = IntLiteral(value=10)
		node = Assignment(target=target, value=value)
		assert node.target == target
		assert node.value == value


class TestFunctionCall:
	def test_no_args(self) -> None:
		node = FunctionCall(name="foo")
		assert node.name == "foo"
		assert node.arguments == []

	def test_with_args(self) -> None:
		args = [IntLiteral(value=1), Identifier(name="x")]
		node = FunctionCall(name="bar", arguments=args)
		assert len(node.arguments) == 2


class TestStatements:
	def test_expr_stmt(self) -> None:
		expr = FunctionCall(name="printf", arguments=[StringLiteral(value="hi")])
		node = ExprStmt(expression=expr)
		assert isinstance(node.expression, FunctionCall)

	def test_return_stmt(self) -> None:
		node = ReturnStmt(expression=IntLiteral(value=0))
		assert isinstance(node.expression, IntLiteral)

	def test_compound_stmt(self) -> None:
		stmts = [
			ReturnStmt(expression=IntLiteral(value=0)),
			ExprStmt(expression=FunctionCall(name="f")),
		]
		node = CompoundStmt(statements=stmts)
		assert len(node.statements) == 2

	def test_if_stmt_no_else(self) -> None:
		node = IfStmt(
			condition=Identifier(name="x"),
			then_branch=ReturnStmt(expression=IntLiteral(value=1)),
		)
		assert node.else_branch is None

	def test_if_stmt_with_else(self) -> None:
		node = IfStmt(
			condition=Identifier(name="x"),
			then_branch=ReturnStmt(expression=IntLiteral(value=1)),
			else_branch=ReturnStmt(expression=IntLiteral(value=0)),
		)
		assert node.else_branch is not None

	def test_while_stmt(self) -> None:
		node = WhileStmt(
			condition=BinaryOp(left=Identifier(name="i"), op="<", right=IntLiteral(value=10)),
			body=CompoundStmt(statements=[]),
		)
		assert isinstance(node.condition, BinaryOp)

	def test_for_stmt(self) -> None:
		node = ForStmt(
			init=VarDecl(type_spec=TypeSpec(base_type="int"), name="i", initializer=IntLiteral(value=0)),
			condition=BinaryOp(left=Identifier(name="i"), op="<", right=IntLiteral(value=10)),
			update=UnaryOp(op="++", operand=Identifier(name="i"), prefix=False),
			body=CompoundStmt(statements=[]),
		)
		assert node.init is not None
		assert node.condition is not None
		assert node.update is not None

	def test_for_stmt_minimal(self) -> None:
		node = ForStmt(body=CompoundStmt(statements=[]))
		assert node.init is None
		assert node.condition is None
		assert node.update is None


class TestDeclarations:
	def test_param_decl(self) -> None:
		node = ParamDecl(type_spec=TypeSpec(base_type="int"), name="argc")
		assert node.name == "argc"
		assert node.type_spec.base_type == "int"

	def test_var_decl_no_init(self) -> None:
		node = VarDecl(type_spec=TypeSpec(base_type="int"), name="x")
		assert node.initializer is None

	def test_var_decl_with_init(self) -> None:
		node = VarDecl(
			type_spec=TypeSpec(base_type="int"),
			name="x",
			initializer=IntLiteral(value=5),
		)
		assert isinstance(node.initializer, IntLiteral)

	def test_function_decl(self) -> None:
		func = FunctionDecl(
			return_type=TypeSpec(base_type="int"),
			name="main",
			params=[],
			body=CompoundStmt(statements=[ReturnStmt(expression=IntLiteral(value=0))]),
		)
		assert func.name == "main"
		assert func.return_type.base_type == "int"
		assert len(func.body.statements) == 1

	def test_function_decl_with_params(self) -> None:
		func = FunctionDecl(
			return_type=TypeSpec(base_type="int"),
			name="add",
			params=[
				ParamDecl(type_spec=TypeSpec(base_type="int"), name="a"),
				ParamDecl(type_spec=TypeSpec(base_type="int"), name="b"),
			],
			body=CompoundStmt(statements=[]),
		)
		assert len(func.params) == 2


class TestArraySubscript:
	def test_construction(self) -> None:
		arr = Identifier(name="arr")
		idx = IntLiteral(value=0)
		node = ArraySubscript(array=arr, index=idx)
		assert isinstance(node.array, Identifier)
		assert isinstance(node.index, IntLiteral)

	def test_nested_subscript(self) -> None:
		"""arr[i][j] => ArraySubscript(ArraySubscript(arr, i), j)"""
		inner = ArraySubscript(
			array=Identifier(name="arr"),
			index=Identifier(name="i"),
		)
		outer = ArraySubscript(array=inner, index=Identifier(name="j"))
		assert isinstance(outer.array, ArraySubscript)

	def test_accept(self) -> None:
		visitor = ASTVisitor()
		node = ArraySubscript(
			array=Identifier(name="a"),
			index=IntLiteral(value=0),
		)
		assert node.accept(visitor) is None


class TestArrayDecl:
	def test_var_decl_with_array_sizes(self) -> None:
		node = VarDecl(
			type_spec=TypeSpec(base_type="int"),
			name="arr",
			array_sizes=[IntLiteral(value=10)],
		)
		assert node.array_sizes is not None
		assert len(node.array_sizes) == 1

	def test_var_decl_multi_dim(self) -> None:
		node = VarDecl(
			type_spec=TypeSpec(base_type="int"),
			name="matrix",
			array_sizes=[IntLiteral(value=3), IntLiteral(value=4)],
		)
		assert len(node.array_sizes) == 2

	def test_var_decl_no_array(self) -> None:
		node = VarDecl(type_spec=TypeSpec(base_type="int"), name="x")
		assert node.array_sizes is None


class TestProgram:
	def test_empty_program(self) -> None:
		prog = Program(declarations=[])
		assert prog.declarations == []

	def test_program_with_declarations(self) -> None:
		func = FunctionDecl(
			return_type=TypeSpec(base_type="int"),
			name="main",
			body=CompoundStmt(statements=[ReturnStmt(expression=IntLiteral(value=0))]),
		)
		var = VarDecl(type_spec=TypeSpec(base_type="int", pointer_count=1), name="global_ptr")
		prog = Program(declarations=[var, func])
		assert len(prog.declarations) == 2
		assert isinstance(prog.declarations[0], VarDecl)
		assert isinstance(prog.declarations[1], FunctionDecl)


# --- Equality tests (dataclass default __eq__) ---


class TestEquality:
	def test_int_literal_equal(self) -> None:
		a = IntLiteral(value=42)
		b = IntLiteral(value=42)
		assert a == b

	def test_int_literal_not_equal(self) -> None:
		a = IntLiteral(value=1)
		b = IntLiteral(value=2)
		assert a != b

	def test_type_spec_equal(self) -> None:
		a = TypeSpec(base_type="int", pointer_count=1)
		b = TypeSpec(base_type="int", pointer_count=1)
		assert a == b

	def test_type_spec_not_equal(self) -> None:
		a = TypeSpec(base_type="int", pointer_count=0)
		b = TypeSpec(base_type="int", pointer_count=1)
		assert a != b

	def test_binary_op_equal(self) -> None:
		a = BinaryOp(left=IntLiteral(value=1), op="+", right=IntLiteral(value=2))
		b = BinaryOp(left=IntLiteral(value=1), op="+", right=IntLiteral(value=2))
		assert a == b

	def test_nested_equality(self) -> None:
		stmt = ReturnStmt(expression=BinaryOp(
			left=Identifier(name="x"),
			op="+",
			right=IntLiteral(value=1),
		))
		stmt2 = ReturnStmt(expression=BinaryOp(
			left=Identifier(name="x"),
			op="+",
			right=IntLiteral(value=1),
		))
		assert stmt == stmt2

	def test_location_affects_equality(self) -> None:
		a = IntLiteral(value=42, loc=SourceLocation(1, 1))
		b = IntLiteral(value=42, loc=SourceLocation(2, 3))
		assert a != b


# --- Visitor pattern tests ---


class TestVisitorPattern:
	def test_base_accept_raises(self) -> None:
		base = ASTNode()
		visitor = ASTVisitor()
		try:
			base.accept(visitor)
			assert False, "Should have raised NotImplementedError"
		except NotImplementedError:
			pass

	def test_default_visitor_returns_none(self) -> None:
		visitor = ASTVisitor()
		assert visitor.visit(IntLiteral(value=1)) is None
		assert visitor.visit(Program()) is None
		assert visitor.visit(ReturnStmt()) is None

	def test_custom_visitor(self) -> None:
		class IntCollector(ASTVisitor):
			def __init__(self) -> None:
				self.values: list[int] = []

			def visit_int_literal(self, node: IntLiteral) -> None:
				self.values.append(node.value)

			def visit_binary_op(self, node: BinaryOp) -> None:
				self.visit(node.left)
				self.visit(node.right)

			def visit_return_stmt(self, node: ReturnStmt) -> None:
				self.visit(node.expression)

			def visit_compound_stmt(self, node: CompoundStmt) -> None:
				for stmt in node.statements:
					self.visit(stmt)

			def visit_program(self, node: Program) -> None:
				for decl in node.declarations:
					self.visit(decl)

			def visit_function_decl(self, node: FunctionDecl) -> None:
				self.visit(node.body)

		# Build: int main() { return 1 + 2; }
		tree = Program(declarations=[
			FunctionDecl(
				return_type=TypeSpec(base_type="int"),
				name="main",
				body=CompoundStmt(statements=[
					ReturnStmt(expression=BinaryOp(
						left=IntLiteral(value=1),
						op="+",
						right=IntLiteral(value=2),
					)),
				]),
			),
		])

		collector = IntCollector()
		collector.visit(tree)
		assert collector.values == [1, 2]

	def test_visitor_via_visit_method(self) -> None:
		class NodeCounter(ASTVisitor):
			def __init__(self) -> None:
				self.count = 0

			def visit_identifier(self, node: Identifier) -> int:
				self.count += 1
				return self.count

		counter = NodeCounter()
		result = counter.visit(Identifier(name="x"))
		assert result == 1
		assert counter.count == 1

	def test_all_node_types_accept(self) -> None:
		"""Every concrete node type should dispatch to the visitor without error."""
		visitor = ASTVisitor()
		nodes: list[ASTNode] = [
			Program(),
			FunctionDecl(),
			ParamDecl(),
			VarDecl(),
			CompoundStmt(),
			ReturnStmt(),
			IfStmt(),
			WhileStmt(),
			ForStmt(),
			ExprStmt(),
			BinaryOp(),
			UnaryOp(),
			IntLiteral(),
			StringLiteral(),
			CharLiteral(),
			Identifier(),
			Assignment(),
			FunctionCall(),
			ArraySubscript(),
			TypeSpec(),
			SwitchStmt(),
			CaseClause(),
			TernaryExpr(),
			SizeofExpr(),
			PostfixExpr(),
		]
		for node in nodes:
			assert node.accept(visitor) is None


class TestSwitchStmt:
	def test_construction(self) -> None:
		case1 = CaseClause(value=IntLiteral(value=1), statements=[ReturnStmt(expression=IntLiteral(value=10))])
		default = CaseClause(value=None, statements=[ReturnStmt(expression=IntLiteral(value=0))])
		node = SwitchStmt(expression=Identifier(name="x"), cases=[case1, default])
		assert len(node.cases) == 2
		assert node.cases[0].value is not None
		assert node.cases[1].value is None

	def test_case_clause(self) -> None:
		case = CaseClause(value=IntLiteral(value=42), statements=[])
		assert isinstance(case.value, IntLiteral)
		assert case.statements == []


class TestTernaryExpr:
	def test_construction(self) -> None:
		node = TernaryExpr(
			condition=Identifier(name="x"),
			true_expr=IntLiteral(value=1),
			false_expr=IntLiteral(value=0),
		)
		assert isinstance(node.condition, Identifier)
		assert isinstance(node.true_expr, IntLiteral)
		assert isinstance(node.false_expr, IntLiteral)


class TestSizeofExpr:
	def test_sizeof_type(self) -> None:
		node = SizeofExpr(type_operand=TypeSpec(base_type="int"))
		assert node.type_operand is not None
		assert node.operand is None

	def test_sizeof_expr(self) -> None:
		node = SizeofExpr(operand=Identifier(name="x"))
		assert node.operand is not None
		assert node.type_operand is None


class TestPostfixExpr:
	def test_increment(self) -> None:
		node = PostfixExpr(operand=Identifier(name="i"), op="++")
		assert node.op == "++"
		assert isinstance(node.operand, Identifier)

	def test_decrement(self) -> None:
		node = PostfixExpr(operand=Identifier(name="i"), op="--")
		assert node.op == "--"
