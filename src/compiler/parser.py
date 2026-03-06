"""Recursive descent parser for a minimal C subset.

Consumes tokens from the lexer and builds an AST. Uses precedence climbing
for expression parsing.
"""

from __future__ import annotations

from compiler.ast_nodes import (
	ArraySubscript,
	Assignment,
	ASTNode,
	BinaryOp,
	BreakStmt,
	CaseClause,
	CastExpr,
	CharLiteral,
	CommaExpr,
	CompoundLiteral,
	CompoundStmt,
	ContinueStmt,
	DesignatedInit,
	DoWhileStmt,
	EnumConstant,
	EnumDecl,
	ExprStmt,
	FloatLiteral,
	ForStmt,
	FunctionCall,
	FunctionDecl,
	GotoStmt,
	Identifier,
	IfStmt,
	InitializerList,
	IntLiteral,
	LabelStmt,
	MemberAccess,
	ParamDecl,
	PostfixExpr,
	Program,
	ReturnStmt,
	SizeofExpr,
	SourceLocation,
	StaticAssertDecl,
	StringLiteral,
	StructDecl,
	StructMember,
	SwitchStmt,
	TernaryExpr,
	TypedefDecl,
	TypeSpec,
	UnaryOp,
	UnionDecl,
	VaCopyExpr,
	VaArgExpr,
	VaEndExpr,
	VaStartExpr,
	VarDecl,
	WhileStmt,
)
from compiler.lexer import Lexer, interpret_c_escapes
from compiler.tokens import Token, TokenType


class ParseError(Exception):
	"""Error raised when the parser encounters invalid syntax."""

	def __init__(self, message: str, line: int, column: int) -> None:
		self.line = line
		self.column = column
		super().__init__(f"{message} at {line}:{column}")


# Operator precedence table for binary operators (higher = tighter binding)
_PRECEDENCE: dict[TokenType, int] = {
	TokenType.OR: 1,
	TokenType.AND: 2,
	TokenType.PIPE: 3,
	TokenType.CARET: 4,
	TokenType.AMPERSAND: 5,
	TokenType.EQUAL: 6,
	TokenType.NOT_EQUAL: 6,
	TokenType.LESS: 7,
	TokenType.GREATER: 7,
	TokenType.LESS_EQUAL: 7,
	TokenType.GREATER_EQUAL: 7,
	TokenType.LSHIFT: 8,
	TokenType.RSHIFT: 8,
	TokenType.PLUS: 9,
	TokenType.MINUS: 9,
	TokenType.STAR: 10,
	TokenType.SLASH: 10,
	TokenType.PERCENT: 10,
}

_BINOP_STR: dict[TokenType, str] = {
	TokenType.PLUS: "+",
	TokenType.MINUS: "-",
	TokenType.STAR: "*",
	TokenType.SLASH: "/",
	TokenType.PERCENT: "%",
	TokenType.AMPERSAND: "&",
	TokenType.PIPE: "|",
	TokenType.CARET: "^",
	TokenType.LSHIFT: "<<",
	TokenType.RSHIFT: ">>",
	TokenType.EQUAL: "==",
	TokenType.NOT_EQUAL: "!=",
	TokenType.LESS: "<",
	TokenType.GREATER: ">",
	TokenType.LESS_EQUAL: "<=",
	TokenType.GREATER_EQUAL: ">=",
	TokenType.AND: "&&",
	TokenType.OR: "||",
}

_COMPOUND_ASSIGN: dict[TokenType, str] = {
	TokenType.PLUS_ASSIGN: "+",
	TokenType.MINUS_ASSIGN: "-",
	TokenType.STAR_ASSIGN: "*",
	TokenType.SLASH_ASSIGN: "/",
	TokenType.PERCENT_ASSIGN: "%",
	TokenType.AMP_ASSIGN: "&",
	TokenType.PIPE_ASSIGN: "|",
	TokenType.CARET_ASSIGN: "^",
	TokenType.LSHIFT_ASSIGN: "<<",
	TokenType.RSHIFT_ASSIGN: ">>",
}

_TYPE_KEYWORDS: set[TokenType] = {
	TokenType.INT, TokenType.CHAR, TokenType.VOID, TokenType.STRUCT, TokenType.ENUM,
	TokenType.FLOAT, TokenType.DOUBLE, TokenType.UNION, TokenType.BOOL,
	TokenType.LONG, TokenType.SHORT, TokenType.SIGNED, TokenType.UNSIGNED,
}

_QUALIFIER_KEYWORDS: set[TokenType] = {TokenType.CONST, TokenType.VOLATILE}

_STORAGE_CLASS_KEYWORDS: set[TokenType] = {
	TokenType.STATIC, TokenType.EXTERN, TokenType.AUTO, TokenType.REGISTER,
}

_FUNCTION_SPECIFIER_KEYWORDS: set[TokenType] = {TokenType.INLINE}

_TYPE_SPECIFIER_START: set[TokenType] = _TYPE_KEYWORDS | _QUALIFIER_KEYWORDS | _STORAGE_CLASS_KEYWORDS | _FUNCTION_SPECIFIER_KEYWORDS


class Parser:
	"""Recursive descent parser that builds AST nodes from a token stream."""

	_PREDEFINED_TYPEDEF_NAMES: set[str] = {
		"int8_t", "uint8_t", "int16_t", "uint16_t",
		"int32_t", "uint32_t", "int64_t", "uint64_t",
		"intptr_t", "uintptr_t", "ptrdiff_t",
		"size_t", "ssize_t", "nullptr_t",
	}

	def __init__(self, tokens: list[Token]) -> None:
		self.tokens = tokens
		self.pos = 0
		self._typedef_names: set[str] = set(self._PREDEFINED_TYPEDEF_NAMES)
		self._last_storage_class: str | None = None
		self._anon_counter: int = 0
		self._pending_decls: list[ASTNode] = []

	@classmethod
	def from_source(cls, source: str) -> Parser:
		"""Create a parser directly from C source code."""
		tokens = Lexer(source).tokenize()
		return cls(tokens)

	# -- Token helpers -------------------------------------------------------

	def _current(self) -> Token:
		return self.tokens[self.pos]

	def _peek(self, offset: int = 0) -> Token:
		idx = self.pos + offset
		if idx >= len(self.tokens):
			return self.tokens[-1]  # EOF
		return self.tokens[idx]

	def _at_end(self) -> bool:
		return self._current().type == TokenType.EOF

	def _advance(self) -> Token:
		tok = self._current()
		if not self._at_end():
			self.pos += 1
		return tok

	def _check(self, *types: TokenType) -> bool:
		return self._current().type in types

	def _match(self, *types: TokenType) -> Token | None:
		if self._current().type in types:
			return self._advance()
		return None

	def _expect(self, token_type: TokenType, message: str | None = None) -> Token:
		tok = self._current()
		if tok.type != token_type:
			msg = message or f"Expected {token_type.name}, got {tok.type.name} ({tok.value!r})"
			raise ParseError(msg, tok.line, tok.column)
		return self._advance()

	def _loc(self, token: Token) -> SourceLocation:
		return SourceLocation(line=token.line, col=token.column)

	def _error(self, message: str) -> ParseError:
		tok = self._current()
		return ParseError(message, tok.line, tok.column)

	def _is_type_start(self, offset: int = 0) -> bool:
		"""Check if the token at the given offset starts a type specifier."""
		tok = self._peek(offset)
		if tok.type in _TYPE_SPECIFIER_START:
			return True
		return tok.type == TokenType.IDENTIFIER and tok.value in self._typedef_names

	# -- Top-level parsing ---------------------------------------------------

	def parse(self) -> Program:
		"""Parse the full token stream into a Program AST node."""
		declarations: list[ASTNode] = []
		while not self._at_end():
			self._pending_decls.clear()
			result = self._parse_top_level_decl()
			# Emit any inline struct/union declarations before the referencing decl
			declarations.extend(self._pending_decls)
			self._pending_decls.clear()
			if isinstance(result, list):
				declarations.extend(result)
			else:
				declarations.append(result)
		return Program(declarations=declarations, loc=SourceLocation(1, 1))

	def _skip_c23_attributes(self) -> None:
		"""Skip C23 [[...]] attributes."""
		while self._check(TokenType.LBRACKET) and self._peek(1).type == TokenType.LBRACKET:
			self._advance()  # first [
			self._advance()  # second [
			depth = 1
			while depth > 0 and not self._at_end():
				if self._check(TokenType.LBRACKET):
					depth += 1
				elif self._check(TokenType.RBRACKET):
					depth -= 1
				self._advance()
			# consume closing ]
			if self._check(TokenType.RBRACKET):
				self._advance()

	def _parse_top_level_decl(self) -> ASTNode:
		"""Parse a top-level declaration (function, global variable, struct, enum, or typedef)."""
		# Skip C23 [[...]] attributes
		self._skip_c23_attributes()

		# Check for _Static_assert
		if self._check(TokenType.STATIC_ASSERT):
			return self._parse_static_assert()

		# Check for typedef
		if self._check(TokenType.TYPEDEF):
			return self._parse_typedef_decl()

		# Check for standalone struct/enum/union definitions (skip past storage class/qualifiers)
		peek_offset = 0
		while self._peek(peek_offset).type in (_STORAGE_CLASS_KEYWORDS | _QUALIFIER_KEYWORDS | _FUNCTION_SPECIFIER_KEYWORDS):
			peek_offset += 1

		# Check for enum definition: enum name { ... };
		if self._peek(peek_offset).type == TokenType.ENUM and self._peek(peek_offset + 1).type == TokenType.IDENTIFIER and self._peek(peek_offset + 2).type == TokenType.LBRACE:
			if peek_offset == 0:
				return self._parse_enum_decl()

		# Check for struct definition or forward declaration: struct [name] { ... }; or struct name;
		if self._peek(peek_offset).type == TokenType.STRUCT:
			next_tok = self._peek(peek_offset + 1)
			if next_tok.type == TokenType.LBRACE and peek_offset == 0:
				# Anonymous struct: struct { ... } var;
				return self._parse_struct_decl()
			if next_tok.type == TokenType.IDENTIFIER:
				next_after_name = self._peek(peek_offset + 2).type
				if peek_offset == 0 and next_after_name in (TokenType.LBRACE, TokenType.SEMICOLON):
					return self._parse_struct_decl()

		# Check for union definition or forward declaration: union [name] { ... }; or union name;
		if self._peek(peek_offset).type == TokenType.UNION:
			next_tok = self._peek(peek_offset + 1)
			if next_tok.type == TokenType.LBRACE and peek_offset == 0:
				# Anonymous union: union { ... } var;
				return self._parse_union_decl()
			if next_tok.type == TokenType.IDENTIFIER:
				next_after_name = self._peek(peek_offset + 2).type
				if peek_offset == 0 and next_after_name in (TokenType.LBRACE, TokenType.SEMICOLON):
					return self._parse_union_decl()

		self._last_storage_class = None
		type_spec = self._parse_type_spec()
		storage_class = self._last_storage_class

		# Check for function pointer declaration: type (*name)(params)
		if self._is_func_ptr_start():
			fp_type, name_tok = self._parse_func_ptr_type(type_spec)
			initializer = None
			if self._match(TokenType.ASSIGN):
				initializer = self._parse_assignment()
			self._expect(TokenType.SEMICOLON, "Expected ';' after function pointer declaration")
			return VarDecl(
				type_spec=fp_type,
				name=name_tok.value,
				initializer=initializer,
				storage_class=storage_class,
				loc=self._loc(name_tok),
			)

		name_tok = self._expect(TokenType.IDENTIFIER, "Expected declaration name")

		if self._check(TokenType.LPAREN):
			decl = self._parse_function_decl(type_spec, name_tok)
			decl.storage_class = storage_class
			return decl

		# Global variable declaration (possibly array, possibly multi-decl)
		array_sizes = self._parse_array_dimensions()
		initializer = self._parse_optional_initializer()
		first_decl = VarDecl(
			type_spec=type_spec,
			name=name_tok.value,
			initializer=initializer,
			array_sizes=array_sizes,
			storage_class=storage_class,
			loc=self._loc(name_tok),
		)
		if self._match(TokenType.COMMA):
			decls: list[ASTNode] = [first_decl]
			while True:
				decls.append(self._parse_additional_declarator(type_spec, storage_class))
				if not self._match(TokenType.COMMA):
					break
			self._expect(TokenType.SEMICOLON, "Expected ';' after variable declaration")
			return decls
		self._expect(TokenType.SEMICOLON, "Expected ';' after variable declaration")
		return first_decl

	# -- Type specifier ------------------------------------------------------

	def _parse_type_spec(self, allow_storage_class: bool = False) -> TypeSpec:
		"""Parse a type specifier with optional qualifiers, signedness, width modifiers, and storage class.

		Returns a TypeSpec. Storage class is stored on self._last_storage_class
		for the caller to attach to the declaration node.
		"""
		start_tok = self._current()
		qualifiers: list[str] = []
		storage_class: str | None = None
		signedness: str | None = None
		width_modifier: str | None = None
		base_type: str | None = None

		# Collect qualifiers, storage classes, signedness, and width modifiers
		while True:
			tok = self._current()
			if tok.type in _QUALIFIER_KEYWORDS:
				self._advance()
				qualifiers.append(tok.value)
			elif tok.type in _STORAGE_CLASS_KEYWORDS:
				self._advance()
				if storage_class is not None:
					raise self._error(f"Multiple storage classes: '{storage_class}' and '{tok.value}'")
				storage_class = tok.value
			elif tok.type in _FUNCTION_SPECIFIER_KEYWORDS:
				self._advance()  # consume inline as a no-op function specifier
			elif tok.type == TokenType.SIGNED:
				self._advance()
				signedness = "signed"
			elif tok.type == TokenType.UNSIGNED:
				self._advance()
				signedness = "unsigned"
			elif tok.type == TokenType.LONG:
				self._advance()
				if width_modifier == "long":
					width_modifier = "long long"
				else:
					width_modifier = "long"
			elif tok.type == TokenType.SHORT:
				self._advance()
				width_modifier = "short"
			else:
				break

		# Now parse the base type
		tok = self._current()
		if tok.type == TokenType.IDENTIFIER and tok.value in self._typedef_names and signedness is None and width_modifier is None:
			self._advance()
			base_type = tok.value
		elif tok.type in _QUALIFIER_KEYWORDS:
			# Post-base qualifiers (e.g. "int const" - unusual but valid C)
			pass
		elif tok.type in {TokenType.INT, TokenType.CHAR, TokenType.VOID, TokenType.FLOAT, TokenType.DOUBLE, TokenType.BOOL}:
			self._advance()
			base_type = tok.value
		elif tok.type == TokenType.STRUCT:
			self._advance()
			if self._check(TokenType.LBRACE):
				# Anonymous struct: struct { ... }
				anon_name = self._parse_anon_struct_body("struct")
				base_type = f"struct {anon_name}"
			elif self._check(TokenType.IDENTIFIER) and self._peek(1).type == TokenType.LBRACE:
				# Named struct definition used as type: struct Name { ... }
				sname = self._advance().value
				self._parse_struct_or_union_body(sname, "struct")
				base_type = f"struct {sname}"
			else:
				name_tok = self._expect(TokenType.IDENTIFIER, "Expected struct name")
				base_type = f"struct {name_tok.value}"
		elif tok.type == TokenType.UNION:
			self._advance()
			if self._check(TokenType.LBRACE):
				# Anonymous union: union { ... }
				anon_name = self._parse_anon_struct_body("union")
				base_type = f"union {anon_name}"
			elif self._check(TokenType.IDENTIFIER) and self._peek(1).type == TokenType.LBRACE:
				# Named union definition used as type: union Name { ... }
				uname = self._advance().value
				self._parse_struct_or_union_body(uname, "union")
				base_type = f"union {uname}"
			else:
				name_tok = self._expect(TokenType.IDENTIFIER, "Expected union name")
				base_type = f"union {name_tok.value}"
		elif tok.type == TokenType.ENUM:
			self._advance()
			name_tok = self._expect(TokenType.IDENTIFIER, "Expected enum name")
			base_type = f"enum {name_tok.value}"

		# Collect any trailing qualifiers (e.g. "int const *")
		while self._check(*_QUALIFIER_KEYWORDS):
			qualifiers.append(self._advance().value)

		# If no explicit base type but we have signedness/width, default to int
		if base_type is None:
			if signedness is not None or width_modifier is not None:
				base_type = "int"
			else:
				raise self._error(f"Expected type specifier, got {self._current().type.name} ({self._current().value!r})")

		pointer_count = 0
		while self._match(TokenType.STAR):
			pointer_count += 1
			# Consume pointer qualifiers (e.g., int * const p)
			while self._check(*_QUALIFIER_KEYWORDS):
				qualifiers.append(self._advance().value)

		self._last_storage_class = storage_class
		return TypeSpec(
			base_type=base_type,
			pointer_count=pointer_count,
			qualifiers=qualifiers,
			signedness=signedness,
			width_modifier=width_modifier,
			loc=self._loc(start_tok),
		)

	# -- Anonymous struct/union helpers --------------------------------------

	def _gen_anon_name(self, kind: str) -> str:
		"""Generate a unique anonymous struct/union name."""
		self._anon_counter += 1
		return f"__anon_{kind}_{self._anon_counter}"

	def _parse_anon_struct_body(self, kind: str) -> str:
		"""Parse an anonymous struct/union body '{...}' and return the generated name."""
		anon_name = self._gen_anon_name(kind)
		self._parse_struct_or_union_body(anon_name, kind)
		return anon_name

	def _parse_struct_or_union_body(self, name: str, kind: str) -> ASTNode:
		"""Parse a struct/union body '{...}' (consumes '{' through '}').
		Returns the StructDecl or UnionDecl and adds it to pending declarations."""
		start_tok = self._current()
		self._expect(TokenType.LBRACE, f"Expected '{{' after {kind} name")
		members: list[StructMember] = []
		sa_list: list[StaticAssertDecl] = []
		while not self._check(TokenType.RBRACE) and not self._at_end():
			if self._check(TokenType.STATIC_ASSERT):
				sa_list.append(self._parse_static_assert())
				continue
			members.append(self._parse_struct_member())
		self._expect(TokenType.RBRACE, f"Expected '}}' after {kind} members")
		if kind == "struct":
			decl = StructDecl(name=name, members=members, static_asserts=sa_list, loc=self._loc(start_tok))
		else:
			decl = UnionDecl(name=name, members=members, static_asserts=sa_list, loc=self._loc(start_tok))
		self._pending_decls.append(decl)
		return decl

	# -- Struct declaration --------------------------------------------------

	def _parse_struct_member(self) -> StructMember:
		"""Parse a single struct/union member: type [name] [dims] [: bitwidth] ;"""
		member_type = self._parse_type_spec()
		bit_width: int | None = None
		member_name = ""
		loc_tok = self._peek()
		if self._check(TokenType.COLON):
			# Unnamed bitfield: type : width ;
			self._advance()
			bit_width = self._parse_bitfield_width()
			self._expect(TokenType.SEMICOLON, "Expected ';' after struct member")
			return StructMember(
				type_spec=member_type,
				name="",
				bit_width=bit_width,
				loc=self._loc(loc_tok),
			)
		member_name_tok = self._expect(TokenType.IDENTIFIER, "Expected member name")
		member_name = member_name_tok.value
		loc_tok = member_name_tok
		dims: list[ASTNode] = []
		if self._check(TokenType.COLON):
			# Named bitfield: type name : width ;
			self._advance()
			bit_width = self._parse_bitfield_width()
		else:
			while self._match(TokenType.LBRACKET):
				dims.append(self._parse_expression())
				self._expect(TokenType.RBRACKET, "Expected ']' after array dimension")
		self._expect(TokenType.SEMICOLON, "Expected ';' after struct member")
		return StructMember(
			type_spec=member_type,
			name=member_name,
			array_dims=dims,
			bit_width=bit_width,
			loc=self._loc(loc_tok),
		)

	def _parse_bitfield_width(self) -> int:
		"""Parse a bitfield width expression, evaluating it to an integer constant."""
		from compiler.const_eval import ConstExprEvaluator
		if self._check(TokenType.INTEGER_LITERAL):
			tok = self._advance()
			return int(tok.value)
		expr = self._parse_ternary()
		evaluator = ConstExprEvaluator()
		result = evaluator.evaluate(expr)
		if result is not None:
			return int(result)
		return 0

	def _parse_struct_decl(self) -> ASTNode | list[ASTNode]:
		"""Parse 'struct [name] { ... };' or 'struct [name] { ... } var, *ptr;' or forward decl."""
		tok = self._advance()  # consume 'struct'
		if self._check(TokenType.LBRACE):
			# Anonymous struct: struct { ... } var;
			struct_name = self._gen_anon_name("struct")
		else:
			name_tok = self._expect(TokenType.IDENTIFIER, "Expected struct name")
			struct_name = name_tok.value
			if self._match(TokenType.SEMICOLON):
				return StructDecl(name=struct_name, members=[], loc=self._loc(tok))
		self._expect(TokenType.LBRACE, "Expected '{' after struct name")
		members: list[StructMember] = []
		sa_list: list[StaticAssertDecl] = []
		while not self._check(TokenType.RBRACE) and not self._at_end():
			if self._check(TokenType.STATIC_ASSERT):
				sa_list.append(self._parse_static_assert())
				continue
			members.append(self._parse_struct_member())
		self._expect(TokenType.RBRACE, "Expected '}' after struct members")
		struct_node = StructDecl(name=struct_name, members=members, static_asserts=sa_list, loc=self._loc(tok))
		if self._match(TokenType.SEMICOLON):
			return struct_node
		# Trailing declarators: struct S { ... } var, *ptr = init;
		return self._parse_trailing_declarators(struct_node, f"struct {struct_name}", tok)

	# -- Union declaration ---------------------------------------------------

	def _parse_union_decl(self) -> ASTNode | list[ASTNode]:
		"""Parse 'union [name] { ... };' or 'union [name] { ... } var, *ptr;' or forward decl."""
		tok = self._advance()  # consume 'union'
		if self._check(TokenType.LBRACE):
			# Anonymous union: union { ... } var;
			union_name = self._gen_anon_name("union")
		else:
			name_tok = self._expect(TokenType.IDENTIFIER, "Expected union name")
			union_name = name_tok.value
			if self._match(TokenType.SEMICOLON):
				return UnionDecl(name=union_name, members=[], loc=self._loc(tok))
		self._expect(TokenType.LBRACE, "Expected '{' after union name")
		members: list[StructMember] = []
		sa_list: list[StaticAssertDecl] = []
		while not self._check(TokenType.RBRACE) and not self._at_end():
			if self._check(TokenType.STATIC_ASSERT):
				sa_list.append(self._parse_static_assert())
				continue
			members.append(self._parse_struct_member())
		self._expect(TokenType.RBRACE, "Expected '}' after union members")
		union_node = UnionDecl(name=union_name, members=members, static_asserts=sa_list, loc=self._loc(tok))
		if self._match(TokenType.SEMICOLON):
			return union_node
		# Trailing declarators: union U { ... } var, *ptr = init;
		return self._parse_trailing_declarators(union_node, f"union {union_name}", tok)

	def _parse_trailing_declarators(self, type_decl: ASTNode, base_type: str, start_tok: Token) -> list[ASTNode]:
		"""Parse variable declarators after a struct/union definition body."""
		result: list[ASTNode] = [type_decl]
		type_spec = TypeSpec(base_type=base_type, pointer_count=0, qualifiers=[], loc=self._loc(start_tok))
		while True:
			pointer_count = 0
			while self._match(TokenType.STAR):
				pointer_count += 1
			decl_type = TypeSpec(
				base_type=base_type, pointer_count=pointer_count, qualifiers=[],
				signedness=type_spec.signedness, width_modifier=type_spec.width_modifier,
				loc=self._loc(start_tok),
			)
			name_tok = self._expect(TokenType.IDENTIFIER, "Expected variable name after struct/union definition")
			array_sizes = self._parse_array_dimensions()
			initializer = self._parse_optional_initializer()
			result.append(VarDecl(
				type_spec=decl_type, name=name_tok.value, initializer=initializer,
				array_sizes=array_sizes, loc=self._loc(name_tok),
			))
			if not self._match(TokenType.COMMA):
				break
		self._expect(TokenType.SEMICOLON, "Expected ';' after variable declaration")
		return result

	# -- Enum declaration ----------------------------------------------------

	def _parse_enum_decl(self) -> EnumDecl:
		"""Parse 'enum name { A, B = 5, C };'."""
		tok = self._advance()  # consume 'enum'
		name_tok = self._expect(TokenType.IDENTIFIER, "Expected enum name")
		self._expect(TokenType.LBRACE, "Expected '{' after enum name")
		constants: list[EnumConstant] = []
		while not self._check(TokenType.RBRACE) and not self._at_end():
			const_name_tok = self._expect(TokenType.IDENTIFIER, "Expected enumerator name")
			value: ASTNode | None = None
			if self._match(TokenType.ASSIGN):
				value = self._parse_assignment()
			constants.append(EnumConstant(
				name=const_name_tok.value,
				value=value,
				loc=self._loc(const_name_tok),
			))
			if not self._check(TokenType.RBRACE):
				self._expect(TokenType.COMMA, "Expected ',' or '}' in enum body")
		self._expect(TokenType.RBRACE, "Expected '}' after enum constants")
		self._expect(TokenType.SEMICOLON, "Expected ';' after enum definition")
		return EnumDecl(name=name_tok.value, constants=constants, loc=self._loc(tok))

	# -- Static assert -------------------------------------------------------

	def _parse_static_assert(self) -> StaticAssertDecl:
		"""Parse '_Static_assert(constant-expression, string-literal);'."""
		tok = self._advance()  # consume '_Static_assert'
		self._expect(TokenType.LPAREN, "Expected '(' after '_Static_assert'")
		expr = self._parse_assignment()
		self._expect(TokenType.COMMA, "Expected ',' in _Static_assert")
		msg_tok = self._expect(TokenType.STRING_LITERAL, "Expected string literal in _Static_assert")
		msg = msg_tok.value[1:-1]  # strip quotes
		self._expect(TokenType.RPAREN, "Expected ')' after _Static_assert")
		self._expect(TokenType.SEMICOLON, "Expected ';' after _Static_assert")
		return StaticAssertDecl(expression=expr, message=msg, loc=self._loc(tok))

	# -- Typedef declaration -------------------------------------------------

	def _parse_typedef_decl(self) -> TypedefDecl:
		"""Parse 'typedef <type> <name>;' including struct/enum inline definitions."""
		tok = self._advance()  # consume 'typedef'
		struct_decl: StructDecl | None = None
		enum_decl: EnumDecl | None = None

		if self._check(TokenType.STRUCT):
			# Check for inline struct definition: typedef struct [name] { ... } alias;
			if self._peek(1).type == TokenType.LBRACE or (
				self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE
			):
				self._advance()  # consume 'struct'
				struct_name = ""
				if self._check(TokenType.IDENTIFIER):
					struct_name = self._advance().value
				self._expect(TokenType.LBRACE, "Expected '{' in struct definition")
				members: list[StructMember] = []
				while not self._check(TokenType.RBRACE) and not self._at_end():
					members.append(self._parse_struct_member())
				self._expect(TokenType.RBRACE, "Expected '}' after struct members")
				alias_tok = self._expect(TokenType.IDENTIFIER, "Expected typedef name")
				# Use alias as struct name if anonymous
				if not struct_name:
					struct_name = alias_tok.value
				struct_decl = StructDecl(name=struct_name, members=members, loc=self._loc(tok))
				type_spec = TypeSpec(base_type=f"struct {struct_name}", pointer_count=0, loc=self._loc(tok))
				self._expect(TokenType.SEMICOLON, "Expected ';' after typedef declaration")
				self._typedef_names.add(alias_tok.value)
				return TypedefDecl(
					type_spec=type_spec,
					name=alias_tok.value,
					struct_decl=struct_decl,
					loc=self._loc(tok),
				)
			# Otherwise: typedef struct Name alias; (reference to existing struct)
			type_spec = self._parse_type_spec()
		elif self._check(TokenType.ENUM):
			# Check for inline enum definition: typedef enum [name] { ... } alias;
			if self._peek(1).type == TokenType.LBRACE or (
				self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE
			):
				self._advance()  # consume 'enum'
				enum_name = ""
				if self._check(TokenType.IDENTIFIER):
					enum_name = self._advance().value
				self._expect(TokenType.LBRACE, "Expected '{' in enum definition")
				constants: list[EnumConstant] = []
				while not self._check(TokenType.RBRACE) and not self._at_end():
					const_name_tok = self._expect(TokenType.IDENTIFIER, "Expected enumerator name")
					value: ASTNode | None = None
					if self._match(TokenType.ASSIGN):
						value = self._parse_expression()
					constants.append(EnumConstant(
						name=const_name_tok.value,
						value=value,
						loc=self._loc(const_name_tok),
					))
					if not self._check(TokenType.RBRACE):
						self._expect(TokenType.COMMA, "Expected ',' or '}' in enum body")
				self._expect(TokenType.RBRACE, "Expected '}' after enum constants")
				alias_tok = self._expect(TokenType.IDENTIFIER, "Expected typedef name")
				if not enum_name:
					enum_name = alias_tok.value
				enum_decl = EnumDecl(name=enum_name, constants=constants, loc=self._loc(tok))
				type_spec = TypeSpec(base_type=f"enum {enum_name}", pointer_count=0, loc=self._loc(tok))
				self._expect(TokenType.SEMICOLON, "Expected ';' after typedef declaration")
				self._typedef_names.add(alias_tok.value)
				return TypedefDecl(
					type_spec=type_spec,
					name=alias_tok.value,
					enum_decl=enum_decl,
					loc=self._loc(tok),
				)
			type_spec = self._parse_type_spec()
		elif self._check(TokenType.UNION):
			# Check for inline union definition: typedef union [name] { ... } alias;
			if self._peek(1).type == TokenType.LBRACE or (
				self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE
			):
				self._advance()  # consume 'union'
				union_name = ""
				if self._check(TokenType.IDENTIFIER):
					union_name = self._advance().value
				self._expect(TokenType.LBRACE, "Expected '{' in union definition")
				u_members: list[StructMember] = []
				while not self._check(TokenType.RBRACE) and not self._at_end():
					u_members.append(self._parse_struct_member())
				self._expect(TokenType.RBRACE, "Expected '}' after union members")
				alias_tok = self._expect(TokenType.IDENTIFIER, "Expected typedef name")
				if not union_name:
					union_name = alias_tok.value
				union_decl = UnionDecl(name=union_name, members=u_members, loc=self._loc(tok))
				type_spec = TypeSpec(base_type=f"union {union_name}", pointer_count=0, loc=self._loc(tok))
				self._expect(TokenType.SEMICOLON, "Expected ';' after typedef declaration")
				self._typedef_names.add(alias_tok.value)
				return TypedefDecl(
					type_spec=type_spec,
					name=alias_tok.value,
					union_decl=union_decl,
					loc=self._loc(tok),
				)
			type_spec = self._parse_type_spec()
		else:
			type_spec = self._parse_type_spec()

		# Check for function pointer typedef: typedef type (*alias)(params);
		if self._is_func_ptr_start():
			fp_type, alias_tok = self._parse_func_ptr_type(type_spec)
			self._expect(TokenType.SEMICOLON, "Expected ';' after typedef declaration")
			self._typedef_names.add(alias_tok.value)
			return TypedefDecl(
				type_spec=fp_type,
				name=alias_tok.value,
				loc=self._loc(tok),
			)

		alias_tok = self._expect(TokenType.IDENTIFIER, "Expected typedef name")
		self._expect(TokenType.SEMICOLON, "Expected ';' after typedef declaration")
		self._typedef_names.add(alias_tok.value)
		return TypedefDecl(
			type_spec=type_spec,
			name=alias_tok.value,
			loc=self._loc(tok),
		)

	# -- Function pointer helpers --------------------------------------------

	def _is_func_ptr_start(self) -> bool:
		"""Check if the current position starts a function pointer declarator: ( * ..."""
		return self._check(TokenType.LPAREN) and self._peek(1).type == TokenType.STAR

	def _parse_func_ptr_type(self, return_type: TypeSpec) -> tuple[TypeSpec, Token]:
		"""Parse '(*name)(param_types...)' given the return type was already parsed.

		Returns (func_ptr_type_spec, name_token).
		"""
		self._expect(TokenType.LPAREN, "Expected '(' in function pointer declaration")
		self._expect(TokenType.STAR, "Expected '*' in function pointer declaration")
		name_tok = self._expect(TokenType.IDENTIFIER, "Expected function pointer name")
		self._expect(TokenType.RPAREN, "Expected ')' after function pointer name")
		self._expect(TokenType.LPAREN, "Expected '(' for function pointer parameter list")
		param_types = self._parse_func_ptr_param_types()
		self._expect(TokenType.RPAREN, "Expected ')' after function pointer parameters")
		fp_type = TypeSpec(
			base_type="__func_ptr",
			is_function_pointer=True,
			func_ptr_return_type=return_type,
			func_ptr_params=param_types,
			loc=return_type.loc,
		)
		return fp_type, name_tok

	def _parse_func_ptr_param_types(self) -> list[TypeSpec]:
		"""Parse comma-separated parameter types for a function pointer signature.

		Supports both bare types 'int, int' and named params 'int a, int b'.
		"""
		types: list[TypeSpec] = []
		if self._check(TokenType.RPAREN):
			return types
		if self._check(TokenType.VOID) and self._peek(1).type == TokenType.RPAREN:
			self._advance()
			return types
		types.append(self._parse_func_ptr_single_param())
		while self._match(TokenType.COMMA):
			types.append(self._parse_func_ptr_single_param())
		return types

	def _parse_func_ptr_single_param(self) -> TypeSpec:
		"""Parse a single parameter type (optionally with name) for a function pointer."""
		ts = self._parse_type_spec()
		# Consume optional parameter name
		if self._check(TokenType.IDENTIFIER) and self._peek(0).value not in self._typedef_names:
			self._advance()
		return ts

	# -- Function declaration ------------------------------------------------

	def _parse_function_decl(self, return_type: TypeSpec, name_tok: Token) -> FunctionDecl:
		"""Parse function parameters and body (or prototype with ';') after seeing 'type name'."""
		self._expect(TokenType.LPAREN)
		params, is_variadic = self._parse_param_list()
		self._expect(TokenType.RPAREN, "Expected ')' after parameter list")
		if self._match(TokenType.SEMICOLON):
			return FunctionDecl(
				return_type=return_type,
				name=name_tok.value,
				params=params,
				body=None,
				is_variadic=is_variadic,
				loc=self._loc(name_tok),
			)
		body = self._parse_compound_stmt()
		return FunctionDecl(
			return_type=return_type,
			name=name_tok.value,
			params=params,
			body=body,
			is_variadic=is_variadic,
			loc=self._loc(name_tok),
		)

	def _parse_param_list(self) -> tuple[list[ParamDecl], bool]:
		"""Parse a comma-separated parameter list (possibly empty).

		Returns (params, is_variadic) where is_variadic is True if the list
		ends with '...'.
		"""
		params: list[ParamDecl] = []
		is_variadic = False
		if self._check(TokenType.RPAREN):
			return params, is_variadic
		# Handle 'void' as sole parameter meaning no params
		if self._check(TokenType.VOID) and self._peek(1).type == TokenType.RPAREN:
			self._advance()
			return params, is_variadic
		params.append(self._parse_param_decl())
		while self._match(TokenType.COMMA):
			if self._match(TokenType.ELLIPSIS):
				is_variadic = True
				break
			params.append(self._parse_param_decl())
		return params, is_variadic

	def _parse_param_decl(self) -> ParamDecl:
		"""Parse a single parameter declaration: type name, type name[], or type (*name)(params)."""
		type_spec = self._parse_type_spec()
		# Check for function pointer parameter: type (*name)(params)
		if self._is_func_ptr_start():
			fp_type, name_tok = self._parse_func_ptr_type(type_spec)
			return ParamDecl(type_spec=fp_type, name=name_tok.value, loc=self._loc(name_tok))
		# Parameter name is optional (prototypes / extern declarations)
		if not self._check(TokenType.IDENTIFIER):
			return ParamDecl(type_spec=type_spec, name="", loc=type_spec.loc)
		name_tok = self._advance()
		# Array parameter syntax: type name[] or type name[size] -> convert to pointer
		if self._match(TokenType.LBRACKET):
			if not self._check(TokenType.RBRACKET):
				self._parse_expression()  # consume and discard array size
			self._expect(TokenType.RBRACKET, "Expected ']' after array parameter")
			type_spec = TypeSpec(
				base_type=type_spec.base_type,
				pointer_count=type_spec.pointer_count + 1,
				qualifiers=type_spec.qualifiers,
				signedness=type_spec.signedness,
				width_modifier=type_spec.width_modifier,
				loc=type_spec.loc,
			)
		return ParamDecl(type_spec=type_spec, name=name_tok.value, loc=self._loc(name_tok))

	# -- Statements ----------------------------------------------------------

	def _parse_statement(self) -> ASTNode:
		"""Parse a single statement."""
		# Skip C23 [[...]] attributes
		self._skip_c23_attributes()
		# Empty statement (bare ;)
		if self._check(TokenType.SEMICOLON):
			tok = self._advance()
			return ExprStmt(expression=IntLiteral(value=0, loc=self._loc(tok)), loc=self._loc(tok))
		if self._check(TokenType.STATIC_ASSERT):
			return self._parse_static_assert()
		if self._check(TokenType.LBRACE):
			return self._parse_compound_stmt()
		if self._check(TokenType.RETURN):
			return self._parse_return_stmt()
		if self._check(TokenType.IF):
			return self._parse_if_stmt()
		if self._check(TokenType.WHILE):
			return self._parse_while_stmt()
		if self._check(TokenType.FOR):
			return self._parse_for_stmt()
		if self._check(TokenType.DO):
			return self._parse_do_while_stmt()
		if self._check(TokenType.SWITCH):
			return self._parse_switch_stmt()
		if self._check(TokenType.BREAK):
			return self._parse_break_stmt()
		if self._check(TokenType.CONTINUE):
			return self._parse_continue_stmt()
		if self._check(TokenType.GOTO):
			return self._parse_goto_stmt()
		# Labeled statement: identifier followed by colon (not a typedef name)
		if (
			self._check(TokenType.IDENTIFIER)
			and self._current().value not in self._typedef_names
			and self._peek(1).type == TokenType.COLON
		):
			return self._parse_label_stmt()
		# typedef in local scope
		if self._check(TokenType.TYPEDEF):
			return self._parse_typedef_decl()
		# enum keyword: enum definition
		if self._check(TokenType.ENUM) and self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE:
			return self._parse_enum_decl()
		# struct keyword: either a struct definition or a struct variable declaration
		if self._check(TokenType.STRUCT):
			if self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE:
				return self._parse_struct_decl()
			return self._parse_var_decl_stmt()
		# union keyword: either a union definition or a union variable declaration
		if self._check(TokenType.UNION):
			if self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE:
				return self._parse_union_decl()
			return self._parse_var_decl_stmt()
		if self._check(*_TYPE_KEYWORDS):
			return self._parse_var_decl_stmt()
		# Qualifiers, storage classes, and function specifiers start a variable declaration
		if self._check(*_QUALIFIER_KEYWORDS) or self._check(*_STORAGE_CLASS_KEYWORDS) or self._check(*_FUNCTION_SPECIFIER_KEYWORDS):
			return self._parse_var_decl_stmt()
		# Typedef name used as type for variable declaration
		if self._check(TokenType.IDENTIFIER) and self._current().value in self._typedef_names:
			return self._parse_var_decl_stmt()
		return self._parse_expr_stmt()

	def _parse_compound_stmt(self) -> CompoundStmt:
		"""Parse a brace-enclosed block of statements."""
		tok = self._expect(TokenType.LBRACE, "Expected '{'")
		stmts: list[ASTNode] = []
		while not self._check(TokenType.RBRACE) and not self._at_end():
			if self._is_var_decl_start():
				stmts.extend(self._parse_var_decl_list())
			else:
				result = self._parse_statement()
				if isinstance(result, list):
					stmts.extend(result)
				else:
					stmts.append(result)
		self._expect(TokenType.RBRACE, "Expected '}'")
		return CompoundStmt(statements=stmts, loc=self._loc(tok))

	def _parse_return_stmt(self) -> ReturnStmt:
		"""Parse 'return [expr];'."""
		tok = self._advance()  # consume 'return'
		expr: ASTNode | None = None
		if not self._check(TokenType.SEMICOLON):
			expr = self._parse_expression()
		self._expect(TokenType.SEMICOLON, "Expected ';' after return statement")
		# ReturnStmt always has an expression field (ASTNode), use IntLiteral(0) for bare return
		has_expr = expr is not None
		if expr is None:
			expr = IntLiteral(value=0, loc=self._loc(tok))
		return ReturnStmt(expression=expr, has_expression=has_expr, loc=self._loc(tok))

	def _parse_if_stmt(self) -> IfStmt:
		"""Parse 'if (cond) stmt [else stmt]'."""
		tok = self._advance()  # consume 'if'
		self._expect(TokenType.LPAREN, "Expected '(' after 'if'")
		condition = self._parse_expression()
		self._expect(TokenType.RPAREN, "Expected ')' after if condition")
		then_branch = self._parse_statement()
		else_branch: ASTNode | None = None
		if self._match(TokenType.ELSE):
			else_branch = self._parse_statement()
		return IfStmt(
			condition=condition,
			then_branch=then_branch,
			else_branch=else_branch,
			loc=self._loc(tok),
		)

	def _parse_while_stmt(self) -> WhileStmt:
		"""Parse 'while (cond) stmt'."""
		tok = self._advance()  # consume 'while'
		self._expect(TokenType.LPAREN, "Expected '(' after 'while'")
		condition = self._parse_expression()
		self._expect(TokenType.RPAREN, "Expected ')' after while condition")
		body = self._parse_statement()
		return WhileStmt(condition=condition, body=body, loc=self._loc(tok))

	def _parse_for_stmt(self) -> ForStmt:
		"""Parse 'for (init; cond; update) stmt'."""
		tok = self._advance()  # consume 'for'
		self._expect(TokenType.LPAREN, "Expected '(' after 'for'")

		# init (may be a multi-variable declaration)
		init: ASTNode | list[ASTNode] | None = None
		if self._is_type_start():
			init_decls = self._parse_var_decl_list()  # includes semicolon
			init = init_decls if len(init_decls) > 1 else init_decls[0]
		elif not self._check(TokenType.SEMICOLON):
			init = self._parse_expression()
			self._expect(TokenType.SEMICOLON, "Expected ';' in for statement")
		else:
			self._advance()  # consume ';'

		# condition
		condition: ASTNode | None = None
		if not self._check(TokenType.SEMICOLON):
			condition = self._parse_expression()
		self._expect(TokenType.SEMICOLON, "Expected ';' in for statement")

		# update
		update: ASTNode | None = None
		if not self._check(TokenType.RPAREN):
			update = self._parse_expression()
		self._expect(TokenType.RPAREN, "Expected ')' after for clauses")

		body = self._parse_statement()
		return ForStmt(init=init, condition=condition, update=update, body=body, loc=self._loc(tok))

	def _parse_do_while_stmt(self) -> DoWhileStmt:
		"""Parse 'do stmt while (cond);'."""
		tok = self._advance()  # consume 'do'
		body = self._parse_statement()
		self._expect(TokenType.WHILE, "Expected 'while' after do body")
		self._expect(TokenType.LPAREN, "Expected '(' after 'while'")
		condition = self._parse_expression()
		self._expect(TokenType.RPAREN, "Expected ')' after do-while condition")
		self._expect(TokenType.SEMICOLON, "Expected ';' after do-while statement")
		return DoWhileStmt(body=body, condition=condition, loc=self._loc(tok))

	def _parse_break_stmt(self) -> BreakStmt:
		"""Parse 'break;'."""
		tok = self._advance()  # consume 'break'
		self._expect(TokenType.SEMICOLON, "Expected ';' after 'break'")
		return BreakStmt(loc=self._loc(tok))

	def _parse_continue_stmt(self) -> ContinueStmt:
		"""Parse 'continue;'."""
		tok = self._advance()  # consume 'continue'
		self._expect(TokenType.SEMICOLON, "Expected ';' after 'continue'")
		return ContinueStmt(loc=self._loc(tok))

	def _parse_goto_stmt(self) -> GotoStmt:
		"""Parse 'goto label;'."""
		tok = self._advance()  # consume 'goto'
		label_tok = self._expect(TokenType.IDENTIFIER, "Expected label name after 'goto'")
		self._expect(TokenType.SEMICOLON, "Expected ';' after goto statement")
		return GotoStmt(label=label_tok.value, loc=self._loc(tok))

	def _parse_label_stmt(self) -> LabelStmt:
		"""Parse 'label: statement'."""
		label_tok = self._advance()  # consume label identifier
		self._advance()  # consume ':'
		stmt = self._parse_statement()
		return LabelStmt(label=label_tok.value, statement=stmt, loc=self._loc(label_tok))

	def _parse_switch_stmt(self) -> SwitchStmt:
		"""Parse 'switch (expr) { case ...: stmts... default: stmts... }'."""
		tok = self._advance()  # consume 'switch'
		self._expect(TokenType.LPAREN, "Expected '(' after 'switch'")
		expression = self._parse_expression()
		self._expect(TokenType.RPAREN, "Expected ')' after switch expression")
		self._expect(TokenType.LBRACE, "Expected '{' after switch expression")
		cases: list[CaseClause] = []
		# C89: statements before the first case/default are valid (reachable via goto)
		pre_stmts: list[ASTNode] = []
		while (
			not self._check(TokenType.CASE)
			and not self._check(TokenType.DEFAULT)
			and not self._check(TokenType.RBRACE)
			and not self._at_end()
		):
			pre_stmts.append(self._parse_statement())
		if pre_stmts:
			cases.append(CaseClause(value=None, statements=pre_stmts, is_pre_switch=True, loc=self._loc(tok)))
		while not self._check(TokenType.RBRACE) and not self._at_end():
			if self._check(TokenType.CASE):
				case_tok = self._advance()  # consume 'case'
				value = self._parse_assignment()
				self._expect(TokenType.COLON, "Expected ':' after case expression")
				stmts: list[ASTNode] = []
				while (
					not self._check(TokenType.CASE)
					and not self._check(TokenType.DEFAULT)
					and not self._check(TokenType.RBRACE)
					and not self._at_end()
				):
					stmts.append(self._parse_statement())
				cases.append(CaseClause(value=value, statements=stmts, loc=self._loc(case_tok)))
			elif self._check(TokenType.DEFAULT):
				default_tok = self._advance()  # consume 'default'
				self._expect(TokenType.COLON, "Expected ':' after 'default'")
				stmts = []
				while (
					not self._check(TokenType.CASE)
					and not self._check(TokenType.DEFAULT)
					and not self._check(TokenType.RBRACE)
					and not self._at_end()
				):
					stmts.append(self._parse_statement())
				cases.append(CaseClause(value=None, statements=stmts, loc=self._loc(default_tok)))
			else:
				raise self._error("Expected 'case' or 'default' in switch body")
		self._expect(TokenType.RBRACE, "Expected '}' after switch body")
		return SwitchStmt(expression=expression, cases=cases, loc=self._loc(tok))

	def _parse_array_dimensions(self) -> list[ASTNode] | None:
		"""Parse optional array dimensions: [size][size]..."""
		if not self._check(TokenType.LBRACKET):
			return None
		array_sizes: list[ASTNode] = []
		while self._match(TokenType.LBRACKET):
			if self._check(TokenType.RBRACKET):
				array_sizes.append(IntLiteral(value=0, loc=self._loc(self._current())))
			else:
				array_sizes.append(self._parse_expression())
			self._expect(TokenType.RBRACKET, "Expected ']' after array size")
		return array_sizes

	def _parse_optional_initializer(self) -> ASTNode | None:
		"""Parse optional initializer: = expr or = {list}."""
		if not self._match(TokenType.ASSIGN):
			return None
		if self._check(TokenType.LBRACE):
			return self._parse_initializer_list()
		return self._parse_assignment()

	def _parse_additional_declarator(self, first_type_spec: TypeSpec, storage_class: str | None) -> VarDecl:
		"""Parse an additional declarator in a comma-separated declaration list.

		Each declarator can have its own pointer count, array dimensions, and initializer.
		The base type (without pointers) is derived from the first type_spec.
		"""
		pointer_count = 0
		qualifiers: list[str] = []
		while self._match(TokenType.STAR):
			pointer_count += 1
			while self._check(*_QUALIFIER_KEYWORDS):
				qualifiers.append(self._advance().value)
		name_tok = self._expect(TokenType.IDENTIFIER, "Expected variable name")
		array_sizes = self._parse_array_dimensions()
		initializer = self._parse_optional_initializer()
		decl_type = TypeSpec(
			base_type=first_type_spec.base_type,
			pointer_count=pointer_count,
			qualifiers=first_type_spec.qualifiers[:] + qualifiers,
			signedness=first_type_spec.signedness,
			width_modifier=first_type_spec.width_modifier,
			loc=first_type_spec.loc,
		)
		return VarDecl(
			type_spec=decl_type,
			name=name_tok.value,
			initializer=initializer,
			array_sizes=array_sizes,
			storage_class=storage_class,
			loc=self._loc(name_tok),
		)

	def _parse_var_decl_list(self) -> list[VarDecl]:
		"""Parse a variable declaration with optional comma-separated declarators.

		Handles: int a, b; | int *a, b; | int a = 1, b[5], *c; etc.
		Returns a list of VarDecl nodes (one per declarator).
		"""
		self._last_storage_class = None
		type_spec = self._parse_type_spec()
		storage_class = self._last_storage_class

		# Function pointer declaration (no multi-decl support)
		if self._is_func_ptr_start():
			fp_type, name_tok = self._parse_func_ptr_type(type_spec)
			initializer = None
			if self._match(TokenType.ASSIGN):
				initializer = self._parse_assignment()
			self._expect(TokenType.SEMICOLON, "Expected ';' after function pointer declaration")
			return [VarDecl(
				type_spec=fp_type,
				name=name_tok.value,
				initializer=initializer,
				storage_class=storage_class,
				loc=self._loc(name_tok),
			)]

		name_tok = self._expect(TokenType.IDENTIFIER, "Expected variable name")
		# Local extern function declaration: extern type name(params);
		if self._check(TokenType.LPAREN):
			self._advance()  # consume '('
			while not self._check(TokenType.RPAREN) and not self._at_end():
				self._advance()
			self._expect(TokenType.RPAREN, "Expected ')' after function params")
			self._expect(TokenType.SEMICOLON, "Expected ';' after function declaration")
			decl = VarDecl(
				type_spec=type_spec, name=name_tok.value,
				storage_class=storage_class, loc=self._loc(name_tok),
			)
			return [decl]
		array_sizes = self._parse_array_dimensions()
		initializer = self._parse_optional_initializer()
		first_decl = VarDecl(
			type_spec=type_spec,
			name=name_tok.value,
			initializer=initializer,
			array_sizes=array_sizes,
			storage_class=storage_class,
			loc=self._loc(name_tok),
		)
		decls = [first_decl]
		while self._match(TokenType.COMMA):
			decls.append(self._parse_additional_declarator(type_spec, storage_class))
		self._expect(TokenType.SEMICOLON, "Expected ';' after variable declaration")
		return decls

	def _is_var_decl_start(self) -> bool:
		"""Check if current position starts a variable declaration (not typedef/struct/enum/union definition)."""
		if self._check(TokenType.TYPEDEF):
			return False
		if self._check(TokenType.STRUCT, TokenType.UNION):
			if self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE:
				return False
			return True
		if self._check(TokenType.ENUM):
			if self._peek(1).type == TokenType.IDENTIFIER and self._peek(2).type == TokenType.LBRACE:
				return False
			return True
		if self._check(TokenType.INT, TokenType.CHAR, TokenType.VOID, TokenType.FLOAT, TokenType.DOUBLE,
			TokenType.LONG, TokenType.SHORT, TokenType.SIGNED, TokenType.UNSIGNED, TokenType.BOOL):
			return True
		if self._check(*_QUALIFIER_KEYWORDS) or self._check(*_STORAGE_CLASS_KEYWORDS) or self._check(*_FUNCTION_SPECIFIER_KEYWORDS):
			return True
		if self._check(TokenType.IDENTIFIER) and self._current().value in self._typedef_names:
			return True
		return False

	def _parse_var_decl_stmt(self) -> VarDecl:
		"""Parse a single local variable declaration (backward compat for non-compound contexts)."""
		decls = self._parse_var_decl_list()
		return decls[0]

	def _parse_expr_stmt(self) -> ExprStmt:
		"""Parse an expression statement: expr;"""
		tok = self._current()
		expr = self._parse_expression()
		self._expect(TokenType.SEMICOLON, "Expected ';' after expression")
		return ExprStmt(expression=expr, loc=self._loc(tok))

	# -- Initializer list ----------------------------------------------------

	def _parse_initializer_list(self) -> InitializerList:
		"""Parse '{' expr, expr, ... '}' with support for nested braces, trailing comma, and designated initializers."""
		tok = self._expect(TokenType.LBRACE, "Expected '{'")
		elements: list[ASTNode] = []
		while not self._check(TokenType.RBRACE) and not self._at_end():
			if self._check(TokenType.DOT) and self._peek(1).type == TokenType.IDENTIFIER:
				# Designated initializer: .field = expr
				dot_tok = self._advance()
				field_tok = self._expect(TokenType.IDENTIFIER, "Expected field name after '.'")
				self._expect(TokenType.ASSIGN, "Expected '=' after designated field name")
				if self._check(TokenType.LBRACE):
					value = self._parse_initializer_list()
				else:
					value = self._parse_assignment()
				elements.append(DesignatedInit(
					field_name=field_tok.value, value=value, loc=self._loc(dot_tok),
				))
			elif self._check(TokenType.LBRACKET):
				# Designated initializer: [index] = expr
				bracket_tok = self._advance()
				index_expr = self._parse_assignment()
				self._expect(TokenType.RBRACKET, "Expected ']' after index designator")
				self._expect(TokenType.ASSIGN, "Expected '=' after index designator")
				if self._check(TokenType.LBRACE):
					value = self._parse_initializer_list()
				else:
					value = self._parse_assignment()
				elements.append(DesignatedInit(
					index=index_expr, value=value, loc=self._loc(bracket_tok),
				))
			elif self._check(TokenType.LBRACE):
				elements.append(self._parse_initializer_list())
			else:
				elements.append(self._parse_assignment())
			if not self._match(TokenType.COMMA):
				break
		self._expect(TokenType.RBRACE, "Expected '}' after initializer list")
		return InitializerList(elements=elements, loc=self._loc(tok))

	# -- Expressions (precedence climbing) -----------------------------------

	def _parse_expression(self) -> ASTNode:
		"""Parse an expression, handling comma operator at lowest precedence."""
		left = self._parse_assignment()
		while self._match(TokenType.COMMA):
			right = self._parse_assignment()
			left = CommaExpr(left=left, right=right, loc=left.loc)
		return left

	def _parse_assignment(self) -> ASTNode:
		"""Parse assignment or compound assignment (right-associative)."""
		left = self._parse_ternary()
		if self._match(TokenType.ASSIGN):
			value = self._parse_assignment()
			return Assignment(target=left, value=value, loc=left.loc)
		if self._current().type in _COMPOUND_ASSIGN:
			op_tok = self._advance()
			value = self._parse_assignment()
			op_str = _COMPOUND_ASSIGN[op_tok.type]
			rhs = BinaryOp(op=op_str, left=left, right=value, loc=left.loc)
			return Assignment(target=left, value=rhs, loc=left.loc)
		return left

	def _parse_ternary(self) -> ASTNode:
		"""Parse ternary conditional: expr ? expr : expr (right-associative)."""
		expr = self._parse_binop(1)
		if self._match(TokenType.QUESTION):
			# GNU extension: ?: (elvis operator) - omitted true expr means use condition
			if self._check(TokenType.COLON):
				true_expr = expr
			else:
				true_expr = self._parse_expression()
			self._expect(TokenType.COLON, "Expected ':' in ternary expression")
			false_expr = self._parse_ternary()
			return TernaryExpr(
				condition=expr, true_expr=true_expr, false_expr=false_expr, loc=expr.loc
			)
		return expr

	def _parse_binop(self, min_prec: int) -> ASTNode:
		"""Precedence climbing for binary operators."""
		left = self._parse_unary()

		while self._current().type in _PRECEDENCE and _PRECEDENCE[self._current().type] >= min_prec:
			op_tok = self._current()
			op_prec = _PRECEDENCE[op_tok.type]
			self._advance()
			# Left-associative: next level is op_prec + 1
			right = self._parse_binop(op_prec + 1)
			left = BinaryOp(
				left=left,
				op=_BINOP_STR[op_tok.type],
				right=right,
				loc=left.loc,
			)

		return left

	def _parse_unary(self) -> ASTNode:
		"""Parse unary prefix operators: - ! ~ * & ++ -- sizeof, cast expressions, and compound literals."""
		tok = self._current()
		# Cast expression or compound literal: (type)expr or (type){init_list}
		if tok.type == TokenType.LPAREN and self._is_type_start(1):
			self._advance()  # consume '('
			cast_type = self._parse_type_spec()
			# Handle pointer-to-array declarator: (type(*)[N]) or (type(*)[])
			if self._check(TokenType.LPAREN) and self._peek(1).type == TokenType.STAR:
				self._advance()  # consume '('
				self._advance()  # consume '*'
				cast_type.pointer_count += 1
				self._expect(TokenType.RPAREN, "Expected ')' in declarator")
				if self._check(TokenType.LBRACKET):
					self._advance()  # consume '['
					arr_size = 0
					if not self._check(TokenType.RBRACKET):
						size_expr = self._parse_expression()
						if isinstance(size_expr, IntLiteral):
							arr_size = size_expr.value
					self._expect(TokenType.RBRACKET, "Expected ']' after array size")
					cast_type.pointee_array_size = arr_size
			# Handle array type in compound literal: (int[]){...} or (int[N]){...}
			if self._check(TokenType.LBRACKET):
				self._advance()  # consume '['
				if not self._check(TokenType.RBRACKET):
					self._parse_expression()  # consume array size
				self._expect(TokenType.RBRACKET, "Expected ']' after array type")
			self._expect(TokenType.RPAREN, "Expected ')' after cast type")
			# Compound literal: (type){init_list}
			if self._check(TokenType.LBRACE):
				init_list = self._parse_initializer_list()
				node: ASTNode = CompoundLiteral(type_spec=cast_type, init_list=init_list, loc=self._loc(tok))
				return self._parse_postfix_tail(node)
			operand = self._parse_unary()
			return CastExpr(target_type=cast_type, operand=operand, loc=self._loc(tok))
		if tok.type == TokenType.SIZEOF:
			self._advance()
			if self._check(TokenType.LPAREN) and self._is_type_start(1):
				self._advance()  # consume '('
				type_spec = self._parse_type_spec()
				self._expect(TokenType.RPAREN, "Expected ')' after sizeof type")
				return SizeofExpr(type_operand=type_spec, loc=self._loc(tok))
			operand = self._parse_unary()
			return SizeofExpr(operand=operand, loc=self._loc(tok))
		if tok.type == TokenType.INCREMENT:
			self._advance()
			operand = self._parse_unary()
			return UnaryOp(op="++", operand=operand, prefix=True, loc=self._loc(tok))
		if tok.type == TokenType.DECREMENT:
			self._advance()
			operand = self._parse_unary()
			return UnaryOp(op="--", operand=operand, prefix=True, loc=self._loc(tok))
		if tok.type == TokenType.MINUS:
			self._advance()
			operand = self._parse_unary()
			return UnaryOp(op="-", operand=operand, prefix=True, loc=self._loc(tok))
		if tok.type == TokenType.BANG:
			self._advance()
			operand = self._parse_unary()
			return UnaryOp(op="!", operand=operand, prefix=True, loc=self._loc(tok))
		if tok.type == TokenType.TILDE:
			self._advance()
			operand = self._parse_unary()
			return UnaryOp(op="~", operand=operand, prefix=True, loc=self._loc(tok))
		if tok.type == TokenType.STAR:
			self._advance()
			operand = self._parse_unary()
			return UnaryOp(op="*", operand=operand, prefix=True, loc=self._loc(tok))
		if tok.type == TokenType.AMPERSAND:
			self._advance()
			operand = self._parse_unary()
			return UnaryOp(op="&", operand=operand, prefix=True, loc=self._loc(tok))
		return self._parse_postfix()

	def _parse_postfix(self) -> ASTNode:
		"""Parse postfix expressions: function calls, array subscripts, member access."""
		node = self._parse_primary()
		return self._parse_postfix_tail(node)

	def _parse_postfix_tail(self, node: ASTNode) -> ASTNode:
		"""Parse postfix operators on an already-parsed primary/compound-literal node."""
		while True:
			if self._check(TokenType.LPAREN):
				self._advance()  # consume '('
				args: list[ASTNode] = []
				if not self._check(TokenType.RPAREN):
					args.append(self._parse_assignment())
					while self._match(TokenType.COMMA):
						args.append(self._parse_assignment())
				self._expect(TokenType.RPAREN, "Expected ')' after function arguments")
				if isinstance(node, Identifier):
					node = FunctionCall(name=node.name, arguments=args, loc=node.loc)
				else:
					node = FunctionCall(callee=node, arguments=args, loc=node.loc)
			elif self._check(TokenType.LBRACKET):
				self._advance()  # consume '['
				index = self._parse_expression()
				self._expect(TokenType.RBRACKET, "Expected ']' after array subscript")
				node = ArraySubscript(array=node, index=index, loc=node.loc)
			elif self._match(TokenType.DOT):
				member_tok = self._expect(TokenType.IDENTIFIER, "Expected member name after '.'")
				node = MemberAccess(object=node, member=member_tok.value, is_arrow=False, loc=node.loc)
			elif self._match(TokenType.ARROW):
				member_tok = self._expect(TokenType.IDENTIFIER, "Expected member name after '->'")
				node = MemberAccess(object=node, member=member_tok.value, is_arrow=True, loc=node.loc)
			elif self._match(TokenType.INCREMENT):
				node = PostfixExpr(operand=node, op="++", loc=node.loc)
			elif self._match(TokenType.DECREMENT):
				node = PostfixExpr(operand=node, op="--", loc=node.loc)
			else:
				break

		return node

	def _parse_primary(self) -> ASTNode:
		"""Parse primary expressions: literals, identifiers, parenthesized exprs."""
		tok = self._current()

		if tok.type == TokenType.INTEGER_LITERAL:
			self._advance()
			return IntLiteral(value=int(tok.value, 0), suffix=tok.suffix.value, loc=self._loc(tok))

		if tok.type == TokenType.FLOAT_LITERAL:
			self._advance()
			raw = tok.value
			suffix = ""
			if raw.endswith(("f", "F")):
				suffix = "f"
				raw = raw[:-1]
			elif raw.endswith(("l", "L")):
				raw = raw[:-1]
			if raw.startswith(("0x", "0X")):
				val = float.fromhex(raw)
			else:
				val = float(raw)
			return FloatLiteral(value=val, suffix=suffix, loc=self._loc(tok))

		if tok.type == TokenType.CHAR_LITERAL:
			self._advance()
			inner = interpret_c_escapes(tok.value[1:-1])
			return CharLiteral(value=inner, loc=self._loc(tok))

		if tok.type == TokenType.STRING_LITERAL:
			self._advance()
			inner = interpret_c_escapes(tok.value[1:-1])
			while self.pos < len(self.tokens) and self._current().type == TokenType.STRING_LITERAL:
				next_tok = self._current()
				self._advance()
				inner += interpret_c_escapes(next_tok.value[1:-1])
			return StringLiteral(value=inner, loc=self._loc(tok))

		if tok.type == TokenType.IDENTIFIER:
			if tok.value in ("va_start", "__builtin_va_start"):
				self._advance()
				self._expect(TokenType.LPAREN, "Expected '(' after va_start")
				ap = self._parse_assignment()
				self._expect(TokenType.COMMA, "Expected ',' in va_start")
				last_tok = self._expect(TokenType.IDENTIFIER, "Expected parameter name in va_start")
				self._expect(TokenType.RPAREN, "Expected ')' after va_start")
				return VaStartExpr(ap=ap, last_param=last_tok.value, loc=self._loc(tok))
			if tok.value in ("va_arg", "__builtin_va_arg"):
				self._advance()
				self._expect(TokenType.LPAREN, "Expected '(' after va_arg")
				ap = self._parse_assignment()
				self._expect(TokenType.COMMA, "Expected ',' in va_arg")
				arg_type = self._parse_type_spec()
				self._expect(TokenType.RPAREN, "Expected ')' after va_arg")
				return VaArgExpr(ap=ap, arg_type=arg_type, loc=self._loc(tok))
			if tok.value in ("va_end", "__builtin_va_end"):
				self._advance()
				self._expect(TokenType.LPAREN, "Expected '(' after va_end")
				ap = self._parse_assignment()
				self._expect(TokenType.RPAREN, "Expected ')' after va_end")
				return VaEndExpr(ap=ap, loc=self._loc(tok))
			if tok.value in ("va_copy", "__builtin_va_copy"):
				self._advance()
				self._expect(TokenType.LPAREN, "Expected '(' after va_copy")
				dest = self._parse_assignment()
				self._expect(TokenType.COMMA, "Expected ',' in va_copy")
				src = self._parse_assignment()
				self._expect(TokenType.RPAREN, "Expected ')' after va_copy")
				return VaCopyExpr(dest=dest, src=src, loc=self._loc(tok))
			self._advance()
			return Identifier(name=tok.value, loc=self._loc(tok))

		if tok.type == TokenType.NULLPTR:
			self._advance()
			return IntLiteral(value=0, loc=self._loc(tok))

		if tok.type == TokenType.LPAREN:
			self._advance()
			expr = self._parse_expression()
			self._expect(TokenType.RPAREN, "Expected ')' after expression")
			return expr

		raise self._error(f"Unexpected token {tok.type.name} ({tok.value!r})")
