"""Recursive descent parser for a minimal C subset.

Consumes tokens from the lexer and builds an AST. Uses precedence climbing
for expression parsing.
"""

from __future__ import annotations

from compiler.ast_nodes import (
	Assignment,
	ASTNode,
	BinaryOp,
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
	Program,
	ReturnStmt,
	SourceLocation,
	StringLiteral,
	TypeSpec,
	UnaryOp,
	VarDecl,
	WhileStmt,
)
from compiler.lexer import Lexer
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
	TokenType.EQUAL: 3,
	TokenType.NOT_EQUAL: 3,
	TokenType.LESS: 4,
	TokenType.GREATER: 4,
	TokenType.LESS_EQUAL: 4,
	TokenType.GREATER_EQUAL: 4,
	TokenType.PLUS: 5,
	TokenType.MINUS: 5,
	TokenType.STAR: 6,
	TokenType.SLASH: 6,
	TokenType.PERCENT: 6,
}

_BINOP_STR: dict[TokenType, str] = {
	TokenType.PLUS: "+",
	TokenType.MINUS: "-",
	TokenType.STAR: "*",
	TokenType.SLASH: "/",
	TokenType.PERCENT: "%",
	TokenType.EQUAL: "==",
	TokenType.NOT_EQUAL: "!=",
	TokenType.LESS: "<",
	TokenType.GREATER: ">",
	TokenType.LESS_EQUAL: "<=",
	TokenType.GREATER_EQUAL: ">=",
	TokenType.AND: "&&",
	TokenType.OR: "||",
}

_TYPE_KEYWORDS: set[TokenType] = {TokenType.INT, TokenType.CHAR, TokenType.VOID}


class Parser:
	"""Recursive descent parser that builds AST nodes from a token stream."""

	def __init__(self, tokens: list[Token]) -> None:
		self.tokens = tokens
		self.pos = 0

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

	# -- Top-level parsing ---------------------------------------------------

	def parse(self) -> Program:
		"""Parse the full token stream into a Program AST node."""
		declarations: list[ASTNode] = []
		while not self._at_end():
			declarations.append(self._parse_top_level_decl())
		return Program(declarations=declarations, loc=SourceLocation(1, 1))

	def _parse_top_level_decl(self) -> ASTNode:
		"""Parse a top-level declaration (function or global variable)."""
		type_spec = self._parse_type_spec()
		name_tok = self._expect(TokenType.IDENTIFIER, "Expected declaration name")

		if self._check(TokenType.LPAREN):
			return self._parse_function_decl(type_spec, name_tok)

		# Global variable declaration
		initializer = None
		if self._match(TokenType.ASSIGN):
			initializer = self._parse_expression()
		self._expect(TokenType.SEMICOLON, "Expected ';' after variable declaration")
		return VarDecl(
			type_spec=type_spec,
			name=name_tok.value,
			initializer=initializer,
			loc=self._loc(name_tok),
		)

	# -- Type specifier ------------------------------------------------------

	def _parse_type_spec(self) -> TypeSpec:
		"""Parse a type specifier: int, char, void with optional pointer '*'s."""
		tok = self._current()
		if tok.type not in _TYPE_KEYWORDS:
			raise self._error(f"Expected type specifier, got {tok.type.name} ({tok.value!r})")
		self._advance()
		pointer_count = 0
		while self._match(TokenType.STAR):
			pointer_count += 1
		return TypeSpec(base_type=tok.value, pointer_count=pointer_count, loc=self._loc(tok))

	# -- Function declaration ------------------------------------------------

	def _parse_function_decl(self, return_type: TypeSpec, name_tok: Token) -> FunctionDecl:
		"""Parse function parameters and body after seeing 'type name'."""
		self._expect(TokenType.LPAREN)
		params = self._parse_param_list()
		self._expect(TokenType.RPAREN, "Expected ')' after parameter list")
		body = self._parse_compound_stmt()
		return FunctionDecl(
			return_type=return_type,
			name=name_tok.value,
			params=params,
			body=body,
			loc=self._loc(name_tok),
		)

	def _parse_param_list(self) -> list[ParamDecl]:
		"""Parse a comma-separated parameter list (possibly empty)."""
		params: list[ParamDecl] = []
		if self._check(TokenType.RPAREN):
			return params
		# Handle 'void' as sole parameter meaning no params
		if self._check(TokenType.VOID) and self._peek(1).type == TokenType.RPAREN:
			self._advance()
			return params
		params.append(self._parse_param_decl())
		while self._match(TokenType.COMMA):
			params.append(self._parse_param_decl())
		return params

	def _parse_param_decl(self) -> ParamDecl:
		"""Parse a single parameter declaration: type name."""
		type_spec = self._parse_type_spec()
		name_tok = self._expect(TokenType.IDENTIFIER, "Expected parameter name")
		return ParamDecl(type_spec=type_spec, name=name_tok.value, loc=self._loc(name_tok))

	# -- Statements ----------------------------------------------------------

	def _parse_statement(self) -> ASTNode:
		"""Parse a single statement."""
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
		if self._check(*_TYPE_KEYWORDS):
			return self._parse_var_decl_stmt()
		return self._parse_expr_stmt()

	def _parse_compound_stmt(self) -> CompoundStmt:
		"""Parse a brace-enclosed block of statements."""
		tok = self._expect(TokenType.LBRACE, "Expected '{'")
		stmts: list[ASTNode] = []
		while not self._check(TokenType.RBRACE) and not self._at_end():
			stmts.append(self._parse_statement())
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
		if expr is None:
			expr = IntLiteral(value=0, loc=self._loc(tok))
		return ReturnStmt(expression=expr, loc=self._loc(tok))

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

		# init
		init: ASTNode | None = None
		if self._check(*_TYPE_KEYWORDS):
			init = self._parse_var_decl_stmt()  # includes semicolon
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

	def _parse_var_decl_stmt(self) -> VarDecl:
		"""Parse a local variable declaration: type name [= expr];"""
		type_spec = self._parse_type_spec()
		name_tok = self._expect(TokenType.IDENTIFIER, "Expected variable name")
		initializer = None
		if self._match(TokenType.ASSIGN):
			initializer = self._parse_expression()
		self._expect(TokenType.SEMICOLON, "Expected ';' after variable declaration")
		return VarDecl(
			type_spec=type_spec,
			name=name_tok.value,
			initializer=initializer,
			loc=self._loc(name_tok),
		)

	def _parse_expr_stmt(self) -> ExprStmt:
		"""Parse an expression statement: expr;"""
		tok = self._current()
		expr = self._parse_expression()
		self._expect(TokenType.SEMICOLON, "Expected ';' after expression")
		return ExprStmt(expression=expr, loc=self._loc(tok))

	# -- Expressions (precedence climbing) -----------------------------------

	def _parse_expression(self) -> ASTNode:
		"""Parse an expression, handling assignment as lowest precedence."""
		return self._parse_assignment()

	def _parse_assignment(self) -> ASTNode:
		"""Parse assignment: expr = expr (right-associative)."""
		left = self._parse_binop(1)
		if self._match(TokenType.ASSIGN):
			value = self._parse_assignment()  # right-associative
			return Assignment(target=left, value=value, loc=left.loc)
		return left

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
		"""Parse unary prefix operators: - ! ~ * &."""
		tok = self._current()
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
		"""Parse postfix expressions: function calls."""
		node = self._parse_primary()

		while True:
			if self._check(TokenType.LPAREN) and isinstance(node, Identifier):
				self._advance()  # consume '('
				args: list[ASTNode] = []
				if not self._check(TokenType.RPAREN):
					args.append(self._parse_expression())
					while self._match(TokenType.COMMA):
						args.append(self._parse_expression())
				self._expect(TokenType.RPAREN, "Expected ')' after function arguments")
				node = FunctionCall(name=node.name, arguments=args, loc=node.loc)
			else:
				break

		return node

	def _parse_primary(self) -> ASTNode:
		"""Parse primary expressions: literals, identifiers, parenthesized exprs."""
		tok = self._current()

		if tok.type == TokenType.INTEGER_LITERAL:
			self._advance()
			return IntLiteral(value=int(tok.value, 0), loc=self._loc(tok))

		if tok.type == TokenType.CHAR_LITERAL:
			self._advance()
			# Strip surrounding quotes; handle escape sequences
			inner = tok.value[1:-1]
			return CharLiteral(value=inner, loc=self._loc(tok))

		if tok.type == TokenType.STRING_LITERAL:
			self._advance()
			# Strip surrounding quotes
			inner = tok.value[1:-1]
			return StringLiteral(value=inner, loc=self._loc(tok))

		if tok.type == TokenType.IDENTIFIER:
			self._advance()
			return Identifier(name=tok.value, loc=self._loc(tok))

		if tok.type == TokenType.LPAREN:
			self._advance()
			expr = self._parse_expression()
			self._expect(TokenType.RPAREN, "Expected ')' after expression")
			return expr

		raise self._error(f"Unexpected token {tok.type.name} ({tok.value!r})")
