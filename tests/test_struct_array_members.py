"""Tests for struct array members and member name validation."""

from __future__ import annotations

import pytest

from compiler.ast_nodes import (
	CompoundStmt,
	ExprStmt,
	FunctionDecl,
	Identifier,
	IntLiteral,
	MemberAccess,
	Program,
	ReturnStmt,
	SourceLocation,
	StructDecl,
	StructMember,
	TypeSpec,
	UnionDecl,
	VarDecl,
)
from compiler.parser import Parser
from compiler.semantic import SemanticAnalyzer, SemanticError


def loc(line: int = 1, col: int = 1) -> SourceLocation:
	return SourceLocation(line=line, col=col)


def int_type() -> TypeSpec:
	return TypeSpec(base_type="int")


def char_type() -> TypeSpec:
	return TypeSpec(base_type="char")


# ---------------------------------------------------------------------------
# Parser tests: struct members with array dimensions
# ---------------------------------------------------------------------------


class TestParserStructArrayMembers:
	def test_struct_int_array_member(self) -> None:
		"""struct Buf { int data[10]; };"""
		p = Parser.from_source("struct Buf { int data[10]; };")
		prog = p.parse()
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert decl.name == "Buf"
		assert len(decl.members) == 1
		m = decl.members[0]
		assert m.name == "data"
		assert m.type_spec.base_type == "int"
		assert len(m.array_dims) == 1
		assert isinstance(m.array_dims[0], IntLiteral)
		assert m.array_dims[0].value == 10

	def test_struct_char_array_member(self) -> None:
		"""struct Name { char name[32]; };"""
		p = Parser.from_source("struct Name { char name[32]; };")
		prog = p.parse()
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		m = decl.members[0]
		assert m.name == "name"
		assert m.type_spec.base_type == "char"
		assert len(m.array_dims) == 1
		assert isinstance(m.array_dims[0], IntLiteral)
		assert m.array_dims[0].value == 32

	def test_struct_multidim_array_member(self) -> None:
		"""struct Matrix { int m[3][4]; };"""
		p = Parser.from_source("struct Matrix { int m[3][4]; };")
		prog = p.parse()
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		m = decl.members[0]
		assert m.name == "m"
		assert len(m.array_dims) == 2
		assert isinstance(m.array_dims[0], IntLiteral)
		assert m.array_dims[0].value == 3
		assert isinstance(m.array_dims[1], IntLiteral)
		assert m.array_dims[1].value == 4

	def test_struct_mixed_members(self) -> None:
		"""Struct with both plain and array members."""
		src = "struct Mixed { int x; char buf[64]; int y; };"
		p = Parser.from_source(src)
		prog = p.parse()
		decl = prog.declarations[0]
		assert isinstance(decl, StructDecl)
		assert len(decl.members) == 3
		assert decl.members[0].name == "x"
		assert len(decl.members[0].array_dims) == 0
		assert decl.members[1].name == "buf"
		assert len(decl.members[1].array_dims) == 1
		assert decl.members[2].name == "y"
		assert len(decl.members[2].array_dims) == 0

	def test_union_array_member(self) -> None:
		"""union U { int arr[5]; char bytes[20]; };"""
		p = Parser.from_source("union U { int arr[5]; char bytes[20]; };")
		prog = p.parse()
		decl = prog.declarations[0]
		assert isinstance(decl, UnionDecl)
		assert len(decl.members) == 2
		assert decl.members[0].name == "arr"
		assert len(decl.members[0].array_dims) == 1
		assert decl.members[0].array_dims[0].value == 5
		assert decl.members[1].name == "bytes"
		assert len(decl.members[1].array_dims) == 1
		assert decl.members[1].array_dims[0].value == 20


# ---------------------------------------------------------------------------
# Semantic tests: member name validation
# ---------------------------------------------------------------------------


def _make_struct(name: str, members: list[StructMember]) -> StructDecl:
	return StructDecl(name=name, members=members, loc=loc())


def _make_union(name: str, members: list[StructMember]) -> UnionDecl:
	return UnionDecl(name=name, members=members, loc=loc())


def _wrap_in_function(struct_decl, var_decl, *stmts) -> Program:
	"""Wrap declarations and statements in a minimal function for analysis."""
	return Program(declarations=[
		struct_decl,
		FunctionDecl(
			return_type=int_type(),
			name="test",
			params=[],
			body=CompoundStmt(statements=[
				var_decl,
				*stmts,
				ReturnStmt(expression=IntLiteral(value=0)),
			]),
			loc=loc(),
		),
	])


class TestSemanticMemberValidation:
	def test_valid_member_dot_access(self) -> None:
		"""Accessing a valid member with '.' should not error."""
		prog = _wrap_in_function(
			_make_struct("Foo", [
				StructMember(type_spec=int_type(), name="x"),
				StructMember(type_spec=int_type(), name="y"),
			]),
			VarDecl(
				type_spec=TypeSpec(base_type="struct Foo"),
				name="f",
				loc=loc(),
			),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="f"),
				member="x",
				is_arrow=False,
				loc=loc(3, 1),
			)),
		)
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_invalid_member_dot_access(self) -> None:
		"""Accessing nonexistent member should produce 'struct Foo has no member bar'."""
		prog = _wrap_in_function(
			_make_struct("Foo", [
				StructMember(type_spec=int_type(), name="x"),
			]),
			VarDecl(
				type_spec=TypeSpec(base_type="struct Foo"),
				name="f",
				loc=loc(),
			),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="f"),
				member="bar",
				is_arrow=False,
				loc=loc(3, 1),
			)),
		)
		with pytest.raises(SemanticError, match="struct Foo has no member bar"):
			SemanticAnalyzer().analyze(prog)

	def test_valid_arrow_access(self) -> None:
		"""Accessing a valid member with '->' on a pointer should not error."""
		prog = _wrap_in_function(
			_make_struct("Foo", [
				StructMember(type_spec=int_type(), name="val"),
			]),
			VarDecl(
				type_spec=TypeSpec(base_type="struct Foo", pointer_count=1),
				name="p",
				loc=loc(),
			),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="p"),
				member="val",
				is_arrow=True,
				loc=loc(3, 1),
			)),
		)
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_invalid_arrow_access(self) -> None:
		"""Arrow-accessing nonexistent member should produce 'struct Foo has no member baz'."""
		prog = _wrap_in_function(
			_make_struct("Foo", [
				StructMember(type_spec=int_type(), name="val"),
			]),
			VarDecl(
				type_spec=TypeSpec(base_type="struct Foo", pointer_count=1),
				name="p",
				loc=loc(),
			),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="p"),
				member="baz",
				is_arrow=True,
				loc=loc(3, 1),
			)),
		)
		with pytest.raises(SemanticError, match="struct Foo has no member baz"):
			SemanticAnalyzer().analyze(prog)

	def test_union_valid_member(self) -> None:
		"""Accessing a valid union member should not error."""
		prog = _wrap_in_function(
			_make_union("U", [
				StructMember(type_spec=int_type(), name="i"),
				StructMember(type_spec=char_type(), name="c"),
			]),
			VarDecl(
				type_spec=TypeSpec(base_type="union U"),
				name="u",
				loc=loc(),
			),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="u"),
				member="i",
				is_arrow=False,
				loc=loc(3, 1),
			)),
		)
		errors = SemanticAnalyzer().analyze(prog)
		assert len(errors) == 0

	def test_union_invalid_member(self) -> None:
		"""Accessing nonexistent union member should produce 'union U has no member z'."""
		prog = _wrap_in_function(
			_make_union("U", [
				StructMember(type_spec=int_type(), name="i"),
				StructMember(type_spec=char_type(), name="c"),
			]),
			VarDecl(
				type_spec=TypeSpec(base_type="union U"),
				name="u",
				loc=loc(),
			),
			ExprStmt(expression=MemberAccess(
				object=Identifier(name="u"),
				member="z",
				is_arrow=False,
				loc=loc(3, 1),
			)),
		)
		with pytest.raises(SemanticError, match="union U has no member z"):
			SemanticAnalyzer().analyze(prog)
