"""Entry point for the C compiler: python -m compiler input.c -o output"""

import argparse
import sys
from pathlib import Path

from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.linker import ToolchainError, compile_and_link, compile_to_object
from compiler.optimizer import IROptimizer
from compiler.parser import Parser
from compiler.peephole import PeepholeOptimizer
from compiler.preprocessor import Preprocessor
from compiler.regalloc import allocate_registers
from compiler.semantic import SemanticAnalyzer, SemanticError


def compile_source(source: str, optimize: bool = False) -> str:
	"""Run C source through the full compiler pipeline, returning assembly."""
	preprocessed = Preprocessor().process(source)
	tokens = Lexer(preprocessed).tokenize()
	ast = Parser(tokens).parse()
	errors = SemanticAnalyzer().analyze(ast)
	if errors:
		raise errors[0]
	ir_program = IRGenerator().generate(ast)
	if optimize:
		ir_program = IROptimizer().optimize(ir_program)
	regalloc_maps = allocate_registers(ir_program) if optimize else None
	assembly = CodeGenerator(regalloc_maps=regalloc_maps).generate(ir_program)
	if optimize:
		assembly = PeepholeOptimizer().optimize(assembly)
	return assembly


def _default_output(input_path: str, mode: str) -> str:
	"""Derive default output filename from input path and compilation mode."""
	stem = Path(input_path).stem
	if mode == "asm":
		return stem + ".s"
	elif mode == "obj":
		return stem + ".o"
	else:
		return stem


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="compiler",
		description="A C compiler targeting x86-64 assembly",
	)
	parser.add_argument("input", help="C source file to compile")
	parser.add_argument("-o", "--output", help="output filename")
	parser.add_argument("--optimize", action="store_true", help="enable IR and peephole optimizations")
	parser.add_argument("-S", dest="emit_asm", action="store_true", help="emit assembly only (do not assemble or link)")
	parser.add_argument("-c", dest="compile_only", action="store_true", help="compile and assemble to object file (do not link)")
	parser.add_argument("-l", dest="libraries", action="append", default=[], metavar="LIB", help="link additional library (e.g. -lm)")
	parser.add_argument("--keep-intermediates", action="store_true", help="keep intermediate .s and .o files")

	args = parser.parse_args()

	# Determine compilation mode
	if args.emit_asm:
		mode = "asm"
	elif args.compile_only:
		mode = "obj"
	else:
		mode = "exe"

	# Read source
	try:
		with open(args.input) as f:
			source = f.read()
	except FileNotFoundError:
		print(f"error: file not found: {args.input}", file=sys.stderr)
		sys.exit(1)
	except OSError as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(1)

	# Compile to assembly
	try:
		assembly = compile_source(source, optimize=args.optimize)
	except SemanticError as e:
		print(f"semantic error: {e}", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"compilation error: {e}", file=sys.stderr)
		sys.exit(1)

	# Emit based on mode
	if mode == "asm":
		if args.output:
			with open(args.output, "w") as f:
				f.write(assembly)
		else:
			print(assembly, end="")
	elif mode == "obj":
		output = args.output or _default_output(args.input, "obj")
		try:
			compile_to_object(assembly, output, keep_asm=args.keep_intermediates)
		except ToolchainError as e:
			print(f"error: {e}", file=sys.stderr)
			sys.exit(1)
	else:
		output = args.output or _default_output(args.input, "exe")
		try:
			compile_and_link(
				assembly,
				output,
				libraries=args.libraries or None,
				keep_intermediates=args.keep_intermediates,
			)
		except ToolchainError as e:
			print(f"error: {e}", file=sys.stderr)
			sys.exit(1)


if __name__ == "__main__":
	main()
