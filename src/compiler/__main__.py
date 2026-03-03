"""Entry point for the C compiler: python -m compiler input.c -o output.s"""

import argparse
import sys

from compiler.codegen import CodeGenerator
from compiler.ir_gen import IRGenerator
from compiler.lexer import Lexer
from compiler.optimizer import IROptimizer
from compiler.parser import Parser
from compiler.peephole import PeepholeOptimizer
from compiler.preprocessor import Preprocessor
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
	assembly = CodeGenerator().generate(ir_program)
	if optimize:
		assembly = PeepholeOptimizer().optimize(assembly)
	return assembly


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="compiler",
		description="A C compiler targeting x86-64 assembly",
	)
	parser.add_argument("input", help="C source file to compile")
	parser.add_argument("-o", "--output", help="output assembly file (default: stdout)")
	parser.add_argument("--optimize", action="store_true", help="enable IR and peephole optimizations")

	args = parser.parse_args()

	try:
		with open(args.input) as f:
			source = f.read()
	except FileNotFoundError:
		print(f"error: file not found: {args.input}", file=sys.stderr)
		sys.exit(1)
	except OSError as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(1)

	try:
		assembly = compile_source(source, optimize=args.optimize)
	except SemanticError as e:
		print(f"semantic error: {e}", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"compilation error: {e}", file=sys.stderr)
		sys.exit(1)

	if args.output:
		with open(args.output, "w") as f:
			f.write(assembly)
	else:
		print(assembly, end="")


if __name__ == "__main__":
	main()
