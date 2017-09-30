import ast
import astor
import compiler
import compiler_test_code
import inspect
import unittest
import random

ACCEPTABLE_IR_CLASSES = set([
    compiler.BinOp,
    compiler.CmpOp,
    compiler.UnOp,
    compiler.Ref,
    compiler.FloatConst,
    compiler.IntConst,
    compiler.StrConst,
    compiler.NamedConst,
    compiler.Call,
    compiler.List,
    compiler.Assign,
    compiler.Block,
    compiler.If,
    compiler.For,
    compiler.Return,
    compiler.FuncDef,
    ast.USub,   # Neg
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,  # NE
    ast.Lt,
    ast.Gt,
    ast.LtE,    # LE
    ast.GtE,    # GE
    type(None)
])


class CompilerTest(unittest.TestCase):
    """Test fixture
    
    Most test functions are loaded dynamically for each of the functions
    defined in the compiler_test_code module.
    """

    def testIR_StmtAssign(self):
        ir = compiler.BuildIR(compiler_test_code.Stmt_Assign_Var)
        # TODO(aryap): Make this more useful. Compare it to what you'd expect
        # the IR to actually be.


def CreateSanityCheckFunction(func):
    def SanityCheckFunction(self):
        compiler.Compile(func)
    return SanityCheckFunction


def CreateCompareToInterpreted(func, *args):
    def CompareToInterpreted(self):
        native_result = func(*args)
        interpreted_func = compiler.Compile(func)
        interpreted_result = interpreted_func(*args)
        print("native: %s | interpreted: %s" % (
            str(native_result), str(interpreted_result)))
        self.assertEqual(native_result, interpreted_result)
    return CompareToInterpreted


def CreateIRUsesOnlyKnownClasses(test_func):
    def IRUsesOnlyKnownClasses(self):
        """Make sure no strange classes leaked into our IR.

        This test points to anything that isn't correctly parsed and
        translated.
        """
        seen_classes = set()

        class UsedClassesCollector(ast.NodeVisitor):
            def __init__(self, seen_classes):
                self.seen = seen_classes

            # We don't care about what python primitive types our
            # representations point to, so we interrupt the traversal here.
            def visit_IntConst(self, node):
                pass
            def visit_FloatConst(self, node):
                pass
            def visit_StrConst(self, node):
                pass
            def visit_NamedConst(self, node):
                pass

            def generic_visit(self, node):
                # Collect the type name and visit all the fields.
                if isinstance(node, list):
                    for member in node:
                        self.visit(member)
                    return
                elif isinstance(node, ast.AST):
                    for field in node._fields:
                        self.visit(getattr(node, field))
                self.seen.add(type(node))

        collector = UsedClassesCollector(seen_classes)
        collector.visit(compiler.BuildIR(test_func))

        # Left difference.
        unacceptable = seen_classes.difference(ACCEPTABLE_IR_CLASSES)
        self.assertEqual(0, len(unacceptable), (
            "There are unacceptable classes in the IR: %s" % unacceptable))

    return IRUsesOnlyKnownClasses


def LoadSanityCheckCases(cls):
    functions_under_test = [
        f for _, f in compiler_test_code.__dict__.items()
        if callable(f) ]

    # We don't _really_ want random. We want it to be reproducible.
    # Ok, *I* want it to be reproducible.
    random.seed(0)

    for func in functions_under_test:
        # 1. Sanity check compilation to make sure not errors are raised.
        test = CreateSanityCheckFunction(func)
        # If the test name doesn't start with 'test', it won't be run.
        # (╯°□°)╯︵ ┻━┻
        test.__name__ = "testCompile_%s" % func.__name__
        setattr(cls, test.__name__, test)

        # 2. Test that the IR contains only classes we expect.
        ir_range_check = CreateIRUsesOnlyKnownClasses(func)
        ir_range_check.__name__ = (
            "testIR_UsesOnlyKnownClasses_%s" % func.__name__)
        setattr(cls, ir_range_check.__name__, ir_range_check)

        # 3. Check that compiling and interpreting every function yields the
        # same result as running it in Python itself.

        # But first generate fake arguments (that are always ints) where needed.
        fake_args = [random.randint(0, 100)
                     for _ in inspect.signature(func).parameters]
        interpret_check = CreateCompareToInterpreted(func, *fake_args)
        interpret_check.__name__ = (
            "testInterpretResultsEqualNativeResults_%s" % func.__name__)
        setattr(cls, interpret_check.__name__, interpret_check)


if __name__ == '__main__':
    LoadSanityCheckCases(CompilerTest)
    unittest.main()
