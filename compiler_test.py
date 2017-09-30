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
        """
        It is boring to hand-craft expected or "golden" IRs that we expect
        for each part of the ADT. There is much more value in comparing the
        output of the interpreted functions to the those run by the python
        interpreter itself; we don't really care what the IR looks like along
        the way as long as our interfaces are consistent. Nevertheless, here is
        just such a test.
        """
        ir = compiler.BuildIR(compiler_test_code.Stmt_Assign_Var)

        self.assertTrue(isinstance(ir, compiler.FuncDef))
        self.assertTrue(isinstance(ir.body, compiler.Block))

        body = ir.body.body
        self.assertIsNone(body[0])  # The doc-string ast.Expr is a nothing.
        assignment_0 = body[1]
        self.assertTrue(isinstance(assignment_0, compiler.Assign))
        self.assertTrue(isinstance(assignment_0.ref, compiler.Ref))
        self.assertTrue(isinstance(assignment_0.ref.name, compiler.StrConst))
        self.assertTrue(isinstance(assignment_0.val, compiler.IntConst))
        self.assertEquals('b', assignment_0.ref.name.val)
        self.assertEquals(1, assignment_0.val.val)
        self.assertIsNone(assignment_0.ref.index)
        assignment_1 = body[2]
        self.assertTrue(isinstance(assignment_1, compiler.Assign))
        self.assertTrue(isinstance(assignment_1.ref, compiler.Ref))
        self.assertTrue(isinstance(assignment_1.ref.name, compiler.StrConst))
        self.assertTrue(isinstance(assignment_1.val, compiler.Ref))
        self.assertTrue(isinstance(assignment_1.val.name, compiler.StrConst))
        self.assertEquals('a', assignment_1.ref.name.val)
        self.assertEquals('b', assignment_1.val.name.val)
        self.assertIsNone(assignment_1.ref.index)
        self.assertIsNone(assignment_1.val.index)
        return_statement = body[3]
        self.assertTrue(isinstance(return_statement, compiler.Return))
        self.assertTrue(isinstance(return_statement.val, compiler.Ref))
        self.assertTrue(
            isinstance(return_statement.val.name, compiler.StrConst))
        self.assertEquals('a', return_statement.val.name.val)


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

        for t in range(50):
            # But first generate fake arguments (that are always ints) where
            # needed.
            fake_args = [random.randint(0, 100)
                         for _ in inspect.signature(func).parameters]
            interpret_check = CreateCompareToInterpreted(func, *fake_args)
            interpret_check.__name__ = (
                "testInterpretedResults_int_%s_%d" % (func.__name__, t))
            setattr(cls, interpret_check.__name__, interpret_check)

        # Some of the tests we'd also like to throw floats at. But not all
        # (e.g. you can't extend a list by multiplying it by a float).
        if (func.__name__.startswith('BinOp') or
            func.__name__.startswith('CmpOp')):
            for t in range(50):
                # But first generate fake arguments (that are always ints)
                # where needed.
                fake_args = [random.uniform(0, 100)
                             for _ in inspect.signature(func).parameters]
                interpret_check = CreateCompareToInterpreted(func, *fake_args)
                interpret_check.__name__ = (
                    "testInterpretedResults_float_%s_%d" % (func.__name__, t))
                setattr(cls, interpret_check.__name__, interpret_check)


if __name__ == '__main__':
    LoadSanityCheckCases(CompilerTest)
    unittest.main()
