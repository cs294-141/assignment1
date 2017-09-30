"""We should be able to compile and interpret all the functions here.

This is the goal IR ADT:

Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Str name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop =  EQ |  NE |  LT |  GT |  LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Block body, Block? elseBody)
     | For(Str var, Expr min, Expr max, Block body)
     | Return(Expr val)
     | FuncDef(Str name, Str* args, Stmt body)

We also want to test these generalisations:
Stmt = [...]
     | If(Expr cond, Block body, Block? elseBody)
     | For(Str var, Expr min, Expr max, Block body)
"""

# Define a trivial test program to start
def trivial() -> int:
    return 5


def Stmt_Assign():
    """Test basic assignment of literals to variables.

    This also tests support for three types of literals:
        IntConst, FloatConst, Str, Bool
    """
    a = 1
    b = 2.3
    c = "celine dion"
    d = True
    e = False


def Stmt_Assign_Var():
    """Test assignment of variables to each other."""
    b = 1
    a = b


def Stmt_Return_Float() -> float:
    """Test that we can return floats."""
    return 3.2


def Stmt_Return_Int() -> int:
    """Test that we can return ints."""
    return 25


def Stmt_Return_Str() -> str:
    """Test that we can return the wisdom of our elders."""
    return "my heart will go on"


def Stmt_FuncDef():
    """Test that we can define profound functions."""
    def InnerFunction(important, arguments):
        return important + arguments


def BinOp_Add() -> float:
    """Add numbers."""
    a = 1
    b = 2.5
    c = 3.4 + 2
    d = a + b
    return d + b


def BinOp_Sub() -> float:
    """Subtract numbers."""
    a = 1
    b = 3.0
    c = b - a
    d = 6.2 - 3.2
    return d - c


def BinOp_Mul() -> float:
    """Multiply numbers."""
    a = 2.5
    b = 3.7
    c = a * b
    d = 3.2 * 8.8
    e = a * 0.99
    return c * d * e


def BinOp_Div() -> float:
    """Divide numbers."""
    a = 14.0
    b = 2.5
    c = a / b
    d = 99/11.1
    return d / c


def BinOp_Mod() -> int:
    """Mod this."""
    r = 5 % 3
    return r


def BinOp_And() -> bool:
    a = True and True
    b = True and False
    c = False and True
    d = False and False
    e = a and b and c and d
    return d and e


def BinOp_Or() -> bool:
    a = False or False
    b = False or True
    c = True or False
    d = True or True
    e = a or b or d or c
    return e or a


def BinOp_Chains():
    """Test compound operator expressions."""
    a = 1 + 2 - 4.5 + 6.2 - 2.3
    b = 8 / 3 + 4 * 6
    c = a % 2 * (99 * 100.0)
    d = True and a or b
    e = b * (b + 3.33) * c - 68 * 14.2
    return a + b * c / e - e and d


def CmpOp_Eq():
    a = 1 == 2.3
    b = 2 == 2
    return a == b


def CmpOp_Ne():
    a = 1 != 2.3
    b = 2 != 2
    return a != b


def CmpOp_Lt():
    a = 1 < 2.3
    b = 2 < 2
    return a < b


def CmpOp_Gt():
    a = 1 > 2.3
    b = 2 > 2
    return a > b


def CmpOp_Le():
    a = 1 <= 2.3
    b = 2 <= 2
    return a <= b


def CmpOp_Ge():
    a = 1 >= 2.3
    b = 2 >= 2
    return a >= b


def UnOp_Neg():
    a = -1
    b = -a
    return -a


def UnOp_Not():
    a = True
    b = not True
    c = not a
    return not b or not a


def Op_Chains():
    a = 2.0
    b = -100 < a < 99
    c = 200 >= b <= -1000
    d = 1 < 2 < 3 < 4 < 5 < 6
    e = 200.0 >= 200 >= 100.0 * 2 <= 9000 > 0
    f = a and b and c or d and e
    g = a or not b or c or not a
    return f == True


def If():
    if 1 > 2:
        return True
    elif 3 == 3:
        return False
    else:
        return 7


def For_OneArgumentRange():
    x = 0
    for i in range(2):
        # NOTE(aryap): '+=' is ast.AugAssign, which we don't support.
        x = x + i
    return x


def For_TwoArgumentRange():
    x = 0
    for i in range(4, 9):
        x = x + i
    return x


def For_ThreeArgumentRange():
    x = 0
    for i in range (0, 2, 2):
        x = x + i
    return x


def LessTrivial() -> int:
    some_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    some_list[2] = 5
    b = 4.33 * 6.5
    a = 1 + 2
    a = not a
    f = -a  # This is the 'neg' operator.
    d = 1 < a < b
    e = a * b
    f = 'asdf'
    if 1 > -2:
        for i in range(10):
            #some_list[i] = i
            a = a * a
    return (e and d or True) or (False and 7) or 3


def AddStrings():
    a = "abc"
    b = "def"
    return a + b + "foo"


def Sum_WithDefaults(a, b, c=2):
    return a + b + c + 2


def If_WithArgs(a, b):
    a = 1
    b = 2
    c = a + b + 3
    if c % 6 == 0:
        d = 12
    else:
        d = 24
    return d


def For_WithArgs(a, b, c=9):
    upper = a * b * c
    a = 0
    for i in range(upper):
        a = a + 5
    return a + i


def For_IndexVariableFinalState(a, b):
    for i in range(5):
        a = None
    return i


def FunctionCall():
    def Add(a, b):
        return a + b
    return Add(1, 2)


def TurtlesAllTheWayDown(i):
    def TurtleA(a):
        def TurtleB(b):
            def TurtleC(c):
                return c + 1
            return TurtleC(b) + 1
        return TurtleB(a) + 1
    return TurtleA(i) + 1


def Lists_Basic():
    a = [1, 2, 3, 4]
    a[1] = a[2] + 3
    return a[1]


def Lists_Parameterised(size):
    a = [1] * size
    if size > 0:
        return a[0]
    return None


def ListOfLists():
    a = [1, 2, 3]
    a[1] = [1, 2, 4]
    return a[1][2]


def Fibonacci():
    def f(a):
        if a == 0 or a == 1:
            return 1
        return f(a - 1) + f(a - 2)
    d = 0
    for i in range(10):
        d = d + f(i)
    return d
