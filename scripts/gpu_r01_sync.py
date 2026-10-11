"""F06: defer internal loss checks/count transfers to the checked update loop."""

import ast
import importlib
import inspect


class _DeferredStatistics(ast.NodeTransformer):
    def visit_If(self, node):
        # The caller checks the complete loss before backward and every gradient
        # before the optimizer update. Avoid two additional device-to-host waits.
        if "isnan" in ast.unparse(node.test):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "int"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Attribute)
                and node.args[0].func.attr == "item"):
            return node.args[0].func.value
        return node


class ReducedSyncLoss:
    """Preserve loss arithmetic; statistic counts remain device tensors.

    Use only with the R01 finite-loss/finite-gradient checks. The update loop
    already counts CPU labels and consolidates the loss/norm transfer.
    """
    def __init__(self, model):
        self.model = model
        module = importlib.import_module(type(model).__module__)
        tree = ast.parse(inspect.getsource(module))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SrsRWKV")
        loss = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_loss")
        loss.decorator_list = []
        loss = _DeferredStatistics().visit(loss)
        namespace = dict(module.__dict__)
        tree = ast.fix_missing_locations(ast.Module(body=[loss], type_ignores=[]))
        exec(compile(tree, module.__file__, "exec"), namespace)
        self.loss = namespace["_get_loss"]

    def get_loss(self, batch):
        return self.loss(self.model, batch.start, batch.sub_gather, batch.sub_gather_lens,
                         batch.time_shift_selects, batch.skips, batch.num_data,
                         batch.labels, batch.label_review_th)
