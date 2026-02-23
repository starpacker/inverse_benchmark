import ast
import difflib
from typing import Any, Dict, Optional


class CodeScorer:
    def evaluate(self, gt_code_path: str, generated_code: str) -> Dict[str, Any]:
        gt_code = ""
        try:
            with open(gt_code_path, "r") as f:
                gt_code = f.read()
        except Exception:
            gt_code = ""

        similarity = difflib.SequenceMatcher(a=gt_code, b=generated_code).ratio() if gt_code else None

        metrics = {
            "similarity_to_gt": similarity,
            "generated": self._analyze_code(generated_code),
            "ground_truth": self._analyze_code(gt_code) if gt_code else None,
        }
        return metrics

    def _analyze_code(self, code: str) -> Optional[Dict[str, Any]]:
        if code is None:
            return None

        stripped = code.strip()
        if not stripped:
            return {"lines": 0, "imports": 0, "functions": 0, "classes": 0, "ast_nodes": 0, "branch_points": 0}

        lines = [ln for ln in stripped.splitlines() if ln.strip()]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {
                "lines": len(lines),
                "imports": None,
                "functions": None,
                "classes": None,
                "ast_nodes": None,
                "branch_points": None,
                "syntax_error": True,
            }

        imports = 0
        functions = 0
        classes = 0
        ast_nodes = 0
        branch_points = 0

        for node in ast.walk(tree):
            ast_nodes += 1
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
            elif isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.With,
                    ast.AsyncWith,
                    ast.BoolOp,
                ),
            ):
                branch_points += 1

        return {
            "lines": len(lines),
            "imports": imports,
            "functions": functions,
            "classes": classes,
            "ast_nodes": ast_nodes,
            "branch_points": branch_points,
            "syntax_error": False,
        }

