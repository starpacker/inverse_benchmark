from typing import Any, Dict, List, Tuple


def validate_plan_schema(plan: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    def err(path: str, msg: str):
        errors.append(f"{path}: {msg}")

    def is_str(x: Any) -> bool:
        return isinstance(x, str) and len(x.strip()) > 0

    def is_list_of_str(x: Any) -> bool:
        return isinstance(x, list) and all(isinstance(i, str) for i in x)

    def is_int(x: Any) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    if not isinstance(plan, dict):
        return False, ["$: plan is not a dict"]

    required_top = [
        "algorithm_name",
        "data_preprocessing",
        "objective_function",
        "variables",
        "initialization",
        "main_inverse_process",
        "evaluation",
    ]
    for k in required_top:
        if k not in plan:
            err(f"$.{k}", "missing")

    if "algorithm_name" in plan and not is_str(plan.get("algorithm_name")):
        err("$.algorithm_name", "must be non-empty string")

    dp = plan.get("data_preprocessing")
    if "data_preprocessing" in plan:
        if not isinstance(dp, dict):
            err("$.data_preprocessing", "must be object")
        else:
            if not is_str(dp.get("description")):
                err("$.data_preprocessing.description", "must be non-empty string")
            steps = dp.get("steps")
            if not isinstance(steps, list) or len(steps) == 0:
                err("$.data_preprocessing.steps", "must be non-empty list")
            else:
                for i, s in enumerate(steps):
                    p = f"$.data_preprocessing.steps[{i}]"
                    if not isinstance(s, dict):
                        err(p, "must be object")
                        continue
                    if not is_int(s.get("step_order")):
                        err(f"{p}.step_order", "must be int")
                    if not is_str(s.get("step_name")):
                        err(f"{p}.step_name", "must be non-empty string")
                    if not is_str(s.get("operation")):
                        err(f"{p}.operation", "must be non-empty string")
                    mf = s.get("mathematical_formula")
                    if mf is not None and not isinstance(mf, str):
                        err(f"{p}.mathematical_formula", "must be string or null")
                    if not is_list_of_str(s.get("input_data")):
                        err(f"{p}.input_data", "must be list of strings")
                    if not is_list_of_str(s.get("output_data")):
                        err(f"{p}.output_data", "must be list of strings")
                    if not is_list_of_str(s.get("assumptions")):
                        err(f"{p}.assumptions", "must be list of strings")

    obj = plan.get("objective_function")
    if "objective_function" in plan:
        if not isinstance(obj, dict):
            err("$.objective_function", "must be object")
        else:
            if not is_str(obj.get("full_expression")):
                err("$.objective_function.full_expression", "must be non-empty string")
            if not is_str(obj.get("data_fidelity_term")):
                err("$.objective_function.data_fidelity_term", "must be non-empty string")
            if not is_str(obj.get("regularization_term")):
                err("$.objective_function.regularization_term", "must be non-empty string")
            constraints = obj.get("constraints")
            if constraints is not None and not is_list_of_str(constraints):
                err("$.objective_function.constraints", "must be list of strings or null")

    vars_ = plan.get("variables")
    if "variables" in plan:
        if not isinstance(vars_, dict):
            err("$.variables", "must be object")
        else:
            for k in ["primal", "dual", "constants", "observations"]:
                if not is_list_of_str(vars_.get(k)):
                    err(f"$.variables.{k}", "must be list of strings")

    init = plan.get("initialization")
    if "initialization" in plan:
        if not isinstance(init, list) or len(init) == 0:
            err("$.initialization", "must be non-empty list")
        else:
            for i, it in enumerate(init):
                p = f"$.initialization[{i}]"
                if not isinstance(it, dict):
                    err(p, "must be object")
                    continue
                if not is_str(it.get("variable")):
                    err(f"{p}.variable", "must be non-empty string")
                if not is_str(it.get("value")):
                    err(f"{p}.value", "must be non-empty string")
                if not is_str(it.get("shape")):
                    err(f"{p}.shape", "must be non-empty string")
                if not is_str(it.get("source")):
                    err(f"{p}.source", "must be non-empty string")

    mip = plan.get("main_inverse_process")
    if "main_inverse_process" in plan:
        if not isinstance(mip, dict):
            err("$.main_inverse_process", "must be object")
        else:
            if not is_str(mip.get("algorithm_framework")):
                err("$.main_inverse_process.algorithm_framework", "must be non-empty string")
            loop = mip.get("iteration_loop")
            if not isinstance(loop, list) or len(loop) == 0:
                err("$.main_inverse_process.iteration_loop", "must be non-empty list")
            else:
                for i, step in enumerate(loop):
                    p = f"$.main_inverse_process.iteration_loop[{i}]"
                    if not isinstance(step, dict):
                        err(p, "must be object")
                        continue
                    if not is_int(step.get("step_order")):
                        err(f"{p}.step_order", "must be int")
                    if not is_str(step.get("step_name")):
                        err(f"{p}.step_name", "must be non-empty string")
                    if not is_str(step.get("step_type")):
                        err(f"{p}.step_type", "must be non-empty string")
                    if not is_str(step.get("mathematical_formula")):
                        err(f"{p}.mathematical_formula", "must be non-empty string")
                    if not is_list_of_str(step.get("operator_requirements")):
                        err(f"{p}.operator_requirements", "must be list of strings")
                    if not is_list_of_str(step.get("input_variables")):
                        err(f"{p}.input_variables", "must be list of strings")
                    if not is_list_of_str(step.get("output_variables")):
                        err(f"{p}.output_variables", "must be list of strings")
                    if not isinstance(step.get("computational_notes"), str):
                        err(f"{p}.computational_notes", "must be string (can be empty)")
            sc = mip.get("stopping_criterion")
            if not isinstance(sc, dict):
                err("$.main_inverse_process.stopping_criterion", "must be object")
            else:
                if not is_str(sc.get("type")):
                    err("$.main_inverse_process.stopping_criterion.type", "must be non-empty string")
                if not is_str(sc.get("expression")):
                    err("$.main_inverse_process.stopping_criterion.expression", "must be non-empty string")

    ev = plan.get("evaluation")
    if "evaluation" in plan:
        if not isinstance(ev, dict):
            err("$.evaluation", "must be object")
        else:
            if not is_str(ev.get("description")):
                err("$.evaluation.description", "must be non-empty string")
            metrics = ev.get("metrics")
            if not isinstance(metrics, list) or len(metrics) == 0:
                err("$.evaluation.metrics", "must be non-empty list")
            else:
                for i, m in enumerate(metrics):
                    p = f"$.evaluation.metrics[{i}]"
                    if not isinstance(m, dict):
                        err(p, "must be object")
                        continue
                    if not is_str(m.get("metric_name")):
                        err(f"{p}.metric_name", "must be non-empty string")
                    if not is_str(m.get("definition")):
                        err(f"{p}.definition", "must be non-empty string")
                    if not is_str(m.get("reference_data")):
                        err(f"{p}.reference_data", "must be non-empty string")
                    if not is_str(m.get("output")):
                        err(f"{p}.output", "must be non-empty string")
            pp = ev.get("post_processing")
            if pp is None:
                err("$.evaluation.post_processing", "missing")
            elif not isinstance(pp, list):
                err("$.evaluation.post_processing", "must be list")
            else:
                for i, op in enumerate(pp):
                    p = f"$.evaluation.post_processing[{i}]"
                    if not isinstance(op, dict):
                        err(p, "must be object")
                        continue
                    if not is_str(op.get("operation")):
                        err(f"{p}.operation", "must be non-empty string")
                    if not is_list_of_str(op.get("input_variables")):
                        err(f"{p}.input_variables", "must be list of strings")
                    if not is_list_of_str(op.get("output_variables")):
                        err(f"{p}.output_variables", "must be list of strings")

    return len(errors) == 0, errors

