from __future__ import annotations

from typing import Dict, List

# Closed KC catalog aligned with Evi4PKT / CodeWorkout prompt annotations.
DEFAULT_KC_CATALOG: List[str] = [
    "IfElse",
    "NestedIf",
    "While",
    "For",
    "NestedFor",
    "MathBasic",
    "MathMod",
    "LogicAndNotOr",
    "LogicCompareNum",
    "LogicBoolean",
    "StringFormat",
    "StringConcat",
    "StringIndex",
    "StringLen",
    "StringEqual",
    "CharEqual",
    "ArrayIndex",
    "DefFunction",
]

# Columns in data/metadata/problem_prompts.csv -> catalog names.
PROMPT_CSV_KC_COLUMNS: Dict[str, str] = {
    "If/Else": "IfElse",
    "NestedIf": "NestedIf",
    "While": "While",
    "For": "For",
    "NestedFor": "NestedFor",
    "Math+-*/": "MathBasic",
    "Math%": "MathMod",
    "LogicAndNotOr": "LogicAndNotOr",
    "LogicCompareNum": "LogicCompareNum",
    "LogicBoolean": "LogicBoolean",
    "StringFormat": "StringFormat",
    "StringConcat": "StringConcat",
    "StringIndex": "StringIndex",
    "StringLen": "StringLen",
    "StringEqual": "StringEqual",
    "CharEqual": "CharEqual",
    "ArrayIndex": "ArrayIndex",
    "DefFunction": "DefFunction",
}
