"""
Structural Match Logic for the ~ Operator

Implements recursive structural matching as specified:
- Object matching: RHS keys must exist in LHS with recursive match
- Type matching: LHS must be instance of RHS type
- Selector matching: LHS must be in RHS selector
- Value matching: fallback to equality
"""

from ..types import (
    ShkValue, ShkObject, ShkType, ShkSelector, ShkNumber, ShkOptional,
)


def match_structure(lhs: ShkValue, rhs: ShkValue) -> bool:
    """
    Structural match: lhs ~ rhs

    Rules:
    1. If rhs is ShkObject, check object structure (subset match)
    2. If rhs is ShkType, check type membership
    3. If rhs is ShkSelector, check membership
    4. Otherwise, check value equality
    """
    if isinstance(rhs, ShkType):
        return isinstance(lhs, rhs.mapped_type)

    if isinstance(rhs, ShkSelector):
        if not isinstance(lhs, ShkNumber):
            return False
        from .selector import selector_iter_values
        values = selector_iter_values(rhs)
        return lhs in values

    if isinstance(rhs, ShkObject):
        if not isinstance(lhs, ShkObject):
            return False

        for key in rhs.slots:
            schema_value = rhs.slots[key]

            # Handle optional fields
            if isinstance(schema_value, ShkOptional):
                if key not in lhs.slots:
                    continue  # Missing optional key is OK
                # Key present, check inner schema
                if not match_structure(lhs.slots[key], schema_value.inner):
                    return False
            else:
                # Non-optional field: must exist
                if key not in lhs.slots:
                    return False
                if not match_structure(lhs.slots[key], schema_value):
                    return False

        return True

    return lhs == rhs
