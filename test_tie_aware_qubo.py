#!/usr/bin/env python3
"""test_tie_aware_qubo.py -- automated coverage for the tie-aware argmax encoding.

Run it directly:

    python test_tie_aware_qubo.py

or under pytest if you have it. It needs nothing beyond requirements.txt.

Why this exists
---------------

``bnn_as_qubo.setup_optim_model`` has two misclassification encodings. The
default, strict one asks for ``logit_k > logit_gt`` for every competing class
k. It is what produced every QUBO and every number in the repository, and it is
kept bit-identical. The opt-in tie-aware one, selected with
``args.argmax_tie_aware = True`` or ``BNN_ARGMAX_TIE_AWARE=1``, matches what
``torch.argmax`` actually does: argmax breaks ties towards the LOWEST index, so
a class k < gt that merely *ties* the true class already misclassifies. The
condition becomes

    logit_k >= logit_gt   for k < gt,      logit_k > logit_gt   for k > gt

and it is encoded by subtracting 1 from the residual of each k < gt, which
needs no extra variables and changes only a constant.

That gives a sharp, checkable invariant, and it is the one thing the soundness
argument turns on:

    offset(tie-aware) - offset(strict) == gt

where gt is the true label, because exactly the gt classes k in 0..gt-1 get the
-1 and each shifts the constant term by exactly 1. Nothing else may move: not
the variable count, not the constraint counts, not the constraints for k > gt.

Note that this is NOT the number of ``gt``-kind constraints in
``H.constraints``, which is 1 on every instance (the single
``add_constraint_gt_zero`` over the sign bits). It is the label index.

The 11x11 instance is the interesting case, because its label is 8: eight of
the nine competing classes are below it, so the delta is 8 and a sign error or
an off-by-one would be obvious. The 5x5 instance is the control at the other
extreme: its label is 0, so no class is below it, the delta is 0, and the two
encodings must produce a bit-identical matrix.
"""

import os
import sys

import verify_paper as vp


EXPECTED = {
    # size: (label, expected offset delta, strict offset)
    5: (0, 0, 533),
    11: (8, 8, 8020),
}


def _build_pair(size):
    """(strict model, tie-aware model, Info.txt fields) for one instance."""
    info = vp.read_info(size)
    strict, strict_order = vp.rebuild_qubo(size, info, tie_aware=False)
    tie, tie_order = vp.rebuild_qubo(size, info, tie_aware=True)
    return strict, strict_order, tie, tie_order, info


def _constraint_counts(model):
    return {kind: len(value) for kind, value in model.constraints.items()}


def check_instance(size):
    """Assert the tie-aware invariants for one instance. Returns a summary."""
    label, expected_delta, strict_offset = EXPECTED[size]
    strict, strict_order, tie, tie_order, info = _build_pair(size)

    assert info["label"] == label, (
        f"{size}x{size}: expected label {label} from Info.txt, "
        f"got {info['label']}")

    # The number of competing classes that get the tie-aware -1 is exactly the
    # label index, since the classes are 0..9 and k < gt is the condition.
    n_classes = 10
    shifted = [k for k in range(n_classes) if k != label and k < label]
    assert len(shifted) == label, (
        f"{size}x{size}: {len(shifted)} classes below the label, expected "
        f"{label}")

    strict_qubo, tie_qubo = strict.to_qubo(), tie.to_qubo()

    assert strict_qubo[()] == strict_offset, (
        f"{size}x{size}: strict energy offset is {strict_qubo[()]}, expected "
        f"{strict_offset}; the strict encoding must stay bit-identical")

    delta = tie_qubo[()] - strict_qubo[()]
    assert delta == expected_delta, (
        f"{size}x{size}: tie-aware offset delta is {delta}, expected "
        f"{expected_delta} (one per class below the label {label})")
    assert delta == len(shifted), (
        f"{size}x{size}: tie-aware offset delta {delta} does not equal the "
        f"{len(shifted)} classes below the label")

    # The encoding change must be a constant shift only: no new variables, no
    # new or lost constraints, and the same variable order.
    assert len(strict.variables) == len(tie.variables), (
        f"{size}x{size}: tie-aware changed the variable count from "
        f"{len(strict.variables)} to {len(tie.variables)}")
    assert _constraint_counts(strict) == _constraint_counts(tie), (
        f"{size}x{size}: tie-aware changed the constraint counts from "
        f"{_constraint_counts(strict)} to {_constraint_counts(tie)}")
    assert list(strict_order) == list(tie_order), (
        f"{size}x{size}: tie-aware changed the variable order")

    # And the label 0 case must be a no-op, not merely a zero delta: with no
    # class below the label there is nothing to shift, so the matrices agree.
    identical = strict_qubo.Q == tie_qubo.Q
    if label == 0:
        assert identical, (
            f"{size}x{size}: the label is 0, so no residual gets the -1 and "
            f"the two encodings must produce the same QUBO, but they differ")
    else:
        assert not identical, (
            f"{size}x{size}: the label is {label}, so {label} residuals should "
            f"have changed, but the two QUBOs are identical")

    return (f"{size}x{size}: label {label}, offset {strict_qubo[()]:,} -> "
            f"{tie_qubo[()]:,}, delta {delta} == {len(shifted)} class(es) "
            f"below the label; {len(strict.variables)} variables and "
            f"{_constraint_counts(strict)} unchanged")


def test_tie_aware_offset_delta_11x11():
    """The headline case: label 8, so the offset must move by exactly 8."""
    if not vp.instance_available(11):
        _skip("the 11x11 instance is not extracted; run "
              f"tar xzf {vp.DATA_ARCHIVE}")
        return
    print(check_instance(11))


def test_tie_aware_is_a_no_op_at_label_zero_5x5():
    """The control: label 0, so the tie-aware encoding must change nothing."""
    if not vp.instance_available(5):
        _skip("the 5x5 instance is not present")
        return
    print(check_instance(5))


def test_strict_encoding_is_the_default():
    """Nothing may turn the tie-aware encoding on by accident."""
    from bnn_as_qubo import tie_aware_argmax_enabled

    class _NoFlag:
        pass

    saved = os.environ.pop("BNN_ARGMAX_TIE_AWARE", None)
    try:
        assert tie_aware_argmax_enabled(_NoFlag()) is False, (
            "the tie-aware encoding must be off unless it is asked for")
        assert tie_aware_argmax_enabled(None) is False
    finally:
        if saved is not None:
            os.environ["BNN_ARGMAX_TIE_AWARE"] = saved
    print("strict argmax is the default encoding")


def _skip(reason):
    print(f"SKIPPED: {reason}")


def main():
    # get_args.py parses sys.argv at import time, so hide any of our own flags.
    sys.argv = [sys.argv[0]]

    tests = [
        test_strict_encoding_is_the_default,
        test_tie_aware_is_a_no_op_at_label_zero_5x5,
        test_tie_aware_offset_delta_11x11,
    ]
    failures = 0
    for test in tests:
        print(f"-- {test.__name__}")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"   FAILED: {exc}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} test(s) FAILED")
        return 1
    print(f"all {len(tests)} test(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
