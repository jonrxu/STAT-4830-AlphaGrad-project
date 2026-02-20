import numpy as np
import time
import traceback

from tasks.alphaevolve_cp.utils import run_with_timeout

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


def validate_packing_ae(centers, radii):
    circles = [[float(x), float(y), float(r)] for (x, y), r in zip(centers, radii)]
    try:
        import itertools
        # Check pairwise disjointness.
        for circle1, circle2 in itertools.combinations(circles, 2):
            center_distance = np.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
            radii_sum = circle1[2] + circle2[2]
            assert center_distance >= radii_sum, f"Circles are NOT disjoint: {circle1} and {circle2}."

        # Check all circles lie inside the unit square [0,1]x[0,1].
        for circle in circles:
            assert (0 <= min(circle[0], circle[1]) - circle[2] and max(circle[0],circle[1]) + circle[2] <= 1), f"Circle {circle} is NOT fully inside the unit square."
            
        return True
    except Exception as e:
        print(f"Error in validate_packing: {e}")
        return False



def evaluate(program_path, n=26):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Target value from the paper
    TARGET_VALUE = 2.635 if n==26 else 2.937  # AlphaEvolve result for n=26

    # For constructor-based approaches, a single evaluation is sufficient
    # since the result is deterministic
    start_time = time.time()

    # Use subprocess to run with timeout
    centers, radii, reported_sum = run_with_timeout(
        program_path, timeout_seconds=600  # Single timeout
    )

    end_time = time.time()
    eval_time = end_time - start_time

    # Ensure centers and radii are numpy arrays
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    # Check for NaN values before validation
    if np.isnan(centers).any() or np.isnan(radii).any():
        raise ValueError("NaN values detected in solution")

    # Validate solution
    valid = validate_packing(centers, radii)

    # Check shape and size
    shape_valid = centers.shape == (n, 2) and radii.shape == (n,)
    if not shape_valid:
        valid = False

    # Calculate sum
    sum_radii = np.sum(radii) if valid else 0.0

    # Target ratio (how close we are to the target)
    target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

    # Validity score
    validity = 1.0 if valid else 0.0

    # Combined score - higher is better
    combined_score = target_ratio * validity

    return {
        "sum_radii": float(sum_radii),
        "target_ratio": float(target_ratio),
        "validity": float(validity),
        "eval_time": float(eval_time),
        "combined_score": float(combined_score),
    }


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path, n=26):
    """
    First stage evaluation - quick validation check
    """
    try:
        # Use the simplified subprocess approach
        try:
            centers, radii, sum_radii = run_with_timeout(program_path, timeout_seconds=600)

            # Ensure centers and radii are numpy arrays
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            # Validate solution (shapes and constraints)
            shape_valid = centers.shape == (n, 2) and radii.shape == (n,)
            if not shape_valid:
                print(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}")
                return {"validity": 0.0, "error": "Invalid shapes"}

            valid = validate_packing(centers, radii)

            # Calculate sum
            actual_sum = np.sum(radii) if valid else 0.0

            # Target from paper
            target = 2.635

            # Simple combined score for stage 1
            combined_score = (actual_sum / target) if valid else 0.0

            # Return evaluation metrics
            return {
                "validity": 1.0 if valid else 0.0,
                "sum_radii": float(actual_sum),
                "target_ratio": float(actual_sum / target if valid else 0.0),
                "combined_score": float(combined_score),
            }

        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {"validity": 0.0, "combined_score": 0.0, "error": "Timeout"}
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}

    except Exception as e:
        print(f"Stage 1 evaluation failed completely: {e}")
        print(traceback.format_exc())
        return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage2(program_path, n=26):
    """
    Second stage evaluation - full evaluation
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path, n=n)