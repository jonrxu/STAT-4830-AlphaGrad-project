import numpy as np
import json
import math
from scipy.optimize import minimize, Bounds, NonlinearConstraint, basinhopping
import os
import sys
import random

# Constants
N_CIRCLES = 26
MIN_RADIUS_OPT = 1e-8 # Minimum radius for numerical stability in optimization
TARGET_SUM_RADII = 2.636 # Target sum of radii for n=26, derived from known optimal solutions

# Increased attempts and iterations for better exploration and convergence
NUM_LLM_ATTEMPTS = 30 # Increased number of times to try LLM and optimize for diverse initial guesses
NUM_RANDOM_STARTS = 20 # Increased: More attempts for purely random initial configurations
OPTIMIZER_MAXITER_SLSQP = 20000 # Max iterations for initial SLSQP runs
OPTIMIZER_MAXITER_BASIN = 12000 # Max iterations for local minimizer within basinhopping
OPTIMIZER_FTOL = 1e-12 # Adjusted function tolerance for SciPy optimizer for better precision
OPTIMIZER_XTOL = 1e-12 # Adjusted tolerance for changes in parameters (position/radius)

# Basinhopping parameters for global exploration
BASINHOPPING_NITER = 300 # Increased: Number of global minimization iterations (perturbation steps)
BASINHOPPING_T = 1.5 # Temperature parameter for acceptance criterion (higher allows escaping deeper local minima)
BASINHOPPING_STEPSIZE = 0.05 # Max step size for random perturbations in basinhopping's global steps

# Adjusted perturbation magnitudes for initial guess generation (using uniform distribution for clarity)
PERTURBATION_MAGNITUDE_XY = 0.02 # Max perturbation for x, y coordinates
PERTURBATION_MAGNITUDE_R = 0.004 # Max perturbation for radius

# Additional constant for robust random initial circle generation
MAX_INITIAL_RADIUS_RANDOM_GEN = 0.08 # Max initial radius for random generation to give optimizer more starting room

def _calculate_sum_radii(circles):
    """Calculates the sum of radii for a list of circles, handling potential None/malformed."""
    return sum(c[2] for c in circles if c is not None and len(c) == 3 and isinstance(c[2], (int, float)))

def _is_valid_packing(circles, n_expected, tolerance=1e-7):
    """
    Checks if a packing is valid:
    - Exactly n_expected circles
    - All circles within unit square
    - No overlaps
    - All radii positive
    """
    if len(circles) != n_expected:
        return False, f"Expected {n_expected} circles, got {len(circles)}."

    for i, c1 in enumerate(circles):
        x1, y1, r1 = c1
        # Basic type and non-negativity checks for radius
        if not (isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and isinstance(r1, (int, float))):
            return False, f"Circle {i} has non-numeric coordinates or radius: {c1}"
        if r1 < MIN_RADIUS_OPT - tolerance: # Check against MIN_RADIUS_OPT
            return False, f"Circle {i} has radius {r1} less than min_radius_opt {MIN_RADIUS_OPT}."
        
        # Boundary checks with tolerance for floating point precision
        if not (0 - tolerance <= x1 - r1 and x1 + r1 <= 1 + tolerance and 0 - tolerance <= y1 - r1 and y1 + r1 <= 1 + tolerance):
            return False, f"Circle {i} ({x1:.3f},{y1:.3f},{r1:.3f}) is outside boundaries."

        for j in range(i + 1, len(circles)):
            c2 = circles[j]
            x2, y2, r2 = c2
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            # Use a small tolerance for floating point comparisons to avoid false negatives
            if dist_sq < min_dist_sq - tolerance:
                return False, f"Circles {i} and {j} overlap: c1={c1}, c2={c2}, dist_sq={dist_sq:.5f}, min_dist_sq={min_dist_sq:.5f}."
    return True, "Valid."


def _generate_random_valid_circles(n):
    """
    Generates a list of n circles with random, valid positions and small radii.
    Ensures no immediate overlaps and within bounds.
    This version attempts to place slightly larger circles initially to give the optimizer more room,
    then fills remaining spots with smaller circles.
    """
    circles = []
    max_attempts_per_circle = 700 # Increased attempts for better initial placement
    # Attempt to place 70% of circles with slightly larger radii, then fill the rest with smaller
    num_larger = int(n * 0.7)
    
    for i in range(n):
        placed = False
        current_max_r_gen = MAX_INITIAL_RADIUS_RANDOM_GEN
        if i >= num_larger: # For the last ~30% of circles, use smaller radii to fill gaps
            current_max_r_gen = max(MIN_RADIUS_OPT, MAX_INITIAL_RADIUS_RANDOM_GEN / 2.5) # Use even smaller radii
        
        for _attempt in range(max_attempts_per_circle):
            r = random.uniform(MIN_RADIUS_OPT, current_max_r_gen)
            
            x_min, x_max = r, 1.0 - r
            y_min, y_max = r, 1.0 - r

            if x_min >= x_max - 1e-9 or y_min >= y_max - 1e-9: # This could happen if r is too large (e.g., > 0.5)
                r = MIN_RADIUS_OPT # Fallback to smallest radius
                x_min, x_max = r, 1.0 - r
                y_min, y_max = r, 1.0 - r
                
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            
            new_circle = [float(x), float(y), float(r)]
            
            is_overlapping = False
            for existing_c in circles:
                ex, ey, er = existing_c
                dist_sq = (x - ex)**2 + (y - ey)**2
                min_dist_sq = (r + er)**2
                if dist_sq < min_dist_sq - 1e-8: # Small tolerance for check
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                circles.append(new_circle)
                placed = True
                break
        
        if not placed:
            # If after max_attempts_per_circle, we can't place a non-overlapping circle,
            # just add one with min_radius_opt at a random valid spot.
            r_fallback = MIN_RADIUS_OPT
            x_fallback = random.uniform(r_fallback, 1.0 - r_fallback)
            y_fallback = random.uniform(r_fallback, 1.0 - r_fallback)
            circles.append([float(x_fallback), float(y_fallback), r_fallback])

    return circles[:n] # Ensure exactly N circles


# Analytical Jacobians for constraints - IMPROVED for optimizer performance
def _boundary_constraint_jac(x, n):
    """Jacobian for boundary_constraint_func."""
    jac = np.zeros((n * 4, n * 3))
    for i in range(n):
        # Constraints: (x-r), (1-x-r), (y-r), (1-y-r)
        # Derivatives with respect to xi, yi, ri
        jac[i * 4 + 0, i * 3 + 0] = 1.0  # d(x-r)/dx = 1
        jac[i * 4 + 0, i * 3 + 2] = -1.0 # d(x-r)/dr = -1

        jac[i * 4 + 1, i * 3 + 0] = -1.0 # d(1-x-r)/dx = -1
        jac[i * 4 + 1, i * 3 + 2] = -1.0 # d(1-x-r)/dr = -1

        jac[i * 4 + 2, i * 3 + 1] = 1.0  # d(y-r)/dy = 1
        jac[i * 4 + 2, i * 3 + 2] = -1.0 # d(y-r)/dr = -1

        jac[i * 4 + 3, i * 3 + 1] = -1.0 # d(1-y-r)/dy = -1
        jac[i * 4 + 3, i * 3 + 2] = -1.0 # d(1-y-r)/dr = -1
    return jac

def _overlap_constraint_jac(x, n):
    """Jacobian for overlap_constraint_func."""
    circles_opt = x.reshape(n, 3)
    num_overlap_constraints = n * (n - 1) // 2
    jac = np.zeros((num_overlap_constraints, n * 3))
    
    constraint_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles_opt[i]
            xj, yj, rj = circles_opt[j]

            # Constraint: (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
            # Partial derivatives for circle i
            jac[constraint_idx, i * 3 + 0] = 2 * (xi - xj) # d/dxi
            jac[constraint_idx, i * 3 + 1] = 2 * (yi - yj) # d/dyi
            jac[constraint_idx, i * 3 + 2] = -2 * (ri + rj) # d/dri

            # Partial derivatives for circle j
            jac[constraint_idx, j * 3 + 0] = -2 * (xi - xj) # d/dxj
            jac[constraint_idx, j * 3 + 1] = -2 * (yi - yj) # d/dyj
            jac[constraint_idx, j * 3 + 2] = -2 * (ri + rj) # d/drj
            
            constraint_idx += 1
    return jac


def solve_packing(llm_function, n=N_CIRCLES):
    """
    Solves the circle packing problem using an LLM for initial guesses,
    refines them with local optimization (SciPy SLSQP), and then applies
    SciPy's basinhopping for global exploration to maximize the sum of radii.
    """
    
    # Initialize with a guaranteed valid, albeit suboptimal, solution
    best_circles = _generate_random_valid_circles(n) 
    best_sum_radii = _calculate_sum_radii(best_circles)
    print(f"Initial random baseline sum_radii: {best_sum_radii:.5f}", file=sys.stderr)

    all_initial_guesses = []

    # Define objective and constraint functions once for the optimization loop
    def objective(x_flat):
        circles_opt = x_flat.reshape(n, 3)
        return -np.sum(circles_opt[:, 2])

    def boundary_constraint_func(x_flat):
        circles_opt = x_flat.reshape(n, 3)
        constraints_list = []
        for i in range(n):
            xi, yi, ri = circles_opt[i]
            constraints_list.append(xi - ri)      # x - r >= 0
            constraints_list.append(1 - (xi + ri)) # 1 - (x + r) >= 0
            constraints_list.append(yi - ri)      # y - r >= 0
            constraints_list.append(1 - (yi + ri)) # 1 - (y + r) >= 0
        return np.array(constraints_list)
    
    def overlap_constraint_func(x_flat):
        circles_opt = x_flat.reshape(n, 3)
        constraints_list = []
        for i in range(n):
            for j in range(i + 1, n): # Only check unique pairs
                xi, yi, ri = circles_opt[i]
                xj, yj, rj = circles_opt[j]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist_sq = (ri + rj)**2
                constraints_list.append(dist_sq - min_dist_sq) # Must be >= 0
        return np.array(constraints_list)

    # Define bounds for x, y, r for each circle
    lower_bounds = np.zeros(3 * n)
    upper_bounds = np.ones(3 * n)
    for i in range(n):
        lower_bounds[i * 3 + 2] = MIN_RADIUS_OPT  # r_i >= MIN_RADIUS_OPT
        upper_bounds[i * 3 + 2] = 0.5             # r_i <= 0.5 (max possible radius for a single circle)
    bounds = Bounds(lower_bounds, upper_bounds)

    # Define constraints with analytical Jacobians
    constraints = [
        NonlinearConstraint(boundary_constraint_func, 0, np.inf, jac=lambda x: _boundary_constraint_jac(x, n), hess=None),
        NonlinearConstraint(overlap_constraint_func, 0, np.inf, jac=lambda x: _overlap_constraint_jac(x, n), hess=None)
    ]

    # Phase 1: Collect LLM-generated initial guesses
    for attempt in range(NUM_LLM_ATTEMPTS):
        print(f"\n--- LLM Prompting Attempt {attempt + 1}/{NUM_LLM_ATTEMPTS} ---", file=sys.stderr)
        
        # Enhanced LLM prompt
        prompt = f"""You are an expert at circle packing optimization, focused on maximizing the total sum of radii.
Think step-by-step to arrive at the solution. First, consider the overall dense structure for {n} circles, then refine positions and radii.
Your task is to propose an initial, exceptionally dense arrangement of exactly {n} circles within the unit square [0,1]x[0,1].
This initial configuration is critical for subsequent numerical optimization to achieve the highest possible sum of radii.
The ideal sum for {n} circles is around {TARGET_SUM_RADII:.3f}. Aim for a high sum from the start.

Key strategies for maximizing the sum of radii for N={n} circles:
- **Prioritize Highly Varied Radii**: This is crucial. Use circles of significantly different sizes. Place larger circles in central, open regions to form a dense core, and smaller circles meticulously to fill all interstitial gaps, especially near corners, edges, and between larger circles. Avoid uniform radii; varying sizes is essential for optimal packing.
- **Optimal Dense Structures for N={n}**: For {n} circles, aim for a sophisticated, potentially asymmetric arrangement. A common and effective strategy involves a tightly packed central cluster (e.g., a hexagonal arrangement or a compact square grid), surrounded by additional circles that extend outwards, filling remaining space. Maximize tangencies (contact points) between circles and between circles and the square's boundaries to achieve the highest density. Think about how circles can adaptively fill all available space.
- **Boundary Interaction**: Maximize contact between circles and the square boundaries. Place circles tightly against the edges (e.g., tangent) to utilize space efficiently and allow for larger radii where possible.
- **Precision is Key**: Provide coordinates and radii with high floating-point precision (at least 7-9 decimal places).

Constraints you MUST satisfy:
1.  You MUST provide **exactly {n} circles**. No more, no less. This is critical for parser stability.
2.  All circles must be entirely within the unit square: 0 <= x-r, x+r <= 1 AND 0 <= y-r, y+r <= 1.
3.  No two circles may overlap: the distance between the centers of any two circles (c1 and c2) must be strictly greater than or equal to the sum of their radii (r1 + r2).
4.  All radii must be positive (r > {MIN_RADIUS_OPT:.1e}).
Before finalizing your JSON, mentally (or algorithmically) check if your proposed circles truly satisfy all boundary and non-overlap constraints with high precision. If your response does not strictly adhere to these rules (especially the number of circles and JSON format), it will be discarded or corrected with random small circles, which will significantly reduce the final score.

Your response MUST be a JSON object with a single key "circles", whose value is a list of {n} circle definitions.
Each circle definition must be a list of three floating-point numbers: [x, y, r].

Example of the desired JSON format (for a hypothetical 2 circles, but you must provide {n} circles):
```json
{{
  "circles": [
    [0.250000000, 0.250000000, 0.250000000],
    [0.750000000, 0.750000000, 0.250000000]
  ]
}}
```
A very high-quality and dense initial arrangement with a sum of radii significantly greater than 2.4 (ideally approaching or exceeding {TARGET_SUM_RADII:.3f}) is critical for the subsequent local optimizer to find a near-optimal solution.
"""

        current_initial_circles = []
        try:
            response_text = llm_function(prompt)
            parsed_response = json.loads(response_text)
            
            if "circles" in parsed_response and isinstance(parsed_response["circles"], list):
                llm_circles = []
                for c_data in parsed_response["circles"]:
                    if isinstance(c_data, list) and len(c_data) == 3:
                        try:
                            x, y, r = float(c_data[0]), float(c_data[1]), float(c_data[2])
                            r = max(r, MIN_RADIUS_OPT) # Ensure radii are at least MIN_RADIUS_OPT
                            llm_circles.append([x,y,r])
                        except ValueError:
                            print(f"Malformed circle data skipped: {c_data}", file=sys.stderr)
                            pass
                
                # Adjust circle count: take largest radii if too many, fill with tiny if too few
                if len(llm_circles) > n:
                    llm_circles = sorted(llm_circles, key=lambda c: c[2], reverse=True)[:n]
                    print(f"LLM provided {len(llm_circles)} circles, truncated to {n}.", file=sys.stderr)
                elif len(llm_circles) < n:
                    print(f"LLM provided {len(llm_circles)} circles, expected {n}. Filling missing circles.", file=sys.stderr)
                    # Fill with random valid tiny circles to meet 'n' count
                    for _ in range(n - len(llm_circles)):
                        r_add = MIN_RADIUS_OPT
                        x_add = random.uniform(r_add, 1 - r_add)
                        y_add = random.uniform(r_add, 1 - r_add)
                        llm_circles.append([float(x_add), float(y_add), r_add])
                
                current_initial_circles = llm_circles
                
                is_valid, msg = _is_valid_packing(current_initial_circles, n)
                if not is_valid:
                    print(f"LLM-generated initial packing for attempt {attempt+1} is invalid ({msg}), generating random valid circles for this attempt.", file=sys.stderr)
                    current_initial_circles = _generate_random_valid_circles(n)
                else:
                    print(f"LLM-generated initial packing for attempt {attempt+1} is valid. Sum radii: {_calculate_sum_radii(current_initial_circles):.5f}", file=sys.stderr)

            else:
                print(f"LLM response malformed (missing 'circles' key or not a list) for attempt {attempt+1}, generating random valid circles.", file=sys.stderr)
                current_initial_circles = _generate_random_valid_circles(n)

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Failed to parse LLM response for attempt {attempt+1}: {e}. Generating random valid circles.", file=sys.stderr)
            current_initial_circles = _generate_random_valid_circles(n)
        except Exception as e:
            print(f"An unexpected error occurred during LLM call or processing for attempt {attempt+1}: {e}. Generating random valid circles.", file=sys.stderr)
            current_initial_circles = _generate_random_valid_circles(n)
        
        # Final check to ensure current_initial_circles always has 'n' items before optimization
        if len(current_initial_circles) != n:
            print(f"Warning: Initial circles count mismatch ({len(current_initial_circles)} vs {n}) after LLM processing for attempt {attempt+1}, force-generating random valid circles.", file=sys.stderr)
            current_initial_circles = _generate_random_valid_circles(n)
        
        all_initial_guesses.append(current_initial_circles)

    # Add purely random initial guesses
    for attempt in range(NUM_RANDOM_STARTS):
        print(f"\n--- Random Start Attempt {attempt + 1}/{NUM_RANDOM_STARTS} ---", file=sys.stderr)
        all_initial_guesses.append(_generate_random_valid_circles(n))

    # Phase 2: Run SLSQP on all initial guesses (LLM and random)
    print(f"\n--- Initial SLSQP Optimization Phase ({len(all_initial_guesses)} runs) ---", file=sys.stderr)
    for idx, initial_circles_for_opt in enumerate(all_initial_guesses):
        # Perturb the initial circles slightly to help the optimizer explore
        perturbed_circles = []
        for x,y,r in initial_circles_for_opt:
            px = x + random.uniform(-PERTURBATION_MAGNITUDE_XY, PERTURBATION_MAGNITUDE_XY)
            py = y + random.uniform(-PERTURBATION_MAGNITUDE_XY, PERTURBATION_MAGNITUDE_XY)
            pr = r + random.uniform(-PERTURBATION_MAGNITUDE_R, PERTURBATION_MAGNITUDE_R)
            
            # Ensure perturbed values are within reasonable variable bounds before passing to optimizer
            pr = np.clip(pr, MIN_RADIUS_OPT, 0.5)
            px = np.clip(px, pr, 1-pr) # Clip x, y relative to their new radius
            py = np.clip(py, pr, 1-pr)
            
            perturbed_circles.append([px, py, pr])
        
        initial_guess_flattened = np.array(perturbed_circles).flatten()

        result = minimize(objective, initial_guess_flattened, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': OPTIMIZER_MAXITER_SLSQP, 'ftol': OPTIMIZER_FTOL, 'disp': False, 'xtol': OPTIMIZER_XTOL})
        
        optimized_x = result.x.reshape(n, 3)
        current_final_circles = []
        for c_data in optimized_x:
            x, y, r = c_data
            # Post-optimization clamping as a final safety
            r = max(r, MIN_RADIUS_OPT)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            current_final_circles.append([float(x), float(y), float(r)])
            
        is_valid_final, final_msg = _is_valid_packing(current_final_circles, n)
        current_sum_radii = _calculate_sum_radii(current_final_circles)

        if is_valid_final:
            print(f"SLSQP Run {idx+1} - Optimized packing valid. Sum radii: {current_sum_radii:.5f}", file=sys.stderr)
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = current_final_circles
                print(f"New best sum_radii found: {best_sum_radii:.5f}", file=sys.stderr)
        else:
            print(f"SLSQP Run {idx+1} - Optimized packing is invalid ({final_msg}). Skipping this result.", file=sys.stderr)

    # Phase 3: Apply basinhopping to the best solution found so far for global exploration
    print(f"\n--- Basinhopping Global Optimization Phase ---", file=sys.stderr)
    
    # Use the best circles found so far as the starting point for basinhopping
    initial_basinhopping_guess = np.array(best_circles).flatten()

    # Minimizer arguments for basinhopping (local optimizer)
    minimizer_kwargs = {
        "method": 'SLSQP',
        "bounds": bounds,
        "constraints": constraints,
        "options": {'maxiter': OPTIMIZER_MAXITER_BASIN, 'ftol': OPTIMIZER_FTOL / 10, 'disp': False, 'xtol': OPTIMIZER_XTOL / 10}
    }

    bh_result = basinhopping(
        objective,
        initial_basinhopping_guess,
        niter=BASINHOPPING_NITER,
        T=BASINHOPPING_T,
        stepsize=BASINHOPPING_STEPSIZE,
        minimizer_kwargs=minimizer_kwargs,
        disp=False # Set to True for detailed basinhopping output
    )
    
    # Extract the best result from basinhopping
    bh_optimized_x = bh_result.x.reshape(n, 3)
    bh_final_circles = []
    for c_data in bh_optimized_x:
        x, y, r = c_data
        r = max(r, MIN_RADIUS_OPT)
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        bh_final_circles.append([float(x), float(y), float(r)])

    is_valid_bh, bh_msg = _is_valid_packing(bh_final_circles, n)
    bh_sum_radii = _calculate_sum_radii(bh_final_circles)

    if is_valid_bh and bh_sum_radii > best_sum_radii:
        best_sum_radii = bh_sum_radii
        best_circles = bh_final_circles
        print(f"Basinhopping found an improved sum_radii: {best_sum_radii:.5f}", file=sys.stderr)
    elif not is_valid_bh:
        print(f"Basinhopping resulted in an invalid packing ({bh_msg}). Keeping previous best.", file=sys.stderr)
    else:
        print(f"Basinhopping did not improve upon the best local optimum ({bh_sum_radii:.5f}). Keeping previous best.", file=sys.stderr)

    return best_circles

if __name__ == "__main__":
    # Configure Gemini LLM for local testing or use a dummy fallback
    llm_function_to_use = None
    try:
        import google.generativeai as genai
        # API key should be set in environment variable GOOGLE_API_KEY
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
        model = genai.GenerativeModel('gemini-pro')

        def gemini_llm_call(prompt):
            response = model.generate_content(prompt)
            # Access the text content from the response, handling potential empty responses
            if response and response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            return "" # Return empty string if no content
        
        llm_function_to_use = gemini_llm_call
        print("Using Google Gemini for LLM calls.", file=sys.stderr)
    except Exception as e:
        print(f"Could not configure Google Gemini (Error: {e}). Using dummy LLM for testing.", file=sys.stderr)
        # Fallback to a dummy LLM if API key or library is not configured
        def dummy_llm_call(prompt):
            print("Dummy LLM called. Returning a simple random valid configuration.", file=sys.stderr)
            dummy_circles = _generate_random_valid_circles(N_CIRCLES)
            return json.dumps({"circles": dummy_circles})
        llm_function_to_use = dummy_llm_call


    circles = solve_packing(llm_function_to_use, n=N_CIRCLES)
    
    # Final validation of the output before printing
    is_valid, validation_msg = _is_valid_packing(circles, N_CIRCLES)
    if not is_valid:
        print(f"Warning: Final optimized packing is invalid: {validation_msg}. Reverting to a known valid baseline.", file=sys.stderr)
        circles = _generate_random_valid_circles(N_CIRCLES)
        is_valid, validation_msg = _is_valid_packing(circles, N_CIRCLES)
        if not is_valid:
            print(f"Critical Error: Random valid circle generation is also invalid: {validation_msg}. Outputting potentially invalid data.", file=sys.stderr)
            
    result = {
        "circles": circles,
        "sum_radii": float(_calculate_sum_radii(circles))
    }

    print(json.dumps(result))