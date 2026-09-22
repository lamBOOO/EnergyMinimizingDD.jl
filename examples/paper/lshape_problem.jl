# Manufactured non-monotone L-domain problem used by the final paper.

using SpecialFunctions

const REACTION_AMPLITUDE = 12.0
const MESH_GRADING = 0.4
const MIDPOINT_QUADRATURE = Gridap.ReferenceFEs.GenericQuadrature(
  [Gridap.Point(0.5, 0.0), Gridap.Point(0.0, 0.5), Gridap.Point(0.5, 0.5)],
  fill(1 / 6, 3),
  "triangle edge-midpoint rule",
)

function exact_solution(x)
  x1, x2 = x[1], x[2]
  radius_squared = x1^2 + x2^2
  iszero(radius_squared) && return 0.0
  return 2 * x1 * x2 * (1 - x1^2) * (1 - x2^2) * radius_squared^(-2 / 3)
end

function minus_laplacian_exact(x)
  x1, x2 = x[1], x[2]
  radius_squared = x1^2 + x2^2
  iszero(radius_squared) && return 0.0
  polynomial = 2 * x1 * x2 * (1 - x1^2) * (1 - x2^2)
  derivative_x1 = 2 * (1 - 3x1^2) * x2 * (1 - x2^2)
  derivative_x2 = 2 * x1 * (1 - x1^2) * (1 - 3x2^2)
  laplacian_polynomial =
    -12x1 * x2 * (1 - x2^2) - 12x2 * x1 * (1 - x1^2)
  return -radius_squared^(-2 / 3) * laplacian_polynomial +
         (8 / 3) * radius_squared^(-5 / 3) *
         (x1 * derivative_x1 + x2 * derivative_x2) -
         (16 / 9) * polynomial * radius_squared^(-5 / 3)
end
