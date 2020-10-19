from scipy.optimize import linprog
import math

# Uses this tutorial: https://realpython.com/linear-programming-python/
# Ultimately uses the following math:
#  1. An equation for two points can be written for each camera from the camera to the object:
#       y1 = tan(angle) * (x1 - x2) + y2
#  2. We add an error term to each equation so we can take into account more cameras than we strictly need.
#  3. We minimize the sum of those error terms.
#
# The angle must be the absolute angle, so it needs to take into account the angle the camera is positioned at AND the
#  angle measured by the camera.


class Triangulator:
    def __init__(self):
        pass

    def calculate_relative(self, inputs):
        """
        Calculates an optimal object location based on multiple camera measurements in relative coordinates.

        Expects inputs as:
        {
          "cx": camera x coordinate
          "cy": camera y coordinate
          "ca": camera angle in degrees (absolute)
          "ma": measured angle relative to the camera's angle in degrees
        }

        :param inputs: See above
        :return: (x, y, error) for the point.
        """

        absolute_inputs = [
            {
                "cx": input["cx"],
                "cy": input["cy"],
                "a": math.tan(math.radians(input["ca"]) + math.radians(input["ma"])),
            }
            for input in inputs
        ]

        return self.calculate_absolute(absolute_inputs)

    def calculate_absolute(self, inputs):
        """
        Calculates an optimal object location based on multiple camera measurements in absolute coordinates.

        Expects inputs as:
        {
          "cx": camera x coordinate
          "cy": camera y coordinate
          "a": tangent of the camera's measurement angle (absolute viewing angle in radians)
        }

        :param inputs: See above
        :return: (x, y, error) for the point.
        """

        # x, y, [error term for each camera]
        minimizer = [0, 0] + [1]*len(inputs)
        lhs_eq = []
        rhs_eq = []

        for idx, input in enumerate(inputs):
            lhs_eq_for_input = [-1*input["a"], 1] + [0]*len(inputs)
            lhs_eq_for_input[idx + 2] = -1
            lhs_eq.append(lhs_eq_for_input)

            rhs_eq.append(
                -1 * input["a"] * input["cx"] + input["cy"]
            )

        opt = linprog(
            c=minimizer,
            A_ub=None,
            b_ub=None,
            A_eq=lhs_eq,
            b_eq=rhs_eq,
            bounds=[],
            method="simplex"
        )

        # Solution array is [x, y, ..error components..]
        solution = opt.x
        error = sum([x for idx, x in enumerate(solution) if idx > 1])
        return solution[1], solution[0], error


inputs_relative = [
    {
        "cx": 1,
        "cy": 0,
        "ca": 45,
        "ma": 30,
    },
    {
        "cx": 4,
        "cy": 0,
        "ca": 0,
        "ma": 30,
    },
    {
        "cx": 7,
        "cy": 0,
        "ca": -45,
        "ma": 0,
    },
]

t = Triangulator()
x, y, error = t.calculate_relative(inputs=inputs_relative)
print("Solution: " + str(x) + ", " + str(y) + " with error: " + str(error))