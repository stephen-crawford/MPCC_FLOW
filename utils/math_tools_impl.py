import random

import casadi as cd
import numpy as np
from scipy.stats import chi2

from utils.utils import LOG_WARN, LOG_DEBUG, LOG_INFO


###### General math utilities ######

def distance(a, b):
    """
Calculate Euclidean distance between two points.
"""
    LOG_DEBUG(f"math_tools.distance: Computing distance between {type(a).__name__} and {type(b).__name__}")
    a_array = np.array(a)
    b_array = np.array(b)
    if a_array.shape != b_array.shape:
        LOG_WARN(f"math_tools.distance: Shape mismatch: {a_array.shape} vs {b_array.shape}")
    result = np.sqrt(np.sum((a_array - b_array) ** 2))
    LOG_DEBUG(f"math_tools.distance: Result = {result}")
    return result


def safe_norm(x, y):
    norm = cd.sqrt(x ** 2 + y ** 2)
    # Add a small epsilon to prevent division by zero
    safe_norm = cd.fmax(norm, 1e-6)
    return safe_norm


def linspace(start, end, num):
    """
Create an evenly spaced array of numbers.
"""
    if num == 0:
        return []
    if num == 1:
        return [end]
    if num == 2:
        return [start, end]

    delta = (end - start) / (num - 1)
    result = [start]
    for i in range(1, num - 1):
        result.append(start + delta * i)
    result.append(end)
    return result


def rotation_matrix(angle):
    return np.array([[cd.cos(angle), -cd.sin(angle)],
                    [cd.sin(angle), cd.cos(angle)]])

def casadi_rotation_matrix(theta):
    return cd.vertcat(
        cd.horzcat(cd.cos(theta), -cd.sin(theta)),
        cd.horzcat(cd.sin(theta),  cd.cos(theta))
    )

def angular_difference(angle1, angle2):
    """
Calculate the shortest angular difference between two angles.
"""
    diff = math.fmod(angle2 - angle1, 2 * math.pi)

    if diff > math.pi:
        diff -= 2 * math.pi  # Shift difference to be within [-pi, pi]
    elif diff < -math.pi:
        diff += 2 * math.pi

    return diff


def bisection(low, high, func, tol):
    """
Find a root of a function using the bisection method.
"""
    if low > high:
        raise RuntimeError("Bisection low value was higher than the high value!")

    value_low = func(low)

    for iterations in range(1000):
        mid = (low + high) / 2.0
        value_mid = func(mid)

        if abs(value_mid) < tol or (high - low) / 2.0 < tol:
            return mid

        # Extra check for integers
        if high - low == 1:
            return high

        if np.sign(value_mid) == np.sign(value_low):
            low = mid
            value_low = value_mid
        else:
            high = mid

    raise RuntimeError("Bisection failed!")


def sgn(val):
    """
Return the sign of a value.
"""
    return 1 if val > 0 else -1 if val < 0 else 0


def chi_square_quantile(dof: int, alpha: float) -> float:
    """Compute chi-square quantile for given degrees of freedom and confidence level."""
    return chi2.ppf(alpha, dof)


def haar_difference_without_abs(a, b):
    """Calculate the smallest angle between two angles in a way compatible with CasADi symbolics."""
    diff = a - b
    # Use casadi's mod function for symbolic compatibility instead of % operator
    return diff - 2 * cd.pi * cd.floor((diff + cd.pi) / (2 * cd.pi))

def angle_to_quaternion(angle):
    """
Convert a yaw angle to a quaternion.
"""
    # In Python, we can use scipy or transformations library, but here's a basic implementation
    cy = math.cos(angle * 0.5)
    sy = math.sin(angle * 0.5)
    result = {
        'x': 0.0,
        'y': 0.0,
        'z': sy,
        'w': cy
    }
    return result

def quaternion_to_angle(pose_or_quaternion):
    """
Extract yaw angle from a quaternion or pose with quaternion.
"""
    # Determine if input is a pose or quaternion
    if hasattr(pose_or_quaternion, 'orientation'):  # It's a pose
        q = pose_or_quaternion.orientation
    else:  # It's a quaternion
        q = pose_or_quaternion

    # Extract quaternion components
    if hasattr(q, 'y'):  # Object with attributes
        ysqr = q.y * q.y
        t3 = 2.0 * (q.w * q.z + q.x * q.y)
        t4 = 1.0 - 2.0 * (ysqr + q.z * q.z)
    else:  # Dictionary-like
        ysqr = q['y'] * q['y']
        t3 = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
        t4 = 1.0 - 2.0 * (ysqr + q['z'] * q['z'])

    return math.atan2(t3, t4)

class BandMatrix:
    """Band matrix implementation with LU decomposition for solving linear systems."""

    def __init__(self, dim=0, n_upper=0, n_lower=0):
        """Initialize a band matrix with given dimensions.

        Args:
            dim (int): Matrix dimension
            n_upper (int): Number of upper diagonals
            n_lower (int): Number of lower diagonals
        """
        if dim > 0:
            self.resize(dim, n_upper, n_lower)
        else:
            self.m_upper = []
            self.m_lower = []

    def resize(self, dim, n_upper, n_lower):
        """Resize the band matrix.

        Args:
            dim (int): Matrix dimension
            n_upper (int): Number of upper diagonals
            n_lower (int): Number of lower diagonals
        """
        assert dim > 0
        assert n_upper >= 0
        assert n_lower >= 0

        self.m_upper = [np.zeros(dim) for _ in range(n_upper + 1)]
        self.m_lower = [np.zeros(dim) for _ in range(n_lower + 1)]

    def dim(self):
        """Get the dimension of the matrix."""
        if len(self.m_upper) > 0:
            return len(self.m_upper[0])
        else:
            return 0

    def num_upper(self):
        """Get the number of upper diagonals."""
        return len(self.m_upper) - 1

    def num_lower(self):
        """Get the number of lower diagonals."""
        return len(self.m_lower) - 1

    def get_element(self, i, j):
        """Get the element at position (i, j)."""
        k = j - i  # which band is the entry
        assert 0 <= i < self.dim() and 0 <= j < self.dim()
        assert -self.num_lower() <= k <= self.num_upper()

        # k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if k >= 0:
            return self.m_upper[k][i]
        else:
            return self.m_lower[-k][i]

    def set_element(self, i, j, value):
        """Set the element at position (i, j)."""
        k = j - i  # which band is the entry
        assert 0 <= i < self.dim() and 0 <= j < self.dim()
        assert -self.num_lower() <= k <= self.num_upper()

        # k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if k >= 0:
            self.m_upper[k][i] = value
        else:
            self.m_lower[-k][i] = value

    def get_saved_diag(self, i):
        """Get the saved diagonal element."""
        assert 0 <= i < self.dim()
        return self.m_lower[0][i]

    def set_saved_diag(self, i, value):
        """Set the saved diagonal element."""
        assert 0 <= i < self.dim()
        self.m_lower[0][i] = value

    def lu_decompose(self):
        """LU decomposition of a band matrix."""
        n = self.dim()

        # Preconditioning: normalize column i so that a_ii=1
        for i in range(n):
            assert self.get_element(i, i) != 0.0
            self.set_saved_diag(i, 1.0 / self.get_element(i, i))

            j_min = max(0, i - self.num_lower())
            j_max = min(n - 1, i + self.num_upper())

            for j in range(j_min, j_max + 1):
                self.set_element(i, j, self.get_element(i, j) * self.get_saved_diag(i))

            self.set_element(i, i, 1.0)  # prevents rounding errors

        # Gauss LR-Decomposition
        for k in range(n):
            i_max = min(n - 1, k + self.num_lower())

            for i in range(k + 1, i_max + 1):
                assert self.get_element(k, k) != 0.0
                x = -self.get_element(i, k) / self.get_element(k, k)
                self.set_element(i, k, -x)  # assembly part of L

                j_max = min(n - 1, k + self.num_upper())
                for j in range(k + 1, j_max + 1):
                    # assembly part of R
                    new_val = self.get_element(i, j) + x * self.get_element(k, j)
                    self.set_element(i, j, new_val)

    def l_solve(self, b):
        """Solve Ly=b."""
        assert self.dim() == len(b)
        n = self.dim()
        x = np.zeros(n)

        for i in range(n):
            sum_val = 0
            j_start = max(0, i - self.num_lower())

            for j in range(j_start, i):
                sum_val += self.get_element(i, j) * x[j]

            x[i] = (b[i] * self.get_saved_diag(i)) - sum_val

        return x

    def r_solve(self, b):
        """Solve Rx=y."""
        assert self.dim() == len(b)
        n = self.dim()
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            sum_val = 0
            j_stop = min(n - 1, i + self.num_upper())

            for j in range(i + 1, j_stop + 1):
                sum_val += self.get_element(i, j) * x[j]

            x[i] = (b[i] - sum_val) / self.get_element(i, i)

        return x

    def lu_solve(self, b, is_lu_decomposed=False):
        """Solve Ax=b using LU decomposition."""
        assert self.dim() == len(b)

        if not is_lu_decomposed:
            self.lu_decompose()

        y = self.l_solve(b)
        x = self.r_solve(y)

        return x

##### Geometry classes #####

class Halfspace:
    def __init__(self, A: np.ndarray, b: float):
        """Halfspace defined by Ax <= b"""
        self.A = A
        self.b = b


class Clothoid2D:
    """2D clothoid implementation"""

    def __init__(self, waypoints_x, waypoints_y, waypoints_angle, sample_distance):
        self._length = 0.0
        self._x = []
        self._y = []
        self._s = []
        self.fit_clothoid(waypoints_x, waypoints_y, waypoints_angle, sample_distance)

    def get_points_on_clothoid(self):
        """Get points on the clothoid"""
        return self._x, self._y, self._s

    def get_length(self):
        """Get total length of the clothoid"""
        return self._length

    def fit_clothoid(self, waypoints_x, waypoints_y, waypoints_angle, sample_distance):
        """Fit a clothoid to the waypoints"""
        # Note: This is a simplified implementation
        # In a full implementation, you would need the actual Clothoid::buildClothoid function
        # which involves solving for clothoid parameters (curvature, curvature rate)

        self._s = [0.0]

        n_waypoints = len(waypoints_x)

        for i in range(n_waypoints - 1):
            # Here we would normally build a clothoid
            # For now, linear approximation (this is not a true clothoid)
            x1, y1, angle1 = waypoints_x[i], waypoints_y[i], waypoints_angle[i]
            x2, y2, angle2 = waypoints_x[i + 1], waypoints_y[i + 1], waypoints_angle[i + 1]

            # Calculate segment length (simplified)
            dx, dy = x2 - x1, y2 - y1
            L = np.sqrt(dx * dx + dy * dy)

            # Sample points along this segment
            n_clothoid = max(int(np.ceil(L / sample_distance)), 2)
            X = []
            Y = []

            for j in range(n_clothoid):
                t = j / (n_clothoid - 1)
                # Linear interpolation (this is NOT a true clothoid)
                X.append(x1 + t * dx)
                Y.append(y1 + t * dy)

            self._length += L

            # Add points to the overall clothoid
            if i == 0:
                # Add all points for the first segment
                self._x.extend(X)
                self._y.extend(Y)
            else:
                # Skip duplicate initial point for later segments
                self._x.extend(X[1:])
                self._y.extend(Y[1:])

            # Add distances
            for j in range(1, n_clothoid):
                self._s.append(self._s[-1] + L / (n_clothoid - 1))




###### Decomposition Utils ######

"""
Decomposition Library - Python Port
A geometric decomposition library for path planning using ellipsoids and polyhedra.
Converted from C++ to Python using NumPy for linear algebra operations.
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Constants and type aliases
EPSILON = 1e-10


# ANSI color codes for printing
class Colors:
    RED = "\x1b[1;31m"
    GREEN = "\x1b[1;32m"
    YELLOW = "\x1b[1;33m"
    BLUE = "\x1b[1;34m"
    MAGENTA = "\x1b[1;35m"
    CYAN = "\x1b[1;36m"
    RESET = "\x1b[0m"


# Type aliases for clarity
Vec2f = np.ndarray  # shape (2,)
Vec3f = np.ndarray  # shape (3,)
Mat2f = np.ndarray  # shape (2, 2)
Mat3f = np.ndarray  # shape (3, 3)
Quatf = np.ndarray  # quaternion as [w, x, y, z]


class DataUtils:
    """Utility functions for data manipulation"""

    @staticmethod
    def transform_vec(vectors: List[np.ndarray], transform: np.ndarray) -> List[np.ndarray]:
        """Transform a list of vectors using given transformation matrix"""
        return [transform @ v for v in vectors]

    @staticmethod
    def total_distance(vectors: List[np.ndarray]) -> float:
        """Calculate total distance along a path"""
        if len(vectors) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(1, len(vectors)):
            total_dist += np.linalg.norm(vectors[i] - vectors[i - 1])
        return total_dist


@dataclass
class Hyperplane:
    """Hyperplane class defined by a point and normal vector"""
    p: np.ndarray  # Point on the plane
    n: np.ndarray  # Normal vector (directional)

    def signed_dist(self, pt: np.ndarray) -> float:
        """Calculate signed distance from point to hyperplane"""
        return np.dot(self.n, pt - self.p)

    def dist(self, pt: np.ndarray) -> float:
        """Calculate absolute distance from point to hyperplane"""
        return abs(self.signed_dist(pt))


class Polyhedron:
    """Polyhedron class defined by hyperplanes"""

    def __init__(self, hyperplanes: Optional[List[Hyperplane]] = None):
        self.hyperplanes_ = hyperplanes or []

    def add(self, hyperplane: Hyperplane):
        """Add a hyperplane to the polyhedron"""
        self.hyperplanes_.append(hyperplane)

    def inside(self, pt: np.ndarray) -> bool:
        """Check if point is inside polyhedron (non-exclusive)"""
        for hp in self.hyperplanes_:
            if hp.signed_dist(pt) > EPSILON:
                return False
        return True

    def points_inside(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Filter points that are inside the polyhedron"""
        return [pt for pt in points if self.inside(pt)]

    def hyperplanes(self) -> List[Hyperplane]:
        """Get the hyperplane array"""
        return self.hyperplanes_

    def cal_normals(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Calculate normals for visualization"""
        return [(hp.p, hp.n) for hp in self.hyperplanes_]

    def __str__(self):
        return f"Polyhedron with {str(self.hyperplanes())}"

    def __repr__(self):
        return self.__str__()


class LinearConstraint:
    """Linear constraint representation as Ax <= b"""

    def __init__(self, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None,
                 p0: Optional[np.ndarray] = None, hyperplanes: Optional[List[Hyperplane]] = None):
        if A is not None and b is not None:
            self.A_ = A
            self.b_ = b
        elif p0 is not None and hyperplanes is not None:
            self._construct_from_hyperplanes(p0, hyperplanes)
        else:
            self.A_ = np.array([])
            self.b_ = np.array([])

    def _construct_from_hyperplanes(self, p0: np.ndarray, hyperplanes: List[Hyperplane]):
        """Construct from inside point and hyperplanes"""
        size = len(hyperplanes)
        dim = len(p0)
        A = np.zeros((size, dim))
        b = np.zeros(size)

        for i, hp in enumerate(hyperplanes):
            n = hp.n.copy()
            c = np.dot(hp.p, n)
            if np.dot(n, p0) - c > 0:
                n = -n
                c = -c
            A[i] = n
            b[i] = c

        self.A_ = A
        self.b_ = b

    def inside(self, pt: np.ndarray) -> bool:
        """Check if point satisfies constraints"""
        d = self.A_ @ pt - self.b_
        return np.all(d <= 0)

    def A(self) -> np.ndarray:
        return self.A_

    def b(self) -> np.ndarray:
        return self.b_

    def __str__(self):
        return f"LinearConstraint with A: {self.A_}, b: {self.b_}"

    def __repr__(self):
        return self.__str__()


class Ellipsoid:
    """Ellipsoid class defined by matrix C and center d"""

    def __init__(self, C: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None):
        self.C_ = C if C is not None else np.eye(2)
        self.d_ = d if d is not None else np.zeros(2)

    def dist(self, pt: np.ndarray) -> float:
        """Calculate distance to center (normalized by ellipsoid)"""
        LOG_DEBUG("Trying to calculate distance for ellipsoid with C: {}, and d: {}".format(self.C_, self.d_))
        try:
            C_inv = np.linalg.inv(self.C_)
        except np.linalg.LinAlgError:
            return 100

        return np.linalg.norm(C_inv @ (pt - self.d_))

    def inside(self, pt: np.ndarray) -> bool:
        """Check if point is inside ellipsoid (non-exclusive)"""
        return self.dist(pt) <= 1.0

    def points_inside(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Filter points inside ellipsoid"""
        return [pt for pt in points if self.inside(pt)]

    def closest_point(self, points: List[np.ndarray]) -> np.ndarray:
        """Find closest point from a list"""
        if not points:
            return np.zeros_like(self.d_)

        min_dist = float('inf')
        closest_pt = points[0]

        for pt in points:
            d = self.dist(pt)
            if d < min_dist:
                min_dist = d
                closest_pt = pt

        return closest_pt

    def closest_hyperplane(self, points: List[np.ndarray]) -> Hyperplane:
        """Find closest hyperplane from closest point"""
        closest_pt = self.closest_point(points)
        C_inv = np.linalg.inv(self.C_)
        n = C_inv @ C_inv.T @ (closest_pt - self.d_)
        n_normalized = n / np.linalg.norm(n)
        return Hyperplane(closest_pt, n_normalized)

    def sample_2d(self, num: int) -> List[np.ndarray]:
        """Sample points along 2D ellipsoid contour"""
        if len(self.d_) != 2:
            raise ValueError("sample_2d only works for 2D ellipsoids")

        points = []
        dyaw = 2 * math.pi / num

        for i in range(num):
            yaw = i * dyaw
            pt = np.array([math.cos(yaw), math.sin(yaw)])
            points.append(self.C_ @ pt + self.d_)

        return points

    def volume(self) -> float:
        """Get ellipsoid volume (determinant of C)"""
        return np.linalg.det(self.C_)

    def C(self) -> np.ndarray:
        return self.C_

    def d(self) -> np.ndarray:
        return self.d_

    def __str__(self):
        return f"Ellipsoid with C: {self.C_}, d: {self.d_}"

    def __repr__(self):
        return self.__str__()

class GeometricUtils:
    """Geometric utility functions"""

    @staticmethod
    def eigen_values(A: np.ndarray) -> np.ndarray:
        """Calculate eigenvalues of matrix"""
        return np.linalg.eigvals(A)

    @staticmethod
    def vec2_to_rotation(v: np.ndarray) -> np.ndarray:
        """Calculate 2D rotation matrix from vector"""
        yaw = math.atan2(v[1], v[0])
        return np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])

    @staticmethod
    def vec3_to_rotation(v: np.ndarray) -> np.ndarray:
        """Calculate 3D rotation matrix from vector (zero roll)"""
        rpy = np.array([0,
                        math.atan2(-v[2], np.linalg.norm(v[:2])),
                        math.atan2(v[1], v[0])])

        # Convert RPY to rotation matrix
        roll, pitch, yaw = rpy

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])

        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])

        return Rz @ Ry @ Rx

    @staticmethod
    def sort_pts_2d(pts: List[np.ndarray]) -> List[np.ndarray]:
        """Sort 2D points in counter-clockwise order"""
        if not pts:
            return pts

        # Calculate center
        avg = np.mean(pts, axis=0)

        # Sort by angle
        def angle_key(pt):
            return math.atan2(pt[1] - avg[1], pt[0] - avg[0])

        return sorted(pts, key=angle_key)

    @staticmethod
    def line_intersect_2d(line1: Tuple[np.ndarray, np.ndarray],
                          line2: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, np.ndarray]:
        """Find intersection between two 2D lines"""
        v1, p1 = line1
        v2, p2 = line2

        a1 = -v1[1]
        b1 = v1[0]
        c1 = a1 * p1[0] + b1 * p1[1]

        a2 = -v2[1]
        b2 = v2[0]
        c2 = a2 * p2[0] + b2 * p2[1]

        det = a1 * b2 - a2 * b1
        if abs(det) < EPSILON:
            return False, np.zeros(2)

        x = (c1 * b2 - c2 * b1) / det
        y = (c1 * a2 - c2 * a1) / det

        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return False, np.zeros(2)

        return True, np.array([x, y])


class DecompBase(ABC):
    """Base class for decomposition algorithms"""

    def __init__(self):
        self.obs_ = []
        self.ellipsoid_ = Ellipsoid()
        self.polyhedron_ = Polyhedron()
        self.local_bbox_ = np.zeros(2)  # Will be resized based on dimension

    def set_local_bbox(self, bbox: np.ndarray):
        """Set local bounding box"""
        self.local_bbox_ = bbox.copy()

    def set_obs(self, obs: List[np.ndarray]):
        """Set obstacle points (filtered by local bbox)"""
        # Create polyhedron from local bbox and filter points
        poly = Polyhedron()
        self.add_local_bbox(poly)
        self.obs_ = poly.points_inside(obs)

    def get_obs(self) -> List[np.ndarray]:
        return self.obs_

    def get_ellipsoid(self) -> Ellipsoid:
        return self.ellipsoid_

    def get_polyhedron(self) -> Polyhedron:
        return self.polyhedron_

    @abstractmethod
    def dilate(self, radius: float = 0):
        """Inflate the shape"""
        pass

    @abstractmethod
    def add_local_bbox(self, polyhedron: Polyhedron):
        """Add local bounding box constraints"""
        pass

    def find_polyhedron(self):
        """Find polyhedron from ellipsoid and obstacles"""
        poly = Polyhedron()
        obs_remain = self.obs_.copy()

        while obs_remain:
            hyperplane = self.ellipsoid_.closest_hyperplane(obs_remain)
            poly.add(hyperplane)

            # Filter points behind the hyperplane
            obs_tmp = []
            for pt in obs_remain:
                if hyperplane.signed_dist(pt) < 0:
                    obs_tmp.append(pt)
            obs_remain = obs_tmp

        self.polyhedron_ = poly


class LineSegment(DecompBase):
    """Line segment decomposition"""

    def __init__(self, p1: Optional[np.ndarray] = None, p2: Optional[np.ndarray] = None):
        super().__init__()
        self.p1_ = p1 if p1 is not None else np.zeros(2)
        self.p2_ = p2 if p2 is not None else np.zeros(2)
        self.local_bbox_ = np.zeros(len(self.p1_))

    def dilate(self, radius: float = 0):
        """Dilate the line segment"""

        if np.linalg.norm(self.p1_ - self.p2_) < 1e-4:
            # Degenerate segment: use dummy ellipsoid and polyhedron
            dummy_C = np.zeros((2, 2))
            dummy_d = self.p1_.copy()
            self.ellipsoid_ = Ellipsoid(dummy_C, dummy_d)

            dummy_poly = Polyhedron()
            dummy_poly.add(Hyperplane(self.p1_, np.array([1.0, 0.0])))  # a1 = 1
            dummy_poly.add(Hyperplane(self.p1_, np.array([-1.0, 0.0])))  # a1 = -1
            dummy_poly.add(Hyperplane(self.p1_, np.array([0.0, 1.0])))  # a2 = 1
            dummy_poly.add(Hyperplane(self.p1_, np.array([0.0, -1.0])))  # a2 = -1
            self.polyhedron_ = dummy_poly
            return

        self._find_ellipsoid(radius)
        self.find_polyhedron()
        self.add_local_bbox(self.polyhedron_)

    def get_line_segment(self) -> List[np.ndarray]:
        """Get the line segment endpoints"""
        return [self.p1_, self.p2_]

    def add_local_bbox(self, polyhedron: Polyhedron):
        """Add local bounding box constraints"""
        if np.linalg.norm(self.local_bbox_) == 0:
            return

        # Direction along the line
        direction = self.p2_ - self.p1_
        if np.linalg.norm(direction) == 0:
            return
        direction = direction / np.linalg.norm(direction)

        # Perpendicular direction
        if len(self.p1_) == 2:
            dir_h = np.array([-direction[1], direction[0]])
        else:  # 3D case
            # Find perpendicular vector
            if abs(direction[2]) < 0.9:
                dir_h = np.cross(direction, np.array([0, 0, 1]))
            else:
                dir_h = np.cross(direction, np.array([1, 0, 0]))
            dir_h = dir_h / np.linalg.norm(dir_h)

        # Add constraints along width (perpendicular to line)
        if len(self.local_bbox_) > 1:
            pp1 = self.p1_ + dir_h * self.local_bbox_[1]
            pp2 = self.p1_ - dir_h * self.local_bbox_[1]
            polyhedron.add(Hyperplane(pp1, dir_h))
            polyhedron.add(Hyperplane(pp2, -dir_h))

        # Add constraints along length
        if len(self.local_bbox_) > 0:
            pp3 = self.p2_ + direction * self.local_bbox_[0]
            pp4 = self.p1_ - direction * self.local_bbox_[0]
            polyhedron.add(Hyperplane(pp3, direction))
            polyhedron.add(Hyperplane(pp4, -direction))

        # Add constraints along height (3D only)
        if len(self.p1_) == 3 and len(self.local_bbox_) > 2:
            dir_v = np.cross(direction, dir_h)
            pp5 = self.p1_ + dir_v * self.local_bbox_[2]
            pp6 = self.p1_ - dir_v * self.local_bbox_[2]
            polyhedron.add(Hyperplane(pp5, dir_v))
            polyhedron.add(Hyperplane(pp6, -dir_v))

    def _find_ellipsoid(self, offset_x: float):
        """Find ellipsoid for line segment"""
        f = np.linalg.norm(self.p1_ - self.p2_) / 2
        dim = len(self.p1_)

        # Initial ellipsoid
        C = f * np.eye(dim)
        axes = np.full(dim, f)
        C[0, 0] += offset_x
        axes[0] += offset_x

        # Scale if needed
        if axes[0] > 0:
            ratio = axes[1] / axes[0]
            axes *= ratio
            C *= ratio

        # Rotation matrix
        if dim == 2:
            R = GeometricUtils.vec2_to_rotation(self.p2_ - self.p1_)
        else:
            R = GeometricUtils.vec3_to_rotation(self.p2_ - self.p1_)

        C = R @ C @ R.T
        center = (self.p1_ + self.p2_) / 2

        ellipsoid = Ellipsoid(C, center)
        obs_inside = ellipsoid.points_inside(self.obs_)

        # Iteratively adjust ellipsoid to avoid obstacles
        while obs_inside:
            pw = ellipsoid.closest_point(obs_inside)
            p = R.T @ (pw - ellipsoid.d())  # Transform to ellipsoid frame

            # Adjust short axes
            if p[0] < axes[0]:
                axes[1] = abs(p[1]) / math.sqrt(1 - (p[0] / axes[0]) ** 2)

            # Update ellipsoid
            new_C = np.diag(axes)
            ellipsoid.C_ = R @ new_C @ R.T

            # Filter remaining obstacles
            obs_new = []
            for pt in obs_inside:
                if 1 - ellipsoid.dist(pt) > EPSILON:
                    obs_new.append(pt)
            obs_inside = obs_new

        self.ellipsoid_ = ellipsoid


class SeedDecomp(DecompBase):
    """Seed point decomposition"""

    def __init__(self, p: Optional[np.ndarray] = None):
        super().__init__()
        self.p_ = p if p is not None else np.zeros(2)
        self.local_bbox_ = np.zeros(len(self.p_))

    def dilate(self, radius: float = 0):
        """Dilate around seed point with sphere"""
        dim = len(self.p_)
        self.ellipsoid_ = Ellipsoid(radius * np.eye(dim), self.p_)
        self.find_polyhedron()
        self.add_local_bbox(self.polyhedron_)

    def get_seed(self) -> np.ndarray:
        return self.p_

    def add_local_bbox(self, polyhedron: Polyhedron):
        """Add bounding box constraints around seed"""
        if np.linalg.norm(self.local_bbox_) == 0:
            return

        dim = len(self.p_)

        # X direction
        if len(self.local_bbox_) > 0:
            dir_x = np.zeros(dim)
            dir_x[0] = 1
            pp1 = self.p_ + dir_x * self.local_bbox_[0]
            pp2 = self.p_ - dir_x * self.local_bbox_[0]
            polyhedron.add(Hyperplane(pp1, dir_x))
            polyhedron.add(Hyperplane(pp2, -dir_x))

        # Y direction
        if len(self.local_bbox_) > 1:
            dir_y = np.zeros(dim)
            dir_y[1] = 1
            pp3 = self.p_ + dir_y * self.local_bbox_[1]
            pp4 = self.p_ - dir_y * self.local_bbox_[1]
            polyhedron.add(Hyperplane(pp3, dir_y))
            polyhedron.add(Hyperplane(pp4, -dir_y))

        # Z direction (3D only)
        if dim == 3 and len(self.local_bbox_) > 2:
            dir_z = np.zeros(dim)
            dir_z[2] = 1
            pp5 = self.p_ + dir_z * self.local_bbox_[2]
            pp6 = self.p_ - dir_z * self.local_bbox_[2]
            polyhedron.add(Hyperplane(pp5, dir_z))
            polyhedron.add(Hyperplane(pp6, -dir_z))


class EllipsoidDecomp:
    """Ellipsoid decomposition for path planning"""

    def __init__(self, origin: Optional[np.ndarray] = None, dim: Optional[np.ndarray] = None):
        if origin is not None and dim is not None:
            self.global_bbox_min_ = origin.copy()
            self.global_bbox_max_ = origin + dim
        else:
            self.global_bbox_min_ = np.zeros(2)
            self.global_bbox_max_ = np.zeros(2)

        self.obs_ = []
        self.local_bbox_ = np.zeros(2)
        self.path_ = []
        self.ellipsoids_ = []
        self.polyhedrons_ = []
        self.lines_ = []

    def set_obs(self, obs: List[np.ndarray]):
        """Set obstacle points"""
        self.obs_ = obs

    def set_local_bbox(self, bbox: np.ndarray):
        """Set local bounding box"""
        self.local_bbox_ = bbox.copy()

    def get_path(self) -> List[np.ndarray]:
        return self.path_

    def get_polyhedrons(self) -> List[Polyhedron]:
        return self.polyhedrons_

    def get_ellipsoids(self) -> List[Ellipsoid]:
        return self.ellipsoids_

    def get_constraints(self) -> List[LinearConstraint]:
        """Get linear constraints for Safe Flight Corridor"""
        constraints = []
        for i in range(len(self.polyhedrons_)):
            pt = (self.path_[i] + self.path_[i + 1]) / 2
            constraint = LinearConstraint(p0=pt, hyperplanes=self.polyhedrons_[i].hyperplanes())
            if np.linalg.norm(self.path_[i] - self.path_[i + 1]) < 1e-4:
                size = len(self.polyhedrons_[i].hyperplanes())
                dim = len(pt)
                A = np.zeros((size, dim))
                b = np.zeros(size)
                for i, hp in enumerate(self.polyhedrons_[i].hyperplanes()):
                    n = hp.n.copy()
                    c = np.dot(hp.p, n)
                    if np.dot(n, pt) - c > 0:
                        n = -n
                    A[i] = n
                    b[i] = 100.0
                LOG_DEBUG("Norm is small so using dummy constraint")
                constraint = LinearConstraint(A, b)
            constraints.append(constraint)
        return constraints

    def set_constraints(self, constraints_out: list, offset: float = 0.0) -> None:
        """
        Set constraints, primarily for compatibility with original code.

        Args:
            constraints_out: Output list to store constraints
            offset: Offset value
        """
        constraints = self.get_constraints()

        # Make sure the output list has enough space
        while len(constraints_out) < len(constraints):
            constraints_out.append(None)

        # Copy constraints to output
        for i, constraint in enumerate(constraints):
            constraints_out[i] = constraint

    def dilate(self, path: List[np.ndarray], offset_x: float = 0):
        """Perform decomposition along path"""
        if len(path) < 2:
            return

        N = len(path) - 1
        self.lines_ = []
        self.ellipsoids_ = []
        self.polyhedrons_ = []

        # Create line segments and decompose
        for i in range(N):

            line = LineSegment(path[i], path[i + 1])
            line.set_local_bbox(self.local_bbox_)
            line.set_obs(self.obs_)
            line.dilate(offset_x)

            self.lines_.append(line)
            self.ellipsoids_.append(line.get_ellipsoid())
            self.polyhedrons_.append(line.get_polyhedron())

        self.path_ = path

        # Add global bounding box if specified
        if (np.linalg.norm(self.global_bbox_min_) != 0 or
                np.linalg.norm(self.global_bbox_max_) != 0):
            for polyhedron in self.polyhedrons_:
                self._add_global_bbox(polyhedron)

    def _add_global_bbox(self, polyhedron: Polyhedron):
        """Add global bounding box constraints"""
        dim = len(self.global_bbox_min_)

        if dim == 2:
            # X bounds
            polyhedron.add(Hyperplane(np.array([self.global_bbox_max_[0], 0]), np.array([1, 0])))
            polyhedron.add(Hyperplane(np.array([self.global_bbox_min_[0], 0]), np.array([-1, 0])))
            # Y bounds
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_max_[1]]), np.array([0, 1])))
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_min_[1]]), np.array([0, -1])))

        elif dim == 3:
            # X bounds
            polyhedron.add(Hyperplane(np.array([self.global_bbox_max_[0], 0, 0]), np.array([1, 0, 0])))
            polyhedron.add(Hyperplane(np.array([self.global_bbox_min_[0], 0, 0]), np.array([-1, 0, 0])))
            # Y bounds
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_max_[1], 0]), np.array([0, 1, 0])))
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_min_[1], 0]), np.array([0, -1, 0])))
            # Z bounds
            polyhedron.add(Hyperplane(np.array([0, 0, self.global_bbox_max_[2]]), np.array([0, 0, 1])))
            polyhedron.add(Hyperplane(np.array([0, 0, self.global_bbox_min_[2]]), np.array([0, 0, -1])))


class IterativeDecomp(EllipsoidDecomp):
    """Iterative decomposition for safer corridors"""

    def dilate_iter(self, path_raw: List[np.ndarray], iter_num: int = 5,
                    res: float = 0, offset_x: float = 0):
        """Iterative decomposition with path simplification"""
        # Downsample if resolution specified
        path = self._downsample(path_raw, res) if res > 0 else path_raw

        # Initial decomposition
        self.dilate(path, offset_x)
        new_path = self._simplify(path)

        # Iterate
        for i in range(iter_num):
            if len(new_path) == len(path):
                break
            path = new_path
            self.dilate(path, offset_x)
            new_path = self._simplify(path)

    def _downsample(self, path: List[np.ndarray], d: float) -> List[np.ndarray]:
        """Uniformly sample path into segments of length d"""
        if len(path) < 2:
            return path

        new_path = []
        for i in range(1, len(path)):
            dist = np.linalg.norm(path[i] - path[i - 1])
            cnt = int(np.ceil(dist / d))
            for j in range(cnt):
                new_path.append(path[i - 1] + j * (path[i] - path[i - 1]) / cnt)

        new_path.append(path[-1])
        return new_path

    def _cal_closest_dist(self, pt: np.ndarray, polyhedron: Polyhedron) -> float:
        """Calculate closest distance to polyhedron"""
        min_dist = float('inf')
        for hp in polyhedron.hyperplanes():
            d = abs(np.dot(hp.n, pt - hp.p))
            if d < min_dist:
                min_dist = d
        return min_dist

    def _simplify(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Remove redundant waypoints"""
        if len(path) <= 2:
            return path

        ref_pt = path[0]
        new_path = [ref_pt]

        for i in range(2, len(path)):
            if (self.polyhedrons_[i - 1].inside(ref_pt) and
                    self._cal_closest_dist(ref_pt, self.polyhedrons_[i - 1]) > 0.1):
                continue
            else:
                ref_pt = path[i - 1]
                new_path.append(ref_pt)

        new_path.append(path[-1])
        return new_path

###### Spline Utils ######
"""
Python implementation of spline utilities based on:
- https://github.com/ttk592/spline (cubic spline library)
- https://github.com/tud-amr/mpc_planner (MPC planner with Spline2D)
- https://github.com/oscardegroot/ros_tools (ROS tools)
"""


class TKSpline:
    """
    Python implementation of cubic spline interpolation (NUMERIC ONLY).
    Based on ttk592/spline library (https://github.com/ttk592/spline).
    
    This implements natural cubic splines with C2 continuity.
    
    IMPORTANT: This class is for NUMERIC evaluation only. For symbolic computation
    (CasADi MX/SX), use Spline2D with parameter dictionaries instead.
    
    Usage:
        - Numeric evaluation: Use TKSpline for post-optimization evaluation
        - Symbolic optimization: Use Spline2D(params, num_segments, s) with CasADi symbols
    """
    
    def __init__(self, x, y, boundary_type='natural'):
        """
        Initialize cubic spline from data points.
        
        Args:
            x: Array of x-coordinates (parameter values, must be strictly increasing)
            y: Array of y-coordinates (function values)
            boundary_type: 'natural' (default), 'clamped', or 'periodic'
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for spline interpolation")
        
        # Check that x is strictly increasing
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        
        self.x = x.copy()
        self.y = y.copy()
        self.n = len(x) - 1  # Number of intervals
        self.boundary_type = boundary_type
        
        # Compute spline coefficients
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        """Compute cubic spline coefficients using tridiagonal matrix algorithm."""
        n = self.n
        h = np.diff(self.x)  # Interval lengths
        delta = np.diff(self.y) / h  # Divided differences
        
        # Set up tridiagonal system: A * m = b
        # where m are second derivatives at knots
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        
        # Interior points
        for i in range(1, n):
            A[i, i - 1] = h[i - 1] / 6.0
            A[i, i] = (h[i - 1] + h[i]) / 3.0
            A[i, i + 1] = h[i] / 6.0
            b[i] = delta[i] - delta[i - 1]
        
        # Boundary conditions
        if self.boundary_type == 'natural':
            # Natural spline: second derivative = 0 at endpoints
            A[0, 0] = 1.0
            A[n, n] = 1.0
            b[0] = 0.0
            b[n] = 0.0
        elif self.boundary_type == 'clamped':
            # Clamped spline: first derivative specified at endpoints
            # For now, use natural as default (can be extended)
            A[0, 0] = 1.0
            A[n, n] = 1.0
            b[0] = 0.0
            b[n] = 0.0
        elif self.boundary_type == 'periodic':
            # Periodic spline: C2 continuity at endpoints
            A[0, 0] = (h[n - 1] + h[0]) / 3.0
            A[0, 1] = h[0] / 6.0
            A[0, n] = h[n - 1] / 6.0
            b[0] = delta[0] - delta[n - 1]
            
            A[n, 0] = h[n - 1] / 6.0
            A[n, n - 1] = h[n - 2] / 6.0
            A[n, n] = (h[n - 2] + h[n - 1]) / 3.0
            b[n] = delta[n - 1] - delta[n - 2]
        
        # Solve tridiagonal system
        self.m = self._solve_tridiagonal(A, b)
        
        # Compute polynomial coefficients for each segment
        # For segment i: S_i(t) = a_i + b_i*t + c_i*t^2 + d_i*t^3
        # where t = (x - x[i]) / h[i] is normalized parameter in [0, 1]
        self.a = self.y[:-1].copy()
        self.b = np.zeros(n)
        self.c = np.zeros(n)
        self.d = np.zeros(n)
        
        for i in range(n):
            self.b[i] = delta[i] - h[i] * (2 * self.m[i] + self.m[i + 1]) / 6.0
            self.c[i] = self.m[i] / 2.0
            self.d[i] = (self.m[i + 1] - self.m[i]) / (6.0 * h[i])
    
    def _solve_tridiagonal(self, A, b):
        """Solve tridiagonal system using Thomas algorithm."""
        n = len(b)
        # Forward elimination
        for i in range(1, n):
            if abs(A[i - 1, i - 1]) < 1e-10:
                raise ValueError("Singular matrix in tridiagonal solve")
            factor = A[i, i - 1] / A[i - 1, i - 1]
            A[i, i] -= factor * A[i - 1, i]
            if i < n - 1:
                A[i, i + 1] -= factor * A[i - 1, i + 1]
            b[i] -= factor * b[i - 1]
        
        # Back substitution
        x = np.zeros(n)
        x[n - 1] = b[n - 1] / A[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (b[i] - A[i, i + 1] * x[i + 1]) / A[i, i]
        
        return x
    
    def _find_segment(self, x_val):
        """Find which segment contains x_val."""
        # Handle boundary cases
        if x_val <= self.x[0]:
            return 0
        if x_val >= self.x[-1]:
            return self.n - 1
        
        # Binary search for efficiency
        left = 0
        right = self.n
        while right - left > 1:
            mid = (left + right) // 2
            if x_val < self.x[mid]:
                right = mid
            else:
                left = mid
        
        return left
    
    def __call__(self, x_val):
        """Evaluate spline at point x_val (callable interface for compatibility)."""
        return self.at(x_val)
    
    def derivative(self, n=1):
        """
        Return derivative function (for compatibility with scipy CubicSpline interface).
        
        Args:
            n: Derivative order (1 or 2)
        
        Returns:
            Callable derivative function
        """
        if n == 1:
            return self.deriv
        elif n == 2:
            return self.deriv2
        else:
            raise ValueError(f"TKSpline only supports derivatives of order 1 or 2, got {n}")
    
    def at(self, x_val):
        """
        Evaluate spline at point x_val.
        Supports both scalar and array inputs.
        
        When x_val is beyond the spline bounds, returns the final point (C++ reference behavior).
        This allows vehicle to progress beyond path length while path evaluation stays valid.
        """
        x_val = np.asarray(x_val)
        scalar_input = x_val.ndim == 0
        x_val = np.atleast_1d(x_val)
        
        result = np.zeros_like(x_val)
        for i, x in enumerate(x_val):
            # Clamp x to valid range for evaluation (C++ reference: return final point beyond bounds)
            if x >= self.x[-1]:
                # Beyond end: return final point
                result[i] = self.y[-1]
            elif x <= self.x[0]:
                # Before start: return first point
                result[i] = self.y[0]
            else:
                # Within bounds: evaluate normally
                seg_idx = self._find_segment(x)
                t = (x - self.x[seg_idx]) / (self.x[seg_idx + 1] - self.x[seg_idx])
                result[i] = (self.a[seg_idx] + 
                            self.b[seg_idx] * t + 
                            self.c[seg_idx] * t * t + 
                            self.d[seg_idx] * t * t * t)
        
        return result[0] if scalar_input else result
    
    def find_closest_s(self, position, num_samples=200):
        """
        Find the closest point on the spline to a given position.
        
        Reference: https://github.com/ttk592/spline - spline parameter represents arc length
        Reference: https://github.com/tud-amr/mpc_planner - spline tracks closest point on path
        
        Args:
            position: 2D position (x, y) as numpy array or list
            num_samples: Number of samples to search (default: 200)
        
        Returns:
            s: Arc length parameter of closest point on spline
            closest_point: (x, y) coordinates of closest point
            distance: Distance from position to closest point
        """
        position = np.asarray(position, dtype=float)
        if position.shape[0] < 2:
            return self.x[0], np.array([self.x[0], self.y[0]]), float('inf')
        
        pos_x, pos_y = float(position[0]), float(position[1])
        
        # Sample spline parameter space (x is the parameter, y is the function value)
        x_min = float(self.x[0])
        x_max = float(self.x[-1])
        x_samples = np.linspace(x_min, x_max, num_samples)
        
        # Evaluate spline at sample points (y = spline(x))
        y_samples = self.at(x_samples)
        
        # Compute distances in (x, y) space
        distances = np.sqrt((x_samples - pos_x)**2 + (y_samples - pos_y)**2)
        
        # Find minimum distance
        min_idx = np.argmin(distances)
        closest_x = float(x_samples[min_idx])
        closest_y = float(y_samples[min_idx])
        closest_point = np.array([closest_x, closest_y])
        min_distance = float(distances[min_idx])
        
        # Refine using binary search around the closest point
        if min_idx > 0 and min_idx < len(x_samples) - 1:
            # Search in the segment around the closest point
            x_left = x_samples[max(0, min_idx - 1)]
            x_right = x_samples[min(len(x_samples) - 1, min_idx + 1)]
            
            # Binary search refinement
            for _ in range(10):  # 10 iterations for refinement
                x_mid = (x_left + x_right) / 2.0
                y_mid = float(self.at(x_mid))
                dist_mid = np.sqrt((x_mid - pos_x)**2 + (y_mid - pos_y)**2)
                
                x_left_mid = (x_left + x_mid) / 2.0
                y_left_mid = float(self.at(x_left_mid))
                dist_left_mid = np.sqrt((x_left_mid - pos_x)**2 + (y_left_mid - pos_y)**2)
                
                x_right_mid = (x_mid + x_right) / 2.0
                y_right_mid = float(self.at(x_right_mid))
                dist_right_mid = np.sqrt((x_right_mid - pos_x)**2 + (y_right_mid - pos_y)**2)
                
                if dist_left_mid < dist_mid and dist_left_mid < dist_right_mid:
                    x_right = x_mid
                    min_distance = dist_left_mid
                    closest_x = x_left_mid
                    closest_y = y_left_mid
                    closest_point = np.array([x_left_mid, y_left_mid])
                elif dist_right_mid < dist_mid:
                    x_left = x_mid
                    min_distance = dist_right_mid
                    closest_x = x_right_mid
                    closest_y = y_right_mid
                    closest_point = np.array([x_right_mid, y_right_mid])
                else:
                    min_distance = dist_mid
                    closest_x = x_mid
                    closest_y = y_mid
                    closest_point = np.array([x_mid, y_mid])
                    break
        
        # Return x as the parameter (s) - for TKSpline, x is the parameter
        return closest_x, closest_point, min_distance
    
    def deriv(self, x_val):
        """
        Evaluate first derivative at point x_val.
        
        When x_val is beyond the spline bounds, returns derivative at final point (C++ reference behavior).
        """
        x_val = np.asarray(x_val)
        scalar_input = x_val.ndim == 0
        x_val = np.atleast_1d(x_val)
        
        result = np.zeros_like(x_val)
        for i, x in enumerate(x_val):
            # Clamp x to valid range for evaluation (C++ reference: return final derivative beyond bounds)
            if x >= self.x[-1]:
                # Beyond end: return derivative at final point
                seg_idx = self.n - 1
                h = self.x[seg_idx + 1] - self.x[seg_idx]
                t = 1.0  # Final point of last segment
                result[i] = (self.b[seg_idx] + 
                            2 * self.c[seg_idx] * t + 
                            3 * self.d[seg_idx] * t * t) / h
            elif x <= self.x[0]:
                # Before start: return derivative at first point
                seg_idx = 0
                h = self.x[seg_idx + 1] - self.x[seg_idx]
                t = 0.0  # First point of first segment
                result[i] = (self.b[seg_idx] + 
                            2 * self.c[seg_idx] * t + 
                            3 * self.d[seg_idx] * t * t) / h
            else:
                # Within bounds: evaluate normally
                seg_idx = self._find_segment(x)
                h = self.x[seg_idx + 1] - self.x[seg_idx]
                t = (x - self.x[seg_idx]) / h
                # Derivative: b + 2*c*t + 3*d*t^2, scaled by 1/h
                result[i] = (self.b[seg_idx] + 
                            2 * self.c[seg_idx] * t + 
                            3 * self.d[seg_idx] * t * t) / h
        
        return result[0] if scalar_input else result
    
    def deriv2(self, x_val):
        """
        Evaluate second derivative at point x_val.
        """
        x_val = np.asarray(x_val)
        scalar_input = x_val.ndim == 0
        x_val = np.atleast_1d(x_val)
        
        result = np.zeros_like(x_val)
        for i, x in enumerate(x_val):
            seg_idx = self._find_segment(x)
            h = self.x[seg_idx + 1] - self.x[seg_idx]
            t = (x - self.x[seg_idx]) / h
            # Second derivative: 2*c + 6*d*t, scaled by 1/h^2
            result[i] = (2 * self.c[seg_idx] + 
                        6 * self.d[seg_idx] * t) / (h * h)
        
        return result[0] if scalar_input else result


class SplineSegment:
    """
    Spline segment class for symbolic computation with parameter dictionaries.
    Used when spline coefficients are provided via parameter dictionary.
    
    Reference: https://github.com/tud-amr/mpc_planner
    The spline uses a cubic polynomial: y = a*t^3 + b*t^2 + c*t + d
    where t is the normalized parameter [0,1] within the segment.
    """

    def __init__(self, param, name, spline_nr, num_segments=None):
        # Retrieve spline values from the parameters (stored as multi parameter by name)
        self.a = param.get(f"{name}_{spline_nr}_a")
        self.b = param.get(f"{name}_{spline_nr}_b")
        self.c = param.get(f"{name}_{spline_nr}_c")
        self.d = param.get(f"{name}_{spline_nr}_d")

        # CRITICAL FIX: s_start uses "path_{spline_nr}_start" NOT "{name}_{spline_nr}_start"
        # because path_0_start is shared between path_x and path_y
        self.s_start = param.get(f"path_{spline_nr}_start")
        
        # Store num_segments to compute segment length
        self.num_segments = num_segments
        
        # Compute segment length from actual segment boundaries
        # Try to get next segment's start to compute segment length
        next_s_start = param.get(f"path_{spline_nr + 1}_start")
        if next_s_start is not None and self.s_start is not None:
            # Segment length is the difference between consecutive segment starts
            import casadi as cd
            if isinstance(next_s_start, (cd.MX, cd.SX)) or isinstance(self.s_start, (cd.MX, cd.SX)):
                self.segment_length = next_s_start - self.s_start
            else:
                self.segment_length = float(next_s_start) - float(self.s_start)
        elif self.num_segments is not None and self.num_segments > 0:
            # Fallback: assume uniform segments if we know num_segments
            # For uniform segments: segment_length = 1.0 / num_segments
            self.segment_length = 1.0 / float(self.num_segments)
        else:
            # Last resort fallback: assume 10 segments
            self.segment_length = 1.0 / 10.0
        
        # Debug: Log if s_start is None
        if self.s_start is None:
            import logging
            logger = logging.getLogger("integration_test")
            logger.warning(f"SplineSegment.__init__: s_start is None for {name}_{spline_nr}! Looking for 'path_{spline_nr}_start'")

    def at(self, spline_index):
        """
        Evaluate spline at normalized parameter spline_index [0,1].
        
        Args:
            spline_index: Normalized path parameter [0,1] (can be CasADi symbolic)
        
        Returns:
            Spline value at spline_index
        """
        # Convert global normalized s [0,1] to segment-local normalized t [0,1]
        # t = (s - s_start) / segment_length
        import casadi as cd
        if isinstance(spline_index, (cd.MX, cd.SX)) or isinstance(self.s_start, (cd.MX, cd.SX)):
            # CasADi symbolic mode
            t = (spline_index - self.s_start) / cd.fmax(self.segment_length, 1e-10)
        else:
            # Numeric mode
            t = (spline_index - self.s_start) / max(self.segment_length, 1e-10)
        
        # Evaluate cubic polynomial: y = a*t^3 + b*t^2 + c*t + d
        return self.a * t * t * t + self.b * t * t + self.c * t + self.d

    def deriv(self, spline_index):
        """
        Evaluate first derivative at normalized parameter spline_index [0,1].
        
        Args:
            spline_index: Normalized path parameter [0,1] (can be CasADi symbolic)
        
        Returns:
            First derivative value at spline_index
        """
        # Convert global normalized s [0,1] to segment-local normalized t [0,1]
        import casadi as cd
        if isinstance(spline_index, (cd.MX, cd.SX)) or isinstance(self.s_start, (cd.MX, cd.SX)):
            # CasADi symbolic mode
            t = (spline_index - self.s_start) / cd.fmax(self.segment_length, 1e-10)
            # Derivative: y' = 3*a*t^2 + 2*b*t + c
            # But we need to account for dt/ds = 1/segment_length
            dt_ds = 1.0 / cd.fmax(self.segment_length, 1e-10)
        else:
            # Numeric mode
            t = (spline_index - self.s_start) / max(self.segment_length, 1e-10)
            dt_ds = 1.0 / max(self.segment_length, 1e-10)
        
        # Derivative with respect to s: dy/ds = dy/dt * dt/ds
        # dy/dt = 3*a*t^2 + 2*b*t + c
        dy_dt = 3 * self.a * t * t + 2 * self.b * t + self.c
        return dy_dt * dt_ds

    def deriv2(self, spline_index):
        """
        Evaluate second derivative at normalized parameter spline_index [0,1].
        
        Args:
            spline_index: Normalized path parameter [0,1] (can be CasADi symbolic)
        
        Returns:
            Second derivative value at spline_index
        """
        # Convert global normalized s [0,1] to segment-local normalized t [0,1]
        import casadi as cd
        if isinstance(spline_index, (cd.MX, cd.SX)) or isinstance(self.s_start, (cd.MX, cd.SX)):
            # CasADi symbolic mode
            t = (spline_index - self.s_start) / cd.fmax(self.segment_length, 1e-10)
            dt_ds = 1.0 / cd.fmax(self.segment_length, 1e-10)
        else:
            # Numeric mode
            t = (spline_index - self.s_start) / max(self.segment_length, 1e-10)
            dt_ds = 1.0 / max(self.segment_length, 1e-10)
        
        # Second derivative with respect to s: d2y/ds2 = d2y/dt2 * (dt/ds)^2
        # d2y/dt2 = 6*a*t + 2*b
        d2y_dt2 = 6 * self.a * t + 2 * self.b
        return d2y_dt2 * dt_ds * dt_ds


class Spline:
    """
    Multi-segment spline with sigmoid blending between segments.
    
    This class is used for SYMBOLIC computation (CasADi MX/SX) during optimization.
    It works with parameter dictionaries that contain spline coefficients.
    
    For numeric evaluation, use TKSpline directly.
    
    Reference: https://github.com/tud-amr/mpc_planner
    """
    
    def __init__(self, params, name, num_segments, s):
        """
        Initialize spline from parameter dictionary.
        
        Args:
            params: Parameter dictionary containing spline coefficients
            name: Parameter name prefix (e.g., "path_x")
            num_segments: Number of spline segments
            s: Current s value for sigmoid blending (can be CasADi symbolic)
        """
        self.splines = []  # Classes containing the splines
        self.lambdas = []  # Merges splines
        self.num_segments = num_segments
        for i in range(num_segments):
            self.splines.append(SplineSegment(params, f"{name}", i, num_segments=num_segments))

            # No lambda for the first segment (it is not glued to anything prior)
            if i > 0 and s is not None:
                # Sigmoid blending: smoothly transition between segments
                # Reference: https://github.com/tud-amr/mpc_planner
                import casadi as cd
                if isinstance(s, (cd.MX, cd.SX)):
                    # CasADi symbolic mode
                    self.lambdas.append(1.0 / (1.0 + cd.exp((s - self.splines[-1].s_start + 0.02) / 0.1)))
                else:
                    # Numeric mode
                    self.lambdas.append(1.0 / (1.0 + np.exp((s - self.splines[-1].s_start + 0.02) / 0.1)))

    def at(self, s):
        """
        Evaluate spline at parameter s.
        Supports both numeric and CasADi symbolic s.
        """
        # Iteratively glue segments together
        value = self.splines[-1].at(s)
        for k in range(len(self.splines) - 1, 0, -1):
            value = self.lambdas[k - 1] * self.splines[k - 1].at(s) + (1.0 - self.lambdas[k - 1]) * value
        return value

    def deriv(self, s):
        """
        Evaluate first derivative at parameter s.
        Supports both numeric and CasADi symbolic s.
        """
        value = self.splines[-1].deriv(s)
        for k in range(len(self.splines) - 1, 0, -1):
            value = self.lambdas[k - 1] * self.splines[k - 1].deriv(s) + (1.0 - self.lambdas[k - 1]) * value
        return value

    def deriv2(self, s):
        """
        Evaluate second derivative at parameter s.
        Supports both numeric and CasADi symbolic s.
        """
        value = self.splines[-1].deriv2(s)
        for k in range(len(self.splines) - 1, 0, -1):
            value = self.lambdas[k - 1] * self.splines[k - 1].deriv2(s) + (1.0 - self.lambdas[k - 1]) * value
        return value


class Spline2D:
    """
    2D spline combining x and y splines for SYMBOLIC computation.
    Based on tud-amr/mpc_planner implementation.
    
    This class is designed for use with CasADi symbolic variables (MX/SX) during
    optimization. It uses parameter dictionaries containing spline coefficients.
    
    Usage:
        - Symbolic optimization: Spline2D(params, num_segments, s) where s is CasADi symbolic
        - Numeric evaluation: Use TKSpline directly or create separate numeric splines
    
    Reference: https://github.com/tud-amr/mpc_planner
    """

    def __init__(self, params, num_segments, s):
        """
        Initialize 2D spline from parameter dictionary.
        
        Args:
            params: Parameter dictionary containing spline coefficients
                   (path_x_{i}_a, path_x_{i}_b, path_x_{i}_c, path_x_{i}_d,
                    path_y_{i}_a, path_y_{i}_b, path_y_{i}_c, path_y_{i}_d,
                    path_{i}_start for each segment i)
            num_segments: Number of spline segments
            s: Current s value (can be CasADi symbolic MX/SX for optimization)
        """
        self.spline_x = Spline(params, "path_x", num_segments, s)
        self.spline_y = Spline(params, "path_y", num_segments, s)

    def at(self, s):
        """Evaluate spline at parameter s, returns (x, y)."""
        return self.spline_x.at(s), self.spline_y.at(s)

    def deriv(self, s):
        """Evaluate first derivative at parameter s, returns (dx, dy)."""
        return self.spline_x.deriv(s), self.spline_y.deriv(s)

    def deriv_normalized(self, s):
        """
        Evaluate normalized first derivative (unit tangent vector).
        Returns (dx_norm, dy_norm) where ||(dx_norm, dy_norm)|| = 1.
        """
        dx = self.spline_x.deriv(s)
        dy = self.spline_y.deriv(s)
        
        # Handle both CasADi symbolic and numeric types
        if isinstance(dx, (cd.MX, cd.SX)) or isinstance(dy, (cd.MX, cd.SX)):
            path_norm = cd.sqrt(dx * dx + dy * dy)
            path_norm = cd.fmax(path_norm, 1e-6)  # Prevent division by zero
            return dx / path_norm, dy / path_norm
        else:
            path_norm = np.sqrt(dx * dx + dy * dy)
            path_norm = max(path_norm, 1e-6)  # Prevent division by zero
            return dx / path_norm, dy / path_norm

    def deriv2(self, s):
        """Evaluate second derivative at parameter s, returns (d2x, d2y)."""
        return self.spline_x.deriv2(s), self.spline_y.deriv2(s)

    def get_curvature(self, s):
        """
        Compute path curvature using the standard formula:
        curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        
        Reference: https://github.com/tud-amr/mpc_planner
        """
        # First derivatives
        path_x_deriv = self.spline_x.deriv(s)
        path_y_deriv = self.spline_y.deriv(s)
        
        # Second derivatives
        path_x_deriv2 = self.spline_x.deriv2(s)
        path_y_deriv2 = self.spline_y.deriv2(s)
        
        # Handle both CasADi symbolic and numeric types
        if isinstance(path_x_deriv, (cd.MX, cd.SX)) or isinstance(path_y_deriv, (cd.MX, cd.SX)):
            # CasADi symbolic mode
            # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = cd.fabs(path_x_deriv * path_y_deriv2 - path_y_deriv * path_x_deriv2)
            denominator = cd.sqrt((path_x_deriv * path_x_deriv + path_y_deriv * path_y_deriv) ** 3)
            
            # Add small epsilon to denominator to prevent division by zero
            denominator = cd.fmax(denominator, 1e-10)
            
            curvature = numerator / denominator
            
            # Clamp curvature to reasonable bounds
            curvature = cd.fmax(curvature, 1e-5)  # Minimum curvature (maximum radius)
            curvature = cd.fmin(curvature, 10.0)   # Maximum curvature (minimum radius)
        else:
            # Numeric mode
            numerator = abs(path_x_deriv * path_y_deriv2 - path_y_deriv * path_x_deriv2)
            denominator = (path_x_deriv * path_x_deriv + path_y_deriv * path_y_deriv) ** 1.5
            
            # Add small epsilon to denominator to prevent division by zero
            denominator = max(denominator, 1e-10)
            
            curvature = numerator / denominator
            
            # Clamp curvature to reasonable bounds
            curvature = max(curvature, 1e-5)  # Minimum curvature (maximum radius)
            curvature = min(curvature, 10.0)   # Maximum curvature (minimum radius)
        
        return curvature


class Spline3D:
    """
    3D spline combining x, y, and z splines for SYMBOLIC computation.
    Based on tud-amr/mpc_planner implementation, extended for 3D paths.
    
    This class is designed for use with CasADi symbolic variables (MX/SX) during
    optimization. It uses parameter dictionaries containing spline coefficients.
    
    Usage:
        - Symbolic optimization: Spline3D(params, num_segments, s) where s is CasADi symbolic
        - Numeric evaluation: Use TKSpline directly or create separate numeric splines
    
    Reference: https://github.com/tud-amr/mpc_planner (extended to 3D)
    """

    def __init__(self, params, num_segments, s):
        """
        Initialize 3D spline from parameter dictionary.
        
        Args:
            params: Parameter dictionary containing spline coefficients
                   (path_x_{i}_a, path_x_{i}_b, path_x_{i}_c, path_x_{i}_d,
                    path_y_{i}_a, path_y_{i}_b, path_y_{i}_c, path_y_{i}_d,
                    path_z_{i}_a, path_z_{i}_b, path_z_{i}_c, path_z_{i}_d,
                    path_{i}_start for each segment i)
            num_segments: Number of spline segments
            s: Current s value (can be CasADi symbolic MX/SX for optimization)
        """
        self.spline_x = Spline(params, "path_x", num_segments, s)
        self.spline_y = Spline(params, "path_y", num_segments, s)
        self.spline_z = Spline(params, "path_z", num_segments, s)

    def at(self, s):
        """Evaluate spline at parameter s, returns (x, y, z)."""
        return self.spline_x.at(s), self.spline_y.at(s), self.spline_z.at(s)

    def deriv(self, s):
        """Evaluate first derivative at parameter s, returns (dx, dy, dz)."""
        return self.spline_x.deriv(s), self.spline_y.deriv(s), self.spline_z.deriv(s)

    def deriv_normalized(self, s):
        """
        Evaluate normalized first derivative (unit tangent vector).
        Returns (dx_norm, dy_norm, dz_norm) where ||(dx_norm, dy_norm, dz_norm)|| = 1.
        """
        dx = self.spline_x.deriv(s)
        dy = self.spline_y.deriv(s)
        dz = self.spline_z.deriv(s)
        
        # Handle both CasADi symbolic and numeric types
        if isinstance(dx, (cd.MX, cd.SX)) or isinstance(dy, (cd.MX, cd.SX)) or isinstance(dz, (cd.MX, cd.SX)):
            path_norm = cd.sqrt(dx * dx + dy * dy + dz * dz)
            path_norm = cd.fmax(path_norm, 1e-6)  # Prevent division by zero
            return dx / path_norm, dy / path_norm, dz / path_norm
        else:
            path_norm = np.sqrt(dx * dx + dy * dy + dz * dz)
            path_norm = max(path_norm, 1e-6)  # Prevent division by zero
            return dx / path_norm, dy / path_norm, dz / path_norm

    def deriv2(self, s):
        """Evaluate second derivative at parameter s, returns (d2x, d2y, d2z)."""
        return self.spline_x.deriv2(s), self.spline_y.deriv2(s), self.spline_z.deriv2(s)

    def get_curvature(self, s):
        """
        Compute 3D path curvature using the formula:
        curvature = ||r' × r''|| / ||r'||^3
        
        where r' = (x', y', z') and r'' = (x'', y'', z'')
        
        Reference: https://github.com/tud-amr/mpc_planner (extended to 3D)
        """
        # First derivatives
        path_x_deriv = self.spline_x.deriv(s)
        path_y_deriv = self.spline_y.deriv(s)
        path_z_deriv = self.spline_z.deriv(s)
        
        # Second derivatives
        path_x_deriv2 = self.spline_x.deriv2(s)
        path_y_deriv2 = self.spline_y.deriv2(s)
        path_z_deriv2 = self.spline_z.deriv2(s)
        
        # Handle both CasADi symbolic and numeric types
        if isinstance(path_x_deriv, (cd.MX, cd.SX)) or isinstance(path_y_deriv, (cd.MX, cd.SX)) or isinstance(path_z_deriv, (cd.MX, cd.SX)):
            # CasADi symbolic mode
            # Cross product: r' × r''
            cross_x = path_y_deriv * path_z_deriv2 - path_z_deriv * path_y_deriv2
            cross_y = path_z_deriv * path_x_deriv2 - path_x_deriv * path_z_deriv2
            cross_z = path_x_deriv * path_y_deriv2 - path_y_deriv * path_x_deriv2
            
            # Magnitude of cross product
            cross_mag = cd.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
            
            # Magnitude of first derivative
            deriv_mag = cd.sqrt(path_x_deriv * path_x_deriv + path_y_deriv * path_y_deriv + path_z_deriv * path_z_deriv)
            deriv_mag_cubed = deriv_mag ** 3
            
            # Add small epsilon to denominator to prevent division by zero
            deriv_mag_cubed = cd.fmax(deriv_mag_cubed, 1e-10)
            
            curvature = cross_mag / deriv_mag_cubed
            
            # Clamp curvature to reasonable bounds
            curvature = cd.fmax(curvature, 1e-5)  # Minimum curvature (maximum radius)
            curvature = cd.fmin(curvature, 10.0)   # Maximum curvature (minimum radius)
        else:
            # Numeric mode
            # Cross product: r' × r''
            cross_x = path_y_deriv * path_z_deriv2 - path_z_deriv * path_y_deriv2
            cross_y = path_z_deriv * path_x_deriv2 - path_x_deriv * path_z_deriv2
            cross_z = path_x_deriv * path_y_deriv2 - path_y_deriv * path_x_deriv2
            
            # Magnitude of cross product
            cross_mag = np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
            
            # Magnitude of first derivative
            deriv_mag = np.sqrt(path_x_deriv * path_x_deriv + path_y_deriv * path_y_deriv + path_z_deriv * path_z_deriv)
            deriv_mag_cubed = deriv_mag ** 3
            
            # Add small epsilon to denominator to prevent division by zero
            deriv_mag_cubed = max(deriv_mag_cubed, 1e-10)
            
            curvature = cross_mag / deriv_mag_cubed
            
            # Clamp curvature to reasonable bounds
            curvature = max(curvature, 1e-5)  # Minimum curvature (maximum radius)
            curvature = min(curvature, 10.0)   # Maximum curvature (minimum radius)
        
        return curvature


def create_numeric_spline2d(s_vals, x_vals, y_vals):
    """
    Helper function to create numeric 2D splines from path data.
    Uses TKSpline for numeric evaluation (post-optimization).
    
    IMPORTANT: This is for NUMERIC evaluation only. For symbolic optimization,
    use Spline2D(params, num_segments, s) with parameter dictionaries.
    
    Args:
        s_vals: Array of parameter values (arc length or normalized [0,1])
        x_vals: Array of x-coordinates
        y_vals: Array of y-coordinates
    
    Returns:
        Tuple of (TKSpline for x, TKSpline for y)
    """
    return TKSpline(s_vals, x_vals), TKSpline(s_vals, y_vals)


def create_numeric_spline3d(s_vals, x_vals, y_vals, z_vals):
    """
    Helper function to create numeric 3D splines from path data.
    Uses TKSpline for numeric evaluation (post-optimization).
    
    IMPORTANT: This is for NUMERIC evaluation only. For symbolic optimization,
    use Spline3D(params, num_segments, s) with parameter dictionaries.
    
    Args:
        s_vals: Array of parameter values (arc length or normalized [0,1])
        x_vals: Array of x-coordinates
        y_vals: Array of y-coordinates
        z_vals: Array of z-coordinates
    
    Returns:
        Tuple of (TKSpline for x, TKSpline for y, TKSpline for z)
    """
    return TKSpline(s_vals, x_vals), TKSpline(s_vals, y_vals), TKSpline(s_vals, z_vals)


def evaluate_path_at(s_val, reference_path, symbolic=False):
    """
    Evaluate reference path at parameter s_val.
    Automatically uses the appropriate spline type based on context.
    
    Args:
        s_val: Parameter value (arc length or normalized [0,1])
        reference_path: ReferencePath object with x_spline and y_spline
        symbolic: If True, expects s_val to be CasADi symbolic and returns symbolic expressions
    
    Returns:
        Tuple of (x, y) coordinates
    """
    import casadi as cd
    
    if symbolic or isinstance(s_val, (cd.MX, cd.SX)):
        # Symbolic evaluation: should use Spline2D with parameter dictionaries
        # This function is for numeric evaluation only
        raise ValueError("For symbolic evaluation, use Spline2D(params, num_segments, s) with parameter dictionaries")
    
    # Numeric evaluation: use TKSpline
    x = reference_path.x_spline.at(float(s_val))
    y = reference_path.y_spline.at(float(s_val))
    return x, y


def evaluate_path_derivative_at(s_val, reference_path, order=1, symbolic=False):
    """
    Evaluate reference path derivative at parameter s_val.
    Automatically uses the appropriate spline type based on context.
    
    Args:
        s_val: Parameter value (arc length or normalized [0,1])
        reference_path: ReferencePath object with x_spline and y_spline
        order: Derivative order (1 or 2)
        symbolic: If True, expects s_val to be CasADi symbolic and returns symbolic expressions
    
    Returns:
        Tuple of (dx, dy) derivatives
    """
    import casadi as cd
    
    if symbolic or isinstance(s_val, (cd.MX, cd.SX)):
        # Symbolic evaluation: should use Spline2D with parameter dictionaries
        raise ValueError("For symbolic evaluation, use Spline2D(params, num_segments, s).deriv(s)")
    
    # Numeric evaluation: use TKSpline
    if order == 1:
        dx = reference_path.x_spline.deriv(float(s_val))
        dy = reference_path.y_spline.deriv(float(s_val))
    elif order == 2:
        dx = reference_path.x_spline.deriv2(float(s_val))
        dy = reference_path.y_spline.deriv2(float(s_val))
    else:
        raise ValueError(f"Derivative order must be 1 or 2, got {order}")
    
    return dx, dy


class DouglasRachford:
    """
    Python implementation of the Douglas-Rachford projection method
    for collision avoidance in motion planning.
    """

    def douglas_rachford_projection(self, position, delta, anchor, r, starting_pose):
        """
        Performs Douglas-Rachford projection.

        Args:
            position (np.ndarray): Point to project (modified in-place)
            delta (np.ndarray): Obstacle position
            anchor (np.ndarray): Anchor position (reference obstacle)
            r (float): Collision radius
            starting_pose (np.ndarray): Starting pose for projection
        """
        position[:] = (position + self._reflect(self._reflect(position, anchor, r, position), delta, r, starting_pose)) / 2.0

    def _project(self, position, delta, radius, starting_pose):
        """
        Projects point `position` onto a circle of radius `radius` centered at `delta`,
        in the direction from `starting_pose` to `delta`, if it's within the radius.
        """
        diff = position - delta
        dist = np.linalg.norm(diff)

        if dist < radius:
            direction = (delta - starting_pose)
            norm_dir = np.linalg.norm(direction)
            if norm_dir == 0:
                return position.copy()  # Avoid division by zero
            return delta - direction / norm_dir * radius
        else:
            return position.copy()

    def _reflect(self, position, delta, r, starting_pose):
        """
        Reflects point p across the projection onto the collision-free region.

        Args:
            position (np.ndarray): Point to reflect
            delta (np.ndarray): Obstacle center
            r (float): Collision radius
            starting_pose (np.ndarray): Starting pose for projection

        Returns:
            np.ndarray: Reflected point
        """
        return 2.0 * self._project(position, delta, r, starting_pose) - position

###### Probability Utils ######

class RandomGenerator:
    """Python equivalent of RosTools::RandomGenerator."""

    def __init__(self, seed: int = -1):
        if seed == -1:
            self.rng_double = np.random.default_rng()
            self.rng_int = np.random.default_rng()
            self.rng_gaussian = np.random.default_rng()
        else:
            self.rng_double = np.random.default_rng(seed)
            self.rng_int = np.random.default_rng(seed)
            self.rng_gaussian = np.random.default_rng(seed)

        self.epsilon = np.finfo(float).eps

    def Double(self) -> float:
        """Uniform [0, 1) double."""
        return self.rng_double.uniform(0.0, 1.0)

    def Int(self, max_val: int) -> int:
        """Uniform integer in [0, max_val]."""
        return self.rng_int.integers(0, max_val + 1)

    def Gaussian(self, mean: float, stddev: float) -> float:
        """Normal distribution sample."""
        return self.rng_gaussian.normal(mean, stddev)

    @staticmethod
    def uniform_to_gaussian_two_dim(uniform_variables: np.ndarray):
        """
        Convert a pair of uniform(0,1) samples into Gaussian(0,1) samples
        using Box–Muller transform.
        """
        u1 = uniform_variables[0]
        u2 = uniform_variables[1]

        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2

        uniform_variables[0] = r * np.cos(theta)
        uniform_variables[1] = r * np.sin(theta)

    @staticmethod
    def uniform_to_gaussian_two_dim_array(u):
        u1 = u[..., 0]
        u2 = u[..., 1]
        r = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2
        u[..., 0] = r * np.cos(theta)
        u[..., 1] = r * np.sin(theta)
        return u

    def BivariateGaussian(self,
                          mean: np.ndarray,
                          major_axis: float,
                          minor_axis: float,
                          angle: float) -> np.ndarray:
        """
        Sample from a 2D Gaussian with given mean, axes, and rotation.
        mean: np.array([mx, my])
        major_axis, minor_axis: stddevs along principal axes
        angle: rotation (radians)
        """

        # Rotation matrix from heading
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        # Generate uniform random numbers in 2D
        u1 = 0.0
        while u1 <= self.epsilon:
            u1 = self.rng_gaussian.uniform(0.0, 1.0)
        u2 = self.rng_gaussian.uniform(0.0, 1.0)

        uniform_samples = np.array([u1, u2], dtype=float)

        # Convert to Gaussian(0,1)
        self.uniformToGaussian2D(uniform_samples)

        # Construct covariance in principal axis form
        SVD = np.array([
            [major_axis ** 2, 0.0],
            [0.0, minor_axis ** 2]
        ])

        # Sigma = R * SVD * R^T
        Sigma = R @ SVD @ R.T

        # Cholesky decomposition (lower-triangular)
        A = np.linalg.cholesky(Sigma)

        # Transform standard normal to desired covariance + mean
        return A @ uniform_samples + mean


def define_robot_area(length: float, width: float, n_discs: int):
    LOG_INFO(f"math_tools.define_robot_area: length={length}, width={width}, n_discs={n_discs}")
    """Define robot area with discs. Implementation moved to planning.types."""
    from planning.types import define_robot_area as _define_robot_area
    return _define_robot_area(length, width, n_discs)


