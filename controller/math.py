import numpy as np

def wedge(x):
            """Return wedged vector."""
            wedge_x = np.array([[0,-x[2][0], x[1][0]], [x[2][0], 0, -x[0][0]], [-x[1][0], x[0][0], 0]])
            return wedge_x
        
        
def deriv_unit_vector(q, q_dot, q_ddot):
    """derivative of a unit vector"""
    nq = np.linalg.norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * np.dot(np.ravel(q), np.ravel(q_dot)) / nq**3
    u_ddot = q_ddot / nq - q_dot / (nq**3) * (2 * np.dot(np.ravel(q), np.ravel(q_dot))) \
    - q / nq**3 * (np.dot(np.ravel(q_dot), np.ravel(q_dot)) + np.dot(np.ravel(q), np.ravel(q_ddot))) \
    + 3 * q / nq**5 * np.dot(np.ravel(q), np.ravel(q_dot))**2
    return u, u_dot, u_ddot

def normalize(x):
    """Return normalized vector."""
    return x / np.linalg.norm(x)


def vee(R):
    """Convert skew-symmetric matrix to vector"""
    return np.array([R[2,1] - R[1,2], 
                    R[0,2] - R[2,0], 
                    R[1,0] - R[0,1]]) / 2

def skew(v):
    """Convert vector to skew symmetric matrix"""
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])