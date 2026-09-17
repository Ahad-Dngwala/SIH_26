package org.sih26.deadreckoning.fusion

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Dense linear algebra for the 7-state UKF, in plain Kotlin.
 *
 * The stream 2 handoff recommended EJML rather than hand-rolling this, for a good
 * reason: a subtly wrong Cholesky does not crash, it produces plausible trajectories
 * that are quietly wrong. That advice is taken seriously here and answered a
 * different way rather than ignored. This file is hand-rolled, and the mitigation is
 * that the whole filter is replayed against tools/parity/fixture_ukf_reference.json
 * on the JVM, 600 cycles, before it ever reaches a phone. The risk EJML removes is
 * the risk of an unverified port; the fixture removes the same risk, and it also
 * catches errors EJML never could, such as getting filterpy's sigma-point convention
 * backwards.
 *
 * Two concrete reasons to prefer this to the dependency here:
 *   1. Everything the filter needs is a 7x7 Cholesky, a symmetric eigendecomposition
 *      and inverses of 1x1, 2x2 and 4x4 innovation matrices. That is a small, fully
 *      testable surface.
 *   2. The parity harness runs with kotlinc alone, no Gradle and no artifact
 *      resolution, so the gate can be run anywhere by anyone in seconds.
 *
 * If this ever grows past the operations below, switch to EJML and keep the fixture.
 *
 * Convention throughout: a matrix is Array(rows) of DoubleArray(cols), row-major.
 */
object LinAlg {

    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    fun diag(values: DoubleArray): Array<DoubleArray> =
        Array(values.size) { i -> DoubleArray(values.size) { j -> if (i == j) values[i] else 0.0 } }

    fun copy(a: Array<DoubleArray>): Array<DoubleArray> = Array(a.size) { a[it].copyOf() }

    fun transpose(a: Array<DoubleArray>): Array<DoubleArray> {
        val rows = a.size
        val cols = a[0].size
        return Array(cols) { i -> DoubleArray(rows) { j -> a[j][i] } }
    }

    fun matMul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size
        val k = b.size
        val m = b[0].size
        require(a[0].size == k) { "shape mismatch: ${a.size}x${a[0].size} times ${b.size}x${b[0].size}" }
        val out = zeros(n, m)
        for (i in 0 until n) {
            for (p in 0 until k) {
                val aip = a[i][p]
                if (aip == 0.0) continue
                for (j in 0 until m) out[i][j] += aip * b[p][j]
            }
        }
        return out
    }

    fun matVec(a: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        val out = DoubleArray(a.size)
        for (i in a.indices) {
            var acc = 0.0
            for (j in v.indices) acc += a[i][j] * v[j]
            out[i] = acc
        }
        return out
    }

    fun add(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(a[0].size) { j -> a[i][j] + b[i][j] } }

    fun subtract(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(a[0].size) { j -> a[i][j] - b[i][j] } }

    fun scale(a: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(a[0].size) { j -> a[i][j] * s } }

    fun symmetrize(a: Array<DoubleArray>): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(a.size) { j -> (a[i][j] + a[j][i]) / 2.0 } }

    fun addVec(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] + b[it] }

    fun subVec(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] - b[it] }

    fun norm(v: DoubleArray): Double {
        var acc = 0.0
        for (x in v) acc += x * x
        return sqrt(acc)
    }

    fun dot(a: DoubleArray, b: DoubleArray): Double {
        var acc = 0.0
        for (i in a.indices) acc += a[i] * b[i]
        return acc
    }

    /**
     * Lower-triangular Cholesky factor L with A = L * L^T.
     *
     * Note for anyone comparing against the Python reference: filterpy's sigma-point
     * generator uses scipy.linalg.cholesky, which returns the UPPER factor U with
     * A = U^T * U, and then takes the ROWS of U. Row k of U is column k of L, so the
     * sigma-point code below reads columns of this L. Getting that transpose wrong
     * produces a filter that still runs and is wrong in a way no unit test of
     * Cholesky itself would catch, which is exactly why the parity fixture exists.
     */
    fun choleskyLower(a: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size
        val l = zeros(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var sum = a[i][j]
                for (k in 0 until j) sum -= l[i][k] * l[j][k]
                if (i == j) {
                    if (sum <= 0.0) {
                        throw IllegalStateException(
                            "covariance is not positive definite at pivot $i (value $sum). " +
                                "The eigenvalue floor in DualChannelUkf.symmetrizeP should have " +
                                "prevented this; if it did not, the filter has diverged rather " +
                                "than hit a rounding problem."
                        )
                    }
                    l[i][j] = sqrt(sum)
                } else {
                    l[i][j] = sum / l[j][j]
                }
            }
        }
        return l
    }

    /** Gauss-Jordan inverse with partial pivoting. Used only on 1x1, 2x2 and 4x4
     * innovation covariances, so the cubic cost is irrelevant and clarity wins. */
    fun inverse(a: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size
        val work = copy(a)
        val inv = identity(n)

        for (col in 0 until n) {
            var pivotRow = col
            for (row in col until n) {
                if (abs(work[row][col]) > abs(work[pivotRow][col])) pivotRow = row
            }
            if (abs(work[pivotRow][col]) < 1e-300) {
                throw IllegalStateException("singular matrix at column $col")
            }
            if (pivotRow != col) {
                val tmp = work[pivotRow]; work[pivotRow] = work[col]; work[col] = tmp
                val tmpInv = inv[pivotRow]; inv[pivotRow] = inv[col]; inv[col] = tmpInv
            }

            val pivot = work[col][col]
            for (j in 0 until n) {
                work[col][j] /= pivot
                inv[col][j] /= pivot
            }
            for (row in 0 until n) {
                if (row == col) continue
                val factor = work[row][col]
                if (factor == 0.0) continue
                for (j in 0 until n) {
                    work[row][j] -= factor * work[col][j]
                    inv[row][j] -= factor * inv[col][j]
                }
            }
        }
        return inv
    }

    /** Eigenvalues and eigenvectors of a symmetric matrix, by cyclic Jacobi rotations.
     *
     * Returns values and vectors where vectors[i][k] is component i of eigenvector k,
     * so the reconstruction is V * diag(values) * V^T. Jacobi is chosen over anything
     * cleverer because it is short, unconditionally stable on symmetric input, and
     * accurate on the small well-scaled matrices this filter produces. It does not
     * sort its output; nothing here depends on eigenvalue order, since the only use
     * is flooring the values and rebuilding, which is order invariant. */
    fun jacobiEigenSymmetric(
        input: Array<DoubleArray>,
        maxSweeps: Int = 100
    ): Pair<DoubleArray, Array<DoubleArray>> {
        val n = input.size
        val a = symmetrize(input)
        val v = identity(n)

        for (sweep in 0 until maxSweeps) {
            var offDiagonal = 0.0
            for (p in 0 until n - 1) for (q in p + 1 until n) offDiagonal += a[p][q] * a[p][q]
            if (offDiagonal < 1e-300) break

            for (p in 0 until n - 1) {
                for (q in p + 1 until n) {
                    if (abs(a[p][q]) < 1e-300) continue

                    // Rotation angle that zeroes the (p, q) entry. Written via hypot
                    // rather than a tangent so it stays well behaved when the
                    // diagonal entries are nearly equal.
                    val theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                    val t = if (theta >= 0.0) {
                        1.0 / (theta + hypot(1.0, theta))
                    } else {
                        -1.0 / (-theta + hypot(1.0, theta))
                    }
                    val c = 1.0 / hypot(1.0, t)
                    val s = t * c

                    for (k in 0 until n) {
                        val akp = a[k][p]
                        val akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    }
                    for (k in 0 until n) {
                        val apk = a[p][k]
                        val aqk = a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    }
                    for (k in 0 until n) {
                        val vkp = v[k][p]
                        val vkq = v[k][q]
                        v[k][p] = c * vkp - s * vkq
                        v[k][q] = s * vkp + c * vkq
                    }
                }
            }
        }

        val values = DoubleArray(n) { a[it][it] }
        return Pair(values, v)
    }
}
